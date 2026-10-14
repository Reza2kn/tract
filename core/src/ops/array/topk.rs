use std::cmp::Ordering;

use tract_data::itertools::Itertools;
use tract_ndarray::{ArrayViewMutD, Axis, Dimension};

use crate::internal::*;

#[derive(Debug, Clone, new, Default, Hash, PartialEq, Eq)]
pub struct Topk {
    pub axis: usize,
    pub largest: bool,
    pub fallback_k: TDim,
}

impl Op for Topk {
    fn mints_runtime_symbols(&self) -> bool {
        true
    }

    fn name(&self) -> StaticName {
        "Topk".into()
    }

    op_as_typed_op!();
}

impl EvalOp for Topk {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (input, k) = args_2!(inputs);
        let mut output_shape: TVec<usize> = input.shape().into();
        let k = k
            .cast_to::<TDim>()?
            .try_as_plain_ram()?
            .to_scalar::<TDim>()?
            .eval_to_i64(ctx.symbols)?;
        ensure!(k >= 0 && k as usize <= input.shape()[self.axis], "TopK count exceeds its axis");
        let k = k as usize;
        output_shape[self.axis] = k;
        let dt = input.datum_type();
        if dt == TDim::datum_type() {
            if input.shape()[self.axis] == 1 && k == 1 {
                return Ok(tvec!(input, Tensor::zero::<i64>(&output_shape)?.into_tvalue()));
            }
            let values = input
                .try_as_plain_ram()?
                .as_slice::<TDim>()?
                .iter()
                .map(|d| d.eval_to_i64(ctx.symbols))
                .collect::<TractResult<Vec<_>>>()?;
            let concrete = Tensor::from_shape(input.shape(), &values)?.into_tvalue();
            let mut result = self.eval(ctx, tvec!(concrete, tensor0(k as i64).into_tvalue()))?;
            result[0] = result[0].cast_to::<TDim>()?.into_owned().into_tvalue();
            return Ok(result);
        }
        let mut output_values = Tensor::zero_dt(dt, &output_shape)?;
        let mut output_indices = Tensor::zero::<i64>(&output_shape)?;
        let mut iterating_shape = output_shape.clone();
        iterating_shape[self.axis] = 1;
        let mut output_indices_plain = output_indices.try_as_plain_ram_mut()?;
        let mut output_indices_view = output_indices_plain.to_array_view_mut::<i64>()?;
        for coords in tract_ndarray::indices(&*iterating_shape) {
            let mut coords: TVec<usize> = coords.as_array_view().as_slice().unwrap().into();
            dispatch_numbers!(Self::inner_loop_t(dt)(
                self,
                &mut coords,
                &input,
                &mut output_values,
                &mut output_indices_view,
                k
            ))?;
        }
        Ok(tvec!(output_values.into_tvalue(), output_indices.into_tvalue()))
    }
}

impl Topk {
    fn inner_loop_t<T: Datum + PartialOrd>(
        &self,
        coords: &mut [usize],
        input: &Tensor,
        output_values: &mut Tensor,
        output_indices_view: &mut ArrayViewMutD<i64>,
        k: usize,
    ) -> TractResult<()> {
        let mut output_values_plain = output_values.try_as_plain_ram_mut()?;
        let mut output_values_view = output_values_plain.to_array_view_mut::<T>()?;
        let mut view = input.to_plain_array_view::<T>()?;
        for (ix, x) in coords.iter().enumerate() {
            if ix != self.axis {
                view.collapse_axis(Axis(ix), *x);
            }
        }
        for (ix, (argmax, max)) in view
            .iter()
            .cloned()
            .enumerate()
            .sorted_by(|a, b| {
                let ord = { a.1.partial_cmp(&b.1).unwrap_or(Ordering::Less) };
                if self.largest { ord.reverse() } else { ord }
            })
            .take(k)
            .enumerate()
        {
            coords[self.axis] = ix;
            output_values_view[&*coords] = max;
            output_indices_view[&*coords] = argmax as i64;
        }
        Ok(())
    }
}

impl TypedOp for Topk {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact_values = inputs[0].without_value();
        let mut fact_indices = inputs[0].without_value();
        let k: TDim = if let Some(k) = &inputs[1].konst {
            k.cast_to::<TDim>()?.try_as_plain_ram()?.to_scalar::<TDim>()?.clone()
        } else if let Some(k) = &inputs[1].uniform_tdim {
            k.clone()
        } else {
            self.fallback_k.clone()
        };
        if inputs[0].shape[self.axis] == 1.to_dim() && k == 1.to_dim() {
            fact_values.uniform_tdim = inputs[0].uniform_tdim.clone();
        }
        fact_values.shape.set(self.axis, k.clone());
        fact_indices.shape.set(self.axis, k);
        fact_indices.datum_type = i64::datum_type();
        Ok(tvec!(fact_values, fact_indices))
    }

    fn set_symbols(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        subs: &HashMap<Symbol, TDim>,
    ) -> TractResult<TVec<OutletId>> {
        let op = Topk {
            axis: self.axis,
            largest: self.largest,
            fallback_k: self.fallback_k.substitute_all(subs)?,
        };
        let inputs = node.inputs.iter().map(|i| mapping[i]).collect::<TVec<_>>();
        target.wire_node(&node.name, op, &inputs)
    }

    as_op!();
}
