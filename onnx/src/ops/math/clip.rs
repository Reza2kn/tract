use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;

pub fn clip(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    match ctx.onnx_operator_set_version {
        6..=10 => clip_6(ctx, node),
        v if v >= 10 => clip_11(ctx, node),
        _ => bail!("Unsupported operator set for Clip operator"),
    }
}

pub fn clip_6(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let min: Option<f32> = node.get_attr_opt("min")?;
    let max: Option<f32> = node.get_attr_opt("max")?;
    Ok((expand(tract_hir::ops::activations::Clip::new(min, max)), vec![]))
}

pub fn clip_11(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut options = crate::model::optional_inputs(node).skip(1);
    let op = Clip11::new(options.next().unwrap(), options.next().unwrap());
    Ok((expand(op), vec![]))
}

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct Clip11 {
    input_min: Option<usize>,
    input_max: Option<usize>,
}

impl Expansion for Clip11 {
    fn name(&self) -> StaticName {
        "Clip".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(
            inputs,
            1 + self.input_min.is_some() as usize + self.input_max.is_some() as usize,
        )?;
        check_output_arity(outputs, 1)?;
        // Relaxed for streaming cache-len clamps: bounds may be I64 constants while the
        // clamped value is a symbolic TDim and shapes may differ in rank (scalar vs [1]).
        // wire() casts the bounds to the input dtype and materializes symbolic clamps as I64.
        Ok(())
    }

    fn wire(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let dt = model.outlet_fact(inputs[0])?.datum_type;
        let mut wire = inputs[0];
        // Clamping a symbolic TDim value whose consumers expect concrete I64: materialize
        // the clamp in the I64 domain so the wired output matches the downstream fact.
        if dt == TDim::datum_type() {
            wire = model.wire_node(format!("{name}.symbol.cast"), tract_core::ops::cast::cast(i64::datum_type()), &[wire])?[0];
        }
        let dt = model.outlet_fact(wire)?.datum_type;
        if let Some(min) = self.input_min {
            let mut b = inputs[min];
            if model.outlet_fact(b)?.datum_type != dt {
                b = model.wire_node(format!("{name}.min.cast"), tract_core::ops::cast::cast(dt), &[b])?[0];
            }
            wire = wire_with_rank_broadcast(
                format!("{name}.min"),
                model,
                tract_hir::ops::math::max(),
                &[wire, b],
            )?[0];
        }
        if let Some(max) = self.input_max {
            let mut b = inputs[max];
            if model.outlet_fact(b)?.datum_type != dt {
                b = model.wire_node(format!("{name}.max.cast"), tract_core::ops::cast::cast(dt), &[b])?[0];
            }
            wire = wire_with_rank_broadcast(
                format!("{name}.max"),
                model,
                tract_hir::ops::math::min(),
                &[wire, b],
            )?[0];
        }
        Ok(tvec!(wire))
    }
}
