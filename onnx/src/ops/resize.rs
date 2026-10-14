use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_nnef::tract_core::ops::nn::resize::{CoordTransformer, Interpolator};
use tract_nnef::tract_num_traits::Zero as _;
use tract_onnx_opl::resize::{AspectRatio, CoordTransform, Nearest, Resize};

pub fn resize(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let op = match ctx.onnx_operator_set_version {
        10 => resize_10(node)?,
        11..=12 => resize_11(node)?,
        13..=17 => resize_13(node)?,
        18.. => resize_18(node)?,
        v => bail!("Unsupported operator set for Resize operator ({v})"),
    };
    Ok((expand(ResizeInference(op)), vec![]))
}

fn resize_10(node: &NodeProto) -> TractResult<Resize> {
    Ok(Resize {
        coord_transformer: CoordTransform::Plain(CoordTransformer::Asymmetric),
        optional_roi_input: None,
        optional_scales_input: Some(1),
        optional_sizes_input: None,
        ..common(node)?
    })
}

fn resize_11(node: &NodeProto) -> TractResult<Resize> {
    let mut options = crate::model::optional_inputs(node).skip(3);
    Ok(Resize {
        optional_roi_input: Some(1),
        optional_scales_input: Some(2),
        optional_sizes_input: options.next().unwrap(),
        ..common(node)?
    })
}

fn resize_13(node: &NodeProto) -> TractResult<Resize> {
    let mut options = crate::model::optional_inputs(node).skip(1);
    Ok(Resize {
        optional_roi_input: options.next().unwrap(),
        optional_scales_input: options.next().unwrap(),
        optional_sizes_input: options.next().unwrap(),
        ..common(node)?
    })
}

fn resize_18(node: &NodeProto) -> TractResult<Resize> {
    let mut options = crate::model::optional_inputs(node).skip(1);
    Ok(Resize {
        axes: node.get_attr_opt_vec("axes")?,
        antialias: node.get_attr_opt::<i64>("antialias")?.unwrap_or(0) != 0,
        keep_aspect_ratio_policy: AspectRatio::parse(
            node.get_attr_opt("keep_aspect_ratio_policy")?.unwrap_or("stretch"),
        )?,
        optional_roi_input: options.next().unwrap(),
        optional_scales_input: options.next().unwrap(),
        optional_sizes_input: options.next().unwrap(),
        ..common(node)?
    })
}

/// The attributes every Resize opset shares. `axes`, `antialias` and
/// `keep_aspect_ratio_policy` arrive with opset 18 and stay at their neutral
/// value below it.
fn common(node: &NodeProto) -> TractResult<Resize> {
    let extrapolation_value: f32 = node.get_attr_opt("extrapolation_value")?.unwrap_or(0.0);
    Ok(Resize {
        axes: None,
        antialias: false,
        keep_aspect_ratio_policy: AspectRatio::Stretch,
        coord_transformer: coord_transformer_from_node(node)?,
        interpolator: interpolator_from_node(node)?,
        nearest: nearest_from_node(node)?,
        cubic_coeff_a_bits: cubic_coeff_a_from_node(node)?,
        exclude_outside: exclude_outside_from_node(node)?,
        extrapolation_value_bits: extrapolation_value.to_bits(),
        optional_roi_input: None,
        optional_scales_input: None,
        optional_sizes_input: None,
    })
}

fn coord_transformer_from_node(node: &NodeProto) -> TractResult<CoordTransform> {
    CoordTransform::parse(
        node.get_attr_opt("coordinate_transformation_mode")?.unwrap_or("half_pixel"),
    )
}

fn interpolator_from_node(node: &NodeProto) -> TractResult<Interpolator> {
    Interpolator::parse(node.get_attr_opt("mode")?.unwrap_or("nearest"))
}

fn nearest_from_node(node: &NodeProto) -> TractResult<Nearest> {
    Nearest::parse(node.get_attr_opt("nearest_mode")?.unwrap_or("round_prefer_floor"))
}

fn cubic_coeff_a_from_node(node: &NodeProto) -> TractResult<u32> {
    let coeff: f32 = node.get_attr_opt("cubic_coeff_a")?.unwrap_or(-0.75);
    Ok(coeff.to_bits())
}

fn exclude_outside_from_node(node: &NodeProto) -> TractResult<bool> {
    Ok(node.get_attr_opt::<i64>("exclude_outside")?.unwrap_or(0) != 0)
}

#[derive(Clone, Debug)]
struct ResizeInference(Resize);

impl Expansion for ResizeInference {
    fn name(&self) -> StaticName {
        "Resize".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        let op = &self.0;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        if let Some(scales) = op.optional_scales_input {
            s.given(&inputs[scales].shape[0], move |s, len| {
                if len.is_zero() {
                    rules_with_sizes(op, s, inputs, outputs)
                } else {
                    rules_with_scales(op, s, inputs, outputs)
                }
            })
        } else if op.optional_sizes_input.is_some() {
            rules_with_sizes(op, s, inputs, outputs)
        } else {
            bail!("Resize with neither scales nor sizes")
        }
    }

    fn wire(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let facts =
            inputs.iter().map(|i| model.outlet_fact(*i)).collect::<TractResult<Vec<_>>>()?;
        if let Some(scales) = self.0.optional_scales_input.and_then(|i| facts[i].konst.as_deref())
            && self.0.compute_output_shape(&facts[0].shape, Some(scales), None).is_err()
        {
            let shape = symbolic_reciprocal_shape(&self.0, &facts[0].shape, scales)?;
            return model.wire_node(
                name,
                SymbolicScaleResize { resize: self.0.clone(), shape: shape.into() },
                inputs,
            );
        }
        model.wire_node(name, self.0.clone(), inputs)
    }
}

fn rules_with_scales<'r, 'p: 'r, 's: 'r>(
    op: &'s Resize,
    s: &mut Solver<'r>,
    inputs: &'p [TensorProxy],
    outputs: &'p [TensorProxy],
) -> InferenceResult {
    let scales_input = op.optional_scales_input.unwrap();
    let scales = &inputs[scales_input];
    s.equals(&scales.datum_type, f32::datum_type())?;
    s.equals(&scales.rank, 1)?;
    s.given(&inputs[0].rank, move |s, rank| {
        s.equals(&scales.shape[0], op.resized_axes(rank as usize)?.len().to_dim())
    })?;
    s.given_2(&inputs[0].shape, &inputs[scales_input].value, move |s, input_shape, scales| {
        let output_size = op
            .compute_output_shape(&input_shape, Some(scales.as_ref()), None)
            .or_else(|_| symbolic_reciprocal_shape(op, &input_shape, &scales))?;
        let rank = input_shape.len();
        for i in 0..rank {
            s.equals(&outputs[0].shape[i], output_size[i].to_dim())?;
        }
        Ok(())
    })
}

/// Infers integer-reciprocal scaling of symbolic dimensions. The typed operator checks
/// the resulting contract against floating-point output-size calculation at runtime.
fn symbolic_reciprocal_shape(
    op: &Resize,
    input: &[TDim],
    scales: &Tensor,
) -> TractResult<TVec<TDim>> {
    let axes = op.resized_axes(input.len())?;
    let scales = scales.cast_to::<f32>()?;
    let scales = scales.try_as_plain_ram()?.as_slice::<f32>()?;
    ensure!(scales.len() == axes.len(), "Resize scales length must match axes");
    let mut shape: TVec<_> = input.into();
    for (&axis, &scale) in axes.iter().zip(scales) {
        ensure!(scale.is_finite() && scale > 0., "Resize scales must be finite and positive");
        shape[axis] = if let Some(dim) = input[axis].as_usize() {
            ((dim as f32 * scale) as usize).to_dim()
        } else if scale.fract() == 0. {
            input[axis].clone() * scale as usize
        } else {
            let reciprocal = scale.recip();
            ensure!(
                reciprocal.is_finite() && reciprocal.fract() == 0. && reciprocal >= 1.,
                "Unsupported symbolic Resize scale {scale}"
            );
            input[axis].clone() / reciprocal as usize
        };
    }
    Ok(shape)
}

/// Resize with integer-reciprocal scales and a symbolic output shape. Floating-point
/// rounding that disagrees with the inferred dimensions is rejected before consumers run.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct SymbolicScaleResize {
    resize: Resize,
    shape: ShapeFact,
}

impl Op for SymbolicScaleResize {
    fn name(&self) -> StaticName {
        "SymbolicScaleResize".into()
    }
    op_as_typed_op!();
}

impl EvalOp for SymbolicScaleResize {
    op_out_of_plan!();
    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let expected = self.shape.eval_to_usize(ctx.symbols)?;
        let output = self.resize.eval(ctx, inputs)?;
        ensure!(
            output[0].shape() == expected.as_slice(),
            "Resize floating-point output shape {:?} differs from symbolic contract {expected:?}",
            output[0].shape()
        );
        Ok(output)
    }
}

impl TypedOp for SymbolicScaleResize {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].datum_type.fact(self.shape.clone())))
    }
    as_op!();
}

fn rules_with_sizes<'r, 'p: 'r, 's: 'r>(
    op: &'s Resize,
    s: &mut Solver<'r>,
    inputs: &'p [TensorProxy],
    outputs: &'p [TensorProxy],
) -> InferenceResult {
    let sizes = &inputs[op.optional_sizes_input.unwrap()];
    s.equals(&sizes.rank, 1)?;
    s.given(&inputs[0].rank, move |s, rank| {
        s.equals(&sizes.shape[0], op.resized_axes(rank as usize)?.len().to_dim())
    })?;
    s.given_2(&inputs[0].shape, &sizes.value, move |s, input_shape, sizes| {
        let output_size = op.compute_output_shape(&input_shape, None, Some(sizes.as_ref()))?;
        for (i, dim) in output_size.iter().enumerate() {
            s.equals(&outputs[0].shape[i], dim.to_dim())?;
        }
        Ok(())
    })
}
