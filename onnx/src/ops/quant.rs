use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::NodeProto;
use tract_core::ops::cast::cast;
use tract_hir::internal::*;
use tract_hir::ops::quant::*;
pub use tract_onnx_opl::dynamic_quant::DynamicQuantizeLinearU8;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("QuantizeLinear", quantize_linear);
    reg.insert("DequantizeLinear", dequantize_linear);
    reg.insert("DynamicQuantizeLinear", dynamic_quantize_linear);
}

fn quantize_linear(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = quantization_axis(node)?;
    let op = QuantizeLinear::new((node.input.len() == 3).then_some(2), axis);
    Ok((expand(op), vec![]))
}

fn dequantize_linear(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = quantization_axis(node)?;
    let op = DequantizeLinear::new((node.input.len() == 3).then_some(2), axis);
    Ok((expand(op), vec![]))
}

fn quantization_axis(node: &NodeProto) -> TractResult<i64> {
    let block_size = node.get_attr_opt::<i64>("block_size")?.unwrap_or(0);
    ensure!(block_size == 0, "{}: blocked quantization is not supported", node.op_type);
    Ok(node.get_attr_opt("axis")?.unwrap_or(1))
}

/// Reshapes a per-axis scale or zero point so that it broadcasts along `axis` of a rank `rank`
/// tensor, as f32.
fn per_axis_param(param: &Tensor, axis: i64, rank: usize) -> TractResult<Tensor> {
    let axis = if axis < 0 { axis + rank as i64 } else { axis };
    ensure!((0..rank as i64).contains(&axis), "quantization axis {axis} out of range");
    let mut shape = tvec![1; rank];
    shape[axis as usize] = param.len();
    param.cast_to::<f32>()?.into_owned().into_shape(&shape)
}

fn dynamic_quantize_linear(
    _ctx: &ParsingContext,
    _node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let op = DynamicQuantizeLinear::new();
    Ok((expand(op), vec![]))
}

#[derive(Debug, Clone, new, Default, Hash, PartialEq, Eq)]
pub struct QuantizeLinear {
    optional_zero_point_input: Option<usize>,
    axis: i64,
}

impl Expansion for QuantizeLinear {
    fn name(&self) -> StaticName {
        "QuantizeLinear".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        check_input_arity(inputs, 2 + self.optional_zero_point_input.is_some() as usize)?;
        check_output_arity(outputs, 1)?;
        //         s.equals(&inputs[1].rank, 0)?; broken in Onnx test suite
        s.equals(&inputs[1].datum_type, f32::datum_type())?;
        if self.optional_zero_point_input.is_some() {
            s.equals(&outputs[0].datum_type, &inputs[2].datum_type)?;
        //            s.equals(&inputs[2].rank, 0)?; // broken in Onnx test suite
        } else {
            s.equals(&outputs[0].datum_type, u8::datum_type())?;
        }
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        use tract_hir::ops::quant::*;
        let scales =
            target.outlet_fact(inputs[1])?.konst.clone().context("y_scale must be a const")?;
        let zero_point = if self.optional_zero_point_input.is_some() {
            target
                .outlet_fact(inputs[2])?
                .konst
                .as_ref()
                .context("y_zero_point must be a const")?
                .clone()
        } else {
            rctensor0(0u8)
        };
        if scales.len() > 1 || zero_point.len() > 1 {
            return self.wire_per_axis(prefix, target, inputs[0], &scales, &zero_point);
        }
        let scale = scales.try_as_plain_ram()?.as_slice::<f32>()?[0].recip();
        let op: Box<dyn TypedOp> = if zero_point.datum_type() == u8::datum_type() {
            Box::new(quantize_linear_u8(scale, zero_point.try_as_plain_ram()?.as_slice::<u8>()?[0]))
        } else {
            Box::new(quantize_linear_i8(scale, zero_point.try_as_plain_ram()?.as_slice::<i8>()?[0]))
        };
        target.wire_node(prefix, op, &[inputs[0]])
    }
}

impl QuantizeLinear {
    /// `saturate(round_half_to_even(x / scale) + zero_point)` with scale and zero point
    /// broadcast along `axis`.
    fn wire_per_axis(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        x: OutletId,
        scales: &Tensor,
        zero_point: &Tensor,
    ) -> TractResult<TVec<OutletId>> {
        use tract_core::ops::math::{add, div, max, min, round_half_to_even};
        let dt = zero_point.datum_type();
        let (lo, hi) = match dt {
            DatumType::U8 => (u8::MIN as f32, u8::MAX as f32),
            DatumType::I8 => (i8::MIN as f32, i8::MAX as f32),
            _ => bail!("QuantizeLinear: unsupported per-axis output type {dt:?}"),
        };
        let rank = target.outlet_fact(x)?.rank();
        let scale = per_axis_param(scales, self.axis, rank)?;
        let scale = target.add_const(format!("{prefix}.scale"), scale)?;
        let zero_point = per_axis_param(zero_point, self.axis, rank)?;
        let zero_point = target.add_const(format!("{prefix}.zero_point"), zero_point)?;
        let lo =
            target.add_const(format!("{prefix}.lo"), tensor0(lo).broadcast_into_rank(rank)?)?;
        let hi =
            target.add_const(format!("{prefix}.hi"), tensor0(hi).broadcast_into_rank(rank)?)?;
        let mut wire = target.wire_node(format!("{prefix}.cast"), cast(f32::datum_type()), &[x])?;
        wire = target.wire_node(format!("{prefix}.div"), div(), &[wire[0], scale])?;
        wire = target.wire_node(format!("{prefix}.round"), round_half_to_even(), &wire)?;
        wire = target.wire_node(format!("{prefix}.add"), add(), &[wire[0], zero_point])?;
        wire = target.wire_node(format!("{prefix}.max"), max(), &[wire[0], lo])?;
        wire = target.wire_node(format!("{prefix}.min"), min(), &[wire[0], hi])?;
        target.wire_node(prefix, cast(dt), &wire)
    }
}

#[derive(Debug, Clone, new, Default, Hash, PartialEq, Eq)]
pub struct DequantizeLinear {
    optional_zero_point_input: Option<usize>,
    axis: i64,
}

impl Expansion for DequantizeLinear {
    fn name(&self) -> StaticName {
        "DequantizeLinear".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        check_input_arity(inputs, 2 + self.optional_zero_point_input.is_some() as usize)?;
        check_output_arity(outputs, 1)?;
        //         s.equals(&inputs[1].rank, 0)?; broken in Onnx test suite
        s.equals(&inputs[1].datum_type, f32::datum_type())?;
        s.equals(&outputs[0].datum_type, f32::datum_type())?;
        if self.optional_zero_point_input.is_some() {
            s.equals(&inputs[0].datum_type, &inputs[2].datum_type)?;
            //            s.equals(&inputs[2].rank, 0)?; // broken in Onnx test suite
        }
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let scales =
            target.outlet_fact(inputs[1])?.konst.clone().context("y_scale must be a const")?;
        let zero_point = if self.optional_zero_point_input.is_some() {
            target
                .outlet_fact(inputs[2])?
                .konst
                .as_ref()
                .context("y_zero_point must be a const")?
                .clone()
        } else {
            rctensor0(0u8)
        };
        if scales.len() > 1 || zero_point.len() > 1 {
            use tract_core::ops::math::{mul, sub};
            let rank = target.outlet_fact(inputs[0])?.rank();
            let scale = per_axis_param(&scales, self.axis, rank)?;
            let scale = target.add_const(format!("{prefix}.scale"), scale)?;
            let zero_point = per_axis_param(&zero_point, self.axis, rank)?;
            let zero_point = target.add_const(format!("{prefix}.zero_point"), zero_point)?;
            let x = target.wire_node(
                format!("{prefix}.cast"),
                cast(f32::datum_type()),
                &[inputs[0]],
            )?;
            let x = target.wire_node(format!("{prefix}.sub"), sub(), &[x[0], zero_point])?;
            return target.wire_node(prefix, mul(), &[x[0], scale]);
        }
        let scale = scales.try_as_plain_ram()?.as_slice::<f32>()?[0];
        let op: Box<dyn TypedOp> = if zero_point.datum_type() == u8::datum_type() {
            Box::new(DequantizeLinearF32::new(
                scale,
                zero_point.try_as_plain_ram()?.as_slice::<u8>()?[0] as i32,
            ))
        } else if zero_point.datum_type() == i8::datum_type() {
            Box::new(DequantizeLinearF32::new(
                scale,
                zero_point.try_as_plain_ram()?.as_slice::<i8>()?[0] as i32,
            ))
        } else {
            Box::new(DequantizeLinearF32::new(
                scale,
                zero_point.try_as_plain_ram()?.as_slice::<i32>()?[0],
            ))
        };
        target.wire_node(prefix, op, &[inputs[0]])
    }
}

#[derive(Debug, Clone, new, Default, Hash, PartialEq, Eq)]
pub struct DynamicQuantizeLinear {}

impl Expansion for DynamicQuantizeLinear {
    fn name(&self) -> StaticName {
        "DynamicQuantizeLinear".into()
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(3)
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 3)?;
        s.equals(&inputs[0].datum_type, f32::datum_type())?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        s.equals(&outputs[0].datum_type, u8::datum_type())?;
        s.equals(&outputs[1].datum_type, f32::datum_type())?;
        s.equals(&outputs[1].rank, 0)?;
        s.equals(&outputs[2].datum_type, u8::datum_type())?;
        s.equals(&outputs[2].rank, 0)?;

        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let op: Box<dyn TypedOp> = Box::new(DynamicQuantizeLinearU8::new());
        target.wire_node(format!("{prefix}.dynamic_quantize"), op, &[inputs[0]])
    }
}
