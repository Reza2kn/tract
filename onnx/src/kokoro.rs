use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::change_axes::wire_with_rank_broadcast;
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::logic;
use tract_core::ops::math;
use tract_core::transform::ModelTransform;
use tract_hir::internal::*;

register_simple_model_transform!("kokoro_stabilize_phase", StabilizePhase);

/// Opt-in repair of Kokoro's exported phase calculations.
/// Maps zero complex bins to phase zero and keeps the float16 sine-generator phase
/// in float32 until after Sin, avoiding overflow of large finite phase angles.
#[derive(Debug)]
struct StabilizePhase;

impl ModelTransform for StabilizePhase {
    fn name(&self) -> StaticName {
        "kokoro_stabilize_phase".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        widen_sine_phase(model)?;
        let node = model
            .node_by_name("/decoder/decoder/generator/Where_1")
            .or_else(|_| model.node_by_name("/decoder/generator/Where_1"))?;
        ensure!(
            node.op_is::<logic::Iff>() && node.inputs.len() == 3,
            "Expected Kokoro's exported phase selection"
        );
        let atan = model.node(node.inputs[2].node);
        ensure!(
            atan.op_as::<ElementWiseOp>().is_some_and(|op| op.0.is::<math::Atan>()),
            "Expected Atan in Kokoro's exported phase calculation"
        );
        let div = model.node(atan.inputs[0].node);
        let op = div.op_as::<TypedBinOp>().context("Expected a phase ratio")?;
        ensure!(div.inputs.len() == 2, "Expected two phase ratio operands");
        let parts = if op.0.is::<math::Div>() {
            div.inputs.clone()
        } else {
            ensure!(op.0.is::<math::Mul>(), "Expected imaginary/real division");
            let reciprocal = model.node(div.inputs[1].node);
            ensure!(
                reciprocal.op_as::<ElementWiseOp>().is_some_and(|op| op.0.is::<math::Recip>()),
                "Expected reciprocal of the real component"
            );
            vec![div.inputs[0], reciprocal.inputs[0]]
        };
        let mut patch = TypedModelPatch::default();
        let inputs = patch.taps(model, &node.inputs)?;
        let phase =
            patch.wire_node(format!("{}.original_phase", node.name), node.op.clone(), &inputs)?[0];
        let parts = patch.taps(model, &parts)?;
        let rank = model.outlet_fact(node.id.into())?.rank();
        let dt = patch.outlet_fact(parts[0])?.datum_type;
        let zero = patch.add_const("zero_phase", Tensor::zero_dt(dt, &vec![1; rank])?)?;
        let imaginary_zero = wire_with_rank_broadcast(
            "imaginary_zero",
            &mut patch,
            TypedBinOp(logic::comp_eq(), None),
            &[parts[0], zero],
        )?[0];
        let real_zero = wire_with_rank_broadcast(
            "real_zero",
            &mut patch,
            TypedBinOp(logic::comp_eq(), None),
            &[parts[1], zero],
        )?[0];
        let both_zero = patch.wire_node("zero_bin", logic::and(), &[imaginary_zero, real_zero])?[0];
        let stable = patch.wire_node(
            format!("{}.stable_phase", node.name),
            logic::Iff,
            &[both_zero, zero, phase],
        )?[0];
        patch.shunt_outside(model, node.id.into(), stable)?;
        patch.apply(model)
    }
}

fn widen_sine_phase(model: &mut TypedModel) -> TractResult<()> {
    let prefix = "/decoder/generator/m_source/l_sin_gen";
    let Ok(sin) = model.node_by_name(format!("{prefix}/Sin")) else {
        return Ok(());
    };
    if model.outlet_fact(sin.id.into())?.datum_type != DatumType::F16 {
        return Ok(());
    }
    ensure!(sin.op_as::<ElementWiseOp>().is_some_and(|op| op.0.is::<math::Sin>()));
    let resize = model.node(sin.inputs[0].node);
    ensure!(
        resize.name == format!("{prefix}/Resize_1")
            && (resize.op_is::<tract_onnx_opl::resize::Resize>()
                || resize.op_is::<tract_core::ops::nn::resize::Resize>()),
        "Expected sine phase Resize_1, found {resize}"
    );
    let multiply = model.node(resize.inputs[0].node);
    let op = multiply.op_as::<TypedBinOp>().context("Expected sine phase multiplication")?;
    ensure!(op.0.is::<math::Mul>() && multiply.inputs.len() == 2);
    let (phase, factor) = if model.outlet_fact(multiply.inputs[0])?.uniform.is_some() {
        (multiply.inputs[1], multiply.inputs[0])
    } else {
        (multiply.inputs[0], multiply.inputs[1])
    };
    let factor_value = model
        .outlet_fact(factor)?
        .uniform
        .as_ref()
        .context("Expected a uniform phase multiplier")?;
    ensure!(factor_value.cast_to_scalar::<f32>()?.is_finite());
    let cast = model.node(phase.node);
    ensure!(cast.name == format!("{prefix}/CumSum_output_0_cast_to_fp16_node"));
    ensure!(cast.op_as::<tract_core::ops::cast::Cast>().is_some_and(|op| op.to == DatumType::F16));
    ensure!(model.outlet_fact(cast.inputs[0])?.datum_type == DatumType::F32);
    let mut patch = TypedModelPatch::default();
    let phase = patch.tap_model(model, cast.inputs[0])?;
    let factor = patch.tap_model(model, factor)?;
    let factor = patch.wire_node(
        "sine_phase_factor_f32",
        tract_core::ops::cast::cast(DatumType::F32),
        &[factor],
    )?[0];
    let phase = patch.wire_node("sine_phase_f32", math::mul(), &[phase, factor])?[0];
    let mut resize_inputs = tvec![phase];
    resize_inputs.extend(patch.taps(model, &resize.inputs[1..])?);
    let phase = patch.wire_node("sine_phase_resize_f32", resize.op.clone(), &resize_inputs)?;
    let sine = patch.wire_node("sine_phase_sin_f32", math::sin(), &phase)?;
    let output =
        patch.wire_node("sine_phase_to_f16", tract_core::ops::cast::cast(DatumType::F16), &sine)?
            [0];
    patch.shunt_outside(model, sin.id.into(), output)?;
    patch.apply(model)
}
