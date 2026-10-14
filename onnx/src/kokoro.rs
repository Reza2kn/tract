use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::change_axes::wire_with_rank_broadcast;
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::logic;
use tract_core::ops::math;
use tract_core::transform::ModelTransform;
use tract_hir::internal::*;

register_simple_model_transform!("kokoro_stabilize_phase", StabilizePhase);

/// Opt-in repair of Kokoro's exported phase calculation at an exactly zero complex bin.
/// Maps its undefined 0/0 phase to zero, preserving all other values and default import behavior.
#[derive(Debug)]
struct StabilizePhase;

impl ModelTransform for StabilizePhase {
    fn name(&self) -> StaticName {
        "kokoro_stabilize_phase".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        let node = model.node_by_name("/decoder/decoder/generator/Where_1")?;
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
        let zero = patch.add_const("zero_phase", tensor0(0f32).broadcast_into_rank(rank)?)?;
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
