use crate::model::ParsingContext;
use crate::pb::GraphProto;
use crate::pb::NodeProto;
use tract_hir::internal::*;
use tract_onnx_opl::repeat_interleave::RepeatInterleave;

fn producer<'a>(graph: &'a GraphProto, output: &str, op: &str) -> TractResult<&'a NodeProto> {
    let node = graph
        .node
        .iter()
        .find(|n| n.output.iter().any(|o| o == output))
        .with_context(|| format!("Missing {op} producer for {output}"))?;
    ensure!(node.op_type == op, "Expected {op}, found {}", node.op_type);
    ensure!(node.domain.is_empty(), "Expected a standard ONNX operator");
    Ok(node)
}

fn constant(
    ctx: &ParsingContext,
    graph: &GraphProto,
    name: &str,
    expected: &[i64],
) -> TractResult<()> {
    let node = producer(graph, name, "Constant")?;
    let tensor = ctx.load_tensor(node.get_attr("value")?)?;
    let tensor = tensor.cast_to::<i64>()?;
    ensure!(tensor.try_as_plain_ram()?.as_slice::<i64>()? == expected);
    Ok(())
}

fn inputs(node: &NodeProto, count: usize) -> TractResult<&[String]> {
    ensure!(node.input.len() == count, "Unexpected {} arity", node.op_type);
    Ok(&node.input)
}

fn zero_axis(node: &NodeProto) -> TractResult<()> {
    ensure!(node.get_attr_opt::<i64>("axis")?.unwrap_or(0) == 0);
    Ok(())
}

/// Recognize the exported sequence loop for one-dimensional repeat-interleave.
/// Every output dependency is checked before replacing the sequence graph.
fn repeat_loop(ctx: &ParsingContext, concat: &NodeProto) -> TractResult<(Vec<String>, DatumType)> {
    zero_axis(concat)?;
    ensure!(concat.get_attr_opt::<i64>("new_axis")?.unwrap_or(0) == 0);
    let graph = ctx.parent_graphs.last().context("Missing enclosing graph")?;
    let loop_node = producer(graph, &inputs(concat, 1)?[0], "Loop")?;
    ensure!(loop_node.output.len() == 1);
    let outer = inputs(loop_node, 3)?;
    let empty = producer(graph, &outer[2], "SequenceEmpty")?;
    inputs(empty, 0)?;
    let dtype = empty.get_attr_opt::<DatumType>("dtype")?.unwrap_or(DatumType::F32);
    let body: &GraphProto = loop_node.get_attr("body")?;
    ensure!(body.input.len() == 3 && body.output.len() == 2);
    let condition = producer(body, &body.output[0].name, "Cast")?;
    ensure!(condition.get_attr::<i64>("to")? == 9);
    ensure!(inputs(condition, 1)?[0] == outer[1]);
    let insert = producer(body, &body.output[1].name, "SequenceInsert")?;
    let insert_inputs = inputs(insert, 2)?;
    ensure!(insert_inputs[0] == body.input[2].name);
    let reshape = producer(body, &insert_inputs[1], "Reshape")?;
    let reshape_inputs = inputs(reshape, 2)?;
    ensure!(reshape.get_attr_opt::<i64>("allowzero")?.unwrap_or(0) == 0);
    constant(ctx, body, &reshape_inputs[1], &[-1])?;
    let expand = producer(body, &reshape_inputs[0], "Expand")?;
    let expand_inputs = inputs(expand, 2)?;
    let unsqueeze = producer(body, &expand_inputs[0], "Unsqueeze")?;
    let unsqueeze_inputs = inputs(unsqueeze, 2)?;
    constant(ctx, body, &unsqueeze_inputs[1], &[1])?;
    let value_at = producer(body, &unsqueeze_inputs[0], "SequenceAt")?;
    let value_at_inputs = inputs(value_at, 2)?;
    ensure!(value_at_inputs[1] == body.input[0].name);
    let select_shape = producer(body, &expand_inputs[1], "Where")?;
    let select_inputs = inputs(select_shape, 3)?;
    let shape_concat = producer(body, &select_inputs[2], "Concat")?;
    zero_axis(shape_concat)?;
    let shape_inputs = inputs(shape_concat, 3)?;
    constant(ctx, body, &shape_inputs[0], &[1])?;
    constant(ctx, body, &shape_inputs[2], &[])?;
    let repeats_at = producer(body, &shape_inputs[1], "SequenceAt")?;
    let repeats_at_inputs = inputs(repeats_at, 2)?;
    ensure!(repeats_at_inputs[1] == body.input[0].name);
    let ones = producer(body, &select_inputs[1], "ConstantOfShape")?;
    let one = ctx.load_tensor(ones.get_attr("value")?)?;
    ensure!(one.cast_to_scalar::<i64>()? == 1);
    let shape = producer(body, &inputs(ones, 1)?[0], "Shape")?;
    ensure!(shape.attribute.is_empty() && inputs(shape, 1)?[0] == select_inputs[2]);
    let equal = producer(body, &select_inputs[0], "Equal")?;
    let equal_inputs = inputs(equal, 2)?;
    ensure!(equal_inputs[0] == select_inputs[2]);
    let mul = producer(body, &equal_inputs[1], "Mul")?;
    let mul_inputs = inputs(mul, 2)?;
    ensure!(mul_inputs[0] == select_inputs[1]);
    constant(ctx, body, &mul_inputs[1], &[-1])?;
    let values_split = producer(graph, &value_at_inputs[0], "SplitToSequence")?;
    let repeats_split = producer(graph, &repeats_at_inputs[0], "SplitToSequence")?;
    zero_axis(values_split)?;
    zero_axis(repeats_split)?;
    ensure!(values_split.get_attr_opt::<i64>("keepdims")?.unwrap_or(1) == 1);
    ensure!(repeats_split.get_attr_opt::<i64>("keepdims")?.unwrap_or(1) == 1);
    let values = inputs(values_split, 2)?;
    let repeats = inputs(repeats_split, 2)?;
    ensure!(values[1] == repeats[1]);
    Ok((
        vec![
            values[0].clone(),
            repeats[0].clone(),
            outer[0].clone(),
            outer[1].clone(),
            values[1].clone(),
        ],
        dtype,
    ))
}

pub fn concat_from_sequence(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let (closures, dtype) = repeat_loop(ctx, node)
        .context("ConcatFromSequence requires a supported repeat-interleave sequence loop")?;
    let len = ctx.template.symbols.new_with_prefix("repeat");
    ctx.template.symbols.add_assertion(format!("{len} >= 0"))?;
    Ok((Box::new(RepeatLoop { dtype, len }), closures))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RepeatLoop {
    dtype: DatumType,
    len: Symbol,
}

impl Op for RepeatLoop {
    fn name(&self) -> StaticName {
        "RepeatInterleaveLoop".into()
    }
    not_a_typed_op!();
}

impl EvalOp for RepeatLoop {
    not_out_of_plan!();
}

impl InferenceRulesOp for RepeatLoop {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 6)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[1].datum_type, self.dtype)?;
        s.equals(&outputs[0].datum_type, self.dtype)?;
        s.equals(&outputs[0].shape, tvec!(self.len.to_dim()))?;
        for ix in [1, 2, 5] {
            s.equals(&inputs[ix].rank, 1)?;
        }
        s.equals(&inputs[4].datum_type, bool::datum_type())?;
        Ok(())
    }

    fn incorporate(
        &self,
        model: &InferenceModel,
        node: &InferenceNode,
    ) -> TractResult<Option<InferenceModelPatch>> {
        Ok(Some(InferenceModelPatch::replace_single_op(
            model,
            node,
            &node.inputs[1..],
            expand(RepeatExpansion { len: self.len.clone() }),
        )?))
    }

    as_op!();
}

#[derive(Debug, Clone)]
struct RepeatExpansion {
    len: Symbol,
}

impl Expansion for RepeatExpansion {
    fn name(&self) -> StaticName {
        "RepeatInterleave".into()
    }
    fn runs_out_of_plan(&self) -> bool {
        false
    }
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 5)?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].shape, tvec!(self.len.to_dim()))?;
        Ok(())
    }
    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let mut inputs = inputs.to_vec();
        for ix in [1, 2, 4] {
            inputs[ix] = model.wire_node(
                format!("{prefix}.cast-{ix}"),
                tract_core::ops::cast::cast(DatumType::I64),
                &[inputs[ix]],
            )?[0];
        }
        model.wire_node(prefix, RepeatInterleave { len: self.len.clone() }, &inputs)
    }
}
