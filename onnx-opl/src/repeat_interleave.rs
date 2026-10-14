use tract_nnef::internal::*;

/// Expand a one-dimensional tensor by per-element repeat counts.
///
/// The remaining inputs carry an ONNX sequence loop's trip count, initial
/// condition, and unit split sizes. They are checked before allocating the
/// result. `len` is bound from the resulting tensor on each execution.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RepeatInterleave {
    pub len: Symbol,
}

impl Op for RepeatInterleave {
    fn name(&self) -> StaticName {
        "RepeatInterleave".into()
    }

    fn mints_runtime_symbols(&self) -> bool {
        true
    }

    op_as_typed_op!();
}

impl EvalOp for RepeatInterleave {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let values = &inputs[0];
        let repeats = inputs[1].cast_to::<i64>()?;
        let repeats = repeats.try_as_plain_ram()?.as_slice::<i64>()?;
        let trip_count = inputs[2].cast_to_scalar::<i64>()?;
        let condition = inputs[3].cast_to_scalar::<bool>()?;
        ensure!(
            condition && trip_count > 0,
            "ConcatFromSequence cannot concatenate an empty sequence"
        );
        let splits = inputs[4].cast_to::<i64>()?;
        let splits = splits.try_as_plain_ram()?.as_slice::<i64>()?;
        ensure!(values.rank() == 1 && inputs[1].rank() == 1 && inputs[4].rank() == 1);
        ensure!(values.len() == repeats.len() && splits.len() == repeats.len());
        ensure!(splits.iter().all(|s| *s == 1), "Repeat loop requires unit splits");
        ensure!(
            trip_count >= 0 && trip_count as usize <= values.len(),
            "Invalid repeat loop trip count"
        );
        let count = trip_count as usize;
        let len = repeats[..count].iter().try_fold(0usize, |len, repeat| {
            let repeat = if *repeat == -1 { 1 } else { *repeat };
            ensure!(repeat >= 0, "Negative repeat count");
            len.checked_add(usize::try_from(repeat)?).context("Repeat count overflow")
        })?;
        let mut output = Tensor::zero_dt(values.datum_type(), &[len])?;
        let mut position = 0;
        for (ix, repeat) in repeats[..count].iter().enumerate() {
            let repeat = if *repeat == -1 { 1 } else { *repeat };
            for _ in 0..repeat {
                output.assign_slice(position..position + 1, values, ix..ix + 1, 0)?;
                position += 1;
            }
        }
        Ok(tvec!(output.into_tvalue()))
    }
}

impl TypedOp for RepeatInterleave {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 5);
        ensure!(inputs[0].rank() == 1 && inputs[1].rank() == 1 && inputs[4].rank() == 1);
        ensure!(inputs[1].datum_type == i64::datum_type());
        ensure!(
            inputs[2].datum_type == i64::datum_type() && inputs[2].shape.volume() == 1.to_dim()
        );
        ensure!(
            inputs[3].datum_type == bool::datum_type() && inputs[3].shape.volume() == 1.to_dim()
        );
        ensure!(inputs[4].datum_type == i64::datum_type());
        Ok(tvec!(inputs[0].datum_type.fact([self.len.to_dim()])))
    }

    as_op!();
}

pub fn register(registry: &mut Registry) {
    registry.register_dumper(dump);
    registry.register_primitive(
        "tract_onnx_repeat_interleave",
        &[
            TypeName::Any.tensor().named("values"),
            TypeName::Integer.tensor().named("repeats"),
            TypeName::Integer.tensor().named("trip_count"),
            TypeName::Logical.tensor().named("condition"),
            TypeName::Integer.tensor().named("splits"),
            TypeName::String.named("len_symbol"),
        ],
        &[("output", TypeName::Any.tensor())],
        load,
    );
}

fn dump(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &RepeatInterleave,
) -> TractResult<Option<Arc<RValue>>> {
    let inputs = node.inputs.iter().map(|i| ast.mapping[i].clone()).collect::<Vec<_>>();
    Ok(Some(invocation(
        "tract_onnx_repeat_interleave",
        &inputs,
        &[("len_symbol", string(op.len.to_string()))],
    )))
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let inputs = ["values", "repeats", "trip_count", "condition", "splits"]
        .iter()
        .map(|name| invocation.named_arg_as(builder, name))
        .collect::<TractResult<Vec<_>>>()?;
    let name: String = invocation.named_arg_as(builder, "len_symbol")?;
    let len = builder.model.symbols.sym(&name);
    builder.wire(RepeatInterleave { len }, &inputs)
}
