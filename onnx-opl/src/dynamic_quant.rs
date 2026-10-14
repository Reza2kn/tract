pub use tract_core::ops::quant::DynamicQuantizeLinearU8;
use tract_nnef::internal::*;

/// Registers per-tensor dynamic quantization in the ONNX NNEF extension.
pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_onnx_dynamic_quantize_linear",
        &[TypeName::Scalar.tensor().named("input")],
        &[
            ("quantized", TypeName::Integer.tensor()),
            ("scale", TypeName::Scalar.tensor()),
            ("zero_point", TypeName::Integer.tensor()),
        ],
        load,
    );
    registry.register_dumper(dump);
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    builder.wire(DynamicQuantizeLinearU8, &[input])
}

fn dump(
    ast: &mut IntoAst,
    node: &TypedNode,
    _op: &DynamicQuantizeLinearU8,
) -> TractResult<Option<Arc<RValue>>> {
    Ok(Some(invocation(
        "tract_onnx_dynamic_quantize_linear",
        &[ast.mapping[&node.inputs[0]].clone()],
        &[],
    )))
}
