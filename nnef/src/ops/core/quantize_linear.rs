use crate::internal::*;
use crate::ser::*;
use std::any::TypeId;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::quant::DequantizeLinearF32;
use tract_core::ops::quant::QuantizeLinearI8;
use tract_core::ops::quant::QuantizeLinearU8;
use tract_core::ops::quant::Scale;
use tract_core::ops::quant::quantize_linear_i8;
use tract_core::ops::quant::quantize_linear_u8;
use tract_core::ops::quant::scale;

/// Registers quantization, dequantization, and mixed-type rounded scaling in core NNEF.
pub fn register(registry: &mut Registry) {
    registry.register_dumper(dump_scale);
    registry.register_primitive(
        "tract_core_scale",
        &[TypeName::Scalar.tensor().named("scale"), TypeName::Any.tensor().named("input")],
        &[("output", TypeName::Any.tensor())],
        load_scale,
    );
    registry.register_dumper(dump_dequantize);
    registry.register_primitive(
        "tract_core_dequantize_linear",
        &[
            TypeName::Integer.tensor().named("input"),
            TypeName::Scalar.named("scale"),
            TypeName::Integer.named("zero_point"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        load_dequantize,
    );
    for id in [TypeId::of::<QuantizeLinearI8>(), TypeId::of::<QuantizeLinearU8>()] {
        registry.register_element_wise(
            "tract_core_quantize_linear",
            id,
            Box::new(dump),
            vec![
                TypeName::Scalar.tensor().named("input"),
                TypeName::String.named("output_type"),
                TypeName::Scalar.named("inverse_scale"),
                TypeName::Integer.named("zero_point"),
            ],
            load,
        );
    }
}

fn load_scale(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let factor = invocation.named_arg_as(builder, "scale")?;
    let input = invocation.named_arg_as(builder, "input")?;
    builder.wire(scale(), &[factor, input])
}

fn dump_scale(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &TypedBinOp,
) -> TractResult<Option<Arc<RValue>>> {
    if op.0.downcast_ref::<Scale>().is_none() {
        return Ok(None);
    }
    Ok(Some(invocation(
        "tract_core_scale",
        &[ast.mapping[&node.inputs[0]].clone(), ast.mapping[&node.inputs[1]].clone()],
        &[],
    )))
}

fn load_dequantize(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let scale = invocation.named_arg_as(builder, "scale")?;
    let zero: i64 = invocation.named_arg_as(builder, "zero_point")?;
    builder.wire(DequantizeLinearF32::new(scale, i32::try_from(zero)?), &[input])
}

fn dump_dequantize(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &DequantizeLinearF32,
) -> TractResult<Option<Arc<RValue>>> {
    Ok(Some(invocation(
        "tract_core_dequantize_linear",
        &[ast.mapping[&node.inputs[0]].clone()],
        &[("scale", numeric(op.scale)), ("zero_point", numeric(op.zero_point))],
    )))
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let scale = invocation.named_arg_as(builder, "inverse_scale")?;
    let zero: i64 = invocation.named_arg_as(builder, "zero_point")?;
    let dt: String = invocation.named_arg_as(builder, "output_type")?;
    match dt.as_str() {
        "u8" => builder.wire(quantize_linear_u8(scale, u8::try_from(zero)?), &[input]),
        "i8" => builder.wire(quantize_linear_i8(scale, i8::try_from(zero)?), &[input]),
        _ => bail!("Unsupported quantization output type {dt}"),
    }
}

fn dump(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = &node.op_as::<ElementWiseOp>().context("Expected elementwise quantization")?.0;
    let input = ast.mapping[&node.inputs[0]].clone();
    let (dt, scale, zero) = if let Some(op) = op.downcast_ref::<QuantizeLinearI8>() {
        ("i8", op.scale, i32::from(op.zero_point))
    } else {
        let op =
            op.downcast_ref::<QuantizeLinearU8>().context("Expected eight-bit quantization")?;
        ("u8", op.scale, i32::from(op.zero_point))
    };
    Ok(Some(invocation(
        "tract_core_quantize_linear",
        &[input],
        &[
            ("output_type", string(dt)),
            ("inverse_scale", numeric(scale)),
            ("zero_point", numeric(zero)),
        ],
    )))
}
