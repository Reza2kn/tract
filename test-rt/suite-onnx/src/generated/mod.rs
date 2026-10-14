use infra::TestSuite;
use tract_hir::internal::*;
use tract_onnx::pb::*;

mod convolution;
mod dynamic_lstm;
mod dynamic_quantize;
mod float_numerics;
mod fused_matmul;
mod integer_matmul;
mod quantized;

pub fn suite() -> TestSuite {
    TestSuite::default()
        .with("convolution", convolution::suite())
        .with("dynamic_quantize", dynamic_quantize::suite())
        .with("integer_matmul", integer_matmul::suite())
        .with("fused_matmul", fused_matmul::suite())
        .with("float_numerics", float_numerics::suite())
        .with("quantized", quantized::suite())
        .with("dynamic_lstm", dynamic_lstm::suite())
}

fn proto(name: &str, tensor: &Tensor) -> TensorProto {
    TensorProto {
        name: name.into(),
        dims: tensor.shape().iter().map(|d| *d as i64).collect(),
        data_type: match tensor.datum_type() {
            DatumType::F16 => 10,
            DatumType::F32 => 1,
            DatumType::U8 => 2,
            DatumType::I8 => 3,
            DatumType::I32 => 6,
            DatumType::I64 => 7,
            _ => panic!("Unsupported test type"),
        },
        raw_data: tensor.as_bytes().to_vec(),
        ..Default::default()
    }
}

fn info(name: &str, tensor: &Tensor) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: proto(name, tensor).data_type,
                shape: Some(TensorShapeProto {
                    dim: tensor
                        .shape()
                        .iter()
                        .map(|d| tensor_shape_proto::Dimension {
                            value: Some(tensor_shape_proto::dimension::Value::DimValue(*d as i64)),
                            ..Default::default()
                        })
                        .collect(),
                }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn int_attr(name: &str, i: i64) -> AttributeProto {
    AttributeProto { name: name.into(), r#type: 2, i, ..Default::default() }
}
fn float_attr(name: &str, f: f32) -> AttributeProto {
    AttributeProto { name: name.into(), r#type: 1, f, ..Default::default() }
}
fn string_attr(name: &str, s: &str) -> AttributeProto {
    AttributeProto { name: name.into(), r#type: 3, s: s.as_bytes().to_vec(), ..Default::default() }
}

/// Imports an ONNX operator and checks it through the selected runtime, including NNEF runtimes.
fn check(
    node: NodeProto,
    inputs: Vec<(&str, Tensor)>,
    constants: Vec<(&str, Tensor)>,
    expected: Vec<Tensor>,
    runtime: &dyn Runtime,
    approximation: Approximation,
) -> TractResult<()> {
    let graph = GraphProto {
        input: inputs.iter().map(|(name, t)| info(name, t)).collect(),
        output: node.output.iter().zip(&expected).map(|(name, t)| info(name, t)).collect(),
        initializer: constants.iter().map(|(name, t)| proto(name, t)).collect(),
        node: vec![node],
        ..Default::default()
    };
    let proto = ModelProto {
        ir_version: 8,
        graph: Some(graph),
        opset_import: vec![
            OperatorSetIdProto { domain: "".into(), version: 20 },
            OperatorSetIdProto { domain: "com.microsoft".into(), version: 1 },
        ],
        ..Default::default()
    };
    let mut model = tract_onnx::onnx().model_for_proto_model(&proto)?;
    model.analyse(false)?;
    let model = model.incorporate()?.into_typed()?.into_decluttered()?;
    let actual =
        runtime.prepare(model)?.run(inputs.into_iter().map(|(_, t)| t.into_tvalue()).collect())?;
    ensure!(actual.len() == expected.len(), "Output count");
    for (a, b) in actual.iter().zip(&expected) {
        ensure!(a.datum_type() == b.datum_type() && a.shape() == b.shape(), "Output fact");
        a.close_enough(b, approximation)?;
    }
    Ok(())
}

fn byte_tensor(shape: &[usize], values: &[i32], signed: bool) -> Tensor {
    if signed {
        Tensor::from_shape(shape, &values.iter().map(|x| *x as i8).collect::<Vec<_>>()).unwrap()
    } else {
        Tensor::from_shape(shape, &values.iter().map(|x| *x as u8).collect::<Vec<_>>()).unwrap()
    }
}
