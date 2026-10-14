use crate::LibraryName;
use crate::encoder::EncoderExt;
use metal::MTLSize;
use tract_core::internal::*;
use tract_core::ops::einsum::einsum_matmul::EinSumMatMul;
use tract_gpu::tensor::DeviceTensor;
use tract_gpu::tensor::DeviceTensorExt;

/// Integer matrix products with scalar or output-axis zero points and unit quantization scales.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct MetalIntegerMatMul {
    pub op: EinSumMatMul,
}

impl Op for MetalIntegerMatMul {
    fn name(&self) -> StaticName {
        "MetalIntegerMatMul".into()
    }

    op_as_typed_op!();
}

impl EvalOp for MetalIntegerMatMul {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let tensors = inputs
            .iter()
            .map(|input| input.to_device_tensor())
            .collect::<TractResult<TVec<_>>>()?;
        ensure!(tensors.iter().all(|tensor| tensor.len() <= u32::MAX as usize));
        let facts = tensors.iter().map(|t| t.datum_type().fact(t.shape())).collect::<TVec<_>>();
        let fact_refs = facts.iter().collect::<TVec<_>>();
        let shape = self.op.output_facts(&fact_refs)?[0].shape.as_concrete().unwrap().to_vec();
        let output = tract_gpu::turn_handler::make_tensor_for_node(ctx, DatumType::I32, &shape)?;
        if output.len() == 0 {
            return Ok(tvec!(output.into_tensor().into_tvalue()));
        }
        ensure!(output.len() <= u32::MAX as usize - 255);
        let axes = self.op.axes.axes(InOut::Out(0)).collect::<TVec<_>>();
        let stride = |slot: usize, axis: &Axis| -> TractResult<u32> {
            match axis.inputs[slot].first() {
                Some(&position) if tensors[slot].shape()[position] != 1 => {
                    Ok(u32::try_from(tensors[slot].strides()[position])?)
                }
                _ => Ok(0),
            }
        };
        let zero_type = |dt| match dt {
            DatumType::I8 => 0,
            DatumType::U8 => 1,
            DatumType::I32 => 2,
            _ => unreachable!(),
        };
        let mut parameters = vec![
            u32::try_from(output.len())?,
            shape.len() as u32,
            u32::try_from(tensors[0].shape()[self.op.a_k()])?,
            u32::try_from(tensors[0].strides()[self.op.a_k()])?,
            u32::try_from(tensors[1].strides()[self.op.b_k()])?,
            zero_type(tensors[3].datum_type()),
            zero_type(tensors[5].datum_type()),
        ];
        for &dimension in &shape {
            parameters.push(u32::try_from(dimension)?);
        }
        for slot in [0, 1, 3, 5] {
            for axis in &axes {
                parameters.push(stride(slot, axis)?);
            }
        }
        crate::with_metal_stream(|stream| {
            for &slot in &[0, 1, 3, 5] {
                stream.retain_tensor(tensors[slot]);
            }
            stream.retain_tensor(&output);
            let name = format!(
                "integer_mm_{}_{}",
                DeviceTensor::tname(tensors[0].datum_type())?,
                DeviceTensor::tname(tensors[1].datum_type())?
            );
            let pipeline = stream.load_pipeline(LibraryName::IntegerMatMul, &name)?;
            stream.command_buffer().encode(|encoder| {
                encoder.set_compute_pipeline_state(&pipeline);
                for (buffer, slot) in [0, 1, 3, 5].into_iter().enumerate() {
                    encoder.set_metal_tensor(
                        buffer as _,
                        tensors[slot],
                        metal::MTLResourceUsage::Read,
                    );
                }
                encoder.set_metal_tensor(4, &output, metal::MTLResourceUsage::Write);
                encoder.set_slice(5, &parameters);
                encoder.dispatch_thread_groups(
                    MTLSize::new(output.len().div_ceil(256) as _, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            });
            Ok(())
        })?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for MetalIntegerMatMul {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| self.op.output_facts(facts))
    }

    as_op!();
}

crate::register_metal_op!(EinSumMatMul, |source, node, op| {
    rule_if!(op.operating_dt == DatumType::I32 && op.q_params == Some(DatumType::I32));
    rule_if!(node.inputs.len() == 9);
    rule_if!(op.k.to_usize().is_ok_and(|k| k <= u32::MAX as usize));
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts[..2].iter().all(|f| matches!(f.datum_type, DatumType::I8 | DatumType::U8)));
    for (slot, value) in [(2, 0.), (4, 1.), (6, 1.), (7, 0.), (8, 1.)] {
        rule_if_some!(constant = facts[slot].konst.as_ref());
        rule_if!(constant.len() == 1 && constant.cast_to_scalar::<f32>()? == value);
    }
    for slot in [3, 5] {
        rule_if!(
            matches!(facts[slot].datum_type, DatumType::I8 | DatumType::U8)
                || (facts[slot].datum_type == DatumType::I32
                    && facts[slot]
                        .konst
                        .as_ref()
                        .is_some_and(|t| t.cast_to_scalar::<i32>().ok() == Some(0)))
        );
        rule_if!(op.k_axis().inputs[slot].is_empty());
    }
    rule_if!(
        op.axes
            .iter_all_axes()
            .all(|axis| axis.inputs.iter().all(|positions| positions.len() <= 1))
    );
    Ok(Some(Box::new(MetalIntegerMatMul { op: op.clone() })))
});
