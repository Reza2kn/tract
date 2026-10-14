use crate::LibraryName;
use crate::encoder::EncoderExt;
use metal::MTLSize;
use tract_core::internal::*;
use tract_core::ops::quant::DynamicQuantizeLinearU8;
use tract_gpu::tensor::DeviceTensor;
use tract_gpu::tensor::DeviceTensorExt;
use tract_gpu::turn_handler::make_tensor_for_node_output;

/// Per-tensor F32 to U8 quantization on Metal, with scalar scale and zero point outputs.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct MetalDynamicQuantizeLinearU8;

impl Op for MetalDynamicQuantizeLinearU8 {
    fn name(&self) -> StaticName {
        "MetalDynamicQuantizeLinearU8".into()
    }

    op_as_typed_op!();
}

impl EvalOp for MetalDynamicQuantizeLinearU8 {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = inputs[0].to_device_tensor()?;
        let quantized = make_tensor_for_node_output(ctx, 0, DatumType::U8, input.shape())?;
        let scale = make_tensor_for_node_output(ctx, 1, DatumType::F32, &[])?;
        let zero = make_tensor_for_node_output(ctx, 2, DatumType::U8, &[])?;
        let count = u32::try_from(input.len())?;
        ensure!(count <= u32::MAX - 255);
        let groups = count.div_ceil(256).clamp(1, 1024);
        let partials = DeviceTensor::uninitialized::<f32>(&[groups as usize, 2])?;
        crate::with_metal_stream(|stream| {
            for tensor in [input, &quantized, &scale, &zero, &partials] {
                stream.retain_tensor(tensor);
            }
            let reduce = stream.load_pipeline(LibraryName::Quantize, "dynamic_quant_range")?;
            let parameters =
                stream.load_pipeline(LibraryName::Quantize, "dynamic_quant_parameters")?;
            let quantize = stream.load_pipeline(LibraryName::Quantize, "dynamic_quant_u8")?;
            let command_buffer = stream.command_buffer();
            command_buffer.encode(|encoder| {
                encoder.set_compute_pipeline_state(&reduce);
                encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
                encoder.set_metal_tensor(1, &partials, metal::MTLResourceUsage::Write);
                encoder.set_slice(2, &[count, groups]);
                encoder.dispatch_thread_groups(
                    MTLSize::new(groups as _, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            });
            command_buffer.encode(|encoder| {
                encoder.set_compute_pipeline_state(&parameters);
                encoder.set_metal_tensor(0, &partials, metal::MTLResourceUsage::Read);
                encoder.set_metal_tensor(1, &scale, metal::MTLResourceUsage::Write);
                encoder.set_metal_tensor(2, &zero, metal::MTLResourceUsage::Write);
                encoder.set_slice(3, &[groups]);
                encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
            });
            if count > 0 {
                command_buffer.encode(|encoder| {
                    encoder.set_compute_pipeline_state(&quantize);
                    encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
                    encoder.set_metal_tensor(1, &scale, metal::MTLResourceUsage::Read);
                    encoder.set_metal_tensor(2, &zero, metal::MTLResourceUsage::Read);
                    encoder.set_metal_tensor(3, &quantized, metal::MTLResourceUsage::Write);
                    encoder.set_slice(4, &[count]);
                    encoder.dispatch_thread_groups(
                        MTLSize::new(count.div_ceil(256) as _, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                });
            }
            Ok(())
        })?;
        Ok(tvec!(
            quantized.into_tensor().into_tvalue(),
            scale.into_tensor().into_tvalue(),
            zero.into_tensor().into_tvalue()
        ))
    }
}

impl TypedOp for MetalDynamicQuantizeLinearU8 {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            ensure!(facts.len() == 1 && facts[0].datum_type == DatumType::F32);
            DynamicQuantizeLinearU8.output_facts(facts)
        })
    }

    as_op!();
}

crate::register_metal_op!(DynamicQuantizeLinearU8, |source, node, _op| {
    rule_if!(source.node_input_facts(node.id)?[0].datum_type == DatumType::F32);
    Ok(Some(Box::new(MetalDynamicQuantizeLinearU8)))
});
