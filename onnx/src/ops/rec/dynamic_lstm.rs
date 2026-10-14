use crate::model::ParsingContext;
use crate::model::optional_inputs;
use crate::model::optional_outputs;
use crate::ops::math::mat_mul_integer::MatMulInteger;
use crate::ops::quant::DynamicQuantizeLinearU8;
use crate::pb::NodeProto;
use tract_core::ops::array::TypedConcat;
use tract_core::ops::cast::cast;
use tract_core::ops::change_axes::wire_with_rank_broadcast;
use tract_core::ops::lstm_cell::LstmEpilogue;
use tract_core::ops::math;
use tract_core::ops::scan::InputMapping;
use tract_core::ops::scan::OutputMapping;
use tract_core::ops::scan::Scan;
use tract_core::ops::scan::ScanInfo;
use tract_hir::internal::*;
use tract_hir::ops::array::Slice;

/// Import the Microsoft dynamic LSTM's standard-activation, full-sequence variant.
pub fn dynamic_lstm(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let direction = node.get_attr_opt::<String>("direction")?.unwrap_or("forward".into());
    ensure!(matches!(direction.as_str(), "forward" | "reverse" | "bidirectional"));
    ensure!(
        node.get_attr_opt::<i64>("input_forget")?.unwrap_or(0) == 0,
        "DynamicQuantizeLSTM coupled input/forget gates are not supported"
    );
    ensure!(
        node.get_attr_opt::<f32>("clip")?.is_none(),
        "DynamicQuantizeLSTM clip is not supported"
    );
    ensure!(
        node.get_attr_opt_vec::<f32>("activation_alpha")?.is_none()
            && node.get_attr_opt_vec::<f32>("activation_beta")?.is_none(),
        "DynamicQuantizeLSTM activation parameters are not supported"
    );
    let directions = if direction == "bidirectional" { 2 } else { 1 };
    if let Some(activations) = node.get_attr_opt_vec::<String>("activations")? {
        ensure!(
            activations == ["Sigmoid", "Tanh", "Tanh"].repeat(directions),
            "DynamicQuantizeLSTM requires standard activations"
        );
    }
    let slots: TVec<_> = optional_inputs(node).take(12).collect();
    ensure!(
        slots.len() == 12
            && slots[..3].iter().all(Option::is_some)
            && slots[8..].iter().all(Option::is_some),
        "Missing DynamicQuantizeLSTM input"
    );
    ensure!(slots[7].is_none(), "DynamicQuantizeLSTM peepholes are not supported");
    let outputs = optional_outputs(node).take(3).collect();
    Ok((
        expand(DynamicLstm {
            slots,
            outputs,
            directions,
            reverse: direction == "reverse",
            hidden: node.get_attr("hidden_size")?,
        }),
        vec![],
    ))
}

/// Dynamic eight-bit LSTM with sequence-wide input quantization and per-step hidden quantization.
/// Standard activations and absent (or constant full-length) sequence lengths are supported.
#[derive(Debug, Clone)]
struct DynamicLstm {
    slots: TVec<Option<usize>>,
    outputs: TVec<Option<usize>>,
    directions: usize,
    reverse: bool,
    hidden: usize,
}

impl Expansion for DynamicLstm {
    fn name(&self) -> StaticName {
        "DynamicQuantizeLSTM".into()
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(self.outputs.iter().flatten().count())
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, self.slots.iter().flatten().count())?;
        check_output_arity(outputs, self.outputs.iter().flatten().count())?;
        ensure!(self.hidden > 0, "Hidden size must be positive");
        for i in [0, 1, 2] {
            s.equals(&inputs[i].rank, 3)?;
        }
        s.equals(&inputs[0].datum_type, DatumType::F32)?;
        s.equals(&inputs[1].datum_type, &inputs[2].datum_type)?;
        for i in [1, 2] {
            s.given(&inputs[i].datum_type, |_, dt| {
                ensure!(matches!(dt, DatumType::I8 | DatumType::U8), "Expected eight-bit weights");
                Ok(())
            })?;
            s.equals(&inputs[i].shape[0], self.directions.to_dim())?;
            s.equals(&inputs[i].shape[2], (4 * self.hidden).to_dim())?;
        }
        s.equals(&inputs[1].shape[1], &inputs[0].shape[2])?;
        s.equals(&inputs[2].shape[1], self.hidden.to_dim())?;
        for (weight, scale, zero) in [(1, 8, 9), (2, 10, 11)] {
            let scale = self.slots[scale].unwrap();
            let zero = self.slots[zero].unwrap();
            s.equals(&inputs[scale].datum_type, DatumType::F32)?;
            s.equals(&inputs[zero].datum_type, &inputs[weight].datum_type)?;
            s.equals(&inputs[scale].shape, &inputs[zero].shape)?;
            s.given(&inputs[scale].shape, move |_, shape| {
                ensure!(
                    shape == tvec!(self.directions.to_dim())
                        || shape == tvec!(self.directions.to_dim(), (4 * self.hidden).to_dim()),
                    "Expected per-direction or per-channel quantization parameters"
                );
                Ok(())
            })?;
        }
        if let Some(b) = self.slots[3] {
            s.equals(&inputs[b].datum_type, DatumType::F32)?;
            s.equals(
                &inputs[b].shape,
                tvec!(self.directions.to_dim(), (8 * self.hidden).to_dim()),
            )?;
        }
        if let Some(lens) = self.slots[4] {
            s.equals(&inputs[lens].datum_type, DatumType::I32)?;
            s.equals(&inputs[lens].rank, 1)?;
            s.equals(&inputs[lens].shape[0], &inputs[0].shape[1])?;
        }
        for slot in [5, 6] {
            if let Some(input) = self.slots[slot] {
                s.equals(&inputs[input].datum_type, DatumType::F32)?;
                s.equals(&inputs[input].rank, 3)?;
                s.equals(&inputs[input].shape[0], self.directions.to_dim())?;
                s.equals(&inputs[input].shape[1], &inputs[0].shape[1])?;
                s.equals(&inputs[input].shape[2], self.hidden.to_dim())?;
            }
        }
        for (original_slot, output) in self.outputs.iter().enumerate() {
            if let Some(output) = output {
                s.equals(&outputs[*output].datum_type, DatumType::F32)?;
                if original_slot == 0 {
                    s.equals(&outputs[*output].rank, 4)?;
                    s.equals(&outputs[*output].shape[0], &inputs[0].shape[0])?;
                    s.equals(&outputs[*output].shape[1], self.directions.to_dim())?;
                    s.equals(&outputs[*output].shape[2], &inputs[0].shape[1])?;
                    s.equals(&outputs[*output].shape[3], self.hidden.to_dim())?;
                } else {
                    s.equals(&outputs[*output].rank, 3)?;
                    s.equals(&outputs[*output].shape[0], self.directions.to_dim())?;
                    s.equals(&outputs[*output].shape[1], &inputs[0].shape[1])?;
                    s.equals(&outputs[*output].shape[2], self.hidden.to_dim())?;
                }
            }
        }
        Ok(())
    }

    fn wire(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        if let Some(slot) = self.slots[4] {
            let fact = model.outlet_fact(inputs[slot])?;
            let lengths =
                fact.konst.as_ref().context("Non constant sequence lengths are not supported")?;
            let seq = model.outlet_fact(inputs[0])?.shape[0].to_i64()?;
            ensure!(
                lengths.to_plain_array_view::<i32>()?.iter().all(|&n| n as i64 == seq),
                "Only full-length sequences are supported"
            );
        }
        let x = model.wire_node(
            format!("{name}.quantize_input"),
            DynamicQuantizeLinearU8,
            &[inputs[0]],
        )?;
        let mut sides = tvec!();
        for dir in 0..self.directions {
            sides.push(self.wire_direction(&format!("{name}.{dir}"), model, inputs, &x, dir)?);
        }
        let mut result = tvec!();
        for (slot, output) in self.outputs.iter().enumerate() {
            if output.is_some() {
                let out = if self.directions == 1 {
                    sides[0][slot]
                } else {
                    model.wire_node(
                        format!("{name}.merge_{slot}"),
                        TypedConcat::new(usize::from(slot == 0)),
                        &[sides[0][slot], sides[1][slot]],
                    )?[0]
                };
                result.push(out);
            }
        }
        Ok(result)
    }
}

impl DynamicLstm {
    fn direction_input(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
        slot: usize,
        dir: usize,
    ) -> TractResult<OutletId> {
        let x = model.wire_node(
            format!("{name}.slice_{slot}"),
            Slice::new(0, dir, dir + 1),
            &[inputs[self.slots[slot].unwrap()]],
        )?[0];
        Ok(model.wire_node(format!("{name}.squeeze_{slot}"), AxisOp::Rm(0), &[x])?[0])
    }

    fn wire_direction(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
        x: &[OutletId],
        dir: usize,
    ) -> TractResult<TVec<OutletId>> {
        let w = self.direction_input(name, model, inputs, 1, dir)?;
        let ws = self.direction_input(name, model, inputs, 8, dir)?;
        let wz = self.direction_input(name, model, inputs, 9, dir)?;
        let projected = quantized_product(&format!("{name}.input_product"), model, x, w, ws, wz)?;
        let chunk = if self.reverse || dir == 1 { -1 } else { 1 };
        let mut body = TypedModel { symbols: model.symbols.clone(), ..TypedModel::default() };
        let mut outer = tvec!(projected);
        let mut mapping = vec![InputMapping::Scan(ScanInfo { axis: 0, chunk })];
        let mut fact = model.outlet_fact(projected)?.without_value();
        fact.shape.set(0, 1.to_dim());
        let projection = body.add_source("projection", fact)?;
        let projection = body.wire_node("gates", AxisOp::Rm(0), &[projection])?[0];
        let mut recurrent = tvec!();
        for slot in [2, 10, 11] {
            let input = self.direction_input(name, model, inputs, slot, dir)?;
            outer.push(input);
            mapping.push(InputMapping::Full);
            recurrent.push(body.add_source(
                format!("recurrent_{slot}"),
                model.outlet_fact(input)?.without_value(),
            )?);
        }
        let mut states = tvec!();
        for slot in [5, 6] {
            let initial = if self.slots[slot].is_some() {
                let state = self.direction_input(name, model, inputs, slot, dir)?;
                model.wire_node(format!("{name}.initial_{slot}"), AxisOp::Add(0), &[state])?[0]
            } else {
                let batch = model.outlet_fact(inputs[0])?.shape[1].to_usize()?;
                model.add_const(
                    format!("{name}.initial_{slot}"),
                    tensor0(0f32).broadcast_scalar_to_shape(&[1, batch, self.hidden])?,
                )?
            };
            outer.push(initial);
            mapping.push(InputMapping::State);
            let state = body
                .add_source(format!("state_{slot}"), model.outlet_fact(initial)?.without_value())?;
            states
                .push(body.wire_node(format!("squeeze_state_{slot}"), AxisOp::Rm(0), &[state])?[0]);
        }
        let hq = body.wire_node("quantize_hidden", DynamicQuantizeLinearU8, &[states[0]])?;
        let rh = quantized_product(
            "hidden_product",
            &mut body,
            &hq,
            recurrent[0],
            recurrent[1],
            recurrent[2],
        )?;
        let mut preact = body.wire_node("preact", math::add(), &[projection, rh])?[0];
        if self.slots[3].is_some() {
            let b = self.direction_input(name, model, inputs, 3, dir)?;
            let wb =
                model.wire_node(format!("{name}.wb"), Slice::new(0, 0, 4 * self.hidden), &[b])?[0];
            let rb = model.wire_node(
                format!("{name}.rb"),
                Slice::new(0, 4 * self.hidden, 8 * self.hidden),
                &[b],
            )?[0];
            let bias = model.wire_node(format!("{name}.bias"), math::add(), &[wb, rb])?[0];
            outer.push(bias);
            mapping.push(InputMapping::Full);
            let bias = body.add_source("bias", model.outlet_fact(bias)?.without_value())?;
            preact =
                wire_with_rank_broadcast("biased", &mut body, math::add(), &[preact, bias])?[0];
        }
        let cells =
            body.wire_node("cell", LstmEpilogue { hidden: self.hidden }, &[preact, states[1]])?;
        let h = body.wire_node("h", AxisOp::Add(0), &[cells[0]])?[0];
        let c = body.wire_node("c", AxisOp::Add(0), &[cells[1]])?[0];
        body.select_output_outlets(&[h, c])?;
        let outputs = vec![
            OutputMapping {
                state: true,
                scan: Some((0, ScanInfo { axis: 0, chunk })),
                last_value_slot: Some(1),
                full_dim_hint: None,
            },
            OutputMapping {
                state: true,
                scan: None,
                last_value_slot: Some(2),
                full_dim_hint: None,
            },
        ];
        let outputs = model.wire_node(
            format!("{name}.scan"),
            Scan::new(body, mapping, outputs, 0)?,
            &outer,
        )?;
        let y = model.wire_node(format!("{name}.y"), AxisOp::Add(1), &[outputs[0]])?[0];
        Ok(tvec!(y, outputs[1], outputs[2]))
    }
}

fn quantized_product(
    name: &str,
    model: &mut TypedModel,
    x: &[OutletId],
    w: OutletId,
    scale: OutletId,
    zero: OutletId,
) -> TractResult<OutletId> {
    let product = MatMulInteger::new(Some(2), Some(3)).wire(
        &format!("{name}.integer"),
        model,
        &[x[0], w, x[2], zero],
    )?[0];
    let product = model.wire_node(format!("{name}.float"), cast(DatumType::F32), &[product])?[0];
    let scale =
        wire_with_rank_broadcast(format!("{name}.scale"), model, math::mul(), &[x[1], scale])?[0];
    Ok(wire_with_rank_broadcast(name, model, math::mul(), &[product, scale])?[0])
}
