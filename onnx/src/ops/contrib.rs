use crate::model::OnnxOpRegister;
use crate::model::ParsingContext;
use crate::model::optional_inputs;
use crate::pb::NodeProto;
use tract_core::broadcast::multi_broadcast;
use tract_core::ops::cast::cast;
use tract_core::ops::change_axes::AxisOp;
use tract_core::ops::change_axes::wire_with_rank_broadcast;
use tract_core::ops::einsum::EinSum;
use tract_core::ops::math;
use tract_hir::internal::*;
use tract_hir::ops::array::Concat;
use tract_hir::ops::matmul::MatMulInference;
use tract_hir::ops::matmul::compute_shapes;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("QLinearAdd", qlinear_elementwise);
    reg.insert("QLinearLeakyRelu", qlinear_elementwise);
    reg.insert("QLinearSigmoid", qlinear_elementwise);
    reg.insert("QLinearConcat", qlinear_concat);
    reg.insert("FusedMatMul", fused_mat_mul);
}

/// Scalar scale and optional zero point of an eight-bit tensor.
#[derive(Clone, Debug)]
struct QuantParams {
    scale: usize,
    zero: Option<usize>,
}

impl QuantParams {
    fn rules<'r, 'p: 'r>(
        &self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        dt: &'p TensorProxy,
    ) -> InferenceResult {
        s.equals(&inputs[self.scale].datum_type, DatumType::F32)?;
        for slot in Some(self.scale).into_iter().chain(self.zero) {
            s.equals(&inputs[slot].rank, 0)?;
        }
        if let Some(zero) = self.zero {
            s.equals(&inputs[zero].datum_type, &dt.datum_type)?;
        }
        Ok(())
    }

    fn dequantize(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
        x: OutletId,
    ) -> TractResult<OutletId> {
        let mut x = model.wire_node(format!("{name}.cast"), cast(DatumType::F32), &[x])?[0];
        if let Some(zero) = self.zero {
            let z =
                model.wire_node(format!("{name}.zero"), cast(DatumType::F32), &[inputs[zero]])?[0];
            x = wire_with_rank_broadcast(format!("{name}.sub"), model, math::sub(), &[x, z])?[0];
        }
        Ok(wire_with_rank_broadcast(name, model, math::mul(), &[x, inputs[self.scale]])?[0])
    }

    fn quantize(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
        x: OutletId,
        dt: DatumType,
    ) -> TractResult<TVec<OutletId>> {
        let (lo, hi) = match dt {
            DatumType::I8 => (-128f32, 127f32),
            DatumType::U8 => (0f32, 255f32),
            _ => bail!("Expected eight-bit quantized tensor, got {dt:?}"),
        };
        let mut x = wire_with_rank_broadcast(
            format!("{name}.div"),
            model,
            math::div(),
            &[x, inputs[self.scale]],
        )?;
        x = model.wire_node(format!("{name}.round"), math::round_half_to_even(), &x)?;
        if let Some(zero) = self.zero {
            let z =
                model.wire_node(format!("{name}.zero"), cast(DatumType::F32), &[inputs[zero]])?[0];
            x = wire_with_rank_broadcast(format!("{name}.add"), model, math::add(), &[x[0], z])?;
        }
        let lo = model.add_const(format!("{name}.lo"), tensor0(lo))?;
        let hi = model.add_const(format!("{name}.hi"), tensor0(hi))?;
        x = wire_with_rank_broadcast(format!("{name}.min"), model, math::min(), &[x[0], hi])?;
        x = wire_with_rank_broadcast(format!("{name}.max"), model, math::max(), &[x[0], lo])?;
        model.wire_node(name, cast(dt), &x)
    }
}

fn qlinear_elementwise(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut slots = optional_inputs(node);
    let x = slots.next().flatten().context("Missing X")?;
    let xp = QuantParams {
        scale: slots.next().flatten().context("Missing X_scale")?,
        zero: slots.next().flatten(),
    };
    let op = match node.op_type.as_str() {
        "QLinearAdd" => {
            let b = slots.next().flatten().context("Missing B")?;
            let bp = QuantParams {
                scale: slots.next().flatten().context("Missing B_scale")?,
                zero: slots.next().flatten(),
            };
            QuantizedKind::Add(b, bp)
        }
        "QLinearLeakyRelu" => QuantizedKind::LeakyRelu(node.get_attr_opt("alpha")?.unwrap_or(0.01)),
        "QLinearSigmoid" => QuantizedKind::Sigmoid,
        _ => unreachable!(),
    };
    let yp = QuantParams {
        scale: slots.next().flatten().context("Missing output scale")?,
        zero: slots.next().flatten(),
    };
    Ok((expand(QuantizedElementWise { x, xp, yp, op }), vec![]))
}

#[derive(Clone, Debug)]
enum QuantizedKind {
    Add(usize, QuantParams),
    LeakyRelu(f32),
    Sigmoid,
}

/// Microsoft scalar-quantized addition and activations, with nearest-even rounding and saturation.
#[derive(Clone, Debug)]
struct QuantizedElementWise {
    x: usize,
    xp: QuantParams,
    yp: QuantParams,
    op: QuantizedKind,
}

impl Expansion for QuantizedElementWise {
    fn name(&self) -> StaticName {
        match self.op {
            QuantizedKind::Add(..) => "QLinearAdd",
            QuantizedKind::LeakyRelu(..) => "QLinearLeakyRelu",
            QuantizedKind::Sigmoid => "QLinearSigmoid",
        }
        .into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        let arity = 3
            + usize::from(self.xp.zero.is_some())
            + usize::from(self.yp.zero.is_some())
            + if let QuantizedKind::Add(_, ref bp) = self.op {
                2 + usize::from(bp.zero.is_some())
            } else {
                0
            };
        check_input_arity(inputs, arity)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[self.x].datum_type, &outputs[0].datum_type)?;
        s.given(&inputs[self.x].datum_type, |_, dt| {
            ensure!(matches!(dt, DatumType::I8 | DatumType::U8), "Expected eight-bit input");
            Ok(())
        })?;
        self.xp.rules(s, inputs, &inputs[self.x])?;
        self.yp.rules(s, inputs, &outputs[0])?;
        if let QuantizedKind::Add(b, ref bp) = self.op {
            s.equals(&inputs[self.x].datum_type, &inputs[b].datum_type)?;
            bp.rules(s, inputs, &inputs[b])?;
            s.given_2(&inputs[self.x].shape, &inputs[b].shape, |s, a, b| {
                s.equals(&outputs[0].shape, multi_broadcast(&[a, b])?)
            })?;
        } else {
            s.equals(&inputs[self.x].shape, &outputs[0].shape)?;
        }
        Ok(())
    }

    fn wire(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let dt = model.outlet_fact(inputs[self.x])?.datum_type;
        let x = self.xp.dequantize(&format!("{name}.x"), model, inputs, inputs[self.x])?;
        let x = match self.op {
            QuantizedKind::Add(b, ref bp) => {
                let b = bp.dequantize(&format!("{name}.b"), model, inputs, inputs[b])?;
                wire_with_rank_broadcast(format!("{name}.sum"), model, math::add(), &[x, b])?[0]
            }
            QuantizedKind::LeakyRelu(alpha) => model.wire_node(
                format!("{name}.leaky"),
                tract_core::ops::nn::leaky_relu(alpha),
                &[x],
            )?[0],
            QuantizedKind::Sigmoid => {
                model.wire_node(format!("{name}.sigmoid"), tract_core::ops::nn::sigmoid(), &[x])?[0]
            }
        };
        self.yp.quantize(name, model, inputs, x, dt)
    }
}

fn qlinear_concat(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    ensure!(
        node.input.len() >= 5 && (node.input.len() - 2).is_multiple_of(3),
        "QLinearConcat expects tensor/scale/zero-point triples"
    );
    let axis = node.get_attr("axis")?;
    Ok((expand(QuantizedConcat { axis, count: (node.input.len() - 2) / 3 }), vec![]))
}

/// Concatenation with scalar requantization of each input to the output's quantization parameters.
#[derive(Clone, Debug)]
struct QuantizedConcat {
    axis: i64,
    count: usize,
}

impl Expansion for QuantizedConcat {
    fn name(&self) -> StaticName {
        "QLinearConcat".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 2 + 3 * self.count)?;
        check_output_arity(outputs, 1)?;
        QuantParams { scale: 0, zero: Some(1) }.rules(s, inputs, &outputs[0])?;
        s.given_all((0..self.count).map(|i| &inputs[2 + 3 * i].shape), |s, shapes| {
            let rank = shapes[0].len();
            let axis = if self.axis < 0 { self.axis + rank as i64 } else { self.axis };
            ensure!((0..rank as i64).contains(&axis), "QLinearConcat axis out of range");
            let axis = axis as usize;
            let mut shape = shapes[0].clone();
            for other in &shapes[1..] {
                ensure!(other.len() == rank, "QLinearConcat ranks must match");
                for dim in 0..rank {
                    if dim != axis {
                        ensure!(shape[dim] == other[dim], "QLinearConcat dimensions must match");
                    }
                }
                shape[axis] += &other[axis];
            }
            s.equals(&outputs[0].shape, shape)
        })?;
        for i in 0..self.count {
            let x = 2 + 3 * i;
            s.equals(&inputs[x].datum_type, &outputs[0].datum_type)?;
            QuantParams { scale: x + 1, zero: Some(x + 2) }.rules(s, inputs, &inputs[x])?;
        }
        Ok(())
    }

    fn wire(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let mut tensors = tvec!();
        let dt = model.outlet_fact(inputs[1])?.datum_type;
        for i in 0..self.count {
            let x = 2 + 3 * i;
            let x = QuantParams { scale: x + 1, zero: Some(x + 2) }.dequantize(
                &format!("{name}.input{i}"),
                model,
                inputs,
                inputs[x],
            )?;
            tensors.push(
                QuantParams { scale: 0, zero: Some(1) }.quantize(
                    &format!("{name}.requant{i}"),
                    model,
                    inputs,
                    x,
                    dt,
                )?[0],
            );
        }
        Concat::new(self.axis).wire(name, model, &tensors)
    }
}

fn fused_mat_mul(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    Ok((
        expand(FusedMatMul {
            matmul: MatMulInference::default()
                .with_a_trans(node.get_attr_opt("transA")?.unwrap_or(false))
                .with_b_trans(node.get_attr_opt("transB")?.unwrap_or(false)),
            trans_batch_a: node.get_attr_opt("transBatchA")?.unwrap_or(false),
            trans_batch_b: node.get_attr_opt("transBatchB")?.unwrap_or(false),
            alpha: node.get_attr_opt("alpha")?.unwrap_or(1.),
        }),
        vec![],
    ))
}

/// NumPy matrix multiplication with matrix/batch transpositions and a scalar product multiplier.
#[derive(Clone, Debug)]
struct FusedMatMul {
    matmul: MatMulInference,
    trans_batch_a: bool,
    trans_batch_b: bool,
    alpha: f32,
}

impl Expansion for FusedMatMul {
    fn name(&self) -> StaticName {
        "FusedMatMul".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 2)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &inputs[1].datum_type)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.given(&inputs[0].datum_type, |_, dt| {
            ensure!(dt.is_float(), "FusedMatMul expects floating-point operands");
            Ok(())
        })?;
        s.given_2(&inputs[0].shape, &inputs[1].shape, |s, mut a, mut b| {
            ensure!(!a.is_empty() && !b.is_empty(), "FusedMatMul operands must have positive rank");
            if self.trans_batch_a || self.trans_batch_b {
                ensure!(
                    a.len() == b.len() && a.len() >= 3,
                    "Batch transpose requires matching ranks >= 3"
                );
                let rank = a.len();
                if self.trans_batch_a {
                    let axis = a.remove(0);
                    a.insert(rank - 2, axis);
                }
                if self.trans_batch_b {
                    let axis = b.remove(0);
                    b.insert(rank - 2, axis);
                }
            }
            let (_, _, _, shape) =
                compute_shapes(a, b, self.matmul.a_trans, self.matmul.b_trans, false)?;
            s.equals(&outputs[0].shape, shape)
        })?;
        Ok(())
    }

    fn wire(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let mut inputs = tvec!(inputs[0], inputs[1]);
        for (i, trans) in [self.trans_batch_a, self.trans_batch_b].into_iter().enumerate() {
            if trans {
                let rank = model.outlet_fact(inputs[i])?.rank();
                inputs[i] = model.wire_node(
                    format!("{name}.batch{i}"),
                    AxisOp::Move(0, rank - 2),
                    &[inputs[i]],
                )?[0];
            }
        }
        let x = if inputs.iter().all(|i| model.outlet_fact(*i).is_ok_and(|f| f.rank() == 1)) {
            let dt = model.outlet_fact(inputs[0])?.datum_type;
            model.wire_node(format!("{name}.matmul"), EinSum::new("k,k->".parse()?, dt), &inputs)?
        } else {
            self.matmul.wire(&format!("{name}.matmul"), model, &inputs)?
        };
        if self.alpha == 1. {
            return Ok(x);
        }
        let dt = model.outlet_fact(x[0])?.datum_type;
        let alpha = model
            .add_const(format!("{name}.alpha"), tensor0(self.alpha).cast_to_dt(dt)?.into_owned())?;
        wire_with_rank_broadcast(name, model, math::mul(), &[x[0], alpha])
    }
}
