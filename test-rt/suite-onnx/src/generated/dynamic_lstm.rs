use super::*;
use infra::Test;
use proptest::collection::vec;
use proptest::prelude::*;

#[derive(Clone, Debug)]
struct Lstm {
    seq: usize,
    batch: usize,
    k: usize,
    h: usize,
    direction: usize,
    signed: bool,
    channel: bool,
    bias: bool,
    initial: bool,
    x: Vec<f32>,
    w: Vec<i32>,
    r: Vec<i32>,
}
impl Arbitrary for Lstm {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;
    fn arbitrary_with(_: ()) -> Self::Strategy {
        (
            1usize..5,
            1usize..3,
            1usize..4,
            1usize..4,
            0usize..3,
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
        )
            .prop_flat_map(|(seq, batch, k, h, direction, signed, channel, bias, initial)| {
                let dirs = if direction == 2 { 2 } else { 1 };
                (
                    vec(-16i32..17, seq * batch * k),
                    vec(-8i32..9, dirs * k * 4 * h),
                    vec(-8i32..9, dirs * h * 4 * h),
                )
                    .prop_map(move |(x, w, r)| Self {
                        seq,
                        batch,
                        k,
                        h,
                        direction,
                        signed,
                        channel,
                        bias,
                        initial,
                        x: x.into_iter().map(|x| x as f32 / 8.).collect(),
                        w,
                        r,
                    })
            })
            .boxed()
    }
}
fn quantized_reference(x: &[f32]) -> (Vec<i32>, f32) {
    let lo = x.iter().copied().fold(0f32, f32::min);
    let hi = x.iter().copied().fold(0f32, f32::max);
    let scale = if hi == lo { 1. } else { (hi - lo) / 255. };
    let z = (-lo / scale).round_ties_even().clamp(0., 255.);
    (
        x.iter()
            .map(|x| ((x / scale).round_ties_even() + z).clamp(0., 255.) as i32 - z as i32)
            .collect(),
        scale,
    )
}
impl Test for Lstm {
    fn run_with_approx(&self, _: &str, runtime: &dyn Runtime, _: Approximation) -> TractResult<()> {
        let dirs = if self.direction == 2 { 2 } else { 1 };
        let gates = 4 * self.h;
        let z = if self.signed { 0 } else { 110 };
        let w: Vec<_> = self.w.iter().map(|x| x + z).collect();
        let r: Vec<_> = self.r.iter().map(|x| x + z).collect();
        let scales: Vec<_> = (0..dirs * if self.channel { gates } else { 1 })
            .map(|i| if self.channel { (1 + i % 3) as f32 / 64. } else { 1. / 64. })
            .collect();
        let bias: Vec<_> = (0..dirs * 2 * gates).map(|i| (i as i32 % 7 - 3) as f32 / 32.).collect();
        let ih: Vec<_> =
            (0..dirs * self.batch * self.h).map(|i| (i as i32 % 5 - 2) as f32 / 16.).collect();
        let ic: Vec<_> = ih.iter().map(|x| x / 2.).collect();
        let mut yh = vec![0.; dirs * self.batch * self.h];
        let mut yc = yh.clone();
        let mut y = vec![0.; self.seq * yh.len()];
        let (qx, xscale) = quantized_reference(&self.x);
        for dir in 0..dirs {
            let mut hidden = if self.initial {
                ih[dir * self.batch * self.h..(dir + 1) * self.batch * self.h].to_vec()
            } else {
                vec![0.; self.batch * self.h]
            };
            let mut cell = if self.initial {
                ic[dir * self.batch * self.h..(dir + 1) * self.batch * self.h].to_vec()
            } else {
                hidden.clone()
            };
            for step in 0..self.seq {
                let t = if self.direction == 1 || dir == 1 { self.seq - 1 - step } else { step };
                let (qh, hscale) = quantized_reference(&hidden);
                let mut pre = vec![0.; self.batch * gates];
                for b in 0..self.batch {
                    for g in 0..gates {
                        let xp: i32 = (0..self.k)
                            .map(|k| {
                                qx[(t * self.batch + b) * self.k + k]
                                    * self.w[(dir * self.k + k) * gates + g]
                            })
                            .sum();
                        let hp: i32 = (0..self.h)
                            .map(|h| qh[b * self.h + h] * self.r[(dir * self.h + h) * gates + g])
                            .sum();
                        let ws = scales[if self.channel { dir * gates + g } else { dir }];
                        pre[b * gates + g] = xp as f32 * (xscale * ws)
                            + hp as f32 * (hscale * ws)
                            + if self.bias {
                                bias[dir * 2 * gates + g] + bias[dir * 2 * gates + gates + g]
                            } else {
                                0.
                            };
                    }
                }
                for b in 0..self.batch {
                    for h in 0..self.h {
                        let sigmoid = |x: f32| 1. / (1. + (-x).exp());
                        let v = &pre[b * gates..(b + 1) * gates];
                        cell[b * self.h + h] = sigmoid(v[2 * self.h + h]) * cell[b * self.h + h]
                            + sigmoid(v[h]) * v[3 * self.h + h].tanh();
                        hidden[b * self.h + h] =
                            sigmoid(v[self.h + h]) * cell[b * self.h + h].tanh();
                        y[((t * dirs + dir) * self.batch + b) * self.h + h] =
                            hidden[b * self.h + h];
                    }
                }
            }
            yh[dir * self.batch * self.h..(dir + 1) * self.batch * self.h].copy_from_slice(&hidden);
            yc[dir * self.batch * self.h..(dir + 1) * self.batch * self.h].copy_from_slice(&cell);
        }
        let qshape = if self.channel { vec![dirs, gates] } else { vec![dirs] };
        let zero = byte_tensor(&qshape, &vec![z; scales.len()], self.signed);
        let mut constants = vec![
            ("w", byte_tensor(&[dirs, self.k, gates], &w, self.signed)),
            ("r", byte_tensor(&[dirs, self.h, gates], &r, self.signed)),
            ("ws", Tensor::from_shape(&qshape, &scales)?),
            ("rs", Tensor::from_shape(&qshape, &scales)?),
            ("wz", zero.clone()),
            ("rz", zero),
        ];
        if self.bias {
            constants.push(("bias", Tensor::from_shape(&[dirs, 2 * gates], &bias)?));
        }
        if self.initial {
            constants.extend([
                ("ih", Tensor::from_shape(&[dirs, self.batch, self.h], &ih)?),
                ("ic", Tensor::from_shape(&[dirs, self.batch, self.h], &ic)?),
            ]);
        }
        check(
            NodeProto {
                op_type: "DynamicQuantizeLSTM".into(),
                domain: "com.microsoft".into(),
                input: vec![
                    "x",
                    "w",
                    "r",
                    if self.bias { "bias" } else { "" },
                    "",
                    if self.initial { "ih" } else { "" },
                    if self.initial { "ic" } else { "" },
                    "",
                    "ws",
                    "wz",
                    "rs",
                    "rz",
                ]
                .into_iter()
                .map(Into::into)
                .collect(),
                output: vec!["y".into(), "yh".into(), "yc".into()],
                attribute: vec![
                    int_attr("hidden_size", self.h as i64),
                    string_attr(
                        "direction",
                        ["forward", "reverse", "bidirectional"][self.direction],
                    ),
                ],
                ..Default::default()
            },
            vec![("x", Tensor::from_shape(&[self.seq, self.batch, self.k], &self.x)?)],
            constants,
            vec![
                Tensor::from_shape(&[self.seq, dirs, self.batch, self.h], &y)?,
                Tensor::from_shape(&[dirs, self.batch, self.h], &yh)?,
                Tensor::from_shape(&[dirs, self.batch, self.h], &yc)?,
            ],
            runtime,
            Approximation::Custom(2e-5, 2e-5, 0.),
        )
    }
}
pub fn suite() -> TestSuite {
    let mut s = TestSuite::default();
    s.add_arbitrary::<Lstm>("directions_quantization_and_state", ());
    let mut w = vec![0; 24];
    w[23] = -1;
    s.add(
        "bidirectional_single_step",
        Lstm {
            seq: 1,
            batch: 1,
            k: 1,
            h: 3,
            direction: 2,
            signed: true,
            channel: true,
            bias: false,
            initial: false,
            x: vec![0.125],
            w,
            r: vec![0; 72],
        },
    );
    s
}
