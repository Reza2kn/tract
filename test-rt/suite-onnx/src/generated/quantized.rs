use super::*;
use infra::Test;
use proptest::collection::vec;
use proptest::prelude::*;

#[derive(Clone, Debug)]
struct Quantized {
    signed: bool,
    present: [bool; 3],
    x: Vec<i32>,
    b: i32,
    op: usize,
}
impl Arbitrary for Quantized {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;
    fn arbitrary_with(_: ()) -> Self::Strategy {
        (any::<bool>(), any::<[bool; 3]>(), vec(0i32..256, 1..80), 0i32..256, 0usize..3)
            .prop_map(|(signed, present, mut x, mut b, op)| {
                if signed {
                    x.iter_mut().for_each(|x| *x -= 128);
                    b -= 128;
                }
                Self { signed, present, x, b, op }
            })
            .boxed()
    }
}
impl Test for Quantized {
    fn run_with_approx(&self, _: &str, runtime: &dyn Runtime, _: Approximation) -> TractResult<()> {
        let z = if self.signed { -3 } else { 120 };
        let alpha = if self.present[1] { 0.25f32 } else { 0.01f32 };
        let ys = if self.op == 2 && self.present[1] { 1f32 / 256. } else { 1. };
        let yz = if self.op == 2 { if self.signed { -128 } else { 0 } } else { z };
        let mut args = vec!["x", "xs", if self.present[0] { "xz" } else { "" }];
        if self.op == 0 {
            args.extend(["b", "bs", if self.present[1] { "bz" } else { "" }]);
        }
        args.extend(["ys", if self.present[2] { "yz" } else { "" }]);
        let y: Vec<_> = self
            .x
            .iter()
            .map(|x| {
                let x = (*x - if self.present[0] { z } else { 0 }) as f32 * 0.5;
                let value = match self.op {
                    0 => x + (self.b - if self.present[1] { z } else { 0 }) as f32 * 0.25,
                    1 => {
                        if x < 0. {
                            x * alpha
                        } else {
                            x
                        }
                    }
                    _ => 1. / (1. + (-x).exp()),
                };
                ((value / ys).round_ties_even() as i32 + if self.present[2] { yz } else { 0 })
                    .clamp(if self.signed { -128 } else { 0 }, if self.signed { 127 } else { 255 })
            })
            .collect();
        check(
            NodeProto {
                op_type: ["QLinearAdd", "QLinearLeakyRelu", "QLinearSigmoid"][self.op].into(),
                domain: "com.microsoft".into(),
                input: args.into_iter().map(Into::into).collect(),
                output: vec!["y".into()],
                attribute: if self.op == 1 && self.present[1] {
                    vec![float_attr("alpha", alpha)]
                } else {
                    vec![]
                },
                ..Default::default()
            },
            vec![("x", byte_tensor(&[self.x.len()], &self.x, self.signed))],
            vec![
                ("b", byte_tensor(&[], &[self.b], self.signed)),
                ("xs", tensor0(0.5f32)),
                ("bs", tensor0(0.25f32)),
                ("ys", tensor0(ys)),
                ("xz", byte_tensor(&[], &[z], self.signed)),
                ("bz", byte_tensor(&[], &[z], self.signed)),
                ("yz", byte_tensor(&[], &[yz], self.signed)),
            ],
            vec![byte_tensor(&[self.x.len()], &y, self.signed)],
            runtime,
            Approximation::Exact,
        )
    }
}
pub fn suite() -> TestSuite {
    let mut s = TestSuite::default();
    s.add_arbitrary::<Quantized>("elementwise_optional_zero_points", ());
    s.add_arbitrary::<Concat>("concat", ());
    s
}

#[derive(Clone, Debug)]
struct Concat {
    signed: bool,
    x: Vec<i32>,
}
impl Arbitrary for Concat {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;
    fn arbitrary_with(_: ()) -> Self::Strategy {
        (any::<bool>(), vec(0i32..256, 1..40))
            .prop_map(|(signed, mut x)| {
                if signed {
                    x.iter_mut().for_each(|x| *x -= 128);
                }
                Self { signed, x }
            })
            .boxed()
    }
}
impl Test for Concat {
    fn run_with_approx(&self, _: &str, runtime: &dyn Runtime, _: Approximation) -> TractResult<()> {
        let z = if self.signed { -3 } else { 120 };
        let expected: Vec<_> = [0.5f32, 0.25]
            .iter()
            .flat_map(|scale| {
                self.x.iter().map(move |x| {
                    (((x - z) as f32 * scale).round_ties_even() as i32 + z).clamp(
                        if self.signed { -128 } else { 0 },
                        if self.signed { 127 } else { 255 },
                    )
                })
            })
            .collect();
        check(
            NodeProto {
                op_type: "QLinearConcat".into(),
                domain: "com.microsoft".into(),
                input: ["ys", "z", "x", "xs", "z", "x", "bs", "z"]
                    .into_iter()
                    .map(Into::into)
                    .collect(),
                output: vec!["y".into()],
                attribute: vec![int_attr("axis", -1)],
                ..Default::default()
            },
            vec![("x", byte_tensor(&[self.x.len()], &self.x, self.signed))],
            vec![
                ("xs", tensor0(0.5f32)),
                ("bs", tensor0(0.25f32)),
                ("ys", tensor0(1f32)),
                ("z", byte_tensor(&[], &[z], self.signed)),
            ],
            vec![byte_tensor(&[2 * self.x.len()], &expected, self.signed)],
            runtime,
            Approximation::Exact,
        )
    }
}
