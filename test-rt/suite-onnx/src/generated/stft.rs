use super::*;
use infra::Test;
use proptest::collection::vec;
use proptest::prelude::*;

#[derive(Clone, Debug)]
struct Stft {
    rank2: bool,
    complex: bool,
    window: bool,
    frame: usize,
    step: usize,
    x: Vec<f32>,
}
impl Arbitrary for Stft {
    type Parameters = usize;
    type Strategy = BoxedStrategy<Self>;
    fn arbitrary_with(frame: usize) -> Self::Strategy {
        (any::<bool>(), any::<bool>(), any::<bool>(), 1usize..5, vec(-8i32..9, frame + 9))
            .prop_map(move |(rank2, complex, window, step, x)| Self {
                rank2,
                complex: complex && !rank2,
                window,
                frame,
                step,
                x: x.into_iter().map(|x| x as f32 / 32.).collect(),
            })
            .boxed()
    }
}
impl Test for Stft {
    fn run_with_approx(&self, _: &str, runtime: &dyn Runtime, _: Approximation) -> TractResult<()> {
        let channels = if self.complex { 2 } else { 1 };
        let mut values = vec![];
        for x in &self.x {
            values.push(*x);
            if self.complex {
                values.push(-*x / 2.);
            }
        }
        let win: Vec<_> = (0..self.frame)
            .map(|i| {
                (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / (self.frame - 1) as f64).cos())
                    as f32
            })
            .collect();
        let count = (self.x.len() - self.frame) / self.step + 1;
        let bins = if self.complex { self.frame } else { self.frame / 2 + 1 };
        let mut reference = vec![];
        for t in 0..count {
            for bin in 0..bins {
                let mut re = 0f64;
                let mut im = 0f64;
                for (i, window) in win.iter().enumerate() {
                    let w = if self.window { *window as f64 } else { 1. };
                    let a = self.x[t * self.step + i] as f64 * w;
                    let b = if self.complex { -a / 2. } else { 0. };
                    let phase = -std::f64::consts::TAU * (bin * i) as f64 / self.frame as f64;
                    re += a * phase.cos() - b * phase.sin();
                    im += a * phase.sin() + b * phase.cos();
                }
                reference.extend([re as f32, im as f32]);
            }
        }
        let shape =
            if self.rank2 { vec![1, self.x.len()] } else { vec![1, self.x.len(), channels] };
        check(
            NodeProto {
                op_type: "STFT".into(),
                input: vec![
                    "x".into(),
                    "step".into(),
                    if self.window { "window".into() } else { "".into() },
                    "length".into(),
                ],
                output: vec!["y".into()],
                attribute: vec![int_attr("onesided", (!self.complex) as i64)],
                ..Default::default()
            },
            vec![("x", Tensor::from_shape(&shape, &values)?)],
            vec![
                ("step", tensor0(self.step as i64)),
                ("length", tensor0(self.frame as i64)),
                ("window", tensor1(&win)),
            ],
            vec![Tensor::from_shape(&[1, count, bins, 2], &reference)?],
            runtime,
            Approximation::Custom(1e-6, 1e-6, 0.),
        )
    }
}
pub fn suite() -> TestSuite {
    let mut s = TestSuite::default();
    s.add_arbitrary::<Stft>("power_of_two", 8);
    s.add_arbitrary::<Stft>("frame20", 20);
    s
}
