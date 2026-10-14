use super::*;
use infra::Test;
use proptest::collection::vec;
use proptest::prelude::*;

#[derive(Clone, Debug)]
struct DynamicQuantize(Vec<f32>);
impl Arbitrary for DynamicQuantize {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;
    fn arbitrary_with(_: ()) -> Self::Strategy {
        vec(-128i32..129, 1..4097)
            .prop_map(|x| Self(x.into_iter().map(|x| x as f32 / 32.).collect()))
            .boxed()
    }
}
impl Test for DynamicQuantize {
    fn run_with_approx(&self, _: &str, runtime: &dyn Runtime, _: Approximation) -> TractResult<()> {
        let lo = self.0.iter().copied().fold(0f32, f32::min);
        let hi = self.0.iter().copied().fold(0f32, f32::max);
        let scale = if hi == lo { 1. } else { (hi - lo) / 255. };
        let zero = (-lo / scale).round_ties_even().clamp(0., 255.) as u8;
        let expected: Vec<_> = self
            .0
            .iter()
            .map(|x| ((x / scale).round_ties_even() + zero as f32).clamp(0., 255.) as u8)
            .collect();
        check(
            NodeProto {
                op_type: "DynamicQuantizeLinear".into(),
                input: vec!["x".into()],
                output: vec!["y".into(), "scale".into(), "zero".into()],
                ..Default::default()
            },
            vec![("x", tensor1(&self.0))],
            vec![],
            vec![tensor1(&expected), tensor0(scale), tensor0(zero)],
            runtime,
            Approximation::Exact,
        )
    }
}
pub fn suite() -> TestSuite {
    let mut s = TestSuite::default();
    s.add_arbitrary::<DynamicQuantize>("reduction_and_rounding", ());
    s
}
