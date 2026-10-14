use super::*;
use infra::Test;

#[derive(Clone, Debug)]
struct InstanceNormF16;
impl Test for InstanceNormF16 {
    fn run_with_approx(&self, _: &str, runtime: &dyn Runtime, _: Approximation) -> TractResult<()> {
        let x: Vec<f32> = [[7., 9.].repeat(5000), [46., 54.].repeat(5000)].concat();
        let expected: Vec<f32> = [[-1., 1.].repeat(5000), [-0.75, 1.25].repeat(5000)].concat();
        check(
            NodeProto {
                op_type: "InstanceNormalization".into(),
                input: vec!["x".into(), "scale".into(), "bias".into()],
                output: vec!["y".into()],
                ..Default::default()
            },
            vec![("x", Tensor::from_shape(&[1, 2, 10000], &x)?.cast_to::<f16>()?.into_owned())],
            vec![
                ("scale", tensor1(&[f16::ONE, f16::ONE])),
                ("bias", tensor1(&[f16::ZERO, f16::from_f32(0.25)])),
            ],
            vec![Tensor::from_shape(&[1, 2, 10000], &expected)?.cast_to::<f16>()?.into_owned()],
            runtime,
            Approximation::Exact,
        )
    }
}
pub fn suite() -> TestSuite {
    TestSuite::default().with("instance_norm_accumulation", InstanceNormF16)
}
