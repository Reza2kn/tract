use infra::Test;
use infra::TestSuite;
use tract_core::internal::*;
use tract_core::ops::array::Topk;

#[derive(Clone, Debug)]
struct MultiOutputNames;
impl Test for MultiOutputNames {
    fn run_with_approx(&self, _: &str, runtime: &dyn Runtime, _: Approximation) -> TractResult<()> {
        let mut model = TypedModel::default();
        let a = model.add_source("a", f32::fact([3]))?;
        let b = model.add_source("b", f32::fact([3]))?;
        let k = model.add_const("k", tensor0(2i64))?;
        let op = Topk { axis: 0, largest: true, fallback_k: 2.to_dim() };
        let first = model.wire_node("pair", op.clone(), &[a, k])?;
        let second = model.wire_node("pair.1", op, &[b, k])?;
        model.select_output_outlets(&[first[0], first[1], second[0], second[1]])?;
        let actual = runtime.prepare(model)?.run(tvec!(
            tensor1(&[1f32, 3., 2.]).into_tvalue(),
            tensor1(&[7f32, 5., 9.]).into_tvalue()
        ))?;
        let expected =
            [tensor1(&[3f32, 2.]), tensor1(&[1i64, 2]), tensor1(&[9f32, 7.]), tensor1(&[2i64, 0])];
        for (a, b) in actual.iter().zip(&expected) {
            a.close_enough(b, Approximation::Exact)?;
        }
        Ok(())
    }
}

/// Multi-output identifiers must stay distinct after NNEF identifier escaping.
pub fn suite() -> TestSuite {
    TestSuite::default()
        .with("multi_output_names", MultiOutputNames)
        .with("scan_constant_bindings", ScanConstants)
}

#[derive(Clone, Debug)]
struct ScanConstants;
impl Test for ScanConstants {
    fn run_with_approx(&self, _: &str, runtime: &dyn Runtime, _: Approximation) -> TractResult<()> {
        use tract_core::ops::scan::*;
        let mut model = TypedModel::default();
        let input = model.add_source("input", f32::fact([2, 9]))?;
        let mut outputs = vec![];
        for shift in [3f32, 7.] {
            let mut body = TypedModel::default();
            let x = body.add_source("x", f32::fact([1, 9]))?;
            let offset = body.add_const("offset", tensor2(&[[shift; 9]]))?;
            let sum = body.wire_node("sum", tract_core::ops::math::add(), &[x, offset])?;
            body.select_output_outlets(&sum)?;
            let scan = Scan::new(
                body,
                vec![InputMapping::Scan(ScanInfo { axis: 0, chunk: 1 })],
                vec![OutputMapping {
                    state: false,
                    scan: Some((0, ScanInfo { axis: 0, chunk: 1 })),
                    last_value_slot: None,
                    full_dim_hint: None,
                }],
                0,
            )?;
            outputs.extend(model.wire_node(format!("scan_{shift}"), scan, &[input])?);
        }
        model.select_output_outlets(&outputs)?;
        let actual = runtime.prepare(model)?.run(tvec!(tensor2(&[[0f32; 9]; 2]).into_tvalue()))?;
        for (output, shift) in actual.iter().zip([3f32, 7.]) {
            output.close_enough(&tensor2(&[[shift; 9]; 2]), Approximation::Exact)?;
        }
        Ok(())
    }
}
