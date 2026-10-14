use super::OnnxTestCase;
use infra::TestSuite;
use std::path::PathBuf;
use tract_hir::internal::Approximation;

mod nnef_contracts;
mod quantized_ops;

/// Focused import and serialization corner cases with independent reference outputs.
pub fn suite() -> TestSuite {
    TestSuite::default()
        .with("quantized_ops", quantized_ops::suite())
        .with("nnef_contracts", nnef_contracts::suite())
}

fn cases(group: &str, names: &[&str], guards: &[&str]) -> TestSuite {
    let mut suite = TestSuite::default();
    for name in names.iter().chain(guards) {
        let case = OnnxTestCase {
            path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("test-cases")
                .join(group)
                .join(name),
            ignore_output_type: false,
            ignore_output_shapes: false,
            input: None,
            exact: !name.starts_with("dynamic_lstm")
                && !name.starts_with("kokoro_phase_zero")
                && !name.starts_with("fast_gelu"),
            approximation: Some(Approximation::Custom(1e-6, 1e-6, 0.)),
            transform: name.starts_with("kokoro_phase_zero").then_some("kokoro_stabilize_phase"),
        };
        suite.add(name, case);
    }
    suite
}
