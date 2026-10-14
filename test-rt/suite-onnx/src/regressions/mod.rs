use super::{ExpectedFailure, OnnxTestCase};
use infra::TestSuite;
use std::path::PathBuf;
use tract_hir::internal::Approximation;

mod dynamic_lstm;
mod float_numerics;
mod nnef_contracts;
mod quantized_ops;
mod sequence_control;
mod symbolic_shapes;

/// Focused import and serialization corner cases with independent reference outputs.
pub fn suite() -> TestSuite {
    TestSuite::default()
        .with("quantized_ops", quantized_ops::suite())
        .with("dynamic_lstm", dynamic_lstm::suite())
        .with("symbolic_shapes", symbolic_shapes::suite())
        .with("float_numerics", float_numerics::suite())
        .with("nnef_contracts", nnef_contracts::suite())
        .with("sequence_control", sequence_control::suite())
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
        if guards.contains(name) {
            suite.add(name, ExpectedFailure { case, message: "differs from symbolic contract" });
        } else {
            suite.add(name, case);
        }
    }
    suite
}
