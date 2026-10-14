use super::*;

pub fn suite() -> TestSuite {
    cases(
        "kokoro_phase",
        &["kokoro_phase_zero", "kokoro_phase_zero_f16", "kokoro_phase_zero_sine_f16"],
        &[],
    )
}
