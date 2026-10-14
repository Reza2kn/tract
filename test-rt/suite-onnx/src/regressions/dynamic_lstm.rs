use super::*;

pub fn suite() -> TestSuite {
    cases(
        "dynamic_lstm",
        &[
            "dynamic_lstm_zero",
            "dynamic_quantize_zero",
            "dynamic_quantize_nan",
            "dynamic_quantize_empty",
        ],
        &[],
    )
}
