use super::*;

pub fn suite() -> TestSuite {
    cases(
        "quantized_ops",
        &["quantized_matmul_axes", "nonfinite_quantization", "integer_matmul_constant"],
        &[],
    )
}
