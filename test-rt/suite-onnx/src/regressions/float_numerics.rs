use super::*;

pub fn suite() -> TestSuite {
    cases(
        "float_numerics",
        &[
            "fast_gelu_tails",
            "fast_gelu_nonfinite",
            "gemm_f16_scalars",
            "random_like_name",
            "random_f16_distributions",
            "division_f16_subnormal",
        ],
        &[],
    )
}
