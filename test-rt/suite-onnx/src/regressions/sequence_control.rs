use super::*;

pub fn suite() -> TestSuite {
    cases(
        "sequence_control",
        &[
            "sequence_repeat",
            "sequence_repeat_prefix",
            "sequence_repeat_infer_dim",
            "if_constant_rank",
        ],
        &[],
    )
}
