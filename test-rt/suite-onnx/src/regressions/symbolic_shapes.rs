use super::*;

pub fn suite() -> TestSuite {
    cases(
        "symbolic_shapes",
        &[
            "dynamic_range_resize",
            "symbolic_slice_bounds",
            "symbolic_resize_half_reciprocal",
            "folded_gather_shape",
            "folded_gather_shape_empty",
            "symbolic_deconv",
            "symbolic_cumsum",
            "topk_symbolic",
            "dynamic_lstm_symbolic_lengths",
        ],
        &["symbolic_resize_precision_guard", "symbolic_resize_half_precision_guard"],
    )
}
