use super::*;

pub fn suite() -> TestSuite {
    cases("nnef_contracts", &["quantized_conv_nhwc"], &[])
}
