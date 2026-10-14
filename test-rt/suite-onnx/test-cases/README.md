The ONNX suite fetches upstream ONNX backend cases separately. These local cases
are focused regressions, not copies of an external test collection.

Routine convolution, integer matmul, contrib elementwise/matmul, dynamically
quantized LSTM, quantization, and STFT coverage lives in `src/generated`.
Those property tests build and import ONNX graphs, use independent scalar
references (including a float64 DFT), and shrink failures through `infra`.
The same cases run in CPU, Metal, and actual NNEF round-trip runtimes.
`PROPTEST_CASES` and `PROPTEST_RNG_SEED` control the existing property-test runner.

The topic directories retain corner cases for import patterns, non-finite
values, empty tensors, precision thresholds, and serialization contracts.
Their generators use ONNX Runtime references unless an explicit mathematical
reference or invariant is supplied. Random distributions are checked through
seeded statistical bounds, not cross-runtime sample identity.

Each checked model has one readable `test_data_set_0/data.json`. Tensor `dtype`
is ONNX TensorProto's numeric type code. `shape` is the complete tensor shape;
`values` is a flattened tile repeated to fill that shape. Non-finite values use
strings. A `runs` array of `[value, count]` pairs can replace `values` for
long piecewise-constant tiles. Empty tensors have an empty tile. Repeated zeros and boolean masks
therefore do not require thousands of stored samples.

To regenerate a topic in an isolated directory, install `onnx==1.20.1`,
`onnxruntime==1.23.2`, and `numpy==2.4.1`, then run:

```
TRACT_FIXTURE_ROOT=/tmp/tract-fixtures python test-cases/<topic>/generate.py
```

Python and ONNX Runtime are not required to run the checked Rust tests.
