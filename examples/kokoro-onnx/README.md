# Kokoro ONNX on CPU

This example runs `kokoro-v1.0.fp16.onnx` from the
[kokoro-onnx v1.0 model release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0).
That export contains mixed signed/unsigned quantized operators despite its filename.
It takes int64 `tokens` (including zero boundary tokens), a float32 voice `style`
of shape `[1, 256]`, and a float32 `speed` of shape `[1]`. Audio is mono, 24 kHz.

From this directory, download the model and voices:

```sh
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
python3 -m venv .venv
.venv/bin/python -m pip install numpy
.venv/bin/python prepare.py voices-v1.0.bin
cargo run -p example-kokoro-onnx --profile opt-no-lto -- kokoro-v1.0.fp16.onnx inputs.npz audio.npz
.venv/bin/python prepare.py voices-v1.0.bin --output audio.npz
```

`prepare.py` supplies the phonemes for “Hello world” with the `af_sarah` voice.
Use `--voice` and `--speed` to change those inputs. For arbitrary text, use an
external phonemizer/tokenizer with the export's vocabulary and select the voice
style row indexed by the phoneme count minus one. The Rust runner accepts any
compatible input bundle; it does not perform text normalization or phonemization.
Additional input/output pairs on the command line reuse the same compiled model.

The runner uses tract's public Rust API. It discards intermediate shape annotations
and the named output-length annotation so tract can infer the runtime audio length.
Token length is constrained to 2–512, including the boundary tokens. Symbolic
integer-reciprocal resizing checks its inferred size against the actual floating-point
size at runtime; an incompatible size is reported instead of propagated.

This is CPU support for this particular export. Other exports can use different
operators (including `DynamicQuantizeLSTM`), and GPU execution and NNEF export are
not exercised by this example. Floating-point and quantized rounding differences
can change the waveform between runtimes and ONNX Runtime optimization levels.
