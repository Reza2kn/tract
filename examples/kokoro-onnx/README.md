# Kokoro ONNX on CPU and Metal

This example runs the FP32, mixed-quantized, and int8 CPU exports from the
[kokoro-onnx v1.0 model release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0).
`kokoro-v1.0.fp16.onnx` contains mixed signed/unsigned quantized operators despite its filename.
`kokoro-v1.0.int8.onnx` also uses dynamically quantized LSTMs.
The FP16-GPU-named release asset is byte-identical to the FP16-named one;
use `--runtime metal` (with `--features metal` when building the example) to execute supported operations on Metal.
The FP32, FP16-named, and int8 v0.19 exports from the
[original model release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files)
use the same runner and default smoke-test tokens. Use that release's `voices.bin`
with those models.
It takes int64 `tokens` (including zero boundary tokens), a float32 voice `style`
of shape `[1, 256]`, and a float32 `speed` of shape `[1]`. Audio is mono, 24 kHz.

The [v1.1 Chinese release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1)
uses `input_ids` instead of `tokens` and a separate vocabulary and voice archive.
The runner accepts the same `tokens` input bundle for both input names. To prepare
Chinese v1.1 text, use Python 3.11 or 3.12 and its matching Misaki frontend:

```sh
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.1-zh.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.1-zh.bin
curl -L https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/01e7505bd6a7a2ac4975463114c3a7650a9f7218/config.json -o config-v1.1-zh.json
python3.11 -m venv .venv-zh
.venv-zh/bin/python -m pip install numpy 'misaki-fork[zh]'
.venv-zh/bin/python prepare.py voices-v1.1-zh.bin --vocab config-v1.1-zh.json --g2p misaki-zh-v1.1 --text "千里之行，始于足下。" --voice zf_001
cargo run -p example-kokoro-onnx --profile opt-no-lto -- --stabilize-phase kokoro-v1.1-zh.onnx inputs.npz audio.npz
```

The v1.1 release provides FP32, int8, and actual FP16 Chinese graphs and also
updated v1.0 exports. These updated FP16 graphs differ from the mixed-quantized
graph named FP16 in the older v1.0 release. Keep them in separate directories.
V1.1's duration expansion uses a sequence loop. The importer recognizes and
checks that complete repeat-interleave pattern, then lowers it to a tensor
operation; arbitrary ONNX sequence/control-flow graphs remain unsupported.
The export also contains unseeded random operators, so repeated inference and
NNEF round trips need not produce identical waveforms.

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
Use `--voice` and `--speed` to change those inputs. To prepare arbitrary text with
eSpeak NG, supply the export's vocabulary configuration:

```sh
.venv/bin/python -m pip install phonemizer espeakng-loader
curl -L https://raw.githubusercontent.com/thewh1teagle/kokoro-onnx/3596b26764286a7de9d90c363e988d50578918e5/src/kokoro_onnx/config.json -o config.json
.venv/bin/python prepare.py voices-v1.0.bin --vocab config.json --text "This is a complete sentence." --language en-us --voice af_heart
```

For languages that need a different G2P, supply its output with `--phonemes` and
`--vocab`; for example, use Misaki for Japanese and Mandarin. The helper reports
characters it discards outside the export vocabulary and rejects empty inputs or
more than 510 phoneme tokens. Split longer text before preparing each bundle. Voice
style is selected at the row indexed by the surviving phoneme count minus one.
Use the vocabulary and voice archive matching your model version. The Rust runner
accepts any compatible input bundle; it does not perform text normalization or phonemization.
Additional input/output pairs on the command line reuse the same compiled model.

The runner uses tract's public Rust API. It discards intermediate shape annotations
and the named output-length annotation so tract can infer the runtime audio length.
Token length is constrained to 2–512, including the boundary tokens. Symbolic
integer-reciprocal resizing checks its inferred size against the actual floating-point
size at runtime; an incompatible size is reported instead of propagated.

For the full-float `kokoro-v1.0.onnx` export, enable the explicit phase repair:

```sh
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
cargo run -p example-kokoro-onnx --profile opt-no-lto -- --stabilize-phase kokoro-v1.0.onnx inputs.npz audio.npz
```

The graph computes phase with `atan(imaginary / real)`. Exact zero Fourier bins
produce 0/0 and can make the full-float waveform non-finite. `--stabilize-phase`
uses the `kokoro_stabilize_phase` model transform to assign phase zero only when
both components are zero. For the current FP16 graphs, the same option also keeps the
sine-generator phase in float32 through Sin before casting back to float16;
large phase angles can otherwise overflow the exported half-precision path.
These repairs are opt-in, and default import behavior is unchanged. The transform
validates the known Kokoro operator patterns.

The same runner accepts `kokoro-v1.0.int8.onnx` without this option. Dynamic LSTM
lowering supports forward, reverse, and bidirectional execution, per-direction or
per-channel weight scales, optional biases and initial states, and standard
activations. Sequence lengths that cannot be proved to cover the full input, custom activations, clipping, coupled
gates, and peepholes are rejected explicitly.

On Apple platforms, select the Metal runtime explicitly:

```sh
cargo run -p example-kokoro-onnx --profile opt-no-lto --features metal -- --runtime metal --stabilize-phase kokoro-v1.0.int8.onnx inputs.npz audio.npz
```

Metal executes the dynamic quantizers, integer matrix products, and the decoder's
20-point STFT on the GPU. It uses CPU fallback for other unsupported operations,
including quantized convolutions. It does not imply that every operation
runs on the GPU or that inference will be faster for a given input.

To serialize and reload a model before running it, add `--nnef-roundtrip`:

```sh
cargo run -p example-kokoro-onnx --profile opt-no-lto -- --nnef-roundtrip kokoro.tar --stabilize-phase kokoro-v1.0.int8.onnx inputs.npz audio.npz
```

The archive preserves dynamic dimensions, quantization parameters, and the optional
phase transform. It uses tract's core and ONNX NNEF extensions; enable the ONNX
registry when loading it with `tract::nnef()`. Combine this option with
`--runtime metal` (with `--features metal` when building the example) to execute the reloaded graph on Metal.

Outputs are not samplewise identical to ONNX Runtime: FFT residuals can select opposite phase
branches at near-zero bins, and quantized recurrence can amplify small floating-point
differences. Token durations can also differ in int8 exports. Compare speech quality
and timing for your application's inputs; a matching output length alone does not
establish audio equivalence.

`./ci.sh` verifies the older v1.0, updated v1.0, and v1.1 Chinese asset checksums and runs two
voice/speed bundles through NNEF for each export. `./ci-v1.1-zh.sh` checks only
the Chinese release; `./ci-gpu.sh` selects Metal on Apple platforms. Set
`KOKORO_MODEL_DIR`, `KOKORO_CURRENT_V10_MODEL_DIR`, and `KOKORO_V11_MODEL_DIR` to reuse existing download directories.
