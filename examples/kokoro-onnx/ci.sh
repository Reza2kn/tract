#!/bin/sh
set -eu

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT HUP INT TERM
models=${KOKORO_MODEL_DIR:-$work}
runtime=${KOKORO_RUNTIME:-cpu}
example_features=
if [ "$runtime" = metal ]; then
    example_features=metal
fi
python3 -m venv "$work/venv"
"$work/venv/bin/python" -m pip install --quiet numpy
for file in voices-v1.0.bin kokoro-v1.0.fp16.onnx kokoro-v1.0.int8.onnx kokoro-v1.0.onnx; do
    if [ ! -f "$models/$file" ]; then
        curl --fail --location --retry 3 \
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/$file" \
            --output "$models/$file"
    fi
done
"$work/venv/bin/python" - "$models" <<'PY'
import hashlib
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = {
    "kokoro-v1.0.fp16.onnx": "c1610a859f3bdea01107e73e50100685af38fff88f5cd8e5c56df109ec880204",
    "kokoro-v1.0.int8.onnx": "6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb",
    "kokoro-v1.0.onnx": "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    "voices-v1.0.bin": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
}
for filename, sha256 in expected.items():
    actual = hashlib.file_digest((root / filename).open("rb"), "sha256").hexdigest()
    if actual != sha256:
        raise SystemExit(f"Release asset checksum mismatch: {filename}")
PY
"$work/venv/bin/python" prepare.py "$models/voices-v1.0.bin" --input "$work/short.npz"
"$work/venv/bin/python" prepare.py "$models/voices-v1.0.bin" --voice am_adam --speed 0.85 --input "$work/slow.npz"
for variant in fp16 int8 float; do
    case "$variant" in
        float) file=kokoro-v1.0.onnx ;;
        *) file=kokoro-v1.0.$variant.onnx ;;
    esac
    cargo run -p example-kokoro-onnx --profile opt-no-lto --features "$example_features" -- \
        --runtime "$runtime" --stabilize-phase --nnef-roundtrip "$work/model.tar" \
        "$models/$file" "$work/short.npz" "$work/short-$variant.npz" \
        "$work/slow.npz" "$work/slow-$variant.npz"
done

./ci-v1.1-zh.sh
./ci-current-v1.0.sh
