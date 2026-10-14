#!/bin/sh
set -eu

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT HUP INT TERM
models=${KOKORO_V11_MODEL_DIR:-$work}
runtime=${KOKORO_RUNTIME:-cpu}
example_features=
if [ "$runtime" = metal ]; then
    example_features=metal
fi
python3 -m venv "$work/venv"
"$work/venv/bin/python" -m pip install --quiet numpy
for file in voices-v1.1-zh.bin kokoro-v1.1-zh.fp16.onnx kokoro-v1.1-zh.int8.onnx kokoro-v1.1-zh.onnx; do
    if [ ! -f "$models/$file" ]; then
        curl --fail --location --retry 3 \
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/$file" \
            --output "$models/$file"
    fi
done
if [ ! -f "$models/config-v1.1-zh.json" ]; then
    curl --fail --location --retry 3 \
        https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/01e7505bd6a7a2ac4975463114c3a7650a9f7218/config.json \
        --output "$models/config-v1.1-zh.json"
fi
"$work/venv/bin/python" - "$models" <<'PY'
import hashlib
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = {
    "config-v1.1-zh.json": "bc333efa5ce4ceff433c8c8e5d027a1eca0166001e4e4a62bea2d26ff7a46890",
    "kokoro-v1.1-zh.fp16.onnx": "a628ea5d6fbde96d1a85f691a6a00847829937f9e488021ba2c5359bc6ea08b5",
    "kokoro-v1.1-zh.int8.onnx": "11751c087b4bbeed031e2b687b11dda698bd27ba0509472f960b57f835a999f7",
    "kokoro-v1.1-zh.onnx": "859f9ded9f53be16c24857cdab3254a45da53c3afd5ba6ef134c7de3f822e326",
    "voices-v1.1-zh.bin": "14cb6186c99e4f6016871405f62046c5df863ae27465cbdc4ee08be7dd703acd",
}
for filename, sha256 in expected.items():
    actual = hashlib.file_digest((root / filename).open("rb"), "sha256").hexdigest()
    if actual != sha256:
        raise SystemExit(f"Release asset checksum mismatch: {filename}")
PY
"$work/venv/bin/python" prepare.py "$models/voices-v1.1-zh.bin" --vocab "$models/config-v1.1-zh.json" --phonemes "ㄑ言1ㄌㄧ3ㄓ十1ㄒ应2, ㄕ十3ㄩ2ㄗㄨ2ㄒ压4." --voice zf_001 --input "$work/short.npz"
"$work/venv/bin/python" prepare.py "$models/voices-v1.1-zh.bin" --vocab "$models/config-v1.1-zh.json" --phonemes "ㄑ言1ㄌㄧ3ㄓ十1ㄒ应2, ㄕ十3ㄩ2ㄗㄨ2ㄒ压4." --voice zf_002 --speed 0.85 --input "$work/slow.npz"
for variant in fp16 int8 float; do
    case "$variant" in
        float) file=kokoro-v1.1-zh.onnx ;;
        *) file=kokoro-v1.1-zh.$variant.onnx ;;
    esac
    cargo run -p example-kokoro-onnx --profile opt-no-lto --features "$example_features" -- \
        --runtime "$runtime" --stabilize-phase --nnef-roundtrip "$work/model.tar" \
        "$models/$file" "$work/short.npz" "$work/short-$variant.npz" \
        "$work/slow.npz" "$work/slow-$variant.npz"
done
