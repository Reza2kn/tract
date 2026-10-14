"""Prepare a real voice/style smoke test and optionally convert the result to WAV."""

import argparse
from pathlib import Path
import wave

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("voices", type=Path)
parser.add_argument("--voice", default="af_sarah")
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--input", type=Path, default=Path("inputs.npz"))
parser.add_argument("--output", type=Path)
args = parser.parse_args()

# "həlˈoʊ wˈɜːld!" (Hello world), encoded with kokoro-onnx's v1.0 vocabulary.
tokens = [50, 83, 54, 156, 57, 135, 16, 65, 156, 87, 158, 54, 46, 5]
with np.load(args.voices) as voices:
    style = voices[args.voice][len(tokens) - 1].reshape(1, 256).astype(np.float32)
np.savez(args.input, tokens=np.array([[0, *tokens, 0]], np.int64), style=style, speed=np.array([args.speed], np.float32))

if args.output:
    with np.load(args.output) as result:
        audio = result["audio"].ravel()
    with wave.open(str(args.output.with_suffix(".wav")), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
