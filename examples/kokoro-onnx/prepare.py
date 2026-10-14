"""Prepare Kokoro token/style inputs or convert a finite output bundle to 24 kHz WAV."""

import argparse
import json
from pathlib import Path
import wave

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("voices", type=Path)
    parser.add_argument("--voice", default="af_sarah")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--input", type=Path, default=Path("inputs.npz"))
    parser.add_argument("--output", type=Path, help="Convert this output bundle to WAV")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Text to phonemize with the selected G2P")
    source.add_argument("--phonemes", help="Phonemes from an external G2P, such as Misaki")
    parser.add_argument("--language", default="en-us", help="eSpeak NG language for --text")
    parser.add_argument("--g2p", choices=["espeak", "misaki-zh-v1.1"], default="espeak")
    parser.add_argument("--vocab", type=Path, help="The export's config.json (required for custom input)")
    args = parser.parse_args()

    if args.output:
        with np.load(args.output) as result:
            audio = result["audio"].ravel()
        if not np.isfinite(audio).all():
            parser.error("Output audio contains non-finite samples")
        if np.any(np.abs(audio) > 1):
            print("WAV conversion clips samples outside [-1, 1]; the NPZ remains unchanged")
        with wave.open(str(args.output.with_suffix(".wav")), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(np.rint(np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        return

    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("Speed must be finite and positive")
    if args.text is not None or args.phonemes is not None:
        if args.vocab is None:
            parser.error("--text and --phonemes require the export's --vocab config.json")
        with args.vocab.open(encoding="utf-8") as config:
            vocab = json.load(config)["vocab"]
        phonemes = args.phonemes
        if args.text is not None:
            if args.g2p == "misaki-zh-v1.1":
                try:
                    from misaki import zh
                except ImportError:
                    parser.error("Chinese v1.1 text requires: pip install 'misaki-fork[zh]'")
                phonemes, _ = zh.ZHG2P(version="1.1")(args.text.strip())
            else:
                try:
                    import espeakng_loader
                    import phonemizer
                    from phonemizer.backend.espeak.wrapper import EspeakWrapper
                except ImportError:
                    parser.error("--text requires: pip install phonemizer espeakng-loader")
                EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
                EspeakWrapper.set_library(espeakng_loader.get_library_path())
                phonemes = phonemizer.phonemize(
                    args.text.strip(), language=args.language, preserve_punctuation=True, with_stress=True
                )
        dropped = sorted(set(phonemes) - set(vocab))
        if dropped:
            print(f"Characters outside the export vocabulary are discarded: {dropped!r}")
        phonemes = "".join(p for p in phonemes if p in vocab).strip()
        tokens = [vocab[p] for p in phonemes]
    else:
        # "həlˈoʊ wˈɜːld!" in the v1.0 vocabulary.
        tokens = [50, 83, 54, 156, 57, 135, 16, 65, 156, 87, 158, 54, 46, 5]
    if not 1 <= len(tokens) <= 510:
        parser.error(f"Expected 1–510 phoneme tokens, got {len(tokens)}; split longer text first")
    if not all(isinstance(token, int) and token > 0 for token in tokens):
        parser.error("Phoneme token IDs must be positive integers; zero is reserved for boundaries")
    with np.load(args.voices) as voices:
        if args.voice not in voices:
            parser.error(f"Unknown voice {args.voice!r}; available voices: {', '.join(voices.files)}")
        rows = voices[args.voice]
        if rows.shape[0] < len(tokens):
            parser.error("Voice archive has no style row for this phoneme count")
        style = rows[len(tokens) - 1].reshape(1, 256).astype(np.float32)
    if not np.isfinite(style).all():
        parser.error("Voice style contains non-finite values")
    np.savez(
        args.input, tokens=np.array([[0, *tokens, 0]], np.int64),
        style=style, speed=np.array([args.speed], np.float32)
    )
    print(f"Prepared {len(tokens) + 2} tokens with {args.voice}, speed {args.speed}")


if __name__ == "__main__":
    main()
