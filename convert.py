#!/usr/bin/env python3
"""Batch‑convert OGG/Opus audio files to 16‑bit PCM WAV.

This helper sits alongside *main.py* so you can quickly prep datasets for
Whisper. It relies on **ffmpeg** via *pydub* and lets you feed either a list of
files or a directory tree.

Examples
--------
Convert one file:
    python convert.py song.ogg

Convert an entire folder into ./wav, resampling to 16 kHz mono:
    python convert.py ./records --outdir wav --rate 16000 --channels 1

Overwrite existing WAVs:
    python convert.py *.ogg --overwrite
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Iterable, List, Sequence

from pydub import AudioSegment  # type: ignore  # requires ffmpeg in PATH

###############################################################################
# CLI
###############################################################################


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="prepare-audio",
        description="Convert .ogg/opus files to WAV for Whisper/ASR.",
    )
    p.add_argument(
        "inputs",
        nargs="+",
        type=pathlib.Path,
        help="One or more .ogg files or directories containing them.",
    )
    p.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=None,
        help="Directory to place converted WAVs (defaults to alongside original files).",
    )
    p.add_argument(
        "--rate",
        type=int,
        default=16000,
        help="Sample rate for output WAV (Hz).",
    )
    p.add_argument(
        "--channels",
        type=int,
        choices=(1, 2),
        default=1,
        help="Number of output channels (1 = mono, 2 = stereo).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination WAV if it already exists.",
    )
    return p.parse_args(argv)


###############################################################################
# Conversion logic
###############################################################################


def collect_ogg_files(paths: Iterable[pathlib.Path]) -> List[pathlib.Path]:
    """Gather *.ogg files recursively from the given paths."""
    ogg_files: List[pathlib.Path] = []
    for p in paths:
        if p.is_dir():
            # Search for all case variants of .ogg extension
            for pattern in ("*.ogg", "*.OGG", "*.Ogg"):
                ogg_files.extend(p.rglob(pattern))
        elif p.suffix.lower() == ".ogg":
            ogg_files.append(p)
        else:
            print(f"⚠️  Skipping unsupported file {p}", file=sys.stderr)
    return sorted(ogg_files)


def convert_file(
    src: pathlib.Path,
    outdir: pathlib.Path | None,
    rate: int,
    channels: int,
    overwrite: bool = False,
) -> None:
    """Convert *src* OGG to WAV with requested parameters."""
    if rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {rate}")

    dest_dir = outdir if outdir is not None else src.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{src.stem}.wav"

    if dest_path.exists() and not overwrite:
        print(f"✓ {dest_path} exists; skipping (use --overwrite).")
        return

    try:
        audio = AudioSegment.from_file(src)
        audio = audio.set_frame_rate(rate).set_channels(channels).set_sample_width(2)  # 16‑bit
        audio.export(dest_path, format="wav")
        print(f"→ {dest_path}")
    except Exception as exc:  # pragma: no cover
        print(f"❌ Failed to convert {src}: {exc}", file=sys.stderr)


###############################################################################
# Entry point
###############################################################################


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover
    args = parse_args(argv)
    ogg_files = collect_ogg_files(args.inputs)
    if not ogg_files:
        sys.exit("No .ogg files found.")

    for src in ogg_files:
        convert_file(src, args.outdir, args.rate, args.channels, args.overwrite)


if __name__ == "__main__":  # pragma: no cover
    main()
