"""The ``ogg2wav`` command: batch-convert OGG/Opus audio files to 16-bit PCM WAV.

Handy for preparing speech datasets (the transcription backends decode any format themselves).
It relies on **ffmpeg** via *pydub* and takes a list of files or a directory tree
(.ogg, .oga and .opus, any case).

Examples
--------
Convert one file:
    ogg2wav song.ogg

Convert an entire folder into ./wav (keeping sub-folders), resampling to 16 kHz mono:
    ogg2wav ./records --outdir wav --rate 16000 --channels 1

Overwrite existing WAVs:
    ogg2wav *.ogg --overwrite
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Iterable, List, Sequence, Tuple

from pydub import AudioSegment  # type: ignore  # requires ffmpeg in PATH

from .media import collect_files

OGG_EXTENSIONS = frozenset({".ogg", ".oga", ".opus"})

###############################################################################
# CLI
###############################################################################


def positive_int(value: str) -> int:
    """argparse type for integers greater than zero."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ogg2wav",
        description="Convert .ogg/.opus files to WAV for speech recognition.",
    )
    p.add_argument(
        "inputs",
        nargs="+",
        type=pathlib.Path,
        help="One or more .ogg/.opus files or directories containing them.",
    )
    p.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=None,
        help="Directory to place converted WAVs (defaults to alongside original files).",
    )
    p.add_argument(
        "--rate",
        type=positive_int,
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


def collect_ogg_files(paths: Iterable[pathlib.Path]) -> List[Tuple[pathlib.Path, pathlib.Path]]:
    """Gather OGG/Opus files recursively as ``(file, path relative to its input folder)`` pairs."""
    ogg_files: List[Tuple[pathlib.Path, pathlib.Path]] = []
    for path, relative in collect_files(paths, OGG_EXTENSIONS):
        if path.suffix.lower() in OGG_EXTENSIONS:
            ogg_files.append((path, relative))
        else:
            print(f"⚠️  Skipping unsupported file {path}", file=sys.stderr)
    return ogg_files


def convert_file(
    src: pathlib.Path,
    outdir: pathlib.Path | None,
    rate: int,
    channels: int,
    overwrite: bool = False,
    relative: pathlib.Path | None = None,
) -> bool:
    """Convert *src* OGG to WAV with requested parameters.

    With *outdir*, the WAV goes to ``outdir / relative`` (so sub-folders are mirrored
    and same-named files don't collide); otherwise it is written next to *src*.
    Returns False if the conversion failed.
    """
    if rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {rate}")

    if outdir is None:
        dest_dir = src.parent
    else:
        dest_dir = outdir / relative.parent if relative is not None else outdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{src.stem}.wav"

    # stdout gets ASCII only: on Windows, redirected output uses a code page without symbols like ✓ and →.
    if dest_path.exists() and not overwrite:
        print(f"{dest_path} exists; skipping (use --overwrite).")
        return True

    try:
        audio = AudioSegment.from_file(src)
        audio = audio.set_frame_rate(rate).set_channels(channels).set_sample_width(2)  # 16‑bit
        audio.export(dest_path, format="wav")
        print(f"-> {dest_path}")
        return True
    except Exception as exc:  # pydub/ffmpeg raise many error types
        print(f"❌ Failed to convert {src}: {exc}", file=sys.stderr)
        return False


###############################################################################
# Entry point
###############################################################################


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    ogg_files = collect_ogg_files(args.inputs)
    if not ogg_files:
        sys.exit("No .ogg/.opus files found.")

    failed = 0
    for src, relative in ogg_files:
        if not convert_file(src, args.outdir, args.rate, args.channels, args.overwrite, relative):
            failed += 1
    if failed:
        sys.exit(f"{failed} of {len(ogg_files)} files failed to convert.")


if __name__ == "__main__":  # pragma: no cover
    main()
