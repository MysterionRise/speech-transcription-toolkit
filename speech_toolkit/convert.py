"""The ``ogg2wav`` command: batch-convert OGG/Opus audio files to 16-bit PCM WAV.

Handy for preparing speech datasets (the transcription backends decode any format themselves).
It decodes with **ffmpeg** (through :func:`speech_toolkit.media.load_audio`), which must be on PATH,
and takes a list of files or a directory tree (.ogg, .oga and .opus, any case). The path of each WAV
it writes goes to stdout, one per line; skipped files and errors go to stderr.

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
import wave
from collections import Counter
from typing import Iterable, List, Sequence, Tuple

from .errors import AudioDecodeError
from .media import collect_files, load_audio

OGG_EXTENSIONS = frozenset({".ogg", ".oga", ".opus"})

Job = Tuple[pathlib.Path, pathlib.Path]  # (OGG/Opus file, WAV file)

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
        description="Convert .ogg/.opus files to WAV for speech recognition. Prints the path of each WAV it writes.",
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
            print(f"Skipping unsupported file {path}", file=sys.stderr)
    return ogg_files


def plan_outputs(files: Sequence[Tuple[pathlib.Path, pathlib.Path]], outdir: pathlib.Path | None) -> List[Job]:
    """Pair every ``(file, relative path)`` from :func:`collect_ogg_files` with the WAV it becomes.

    The WAV goes next to its file, or with *outdir* to ``outdir / relative`` (so sub-folders are mirrored). Files
    that would share a WAV, such as ``a.ogg`` and ``a.opus``, keep their full name instead: ``a.ogg.wav``,
    ``a.opus.wav``.

    Raises:
        ValueError: If two inputs would still write the same WAV.
    """
    targets = [(outdir / relative if outdir is not None else src).with_suffix(".wav") for src, relative in files]
    counts = Counter(targets)
    targets = [t.with_name(f"{src.name}.wav") if counts[t] > 1 else t for (src, _), t in zip(files, targets)]
    if len(set(targets)) < len(targets):
        raise ValueError("several inputs would write the same WAV file; rename them or use one --outdir per folder")
    return [(src, target) for (src, _), target in zip(files, targets)]


def decode(src: pathlib.Path, rate: int, channels: int) -> bytes:
    """Decode *src* with ffmpeg to 16-bit PCM at *rate* Hz with *channels* interleaved channels, in the machine's
    byte order (as :mod:`wave` expects).

    Raises:
        AudioDecodeError: If ffmpeg is missing or can't decode the file.
    """
    import numpy as np  # imported lazily, like load_audio does: --help shouldn't pay for it

    audio = load_audio(src, sr=rate, channels=channels)
    audio *= 32768  # load_audio divided ffmpeg's 16-bit samples by 32768: this gives them back exactly
    pcm: bytes = audio.astype(np.int16).tobytes()  # rows of interleaved channels, in the machine's byte order
    return pcm


def convert_file(src: pathlib.Path, dest: pathlib.Path, rate: int, channels: int, overwrite: bool = False) -> bool:
    """Convert *src* to a 16-bit WAV at *dest* with the requested sample rate and channels.

    An existing *dest* is kept unless *overwrite*. Prints *dest* to stdout once it is written; skips and errors go
    to stderr. Returns False if the conversion failed.
    """
    if rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {rate}")

    if dest.exists() and not overwrite:
        print(f"{dest} exists; skipping (use --overwrite).", file=sys.stderr)
        return True

    try:
        frames = decode(src, rate, channels)  # before touching dest: a failed decode leaves no empty WAV behind
        dest.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(dest), "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(rate)
            wav.writeframes(frames)
    except (AudioDecodeError, OSError) as e:
        print(f"Error: failed to convert {src}: {e}", file=sys.stderr)
        return False
    print(dest)
    return True


###############################################################################
# Entry point
###############################################################################


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    ogg_files = collect_ogg_files(args.inputs)
    if not ogg_files:
        sys.exit("No .ogg/.opus files found.")
    try:
        jobs = plan_outputs(ogg_files, args.outdir)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    failed = 0
    for src, dest in jobs:
        if not convert_file(src, dest, args.rate, args.channels, args.overwrite):
            failed += 1
    if failed:
        sys.exit(f"{failed} of {len(jobs)} files failed to convert.")


if __name__ == "__main__":  # pragma: no cover
    main()
