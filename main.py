#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline speech-to-text CLI with optional speaker diarization.

Transcribes audio/video with a pluggable backend (Whisper, faster-whisper, Voxtral) and can
label speakers with pyannote.audio. Nothing is sent to a cloud API.

Examples
--------
Transcript to stdout (Whisper turbo by default):
    python main.py audio.mp3

Subtitles, format taken from the file extension:
    python main.py audio.mp3 -o audio.srt

Every audio file in a folder, one .vtt per file:
    python main.py recordings/ --outdir subs -f vtt

Other backends:
    python main.py audio.mp3 -b faster-whisper -m small
    python main.py audio.mp3 -b voxtral

Speaker labels (needs requirements-diarize.txt and a Hugging Face token):
    python main.py meeting.wav --diarize --num-speakers 3
"""

from __future__ import annotations

import argparse
import contextlib
import os
import pathlib
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple

from backends import DEFAULT_BACKEND, TranscriptionBackend, get_backend, get_backend_class, list_backends
from formats import FORMATS, format_for_path, render, to_json
from media import MEDIA_EXTENSIONS, collect_files

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
SAMPLE_RATE = 16000

# Errors that fail one file without stopping the rest of a batch.
FILE_ERRORS = (OSError, RuntimeError, ValueError)

Job = Tuple[pathlib.Path, Optional[pathlib.Path]]  # (input file, output file or None for stdout)

###############################################################################
# Argument parsing
###############################################################################


def positive_int(value: str) -> int:
    """argparse type for integers greater than zero."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe audio offline with Whisper, faster-whisper or Voxtral, "
        "optionally labelling speakers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.mp3                          # transcript to stdout
  %(prog)s audio.mp3 -o audio.srt             # subtitles (format from extension)
  %(prog)s recordings/ --outdir subs -f vtt   # every audio file in a folder
  %(prog)s audio.mp3 -b faster-whisper -m small
  %(prog)s meeting.wav --diarize --num-speakers 3
  %(prog)s --list-models --backend voxtral
""",
    )

    # Positional inputs (optional when using --list-* flags)
    parser.add_argument("audio", type=pathlib.Path, nargs="*", help="Audio/video files or folders to transcribe.")

    # Backend and model
    parser.add_argument(
        "-b",
        "--backend",
        default=DEFAULT_BACKEND,
        help=f"Transcription backend (default: {DEFAULT_BACKEND}). Available: {', '.join(list_backends())}",
    )
    parser.add_argument("-m", "--model", default=None, help="Model name/size (default: the backend's default).")
    parser.add_argument("-l", "--language", default=None, help="Language code, e.g. 'en' (default: auto-detect).")
    parser.add_argument(
        "-t",
        "--task",
        choices=("transcribe", "translate"),
        default="transcribe",
        help="'transcribe' or 'translate' to English (default: transcribe).",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None, help="Force device (default: auto).")
    parser.add_argument(
        "--hf-token",
        metavar="TOKEN",
        help="Hugging Face token for model downloads (default: HF_TOKEN or HUGGINGFACE_TOKEN env var).",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output.")

    # Output
    output = parser.add_argument_group("output")
    output.add_argument("-o", "--output", type=pathlib.Path, help="Write the transcript to this file, not stdout.")
    output.add_argument(
        "-f", "--format", choices=FORMATS, help="Output format (default: from the --output extension, else txt)."
    )
    output.add_argument("--outdir", type=pathlib.Path, help="Write one file per input into this folder.")
    output.add_argument("--json", type=pathlib.Path, help="Also write the full result as JSON to this file.")

    # Speaker diarization
    diarization = parser.add_argument_group("speaker diarization")
    diarization.add_argument("--diarize", action="store_true", help="Label speakers with pyannote.audio.")
    diarization.add_argument("--num-speakers", type=positive_int, help="Exact number of speakers, if known.")
    diarization.add_argument("--min-speakers", type=positive_int, help="Minimum number of speakers.")
    diarization.add_argument("--max-speakers", type=positive_int, help="Maximum number of speakers.")

    # Information flags
    info = parser.add_argument_group("information")
    info.add_argument("--list-backends", action="store_true", help="List available backends and exit.")
    info.add_argument("--list-models", action="store_true", help="List models for the selected backend and exit.")

    args = parser.parse_args(argv)
    if not (args.list_backends or args.list_models):
        _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject option combinations that can't work (exits via parser.error)."""
    if not args.audio:
        parser.error("the following arguments are required: audio")
    if (args.output or args.json) and is_batch(args):
        parser.error("--output and --json take a single input file; use --outdir for several files")
    if (args.num_speakers or args.min_speakers or args.max_speakers) and not args.diarize:
        parser.error("--num-speakers, --min-speakers and --max-speakers need --diarize")


def is_batch(args: argparse.Namespace) -> bool:
    """Batch mode writes one output file per input instead of printing a single transcript."""
    return args.outdir is not None or len(args.audio) > 1 or args.audio[0].is_dir()


def plan_outputs(args: argparse.Namespace, fmt: str) -> List[Job]:
    """Pair every input file with the file its transcript goes to (None means stdout)."""
    if not is_batch(args):
        return [(args.audio[0], args.output)]

    files = collect_files(args.audio, MEDIA_EXTENSIONS)
    targets = [(args.outdir / rel if args.outdir else src).with_suffix(f".{fmt}") for src, rel in files]
    # talk.mp3 and talk.wav would both become talk.<fmt>; keep the full name for those.
    counts = Counter(targets)
    targets = [t.with_name(f"{src.name}.{fmt}") if counts[t] > 1 else t for (src, _), t in zip(files, targets)]
    if len(set(targets)) < len(targets):
        sys.exit("Error: several inputs would write the same output file; rename them or use one --outdir per folder.")
    return [(src, target) for (src, _), target in zip(files, targets)]


###############################################################################
# Transcription
###############################################################################


def configure_hf_token(cli_token: Optional[str]) -> None:
    """Expose the Hugging Face token as HF_TOKEN, which huggingface_hub uses for every model download."""
    token = cli_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token


def load_backend(args: argparse.Namespace) -> TranscriptionBackend:
    """Create the selected backend and load its model (backend default if --model is not given)."""
    backend = get_backend(args.backend)
    model_name = args.model or get_backend_class(args.backend).default_model()
    if not args.quiet:
        print(f"Loading {args.backend} model '{model_name}'...", file=sys.stderr)
    backend.load_model(model_name, device=args.device)
    return backend


def transcribe_file(
    backend: TranscriptionBackend, audio: pathlib.Path, args: argparse.Namespace, pipeline: Any = None
) -> Dict[str, Any]:
    """Transcribe one file and, if a diarization pipeline is given, attach speaker labels."""
    result = backend.transcribe(
        audio_path=audio, language=args.language, task=args.task, verbose=not args.quiet
    ).to_dict()

    if pipeline is not None:
        turns = diarize_audio(
            audio,
            pipeline,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
        result["speaker_segments"] = merge_diarization(result, turns)
    return result


###############################################################################
# Diarization helpers
###############################################################################


def load_diarization_pipeline(device: Optional[str] = None) -> Any:
    """Load the pyannote speaker-diarization pipeline (imported lazily: it is slow to import)."""
    # pyannote.audio 4 sends usage metrics to pyannote.ai by default; stay offline unless the user opted in.
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise ImportError(
            f"speaker diarization needs pyannote.audio ({e}). Install with: pip install -r requirements-diarize.txt"
        ) from e

    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=os.getenv("HF_TOKEN"))
    if pipeline is None:  # pyannote returns None when the gated model can't be downloaded
        raise RuntimeError(
            f"could not download '{DIARIZATION_MODEL}'. Accept its terms at https://hf.co/{DIARIZATION_MODEL} "
            "and set HF_TOKEN (or pass --hf-token)."
        )
    pipeline.to(torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")))
    return pipeline


def diarize_audio(
    audio_path: pathlib.Path,
    pipeline: Any,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[Tuple[float, float, str]]:
    """Return list of (start, end, speaker_label)."""
    import torch
    from whisper import load_audio

    # Decode with ffmpeg here so pyannote needs no audio I/O backend of its own, whatever the format.
    waveform = torch.from_numpy(load_audio(str(audio_path), sr=SAMPLE_RATE)).unsqueeze(0)
    output = pipeline(
        {"waveform": waveform, "sample_rate": SAMPLE_RATE},
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    # pyannote 4 returns both variants; the "exclusive" one has no overlapping turns, which suits transcripts.
    annotation = getattr(output, "exclusive_speaker_diarization", output)
    return [(turn.start, turn.end, speaker) for turn, _, speaker in annotation.itertracks(yield_label=True)]


def merge_diarization(
    transcription_result: Dict[str, Any],
    spk_segments: List[Tuple[float, float, str]],
) -> List[Dict[str, Any]]:
    """Attach to each segment the speaker who talks the most during it."""
    turns = sorted(spk_segments, key=lambda turn: turn[0])

    output: List[Dict[str, Any]] = []
    for seg in transcription_result.get("segments", []):
        if "start" not in seg or "end" not in seg:
            output.append({**seg, "speaker": "unknown"})
            continue
        output.append({**seg, "speaker": _best_speaker(seg["start"], seg["end"], turns)})
    return output


def _best_speaker(start: float, end: float, turns: List[Tuple[float, float, str]]) -> str:
    """Speaker with the most overlap with [start, end]; else the one whose turn contains its midpoint."""
    overlap: Dict[str, float] = {}
    for turn_start, turn_end, speaker in turns:
        shared = min(end, turn_end) - max(start, turn_start)
        if shared > 0:
            overlap[speaker] = overlap.get(speaker, 0.0) + shared
    if overlap:
        return max(overlap, key=lambda speaker: overlap[speaker])

    middle = (start + end) / 2.0
    return next((speaker for turn_start, turn_end, speaker in turns if turn_start <= middle <= turn_end), "unknown")


###############################################################################
# Output helpers
###############################################################################


def write_output(text: str, dest: Optional[pathlib.Path], stdout: TextIO) -> None:
    """Write *text* to *dest*, or print it to *stdout* when there is no destination."""
    if dest is None:
        print(text, file=stdout)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


###############################################################################
# Information display
###############################################################################


def show_backends() -> None:
    """Display available backends and their descriptions."""
    print("Available transcription backends:\n")
    for name in list_backends():
        backend_class = get_backend_class(name)
        default_marker = " (default)" if name == DEFAULT_BACKEND else ""
        print(f"  {name}{default_marker}")
        print(f"    {backend_class.description}")
        print(f"    Models: {', '.join(backend_class.available_models())}")
        print()


def show_models(backend_name: str) -> None:
    """Display available models for a backend."""
    try:
        backend_class = get_backend_class(backend_name)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    print(f"Available models for '{backend_name}' backend:\n")
    models = backend_class.available_models()
    default = backend_class.default_model()
    for model in models:
        default_marker = " (default)" if model == default else ""
        print(f"  {model}{default_marker}")


###############################################################################
# Main program flow
###############################################################################


def run_jobs(args: argparse.Namespace, jobs: List[Job], fmt: str, stdout: TextIO) -> int:
    """Transcribe every job, carrying on after per-file errors. Returns the number of failed files."""
    try:
        # Diarization first: a missing package or HF token should fail before a big model download.
        pipeline = load_diarization_pipeline(args.device) if args.diarize else None
        backend = load_backend(args)
    except (ImportError, *FILE_ERRORS) as e:
        sys.exit(f"Error: {e}")

    failures = 0
    for audio, dest in jobs:
        try:
            result = transcribe_file(backend, audio, args, pipeline)
        except FILE_ERRORS as e:
            failures += 1
            print(f"Error: {audio}: {e}", file=sys.stderr)
            continue

        write_output(render(result, fmt), dest, stdout)
        if args.json:
            write_output(to_json(result), args.json, stdout)
        if dest is not None and not args.quiet:
            print(f"→ {dest}", file=sys.stderr)
    return failures


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    # Handle information flags
    if args.list_backends:
        show_backends()
        return

    if args.list_models:
        show_models(args.backend)
        return

    configure_hf_token(args.hf_token)
    fmt = args.format or format_for_path(args.output)
    jobs = plan_outputs(args, fmt)
    if not jobs:
        sys.exit("Error: no audio files found.")
    missing = [str(audio) for audio, _ in jobs if not audio.is_file()]
    if missing:  # checked before the (slow) model load
        sys.exit(f"Error: file not found: {', '.join(missing)}")

    stdout = sys.stdout
    # Libraries print progress and debug text to stdout; send it to stderr so stdout carries only the transcript.
    with contextlib.redirect_stdout(sys.stderr):
        failures = run_jobs(args, jobs, fmt, stdout)
    if failures:
        sys.exit(1 if len(jobs) == 1 else f"Error: {failures} of {len(jobs)} files failed.")


if __name__ == "__main__":  # pragma: no cover
    main()
