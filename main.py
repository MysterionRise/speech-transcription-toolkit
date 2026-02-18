#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Speech-to-text CLI with optional speaker diarization.

This script provides a unified CLI for speech-to-text transcription using
multiple backends (Whisper, Voxtral) with optional speaker diarization
via pyannote.audio.

Examples
--------
Transcribe with Whisper (default):
    python main.py audio.mp3

Transcribe with Voxtral Mini:
    python main.py audio.mp3 --backend voxtral --model voxtral-mini

Full diarization (requires `pyannote.audio` >=2.1):
    python main.py audio.mp3 --diarize --hf-token $HUGGINGFACE_TOKEN

Write combined transcript to a file while saving the raw JSON:
    python main.py audio.mp3 -o transcript.txt --json result.json --diarize

List available backends and models:
    python main.py --list-backends
    python main.py --list-models
    python main.py --list-models --backend voxtral

"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

from backends import DEFAULT_BACKEND, get_backend, get_backend_class, list_backends

# ``pyannote.audio`` is only required for diarization. Keep it optional.
try:
    from pyannote.audio import Pipeline  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    Pipeline = None  # type: ignore

###############################################################################
# Argument parsing
###############################################################################


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe audio with multiple backend support (Whisper, Voxtral) "
        "and optional speaker diarization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.mp3                          # Transcribe with Whisper (default)
  %(prog)s audio.mp3 --backend voxtral        # Use Voxtral backend
  %(prog)s audio.mp3 -m large --diarize       # Large model with speaker labels
  %(prog)s --list-backends                    # Show available backends
  %(prog)s --list-models --backend voxtral    # Show Voxtral models
""",
    )

    # Positional audio path (optional if using --list-* flags)
    parser.add_argument("audio", type=pathlib.Path, nargs="?", help="Path to the input audio/video file.")

    # Backend selection
    parser.add_argument(
        "-b",
        "--backend",
        default=DEFAULT_BACKEND,
        help=f"Transcription backend to use (default: {DEFAULT_BACKEND}). " f"Available: {', '.join(list_backends())}",
    )

    # Model options
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Model name/size. If not specified, uses backend's default model.",
    )
    parser.add_argument("-l", "--language", default=None, help="Language code spoken in the audio (e.g., 'en', 'es').")
    parser.add_argument(
        "-t",
        "--task",
        choices=("transcribe", "translate"),
        default="transcribe",
        help="'transcribe' or 'translate' to English (default: transcribe).",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None, help="Force device (cpu/cuda).")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output.")

    # Speaker diarization
    parser.add_argument("--diarize", action="store_true", help="Run speaker diarization with pyannote.audio.")
    parser.add_argument(
        "--hf-token",
        metavar="TOKEN",
        help="Hugging Face access token for downloading models. "
        "If omitted, uses HUGGINGFACE_TOKEN environment variable.",
    )

    # Output paths
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Write plain transcript to file.")
    parser.add_argument("--json", type=pathlib.Path, help="Write full result as JSON.")

    # Information flags
    parser.add_argument("--list-backends", action="store_true", help="List available transcription backends and exit.")
    parser.add_argument("--list-models", action="store_true", help="List available models for the selected backend.")

    args = parser.parse_args(argv)

    # Validate: audio is required unless using --list-* flags
    if not args.list_backends and not args.list_models and args.audio is None:
        parser.error("the following arguments are required: audio")

    return args


###############################################################################
# Backend-agnostic transcription
###############################################################################


def run_transcription(args: argparse.Namespace) -> Dict[str, Any]:  # pragma: no cover
    """Run transcription using the selected backend.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Dict containing at minimum 'text' and 'segments' keys.
    """
    if not args.audio.exists():
        sys.exit(f"Error: audio file '{args.audio}' does not exist.")

    # Get and configure backend
    try:
        backend = get_backend(args.backend)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    # Determine model (use backend's default if not specified)
    model_name = args.model or get_backend_class(args.backend).default_model()

    # Load model
    try:
        if not args.quiet:
            print(f"Loading {args.backend} model '{model_name}'...", file=sys.stderr)
        backend.load_model(model_name, device=args.device)
    except (ValueError, RuntimeError, ImportError) as e:
        sys.exit(f"Error loading model: {e}")

    # Run transcription
    try:
        result = backend.transcribe(
            audio_path=args.audio,
            language=args.language,
            task=args.task,
            verbose=not args.quiet,
        )
    except (FileNotFoundError, RuntimeError) as e:
        sys.exit(f"Error during transcription: {e}")

    return result.to_dict()


###############################################################################
# Diarization helpers
###############################################################################


def load_diarization_pipeline(token: Optional[str]) -> "Pipeline":  # type: ignore  # pragma: no cover
    """Load the PyAnnote speaker-diarization pipeline (lazy)."""
    if Pipeline is None:
        sys.exit("pyannote.audio is not installed. Install with 'pip install pyannote.audio'.")

    auth_token = token or os.getenv("HUGGINGFACE_TOKEN")
    if auth_token is None:
        print("Warning: No HF token provided. May fail for private models.", file=sys.stderr)

    # Using the official pretrained pipeline released under Apache-2.0.
    return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=auth_token)  # type: ignore


def diarize_audio(audio_path: pathlib.Path, pipeline: "Pipeline") -> List[Tuple[float, float, str]]:  # type: ignore  # pragma: no cover
    """Return list of (start, end, speaker_label)."""
    diarization = pipeline(str(audio_path))  # returns "pyannote.core.Annotation"

    segments: List[Tuple[float, float, str]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))
    return segments


def merge_diarization(
    transcription_result: Dict[str, Any],
    spk_segments: List[Tuple[float, float, str]],
) -> List[Dict[str, Any]]:
    """Attach speaker labels to each segment via midpoint lookup."""
    # Build timeline index
    indexed: List[Tuple[float, float, str]] = sorted(spk_segments, key=lambda x: x[0])

    output: List[Dict[str, Any]] = []
    for seg in transcription_result.get("segments", []):
        if "start" not in seg or "end" not in seg:
            output.append({**seg, "speaker": "unknown"})
            continue
        mid = (seg["start"] + seg["end"]) / 2.0
        speaker_id = "unknown"
        for s_start, s_end, label in indexed:
            if s_start <= mid <= s_end:
                speaker_id = label
                break
        output.append({**seg, "speaker": speaker_id})
    return output


###############################################################################
# Output helpers
###############################################################################


def write_outputs(result: Dict[str, Any], args: argparse.Namespace) -> None:  # pragma: no cover
    """Write transcript/plain text and optional JSON to disk or stdout."""
    if args.diarize and "speaker_segments" in result:
        # Pretty print with speaker labels
        parts: List[str] = []
        current_spk: Optional[str] = None
        for seg in result["speaker_segments"]:
            spk = seg["speaker"]
            if spk != current_spk:
                parts.append(f"\n[{spk}]")
                current_spk = spk
            parts.append(seg["text"].strip())
        transcript = " ".join(parts).strip()
    else:
        transcript = result.get("text", "").strip()

    # Write / print transcript
    if args.output:
        args.output.write_text(transcript, encoding="utf-8")
    else:
        print(transcript)

    # Raw JSON output if requested
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


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


def main() -> None:  # pragma: no cover
    args = parse_args()

    # Handle information flags
    if args.list_backends:
        show_backends()
        return

    if args.list_models:
        show_models(args.backend)
        return

    # Run transcription
    result = run_transcription(args)

    # Optionally run speaker diarization
    if args.diarize:
        pipeline = load_diarization_pipeline(args.hf_token)
        spk_segments = diarize_audio(args.audio, pipeline)
        speaker_segments = merge_diarization(result, spk_segments)
        result["speaker_segments"] = speaker_segments  # attach for JSON export

    write_outputs(result, args)


if __name__ == "__main__":  # pragma: no cover
    main()
