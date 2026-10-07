"""The ``transcribe`` command: offline speech-to-text with optional speaker labels.

Transcribes audio/video with a pluggable backend (Whisper, faster-whisper, Voxtral, Parakeet,
Canary) and can
label speakers with pyannote.audio. Nothing is sent to a cloud API.

Examples
--------
Transcript to stdout (Whisper turbo by default):
    transcribe audio.mp3

Subtitles, format taken from the file extension:
    transcribe audio.mp3 -o audio.srt

Every audio file in a folder, one .vtt per file:
    transcribe recordings/ --outdir subs -f vtt

Other backends:
    transcribe audio.mp3 -b faster-whisper -m small
    transcribe audio.mp3 -b voxtral

Better accuracy and readable subtitles:
    transcribe talk.mp3 -b faster-whisper --vad --prompt "Kubernetes, Grafana" -o talk.srt --max-line-width 42

Speaker labels (needs the ``diarize`` extra and a Hugging Face token):
    transcribe meeting.wav --diarize --num-speakers 3
"""

from __future__ import annotations

import argparse
import contextlib
import pathlib
import sys
import warnings
from collections import Counter
from typing import List, Optional, Sequence, TextIO, Tuple

from . import __version__
from .api import Transcriber
from .backends import DEFAULT_BACKEND, get_backend_class, list_backends
from .errors import SpeechToolkitError
from .formats import FORMATS, format_for_path, write_text
from .media import MEDIA_EXTENSIONS, collect_files

# Errors whose message says what went wrong, so it is shown alone. In a batch, any other Exception also fails only
# its own file (shown with its type); KeyboardInterrupt still stops the run.
FILE_ERRORS = (SpeechToolkitError, OSError, RuntimeError, ValueError)

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
        description="Transcribe audio offline with Whisper, faster-whisper, Voxtral, Parakeet or Canary, "
        "optionally labelling speakers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.mp3                          # transcript to stdout
  %(prog)s audio.mp3 -o audio.srt             # subtitles (format from extension)
  %(prog)s recordings/ --outdir subs -f vtt   # every audio file in a folder
  %(prog)s audio.mp3 -b faster-whisper -m small
  %(prog)s talk.mp3 -b faster-whisper --vad --prompt "Kubernetes, Grafana"
  %(prog)s talk.mp3 -o talk.srt --max-line-width 42
  %(prog)s meeting.wav --diarize --num-speakers 3
  %(prog)s --list-models --backend voxtral
""",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

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
    output.add_argument(
        "--max-line-width",
        type=positive_int,
        metavar="N",
        help="Split subtitles into lines of at most N characters, 2 lines per cue (srt/vtt).",
    )

    # Accuracy
    accuracy = parser.add_argument_group("accuracy")
    accuracy.add_argument(
        "--prompt",
        metavar="TEXT",
        help="Names, terms or a sample sentence that guide spelling and style (whisper, faster-whisper).",
    )
    accuracy.add_argument(
        "--vad", action="store_true", help="Skip silence first; avoids made-up text in quiet parts (faster-whisper)."
    )
    accuracy.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Add per-word timings to the JSON output (whisper, faster-whisper).",
    )

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


def load_transcriber(args: argparse.Namespace) -> Transcriber:
    """Load the selected backend's model (its default if --model is not given) and, with --diarize, pyannote."""
    model_name = args.model or get_backend_class(args.backend).default_model()
    if not args.quiet:
        print(f"Loading {args.backend} model '{model_name}'...", file=sys.stderr)
    return Transcriber(args.backend, model_name, device=args.device, diarize=args.diarize, hf_token=args.hf_token)


def word_timestamps_option(args: argparse.Namespace, fmt: str, transcriber: Transcriber) -> Optional[bool]:
    """--word-timestamps forces word timings; subtitle line splitting uses them when the backend has them."""
    splits_lines = args.max_line_width and fmt in ("srt", "vtt")
    if args.word_timestamps or (splits_lines and transcriber.supports("word_timestamps")):
        return True
    return None  # the library's default: on for diarization


def write_output(text: str, dest: Optional[pathlib.Path], stdout: TextIO) -> None:
    """Write *text* to *dest*, or print it to *stdout* when there is no destination."""
    if dest is None:
        print(text, file=stdout)
    else:
        write_text(dest, text)


def run_jobs(args: argparse.Namespace, jobs: List[Job], fmt: str, stdout: TextIO) -> int:
    """Transcribe every job, carrying on after any per-file error. Returns the number of failed files."""
    try:
        transcriber = load_transcriber(args)
    except (ImportError, *FILE_ERRORS) as e:
        sys.exit(f"Error: {e}")

    word_timestamps = word_timestamps_option(args, fmt, transcriber)
    failures = 0
    for audio, dest in jobs:
        try:
            result = transcriber.transcribe(
                audio,
                args.language,
                args.task,
                prompt=args.prompt,
                vad=args.vad,
                word_timestamps=word_timestamps,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                verbose=not args.quiet,
            )
            write_output(result.render(fmt, args.max_line_width), dest, stdout)
            if args.json:
                write_output(result.render("json"), args.json, stdout)
        except Exception as e:  # one bad file doesn't stop the batch; KeyboardInterrupt isn't an Exception
            failures += 1
            # Other errors also show their type: a KeyError's message is just the key.
            message = str(e) if isinstance(e, FILE_ERRORS) else f"{type(e).__name__}: {e}".removesuffix(": ")
            print(f"Error: {audio}: {message}", file=sys.stderr)
            continue

        if dest is not None and not args.quiet:
            print(f"→ {dest}", file=sys.stderr)
    return failures


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


def _print_warning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:
    """Show library warnings (e.g. turbo can't translate) as one plain line on stderr."""
    print(f"Warning: {message}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    # Handle information flags
    if args.list_backends:
        show_backends()
        return

    if args.list_models:
        show_models(args.backend)
        return

    stdout = sys.stdout
    # Libraries print progress and debug text to stdout; send it to stderr so stdout carries only the transcript.
    # Warnings, from the format lookup on, print as one line each.
    with warnings.catch_warnings(), contextlib.redirect_stdout(sys.stderr):
        warnings.showwarning = _print_warning
        fmt = args.format or format_for_path(args.output)
        jobs = plan_outputs(args, fmt)
        if not jobs:
            sys.exit("Error: no audio files found.")
        missing = [str(audio) for audio, _ in jobs if not audio.is_file()]
        if missing:  # checked before the (slow) model load
            sys.exit(f"Error: file not found: {', '.join(missing)}")
        failures = run_jobs(args, jobs, fmt, stdout)
    if failures:
        sys.exit(1 if len(jobs) == 1 else f"Error: {failures} of {len(jobs)} files failed.")
