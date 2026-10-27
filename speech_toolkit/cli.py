"""The ``transcribe`` command: offline speech-to-text with optional speaker labels.

Transcribes audio/video with a pluggable backend (faster-whisper by default, Whisper, Voxtral,
Parakeet, Canary) and can label speakers with pyannote.audio. Nothing is sent to a cloud API.

Examples
--------
Transcript to stdout (faster-whisper's turbo model by default):
    transcribe audio.mp3

Subtitles, format taken from the file extension:
    transcribe audio.mp3 -o audio.srt

Every audio file in a folder, one .vtt per file:
    transcribe recordings/ --outdir subs -f vtt

Several formats from one transcription, skipping the files done already:
    transcribe recordings/ --outdir subs -f srt,json --skip-existing

Audio piped into standard input:
    cat talk.mp3 | transcribe -

Other models and backends:
    transcribe audio.mp3 -m small
    transcribe audio.mp3 -b whisper      # needs the ``whisper`` extra
    transcribe audio.mp3 -b voxtral

Better accuracy and readable subtitles:
    transcribe talk.mp3 --vad --prompt "Kubernetes, Grafana" -o talk.srt --max-line-width 42

Speaker labels (needs the ``diarize`` extra and a Hugging Face token):
    transcribe meeting.wav --diarize --num-speakers 3
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import os
import pathlib
import sys
import time
import warnings
from collections import Counter
from typing import AbstractSet, List, Optional, Sequence, TextIO, Tuple

from . import __version__
from .api import TASKS, Transcriber
from .backends import CAPABILITIES, DEFAULT_BACKEND, backends_with, get_backend_class, list_backends
from .diarization import check_speaker_hints
from .errors import SpeechToolkitError, UnsupportedOptionError
from .formats import FORMATS, format_for_path, write_text
from .media import MEDIA_EXTENSIONS, collect_files

# Errors whose message says what went wrong, so it is shown alone. In a batch, any other Exception also fails only
# its own file (shown with its type); KeyboardInterrupt still stops the run.
FILE_ERRORS = (SpeechToolkitError, OSError, RuntimeError, ValueError)

STDIN = pathlib.Path("-")
"""The input ``-``: audio piped into standard input. With --outdir, its outputs are named ``stdin.<format>``."""
STDIN_NAME = "stdin"


@dataclasses.dataclass(frozen=True)
class Output:
    """One rendering of a transcript: format *fmt*, written to *path*, or to stdout when *path* is None."""

    fmt: str
    path: Optional[pathlib.Path] = None


@dataclasses.dataclass(frozen=True)
class Job:
    """An input (a file, or STDIN) and its outputs, all rendered from one transcription."""

    audio: pathlib.Path
    outputs: Tuple[Output, ...]


###############################################################################
# Argument parsing
###############################################################################


def positive_int(value: str) -> int:
    """argparse type for integers greater than zero."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return number


def _supported_by(capability: str) -> str:
    """The registered backends with *capability*, for help text: ``"whisper, faster-whisper"``."""
    return ", ".join(backends_with(capability)) or "no backend"


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
  %(prog)s recordings/ -f srt,json --skip-existing   # two formats; resume a stopped run
  cat audio.mp3 | %(prog)s -                  # audio from standard input
  %(prog)s audio.mp3 -m small                 # a smaller, faster model
  %(prog)s audio.mp3 -b whisper               # OpenAI's Whisper (the whisper extra)
  %(prog)s talk.mp3 --vad --prompt "Kubernetes, Grafana"
  %(prog)s talk.mp3 -o talk.srt --max-line-width 42
  %(prog)s meeting.wav --diarize --num-speakers 3
  %(prog)s --list-models --backend voxtral
""",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # Positional inputs (optional when using --list-* flags)
    parser.add_argument(
        "audio", type=input_path, nargs="*", help="Audio/video files or folders to transcribe; - reads standard input."
    )

    add_model_options(parser)
    parser.add_argument("-l", "--language", default=None, help="Language code, e.g. 'en' (default: auto-detect).")
    parser.add_argument(
        "-t",
        "--task",
        choices=TASKS,
        default="transcribe",
        help="'transcribe' or 'translate' to English (default: transcribe). "
        f"Backends that translate: {_supported_by('translate')}.",
    )
    parser.add_argument(
        "--hf-token",
        metavar="TOKEN",
        help="Hugging Face token for model downloads (default: HF_TOKEN or HUGGINGFACE_TOKEN env var).",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output.")
    add_output_options(parser)
    add_accuracy_options(parser)
    add_diarization_options(parser)
    add_information_options(parser)

    args = parser.parse_args(argv)
    if not (args.list_backends or args.list_models):
        _validate_args(parser, args)
    return args


def add_model_options(parser: argparse.ArgumentParser) -> None:
    """--backend, --model and --device: the model to load (transcribe-server takes them too)."""
    model = parser.add_argument_group("model")
    model.add_argument(
        "-b",
        "--backend",
        default=DEFAULT_BACKEND,
        help=f"Transcription backend (default: {DEFAULT_BACKEND}). Available: {', '.join(list_backends())}",
    )
    model.add_argument("-m", "--model", default=None, help="Model name/size (default: the backend's default).")
    model.add_argument("--device", choices=("cpu", "cuda"), default=None, help="Force device (default: auto).")


def input_path(value: str) -> pathlib.Path:
    """argparse type for inputs: a path, or STDIN itself for ``-`` (so that ``./-`` is still a file named "-")."""
    return STDIN if value == "-" else pathlib.Path(value)


def is_stdin(path: pathlib.Path) -> bool:
    """Whether *path* is the input ``-``: standard input."""
    return path is STDIN


def format_list(value: str) -> Tuple[str, ...]:
    """argparse type for --format: a format, several separated by commas (``srt,json``), or ``all`` of them."""
    names = [name.strip().lower() for name in value.split(",") if name.strip()]
    if not names or any(name not in FORMATS and name != "all" for name in names):
        raise argparse.ArgumentTypeError(
            f"invalid format {value!r}: choose from {', '.join(FORMATS)}, several separated by commas, or all"
        )
    return tuple(FORMATS) if "all" in names else tuple(dict.fromkeys(names))


def add_output_options(parser: argparse.ArgumentParser) -> None:
    """Where transcripts go, and in which formats."""
    output = parser.add_argument_group("output")
    output.add_argument("-o", "--output", type=pathlib.Path, help="Write the transcript to this file, not stdout.")
    output.add_argument(
        "-f",
        "--format",
        type=format_list,
        metavar="FORMAT",
        help=f"Output format: {', '.join(FORMATS)}. Several separated by commas (srt,json), or all, write a file "
        "each from one transcription (default: from the --output extension, else txt).",
    )
    output.add_argument("--outdir", type=pathlib.Path, help="Write each input's output files into this folder.")
    output.add_argument(
        "--json",
        type=pathlib.Path,
        metavar="FILE",
        help="Also write the full result as JSON to this file (one input; for several, add json to --format).",
    )
    output.add_argument(
        "--skip-existing",
        action="store_true",
        help="Keep output files that already exist: transcribe only the inputs with an output missing, and write "
        "just the missing ones (default: overwrite them).",
    )
    output.add_argument(
        "--max-line-width",
        type=positive_int,
        metavar="N",
        help="Split subtitles into lines of at most N columns, 2 lines per cue (srt/vtt). Chinese, Japanese and "
        "Korean characters take two columns.",
    )


def add_accuracy_options(parser: argparse.ArgumentParser) -> None:
    """Optional backend features; their help lists the backends that declare each one."""
    accuracy = parser.add_argument_group("accuracy")
    accuracy.add_argument(
        "--prompt",
        metavar="TEXT",
        help=f"Names, terms or a sample sentence that guide spelling and style ({_supported_by('prompt')}).",
    )
    accuracy.add_argument(
        "--vad",
        action="store_true",
        help=f"Skip silence first; avoids made-up text in quiet parts ({_supported_by('vad')}).",
    )
    accuracy.add_argument(
        "--word-timestamps",
        action="store_true",
        help=f"Add per-word timings to the JSON output ({_supported_by('word_timestamps')}).",
    )


def add_diarization_options(parser: argparse.ArgumentParser) -> None:
    """Speaker labels with pyannote.audio."""
    diarization = parser.add_argument_group("speaker diarization")
    diarization.add_argument("--diarize", action="store_true", help="Label speakers with pyannote.audio.")
    diarization.add_argument("--num-speakers", type=positive_int, help="Exact number of speakers, if known.")
    diarization.add_argument("--min-speakers", type=positive_int, help="Minimum number of speakers.")
    diarization.add_argument("--max-speakers", type=positive_int, help="Maximum number of speakers.")


def add_information_options(parser: argparse.ArgumentParser) -> None:
    """Flags that print information and exit instead of transcribing."""
    info = parser.add_argument_group("information")
    info.add_argument("--list-backends", action="store_true", help="List available backends and exit.")
    info.add_argument("--list-models", action="store_true", help="List models for the selected backend and exit.")


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject option combinations that can't work (exits via parser.error), before any model loads."""
    if not args.audio:
        parser.error("the following arguments are required: audio")
    problem = output_problem(args)
    if problem:
        parser.error(problem)
    try:
        check_speaker_hints(args.num_speakers, args.min_speakers, args.max_speakers, diarize=args.diarize, flags=True)
    except UnsupportedOptionError as e:
        parser.error(str(e))
    # An unknown backend is reported when it's loaded, like any other loading error.
    translators = backends_with("translate")
    if args.task == "translate" and args.backend in list_backends() and args.backend not in translators:
        parser.error(f"the {args.backend} backend can't translate; backends that translate: {', '.join(translators)}")


def output_problem(args: argparse.Namespace) -> Optional[str]:
    """Why the inputs and the output options don't fit together, or None when they do."""
    stdin = any(is_stdin(path) for path in args.audio)
    if stdin and len(args.audio) > 1:
        return "- (standard input) must be the only input"
    if args.output and is_batch(args):
        return "--output takes a single input file and format; use --outdir for several"
    if args.json and several_inputs(args):
        return "--json takes a single input file; for one JSON file per input, add json to --format (-f srt,json)"
    if stdin and is_batch(args) and args.outdir is None:
        return f"several formats from standard input need --outdir, where they are named {STDIN_NAME}.<format>"
    return None


def several_inputs(args: argparse.Namespace) -> bool:
    """Whether the inputs are several files, or a folder."""
    return len(args.audio) > 1 or (not is_stdin(args.audio[0]) and args.audio[0].is_dir())


def is_batch(args: argparse.Namespace) -> bool:
    """Batch mode writes each input's transcript to files instead of printing a single one.

    Several inputs, a folder, several formats and --outdir turn it on.
    """
    return args.outdir is not None or len(args.format or ()) > 1 or several_inputs(args)


def output_file(base: pathlib.Path, fmt: str) -> pathlib.Path:
    """The file format *fmt* of an input goes to: *base*, named after the input, with the format as extension."""
    return base.with_name(f"{base.name}.{fmt}")


def plan_outputs(args: argparse.Namespace, formats: Sequence[str]) -> List[Job]:
    """Pair every input with its outputs: a file per format (or stdout for a single transcript), and --json's file."""
    extra = (Output("json", args.json),) if args.json else ()
    if not is_batch(args):
        return [Job(args.audio[0], (Output(formats[0], args.output), *extra))]

    if is_stdin(args.audio[0]):
        files = [(args.audio[0], pathlib.Path(STDIN_NAME))]
    else:
        files = collect_files(args.audio, MEDIA_EXTENSIONS)
    # An input's outputs are named after it, each with its format's extension: talk.mp3 gives talk.srt and talk.json.
    bases = [(args.outdir / rel if args.outdir else src).with_suffix("") for src, rel in files]
    # talk.mp3 and talk.wav would share theirs; keep the full name for those (talk.mp3.srt).
    counts = Counter(bases)
    bases = [base.with_name(src.name) if counts[base] > 1 else base for (src, _), base in zip(files, bases)]
    if len(set(bases)) < len(bases):
        sys.exit("Error: several inputs would write the same output file; rename them or use one --outdir per folder.")
    return [
        Job(src, (*(Output(fmt, output_file(base, fmt)) for fmt in formats), *extra))
        for (src, _), base in zip(files, bases)
    ]


###############################################################################
# Transcription
###############################################################################


def load_transcriber(args: argparse.Namespace) -> Transcriber:
    """Load the selected backend's model (its default if --model is not given) and, with --diarize, pyannote."""
    model_name = args.model or get_backend_class(args.backend).default_model()
    if not args.quiet:
        print(f"Loading {args.backend} model '{model_name}'...", file=sys.stderr)
    return Transcriber(args.backend, model_name, device=args.device, diarize=args.diarize, hf_token=args.hf_token)


def word_timestamps_option(
    args: argparse.Namespace, formats: AbstractSet[str], transcriber: Transcriber
) -> Optional[bool]:
    """--word-timestamps forces word timings; subtitle line splitting uses them when the backend has them."""
    splits_lines = args.max_line_width and not formats.isdisjoint({"srt", "vtt"})
    if args.word_timestamps or (splits_lines and transcriber.supports("word_timestamps")):
        return True
    return None  # the library's default: on for diarization


def write_output(text: str, dest: Optional[pathlib.Path], stdout: TextIO) -> None:
    """Write *text* to *dest*, or print it to *stdout* when there is no destination.

    The file is written under a temporary name beside *dest*, then renamed, so an error or Ctrl-C while writing
    leaves no half-written file: *dest* is complete, or as it was before.
    """
    if dest is None:
        print(text, file=stdout)
        return
    temporary = dest.with_name(f".{dest.name}.{os.getpid()}.part")
    try:
        write_text(temporary, text)
        temporary.replace(dest)
    finally:
        temporary.unlink(missing_ok=True)  # already gone once renamed


def pending_outputs(job: Job, skip_existing: bool) -> Tuple[Output, ...]:
    """The outputs of *job* to write: all of them, or with --skip-existing those whose file doesn't exist yet."""
    if not skip_existing:
        return job.outputs
    return tuple(output for output in job.outputs if output.path is None or not output.path.exists())


def shown_name(audio: pathlib.Path) -> str:
    """An input as progress lines and error messages show it."""
    return "standard input" if is_stdin(audio) else str(audio)


def elapsed_time(seconds: float) -> str:
    """A duration as the summary shows it: ``12.3 s``, ``4:05`` or ``1:02:03``."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


@dataclasses.dataclass
class Summary:
    """What a run did with its *total* inputs: how many it transcribed (done), skipped and failed, and whether Ctrl-C
    stopped it."""

    total: int
    done: int = 0
    skipped: int = 0
    failed: int = 0
    interrupted: bool = False
    started: float = dataclasses.field(default_factory=time.monotonic)

    def line(self) -> str:
        """The line shown at the end of a run."""
        counts = f"{self.done} done, {self.skipped} skipped, {self.failed} failed"
        elapsed = elapsed_time(time.monotonic() - self.started)
        if self.interrupted:
            left = self.total - self.done - self.skipped - self.failed
            return f"Interrupted after {elapsed}: {counts}, {left} left."
        return f"Finished in {elapsed}: {counts}."


def show_job(job: Job, counter: Optional[str], skipped: bool) -> None:
    """Show the input of *job* on stderr: in batch mode after its *counter* (``[2/5] ``), else only when skipped."""
    if counter is not None or skipped:
        note = " (skipped: its outputs exist)" if skipped else ""
        print(f"{counter or ''}{shown_name(job.audio)}{note}", file=sys.stderr)


def load_or_exit(args: argparse.Namespace) -> Transcriber:
    """The transcriber of load_transcriber(); a model that can't load ends the program with the reason."""
    try:
        return load_transcriber(args)
    except (ImportError, *FILE_ERRORS) as e:
        sys.exit(f"Error: {e}")


def run_jobs(args: argparse.Namespace, jobs: List[Job], stdout: TextIO) -> Summary:
    """Transcribe every job and write its outputs, carrying on after any per-file error. Returns the summary.

    The model loads before the first transcription, so a run with nothing left to do (--skip-existing) loads none.
    In batch mode, each input gets an ``[i/N]`` line on stderr and the run a summary at the end; -q hides them.
    Ctrl-C stops the run and shows the summary; the files written so far are complete.
    """
    batch = is_batch(args)
    summary = Summary(len(jobs))
    formats = {output.fmt for job in jobs for output in job.outputs}
    transcriber: Optional[Transcriber] = None
    word_timestamps: Optional[bool] = None
    try:
        for number, job in enumerate(jobs, start=1):
            outputs = pending_outputs(job, args.skip_existing)
            if not args.quiet:
                show_job(job, f"[{number}/{len(jobs)}] " if batch else None, skipped=not outputs)
            if not outputs:
                summary.skipped += 1
                continue
            if transcriber is None:
                transcriber = load_or_exit(args)
                word_timestamps = word_timestamps_option(args, formats, transcriber)
            if transcribe_job(args, transcriber, job.audio, outputs, word_timestamps, stdout):
                summary.done += 1
            else:
                summary.failed += 1
    except KeyboardInterrupt:
        summary.interrupted = True
    if not args.quiet and (batch or summary.interrupted):
        print(summary.line(), file=sys.stderr)
    return summary


def transcribe_job(
    args: argparse.Namespace,
    transcriber: Transcriber,
    audio: pathlib.Path,
    outputs: Sequence[Output],
    word_timestamps: Optional[bool],
    stdout: TextIO,
) -> bool:
    """Transcribe *audio* (standard input for STDIN) and write each of *outputs*. Returns False after an error."""
    try:
        result = transcriber.transcribe(
            sys.stdin.buffer if is_stdin(audio) else audio,
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
        for output in outputs:
            write_output(result.render(output.fmt, args.max_line_width), output.path, stdout)
            if output.path is not None and not args.quiet:
                print(f"→ {output.path}", file=sys.stderr)
    except Exception as e:  # one bad file doesn't stop the batch; KeyboardInterrupt isn't an Exception
        # Other errors also show their type: a KeyError's message is just the key.
        message = str(e) if isinstance(e, FILE_ERRORS) else f"{type(e).__name__}: {e}".removesuffix(": ")
        print(f"Error: {shown_name(audio)}: {message}", file=sys.stderr)
        return False
    return True


###############################################################################
# Information display
###############################################################################


def show_backends() -> None:
    """Display the backends: description, models, capabilities and whether their packages are installed.

    The installed check looks the packages up without importing them, so listing stays instant.
    """
    print("Available transcription backends:\n")
    for name in list_backends():
        backend_class = get_backend_class(name)
        default_marker = " (default)" if name == DEFAULT_BACKEND else ""
        missing = backend_class.missing_requirements()
        installed = f"no (missing {', '.join(missing)})" if missing else "yes"
        print(f"  {name}{default_marker}")
        print(f"    {backend_class.description}")
        print(f"    Models: {', '.join(backend_class.available_models())}")
        print(f"    Capabilities: {_capability_list(backend_class.capabilities)}")
        print(f"    Installed: {installed}")
        print()


# Capabilities --list-backends leaves out: only the Python API can use them (it passes backends decoded samples).
API_ONLY_CAPABILITIES = frozenset({"waveform"})


def _capability_list(capabilities: AbstractSet[str]) -> str:
    """Capabilities as text, e.g. "translate, language detection": known ones in their usual order, then others.

    Capabilities only the Python API can use, such as "waveform", aren't shown.
    """
    capabilities = set(capabilities) - API_ONLY_CAPABILITIES
    names = [name for name in CAPABILITIES if name in capabilities] + sorted(set(capabilities) - set(CAPABILITIES))
    return ", ".join(name.replace("_", " ") for name in names) or "transcription only"


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


def utf8_when_redirected(stream: TextIO) -> None:
    """On Windows, make *stream* write UTF-8 when it is redirected to a file or a pipe.

    It would use the ANSI code page then, such as cp1252, which can't encode most scripts: a Chinese or Russian
    transcript would stop the program. A console takes any text already, and PYTHONIOENCODING, when set, wins.
    """
    platform = sys.platform  # read at run time, so tests can set it (mypy takes `sys.platform` checks as static)
    if platform != "win32" or os.environ.get("PYTHONIOENCODING"):
        return
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None and not stream.isatty():
        reconfigure(encoding="utf-8")


def check_inputs(jobs: Sequence[Job]) -> None:
    """Exit unless every input file exists and, when the input is ``-``, audio is piped into standard input."""
    missing = [str(job.audio) for job in jobs if not (is_stdin(job.audio) or job.audio.is_file())]
    if missing:
        sys.exit(f"Error: file not found: {', '.join(missing)}")
    if any(is_stdin(job.audio) for job in jobs) and (sys.stdin is None or sys.stdin.isatty()):
        sys.exit("Error: - reads audio piped into standard input, as in: cat talk.mp3 | transcribe -")


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
    utf8_when_redirected(stdout)
    # Libraries print progress and debug text to stdout; send it to stderr so stdout carries only the transcript.
    # Warnings, from the format lookup on, print as one line each.
    with warnings.catch_warnings(), contextlib.redirect_stdout(sys.stderr):
        warnings.showwarning = _print_warning
        formats = args.format or (format_for_path(args.output),)
        jobs = plan_outputs(args, formats)
        if not jobs:
            sys.exit("Error: no audio files found.")
        check_inputs(jobs)  # before the (slow) model load
        summary = run_jobs(args, jobs, stdout)
    if summary.interrupted:
        sys.exit(130)  # the exit status of a program stopped by Ctrl-C
    if summary.failed:
        sys.exit(1 if len(jobs) == 1 else f"Error: {summary.failed} of {len(jobs)} files failed.")
