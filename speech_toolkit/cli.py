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

The language of each file, from its first 30 s, without transcribing:
    transcribe --detect-language recordings/
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import math
import os
import pathlib
import sys
import time
import warnings
from collections import Counter
from typing import AbstractSet, Any, Iterator, List, Mapping, Optional, Sequence, TextIO, Tuple

from . import __version__
from .api import TASKS, Transcriber, check_language_detection, check_load_arguments, language_detectors
from .backends import (
    CAPABILITIES,
    DEFAULT_BACKEND,
    backends_on,
    backends_with,
    backends_with_option,
    get_backend_class,
    list_backends,
    parse_device,
)
from .backends.base import TranscriptionResult
from .backends.config import describe_devices
from .diarization import check_speaker_hints
from .errors import SpeechToolkitError, UnsupportedOptionError
from .formats import FORMATS, SPEAKER_FORMATS, format_for_path, write_text
from .media import MEDIA_EXTENSIONS, collect_files

# Errors whose message says what went wrong, so it is shown alone. In a batch, any other Exception also fails only
# its own file (shown with its type); KeyboardInterrupt still stops the run.
FILE_ERRORS = (SpeechToolkitError, OSError, RuntimeError, ValueError)

STDIN = pathlib.Path("-")
"""The input ``-``: audio piped into standard input. With --outdir, its outputs are named ``stdin.<format>``."""
STDIN_NAME = "stdin"

JSON_EXTENSIONS = frozenset({".json"})
"""What --from-json picks up in folders: results saved as JSON."""

# The options that change what is transcribed, as (dest, flag): --from-json, which transcribes nothing, refuses them.
TRANSCRIPTION_OPTIONS = (
    ("backend", "--backend"),
    ("model", "--model"),
    ("device", "--device"),
    ("compute_type", "--compute-type"),
    ("options", "-O"),
    ("language", "--language"),
    ("task", "--task"),
    ("hf_token", "--hf-token"),
    ("prompt", "--prompt"),
    ("vad", "--vad"),
    ("word_timestamps", "--word-timestamps"),
    ("diarize", "--diarize"),
    ("num_speakers", "--num-speakers"),
    ("min_speakers", "--min-speakers"),
    ("max_speakers", "--max-speakers"),
)


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


def positive_float(value: str) -> float:
    """argparse type for numbers greater than zero, such as 7 or 2.5."""
    number = float(value)
    if not (math.isfinite(number) and number > 0):
        raise argparse.ArgumentTypeError(f"must be a positive number, got {value}")
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
  %(prog)s --detect-language recordings/      # each file's language, without transcribing
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
    add_general_options(parser)
    add_output_options(parser)
    add_accuracy_options(parser)
    add_diarization_options(parser)
    add_information_options(parser)

    args = parser.parse_args(argv)
    if not (args.list_backends or args.list_models):
        if args.detect_language:
            _validate_detection(parser, args)
        _validate_args(parser, args)
    return args


def device_arg(value: str) -> str:
    """argparse type for --device: auto, cpu, cuda, cuda:N or mps, written as given ("CUDA:1" becomes "cuda:1")."""
    try:
        return str(parse_device(value))
    except UnsupportedOptionError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def option_arg(value: str) -> Tuple[str, str]:
    """argparse type for -O: KEY=VALUE, as a (key, value) pair; the backend checks both when it loads."""
    key, equals, text = value.partition("=")
    if not equals or not key.strip():
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, such as cpu_threads=4, got {value!r}")
    return key.strip(), text


def add_model_options(parser: argparse.ArgumentParser) -> None:
    """--backend, --model, --device, --compute-type and -O: the model to load (transcribe-server takes them too)."""
    model = parser.add_argument_group("model")
    model.add_argument(
        "-b",
        "--backend",
        default=DEFAULT_BACKEND,
        help=f"Transcription backend (default: {DEFAULT_BACKEND}). Available: {', '.join(list_backends())}",
    )
    model.add_argument(
        "-m",
        "--model",
        default=None,
        help="Model name/size (default: the backend's default), or a Hugging Face repo id (org/name) or local path; "
        "see --list-models.",
    )
    model.add_argument(
        "--device",
        type=device_arg,
        default=None,
        metavar="DEVICE",
        help="auto (default: the GPU when available), cpu, cuda, cuda:N for the N-th GPU, or mps for an Apple GPU "
        f"({', '.join(backends_on('mps')) or 'no backend'}).",
    )
    model.add_argument(
        "--compute-type",
        metavar="TYPE",
        help="Precision: auto (default: what suits the device), or a type --list-backends shows, such as int8 or "
        f"float16 ({', '.join(backends_with_option('compute_type')) or 'no backend'}).",
    )
    model.add_argument(
        "-O",
        "--option",
        dest="options",
        action="append",
        type=option_arg,
        metavar="KEY=VALUE",
        help="Set a backend option, such as cpu_threads=4; repeat for several. --list-backends lists them.",
    )


def input_path(value: str) -> pathlib.Path:
    """argparse type for inputs: a path, or STDIN itself for ``-`` (so that ``./-`` is still a file named "-")."""
    return STDIN if value == "-" else pathlib.Path(value)


def is_stdin(path: pathlib.Path) -> bool:
    """Whether *path* is the input ``-``: standard input."""
    return path is STDIN


def format_list(value: str) -> Tuple[str, ...]:
    """argparse type for --format: a format, several separated by commas (``srt,json``), or ``all`` of them.

    ``all`` stays in the list, as it stands for formats that depend on other options: see expand_formats().
    """
    names = [name.strip().lower() for name in value.split(",") if name.strip()]
    if not names or any(name not in FORMATS and name != "all" for name in names):
        raise argparse.ArgumentTypeError(
            f"invalid format {value!r}: choose from {', '.join(FORMATS)}, several separated by commas, or all"
        )
    return tuple(dict.fromkeys(names))


def expand_formats(names: Optional[Sequence[str]], diarize: bool) -> Optional[Tuple[str, ...]]:
    """The formats of --format, with ``all`` standing for every format, but for the ones that need speaker labels
    (rttm) unless *diarize* or the list names them too. None (no --format) stays None."""
    if names is None or "all" not in names:
        return None if names is None else tuple(names)
    return tuple(fmt for fmt in FORMATS if diarize or fmt in names or fmt not in SPEAKER_FORMATS)


def speaker_names_arg(value: str) -> Tuple[str, ...]:
    """argparse type for --speaker-names: names separated by commas; an empty one keeps that speaker's label."""
    names = tuple(name.strip() for name in value.split(","))
    if not any(names):
        raise argparse.ArgumentTypeError(f"expected names separated by commas, such as Alice,Bob, got {value!r}")
    return names


def add_output_options(parser: argparse.ArgumentParser) -> None:
    """Where transcripts go, and in which formats."""
    output = parser.add_argument_group("output")
    output.add_argument("-o", "--output", type=pathlib.Path, help="Write the transcript to this file, not stdout.")
    output.add_argument(
        "-f",
        "--format",
        type=format_list,
        metavar="FORMAT",
        help=f"Output format: {', '.join(FORMATS)} (rttm needs --diarize). Several separated by commas (srt,json), "
        "or all (rttm only with --diarize), write a file each from one transcription (default: from the --output "
        "extension, else txt).",
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
        "--from-json",
        action="store_true",
        help="The inputs are results saved as JSON (by --json or -f json), not audio: write them in the formats "
        "asked for without transcribing, so no model loads. Folders are searched for .json files.",
    )
    output.add_argument(
        "--max-line-width",
        type=positive_int,
        metavar="N",
        help="Split subtitles into lines of at most N columns, 2 lines per cue (srt/vtt). Chinese, Japanese and "
        "Korean characters take two columns.",
    )
    output.add_argument(
        "--max-lines",
        type=positive_int,
        metavar="N",
        help="With --max-line-width, at most N lines per cue (default: 2).",
    )
    output.add_argument(
        "--max-cue-duration",
        type=positive_float,
        metavar="SECONDS",
        help="Split subtitle cues that would last longer than this many seconds (srt/vtt).",
    )
    output.add_argument(
        "--speaker-names",
        type=speaker_names_arg,
        metavar="NAMES",
        help='Rename the speakers in the order they first speak, such as "Alice,Bob", in every format; an empty name '
        "keeps that speaker's label (with --diarize, or --from-json).",
    )
    output.add_argument(
        "--timestamps",
        action="store_true",
        help="txt: one line per segment, starting with its time: [00:01:02] text.",
    )
    output.add_argument(
        "--word-level",
        action="store_true",
        help="vtt: mark when each word starts, with <00:00:01.200> tags, so players can highlight the words as they "
        "are spoken. Turns on word timestamps.",
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


def add_general_options(parser: argparse.ArgumentParser) -> None:
    """-q, -v and --log-level: what the command shows on stderr besides the transcript."""
    general = parser.add_argument_group("general")
    amount = general.add_mutually_exclusive_group()
    amount.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Hide progress: loading messages, [i/N] lines and the summary, progress bars (Hugging Face downloads "
        "and transformers' too) and info messages. Warnings and errors still show.",
    )
    amount.add_argument(
        "-v", "--verbose", action="store_true", help="Also show debug messages (the same as --log-level debug)."
    )
    general.add_argument(
        "--log-level",
        type=str.lower,
        choices=LOG_LEVELS,
        metavar="LEVEL",
        help="Show messages from this level up: debug, info, warning or error "
        "(default: info; warning with -q, debug with -v).",
    )


def add_information_options(parser: argparse.ArgumentParser) -> None:
    """Flags that print information and exit instead of transcribing."""
    info = parser.add_argument_group("information")
    info.add_argument("--list-backends", action="store_true", help="List available backends and exit.")
    info.add_argument("--list-models", action="store_true", help="List models for the selected backend and exit.")
    info.add_argument(
        "--detect-language",
        action="store_true",
        help="Print the language of each input, from its first 30 s, and exit without transcribing: a line each, "
        "with the file, the language and its probability separated by tabs, or with -f json a JSON object each "
        f"({', '.join(language_detectors()) or 'no backend'}).",
    )


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject option combinations that can't work (exits via parser.error), before any model loads."""
    if not args.audio:
        parser.error("the following arguments are required: audio")
    args.format = expand_formats(args.format, args.diarize)
    problem = output_problem(args) or rendering_problem(parser, args)
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


def rendering_problem(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Optional[str]:
    """Why the rendering options, and --from-json, don't fit the other options, or None when they do."""
    if args.max_lines and not args.max_line_width:
        return "--max-lines needs --max-line-width: cues are split into lines to a width"
    if args.speaker_names and not (args.diarize or args.from_json):
        return "--speaker-names renames the speakers --diarize labels, so it needs --diarize (or --from-json)"
    if args.from_json:
        if any(is_stdin(path) for path in args.audio):
            return "--from-json reads results saved in files, not standard input"
        given = [flag for dest, flag in TRANSCRIPTION_OPTIONS if getattr(args, dest) != parser.get_default(dest)]
        if given:
            return f"--from-json writes saved results without transcribing, so it doesn't take {', '.join(given)}"
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
        files = collect_files(args.audio, JSON_EXTENSIONS if args.from_json else MEDIA_EXTENSIONS)
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
    """Load the selected backend's model (its default if --model is not given) and, with --diarize, pyannote.

    A model name, --device, --compute-type or -O option the backend can't take fails first, before the message that
    the model loads.
    """
    backend_class = get_backend_class(args.backend)
    model_name = args.model or backend_class.default_model()
    compute_type = getattr(args, "compute_type", None)
    options = dict(getattr(args, "options", None) or ())  # the last -O for a key wins
    check_load_arguments(backend_class, model_name, args.device, compute_type, options)
    if not args.quiet:
        print(f"Loading {args.backend} model '{model_name}'...", file=sys.stderr)
    return Transcriber(
        args.backend,
        model_name,
        device=args.device,
        compute_type=compute_type,
        options=options,
        diarize=args.diarize,
        hf_token=args.hf_token,
    )


def word_timestamps_option(
    args: argparse.Namespace, formats: AbstractSet[str], transcriber: Transcriber
) -> Optional[bool]:
    """--word-timestamps forces word timings, and so does --word-level for vtt (the library warns if the backend has
    none); splitting subtitle cues to a width or a duration uses them when the backend has them."""
    splits_cues = (args.max_line_width or args.max_cue_duration) and not formats.isdisjoint({"srt", "vtt"})
    if args.word_timestamps or (args.word_level and "vtt" in formats):
        return True
    if splits_cues and transcriber.supports("word_timestamps"):
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


def recording_id(audio: pathlib.Path) -> str:
    """The name rttm lines give the recording: its file name without the extension, or ``stdin``."""
    return STDIN_NAME if is_stdin(audio) else audio.stem


def render_output(args: argparse.Namespace, result: TranscriptionResult, fmt: str, audio: pathlib.Path) -> str:
    """*result* of the input *audio*, in format *fmt*, with the output options of *args*."""
    return result.render(
        fmt,
        args.max_line_width,
        max_lines=args.max_lines,
        max_cue_duration=args.max_cue_duration,
        speaker_names=args.speaker_names,
        timestamps=args.timestamps,
        word_level=args.word_level,
        file_id=recording_id(audio),
    )


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

    The model loads before the first transcription, so a run with nothing left to do (--skip-existing) loads none;
    with --from-json, none loads: each input is a saved result, loaded instead.
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
            if transcriber is None and not args.from_json:
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
    transcriber: Optional[Transcriber],
    audio: pathlib.Path,
    outputs: Sequence[Output],
    word_timestamps: Optional[bool],
    stdout: TextIO,
) -> bool:
    """Transcribe *audio* (standard input for STDIN), or without a *transcriber* (--from-json) load the result saved
    in it, and write each of *outputs*. Returns False after an error."""
    try:
        if transcriber is None:
            result = TranscriptionResult.load(audio)
        else:
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
            write_output(render_output(args, result, output.fmt, audio), output.path, stdout)
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
    """Display the backends: description, models, capabilities, whether their packages are installed, and then the
    custom models, devices and options each declares (a line is left out when it declares nothing for it).

    The first lines of each backend stay as they were, in the same places, for scripts that read them. The installed
    check looks the packages up without importing them, so listing stays instant.
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
        custom = backend_class.describe_custom_models()
        if custom:
            print(f"    Custom models: {custom}")
        if backend_class.DEVICES is not None:
            print(f"    Devices: {describe_devices(backend_class.DEVICES)}")
        for option_name, option in backend_class.OPTIONS.items():
            print(f"    -O {option_name}={option.metavar}: {option.describe()}")
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
    custom = backend_class.describe_custom_models()
    if custom:
        print(f"\nCustom models: -m also takes {custom}.")


###############################################################################
# Language detection
###############################################################################

# The options --detect-language refuses, as (dest, flag): it transcribes nothing, and prints its lines to stdout.
DETECTION_REFUSES = (
    ("language", "--language"),
    ("task", "--task"),
    ("prompt", "--prompt"),
    ("vad", "--vad"),
    ("word_timestamps", "--word-timestamps"),
    ("diarize", "--diarize"),
    ("num_speakers", "--num-speakers"),
    ("min_speakers", "--min-speakers"),
    ("max_speakers", "--max-speakers"),
    ("output", "--output"),
    ("outdir", "--outdir"),
    ("json", "--json"),
    ("skip_existing", "--skip-existing"),
    ("max_line_width", "--max-line-width"),
)
# What -f takes with --detect-language: txt, tab-separated lines (the default), or json, a JSON object per line.
DETECTION_FORMATS = ("txt", "json")


def _validate_detection(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject the options --detect-language doesn't take (exits via parser.error)."""
    given = [flag for dest, flag in DETECTION_REFUSES if getattr(args, dest) != parser.get_default(dest)]
    if given:
        parser.error(f"--detect-language only detects the language, so it doesn't take {', '.join(given)}")
    if args.format is not None and tuple(args.format) not in [(fmt,) for fmt in DETECTION_FORMATS]:
        parser.error("--detect-language prints a line per input, or with -f json a JSON object: -f takes txt or json")


def detect_languages(args: argparse.Namespace) -> None:
    """--detect-language: print the language of each input, from its first 30 seconds, without transcribing.

    Each input's line goes to stdout as soon as it is detected (see detection_line()). The inputs are found as for
    transcription: folders are searched for media files, and ``-`` reads standard input. A backend that can't detect
    languages and a missing file are errors before the model loads. A file that fails is reported on stderr and the
    others carry on; the exit status is then non-zero, and 130 after Ctrl-C.
    """
    stdout = sys.stdout
    utf8_when_redirected(stdout)
    as_json = args.format == ("json",)
    # As when transcribing: stdout carries only the lines, other output goes to stderr, a warning is one line.
    with warnings.catch_warnings(), contextlib.redirect_stdout(sys.stderr), show_logs(args):
        warnings.showwarning = _print_warning
        try:
            check_language_detection(args.backend)
        except ValueError as e:  # an unknown backend, or one that can't detect languages
            sys.exit(f"Error: {e}")
        inputs = detection_inputs(args.audio)
        if not inputs:
            sys.exit("Error: no audio files found.")
        check_inputs([Job(audio, ()) for audio in inputs])  # before the (slow) model load
        transcriber = load_or_exit(args)
        try:
            failed = sum(not detect_input(transcriber, audio, as_json, stdout) for audio in inputs)
        except KeyboardInterrupt:
            sys.exit(130)  # the exit status of a program stopped by Ctrl-C
    if failed:
        sys.exit(1 if len(inputs) == 1 else f"Error: {failed} of {len(inputs)} files failed.")


def detection_inputs(paths: Sequence[pathlib.Path]) -> List[pathlib.Path]:
    """The inputs of --detect-language: standard input, or the files given and the media files in the folders given."""
    if paths and is_stdin(paths[0]):
        return [paths[0]]  # the only input: output_problem() checks that
    return [file for file, _ in collect_files(paths, MEDIA_EXTENSIONS)]


def detect_input(transcriber: Transcriber, audio: pathlib.Path, as_json: bool, stdout: TextIO) -> bool:
    """Detect the language of *audio* (standard input for STDIN) and print its line. Returns False after an error."""
    try:
        detection = transcriber.detect_language(sys.stdin.buffer if is_stdin(audio) else audio)
    except Exception as e:  # one bad file doesn't stop the others; KeyboardInterrupt isn't an Exception
        message = str(e) if isinstance(e, FILE_ERRORS) else f"{type(e).__name__}: {e}".removesuffix(": ")
        print(f"Error: {shown_name(audio)}: {message}", file=sys.stderr)
        return False
    print(detection_line(audio, detection, as_json), file=stdout, flush=True)
    return True


def detection_line(audio: pathlib.Path, detection: Mapping[str, Any], as_json: bool) -> str:
    """What --detect-language prints for *audio*: ``file<TAB>language<TAB>probability``, the probability with two
    decimals; or with *as_json*, *detection* as a JSON object after the file, ``{"file": "talk.mp3", "language": "de",
    "probability": 0.97, "candidates": [...]}``. Standard input is the file ``-``."""
    if as_json:
        return json.dumps({"file": str(audio), **detection}, ensure_ascii=False)
    return f"{audio}\t{detection['language']}\t{detection['probability']:.2f}"


###############################################################################
# Messages
###############################################################################

LOG_LEVELS = ("debug", "info", "warning", "error")
# What -q sets before any model loads: no Hugging Face progress bars (model downloads, and transformers' "Loading
# weights", which follows the same setting), and no transformers messages below errors. Both libraries read these
# variables when they are imported.
QUIET_ENVIRONMENT = {"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TRANSFORMERS_VERBOSITY": "error"}


class _LogFormatter(logging.Formatter):
    """Log messages for stderr: info messages as they are, others after their level, like the Warning: lines."""

    PREFIXES = {logging.DEBUG: "Debug: ", logging.WARNING: "Warning: ", logging.ERROR: "Error: "}

    def format(self, record: logging.LogRecord) -> str:
        prefix = self.PREFIXES.get(record.levelno, "Error: " if record.levelno > logging.ERROR else "")
        return prefix + super().format(record)


@contextlib.contextmanager
def show_logs(args: argparse.Namespace) -> Iterator[None]:
    """Show speech_toolkit's log messages on stderr, from the level -q, -v or --log-level ask for, until the block ends.

    Other libraries' messages aren't shown. With -q, Hugging Face's progress bars and transformers' messages below
    errors are hidden too (QUIET_ENVIRONMENT): the block must start before any model loads. The environment is
    restored at the end.
    """
    level = args.log_level or ("debug" if args.verbose else "warning" if args.quiet else "info")
    logger = logging.getLogger("speech_toolkit")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_LogFormatter())
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    saved = {name: os.environ.get(name) for name in QUIET_ENVIRONMENT}
    try:
        if args.quiet:
            _quiet_libraries()
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _quiet_libraries() -> None:
    """Hide Hugging Face's progress bars and transformers' messages below errors (for -q).

    A library imported already, which no longer reads the environment, is told directly: the modules that keep these
    settings are loaded once it has read them.
    """
    os.environ.update(QUIET_ENVIRONMENT)
    transformers_logging = sys.modules.get("transformers.utils.logging")
    if transformers_logging is not None:
        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    hub_utils = sys.modules.get("huggingface_hub.utils")
    if hub_utils is not None:
        hub_utils.disable_progress_bars()


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


def check_outputs(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    """Exit unless every output can be written: rttm needs speaker labels, and --from-json doesn't replace a file it
    reads (unless --skip-existing keeps it)."""
    formats = {output.fmt for job in jobs for output in job.outputs}
    if not (args.diarize or args.from_json) and not formats.isdisjoint(SPEAKER_FORMATS):
        sys.exit("Error: rttm lists the speaker turns --diarize finds, so it needs --diarize")
    if args.from_json and not args.skip_existing:
        sources = {_file_identity(job.audio) for job in jobs}
        replaced = [
            str(output.path)
            for job in jobs
            for output in job.outputs
            if output.path is not None and output.path.exists() and _file_identity(output.path) in sources
        ]
        if replaced:
            sys.exit(
                f"Error: --from-json would replace a result it reads: {', '.join(replaced)}. "
                "Use --outdir, or leave json out of -f."
            )


def _file_identity(path: pathlib.Path) -> Tuple[int, int]:
    """What tells whether two paths name the same file: its device and inode numbers."""
    stat = path.stat()
    return stat.st_dev, stat.st_ino


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    # Handle information flags
    if args.list_backends:
        show_backends()
        return

    if args.list_models:
        show_models(args.backend)
        return

    if args.detect_language:
        detect_languages(args)
        return

    stdout = sys.stdout
    utf8_when_redirected(stdout)
    # Libraries print progress and debug text to stdout; send it to stderr so stdout carries only the transcript.
    # Warnings, from the format lookup on, print as one line each.
    with warnings.catch_warnings(), contextlib.redirect_stdout(sys.stderr), show_logs(args):
        warnings.showwarning = _print_warning
        formats = args.format or (format_for_path(args.output),)
        jobs = plan_outputs(args, formats)
        if not jobs:
            sys.exit(f"Error: no {'JSON' if args.from_json else 'audio'} files found.")
        check_inputs(jobs)  # before the (slow) model load
        check_outputs(args, jobs)
        summary = run_jobs(args, jobs, stdout)
    if summary.interrupted:
        sys.exit(130)  # the exit status of a program stopped by Ctrl-C
    if summary.failed:
        sys.exit(1 if len(jobs) == 1 else f"Error: {summary.failed} of {len(jobs)} files failed.")
