"""Shared pieces of the integration tests: the samples (and the long clip made of them), the packages each test needs,
running the commands, and the checks on their output (word error rate, timestamps, chunks)."""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import pathlib
import re
import subprocess
import sys
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pytest

from tests.integration.devices import check_devices, require_device, require_gpu, sampling_nvidia_smi
from tests.integration.instrument import EVENTS_VARIABLE, NVIDIA_SMI_VARIABLE
from tests.integration.report import REPORT, Report, Run
from tests.integration.wer import edit_distance, normalize

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data"
MANIFEST = DATA / "samples.json"

# CI sets INTEGRATION_REQUIRE_EXTRAS=1: there, a missing package fails the test instead of skipping it.
REQUIRE_EXTRAS = os.getenv("INTEGRATION_REQUIRE_EXTRAS", "") not in ("", "0")

# The modules each backend or feature imports, and what installs them.
PACKAGES: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "whisper": (("whisper",), "openai-whisper"),
    "faster-whisper": (("faster_whisper",), '-e ".[faster-whisper]"'),
    "parakeet": (("torch", "transformers", "librosa"), '-e ".[nvidia]"'),
    "canary": (("torch", "transformers", "librosa"), '-e ".[nvidia]"'),
    "server": (("fastapi", "uvicorn", "python_multipart", "openai"), '-e ".[server]" openai'),
    "diarize": (("torch", "pyannote.audio"), '-e ".[diarize]"'),
    "voxtral": (("torch", "transformers", "mistral_common"), '-e ".[voxtral]"'),
}

# The transcribe options that select each backend and its model.
BACKEND_ARGS: Dict[str, Tuple[str, ...]] = {
    "whisper": ("-m", "tiny"),
    "faster-whisper": ("-b", "faster-whisper", "-m", "tiny"),
    "parakeet": ("-b", "parakeet"),
    "canary": ("-b", "canary", "-l", "en"),  # Canary can't detect the language
}

# The larger models of the full tier (INTEGRATION_TIER=full), as (backend, model); devices.MEMORY_GB has their needs.
FULL_MODELS: Tuple[Tuple[str, str], ...] = (
    ("faster-whisper", "large-v3"),
    ("faster-whisper", "turbo"),
    ("whisper", "turbo"),
    ("whisper", "large-v3"),
    ("parakeet", "parakeet-tdt-0.6b-v2"),
    ("parakeet", "parakeet-tdt-0.6b-v3"),
    ("canary", "canary-1b-v2"),
    ("voxtral", "voxtral-mini"),
)

# The highest word error rate accepted per backend. Every backend transcribes both samples without an error today
# (WER 0); the margins absorb small changes between model and library releases, the widest for the tiny models. A
# broken backend (words missing or glued together, another language, no text) lands well above them.
MAX_WER: Dict[str, float] = {
    "whisper": 0.25,  # tiny
    "faster-whisper": 0.25,  # tiny
    "parakeet": 0.15,
    "canary": 0.15,
    "voxtral": 0.15,  # full tier only
}

COMMAND_TIMEOUT = 900  # seconds, model download included
TIME_TOLERANCE = 0.25  # seconds a timestamp may run past the end of the audio (models round to their resolution)


@dataclasses.dataclass(frozen=True)
class Turn:
    voice: str  # the flite voice that speaks it
    text: str
    start: float
    end: float


@dataclasses.dataclass(frozen=True)
class Sample:
    name: str  # "speech" or "dialogue"
    path: pathlib.Path
    duration: float
    turns: Tuple[Turn, ...]

    @property
    def text(self) -> str:
        """The reference transcript."""
        return " ".join(turn.text for turn in self.turns)


def load_samples() -> Dict[str, Sample]:
    """The samples described in tests/data/samples.json, by name (the file name without extension)."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {
        pathlib.Path(file).stem: Sample(
            pathlib.Path(file).stem, DATA / file, sample["duration"], tuple(Turn(**turn) for turn in sample["turns"])
        )
        for file, sample in manifest.items()
    }


def require(*features: str) -> None:
    """Skip the test unless the packages of every feature (a backend, "server" or "diarize") are installed.

    With INTEGRATION_REQUIRE_EXTRAS=1, a missing package fails the test instead. A package that is installed but
    fails to import always fails the test. With INTEGRATION_DEVICE=cuda, the test fails when there is no GPU, or when
    the library a feature runs on (torch or CTranslate2) can't use it.
    """
    require_gpu()
    for feature in features:
        modules, install = PACKAGES[feature]
        for module in modules:
            reason = f"{feature} needs {module}: pip install {install}"
            if REQUIRE_EXTRAS:
                try:
                    importlib.import_module(module)
                except ModuleNotFoundError:
                    pytest.fail(f"{reason} (INTEGRATION_REQUIRE_EXTRAS is set, so missing packages fail)")
            else:
                pytest.importorskip(module, reason=reason)
    require_device(*features)


def command_env() -> Dict[str, str]:
    """The environment the commands run in: this checkout first on the import path, UTF-8 output."""
    env = dict(os.environ, PYTHONUTF8="1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))
    return env


def instrumented_env(events: pathlib.Path) -> Dict[str, str]:
    """command_env() for a command run through tests.integration.instrument, which appends its events to *events*."""
    env = command_env()
    env[EVENTS_VARIABLE] = str(events)
    nvidia_smi = sampling_nvidia_smi()
    if nvidia_smi:
        env[NVIDIA_SMI_VARIABLE] = nvidia_smi
    return env


class EventLog:
    """The events an instrumented command appends to *path* (see instrument.py), read as the file grows."""

    def __init__(self, path: pathlib.Path, report: Report = REPORT) -> None:
        self.path = path
        self.report = report
        self._read = 0  # characters read so far
        self._load: Optional[Dict[str, Any]] = None  # the last model load

    def record(self) -> int:
        """Check the device of each new model load, and add each new transcription to the report.

        Returns the number of new loads.
        """
        text = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
        complete = text[: text.rfind("\n") + 1]  # a line still being written waits for the next call
        new, self._read = complete[self._read :], len(complete)
        loads = 0
        for line in new.splitlines():
            event = json.loads(line)
            if event["event"] == "load":
                check_devices(event["backend"], event["devices"])
                self._load, loads = event, loads + 1
            elif event["event"] == "transcribe" and self._load is not None:
                self.report.add(run_from_events(self._load, event, self.report.test))
        return loads


def run_from_events(load: Dict[str, Any], transcription: Dict[str, Any], test: str) -> Run:
    """The report's Run for a "transcribe" event, with the model of the "load" event before it."""
    return Run(
        test=test,
        backend=load["backend"],
        model=load["model"],
        devices=load["devices"],
        precision=load["precision"],
        load_seconds=load["seconds"],
        audio_seconds=transcription["audio_seconds"],
        seconds=transcription["seconds"],
        torch_peak_bytes=transcription.get("torch_peak_bytes"),
        sampled_peak_bytes=transcription.get("sampled_peak_bytes"),
    )


def run_module(
    module: str, *args: object, cwd: pathlib.Path, events: Optional[EventLog] = None
) -> subprocess.CompletedProcess:
    """Run ``python -m <module> <args>`` on this checkout, with the interpreter that runs pytest.

    Fails the test if the command fails; otherwise returns it with its stdout and stderr. With *events*, the command
    runs through tests.integration.instrument: the test also fails unless the models it loaded ran on
    INTEGRATION_DEVICE, and the files it transcribed go in the report.
    """
    command = [sys.executable, "-m", module, *map(str, args)]
    env = command_env()
    if events is not None:
        command[2:2] = ["tests.integration.instrument"]  # python -m tests.integration.instrument <module> <args>
        env = instrumented_env(events.path)
    done = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=COMMAND_TIMEOUT,
    )
    shown = " ".join([module, *map(str, args)])
    if done.returncode != 0:
        pytest.fail(f"python -m {shown} exited with {done.returncode}:\n{done.stderr}", pytrace=False)
    if events is not None and not events.record():
        pytest.fail(f"python -m {shown} loaded no model, or the instrumentation recorded none", pytrace=False)
    return done


@dataclasses.dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str  # its lines, joined by newlines


_TIMING = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d\d\d) --> (\d+):(\d\d):(\d\d)[,.](\d\d\d)")


def parse_cues(subtitles: str) -> List[Cue]:
    """The cues of SRT or WebVTT *subtitles*."""
    cues = []
    for block in subtitles.strip().split("\n\n"):
        lines = block.splitlines()
        for i, line in enumerate(lines):
            match = _TIMING.fullmatch(line.strip())
            if match:
                h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, match.groups())
                start, end = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000, h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
                cues.append(Cue(start, end, "\n".join(lines[i + 1 :])))
                break
    return cues


def assert_timeline(spans: Iterable[Tuple[float, float]], duration: float, what: str) -> None:
    """*spans* (start, end) are inside the audio and in order: neither starts nor ends ever go back in time."""
    spans = list(spans)
    assert spans, f"no {what}"
    for start, end in spans:
        fits = 0.0 <= start <= end <= duration + TIME_TOLERANCE
        assert fits, f"{what}: {start:.2f}-{end:.2f} s doesn't fit in the audio (0-{duration:.2f} s)"
    for (start, end), (next_start, next_end) in zip(spans, spans[1:]):
        in_order = start <= next_start and end <= next_end
        assert in_order, f"{what} out of order: {start:.2f}-{end:.2f} s, then {next_start:.2f}-{next_end:.2f} s"


def assert_wer(transcript: str, sample: Sample, backend: str) -> None:
    """The transcript's word error rate on *sample* is within *backend*'s threshold; it is printed for the CI log.

    It also goes in the report, with the test's run.
    """
    reference = normalize(sample.text)
    errors = edit_distance(reference, normalize(transcript))
    score, limit = errors / len(reference), MAX_WER[backend]
    REPORT.note_wer(errors, len(reference))
    print(f"{backend} on {sample.name}: WER {score:.3f} (at most {limit}): {transcript.strip()!r}")
    assert score <= limit, (
        f"{backend} on {sample.name}: WER {score:.2f} is above {limit}\n"
        f"  reference:  {sample.text}\n  transcript: {transcript.strip()}"
    )


def labels_by_voice(turns: Sequence[Turn], segments: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """The speaker labels each voice got: per turn, the label of the segments that cover most of it."""
    labels: Dict[str, Set[str]] = {}
    for turn in turns:
        overlap: Counter = Counter()
        for seg in segments:
            shared = min(seg["end"], turn.end) - max(seg["start"], turn.start)
            if shared > 0:
                overlap[seg["speaker"]] += shared
        assert overlap, f"no speaker segment during {turn.text!r} ({turn.start}-{turn.end} s)"
        labels.setdefault(turn.voice, set()).add(overlap.most_common(1)[0][0])
    return labels


def require_hf_token() -> None:
    """Skip the test unless HF_TOKEN is set: pyannote's diarization model is gated."""
    if not os.getenv("HF_TOKEN"):
        pytest.skip("needs HF_TOKEN with access to pyannote/speaker-diarization-community-1 (a gated model)")


def check_speaker_labels(
    sample: Sample, args: Sequence[object], transcribe: Callable[..., subprocess.CompletedProcess], folder: pathlib.Path
) -> None:
    """Label the speakers of the two-voice *sample* (--diarize, with the model *args* select) and check the labels:
    each voice gets one, the two voices different ones, and the transcript shows them."""
    txt, result_file = folder / f"{sample.name}.txt", folder / f"{sample.name}.json"
    options = ["-q", "--diarize", "--num-speakers", "2", "-o", txt, "--json", result_file]

    run = transcribe(sample.path, *args, *options)

    assert run.stdout == ""
    segments = json.loads(result_file.read_text(encoding="utf-8"))["speaker_segments"]
    assert_timeline(((seg["start"], seg["end"]) for seg in segments), sample.duration, "speaker segments")
    labels = labels_by_voice(sample.turns, segments)
    print(f"speaker labels by voice: {labels}")
    assert all(len(found) == 1 for found in labels.values()), f"a voice got several speaker labels: {labels}"
    speakers: Set[str] = set().union(*labels.values())
    assert len(speakers) == 2, f"both voices got the same speaker label: {labels}"
    transcript = txt.read_text(encoding="utf-8")  # one "[SPEAKER_00] text" line per speaker turn
    assert all(f"[{speaker}] " in transcript for speaker in speakers), transcript


# The long clip: the dialogue and the speech sample, six times over, with a second of silence between parts (104 s).
# That makes several of Canary's, Voxtral's and Whisper's 30 s chunks, and stays under Parakeet's 120 s chunk. It
# ends where the speech does: after the same sentences six times, Whisper turbo fills trailing silence with made-up
# repetitions ("test test test", "Thank you very much." again and again), which isn't what this clip tests.
LONG_CLIP_PARTS = ("dialogue", "speech")
LONG_CLIP_REPEATS = 6
LONG_CLIP_GAP = 1.0  # seconds of silence between parts
# Backends that make one segment per chunk, starting where the chunk does.
ONE_SEGMENT_PER_CHUNK = frozenset({"canary", "voxtral"})


def join_audio(parts: Sequence[Tuple[Sample, "np.ndarray"]], gap: float, rate: int) -> Tuple["np.ndarray", List[Turn]]:
    """Join *parts*, each a sample and its audio at *rate* Hz, with *gap* seconds of silence between them.

    Returns the audio, and the samples' turns moved to where they are in it.
    """
    import numpy as np

    silence = np.zeros(int(gap * rate), dtype=np.float32)
    pieces: List["np.ndarray"] = []
    turns: List[Turn] = []
    offset = 0
    for sample, audio in parts:
        if pieces:
            pieces.append(silence)
            offset += len(silence)
        start = offset / rate
        turns += [
            dataclasses.replace(turn, start=round(turn.start + start, 3), end=round(turn.end + start, 3))
            for turn in sample.turns
        ]
        pieces.append(audio.astype(np.float32))
        offset += len(audio)
    return np.concatenate(pieces), turns


def write_wav(path: pathlib.Path, audio: "np.ndarray", rate: int) -> None:
    """Write *audio*, float samples in [-1, 1], as a mono 16-bit WAV file."""
    import wave

    import numpy as np

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())


def build_long_clip(samples: Dict[str, Sample], folder: pathlib.Path) -> Sample:
    """The long clip (LONG_CLIP_*), written to *folder* as long.wav, with the samples' turns moved into place."""
    from speech_toolkit.media import SAMPLE_RATE, load_audio

    decoded = {name: load_audio(samples[name].path) for name in LONG_CLIP_PARTS}
    parts = [(samples[name], decoded[name]) for name in LONG_CLIP_PARTS] * LONG_CLIP_REPEATS
    audio, turns = join_audio(parts, LONG_CLIP_GAP, SAMPLE_RATE)
    path = folder / "long.wav"
    write_wav(path, audio, SAMPLE_RATE)
    return Sample("long", path, len(audio) / SAMPLE_RATE, tuple(turns))


def chunk_seconds(backend: str) -> float:
    """How much audio *backend* transcribes at once: its chunk length, or the 30 s window of the Whisper models."""
    if backend == "voxtral":
        from speech_toolkit.backends.voxtral_backend import CHUNK_SECONDS

        return float(CHUNK_SECONDS)
    from speech_toolkit.backends import get_backend_class

    return float(getattr(get_backend_class(backend), "CHUNK_SECONDS", 30))


def chunk_spans(path: pathlib.Path, seconds: float) -> List[Tuple[float, float]]:
    """Where the chunked backends cut the audio at *path* into chunks of at most *seconds*: (start, end) pairs."""
    from speech_toolkit.media import SAMPLE_RATE, load_audio, split_audio

    return [(start, start + len(chunk) / SAMPLE_RATE) for start, chunk in split_audio(load_audio(path), seconds)]


def assert_chunks_covered(
    spans: Sequence[Tuple[float, float]], chunks: Sequence[Tuple[float, float]], what: str
) -> None:
    """Some of *spans* (segments or words) fall in every chunk: no chunk's text went missing."""
    for start, end in chunks:
        found = any(span_start < end and span_end > start for span_start, span_end in spans)
        assert found, f"no {what} in the chunk at {start:.2f}-{end:.2f} s"


def assert_no_cut_off_warning(stderr: str) -> None:
    """No backend warned that it cut text off, as Canary and Voxtral do when a chunk reaches their token limit (#27)."""
    pattern = re.compile(r"cut off|truncat", re.IGNORECASE)
    warned = [line for line in stderr.splitlines() if line.startswith("Warning:") and pattern.search(line)]
    assert not warned, "the backend warned that it cut text off:\n" + "\n".join(warned)


def check_long_audio(
    backend: str,
    args: Sequence[object],
    clip: Sample,
    transcribe: Callable[..., subprocess.CompletedProcess],
    folder: pathlib.Path,
) -> Dict[str, Any]:
    """Transcribe the long *clip* with *backend* (selected by *args*), with word timings if it has them, and check:

    - the chunks: a long clip makes several 30 s chunks, every chunk has text, and Canary and Voxtral start a segment
      at each one;
    - the timestamps of segments and words: in order across the chunk boundaries, and inside the audio;
    - the word error rate, against the backend's threshold;
    - that the backend didn't warn that it cut text off.

    Returns the JSON result.
    """
    from speech_toolkit.backends import get_backend_class

    result_file = folder / "long.json"
    words = "word_timestamps" in get_backend_class(backend).capabilities
    run = transcribe(clip.path, *args, "--json", result_file, *(["--word-timestamps"] if words else []))

    assert_no_cut_off_warning(run.stderr)
    result: Dict[str, Any] = json.loads(result_file.read_text(encoding="utf-8"))
    segments = [(seg["start"], seg["end"]) for seg in result["segments"]]
    assert_timeline(segments, clip.duration, "segments")
    seconds = chunk_seconds(backend)
    chunks = chunk_spans(clip.path, seconds)
    assert seconds > clip.duration or len(chunks) >= 3, f"{clip.duration:.0f} s made only {len(chunks)} chunks"
    assert_chunks_covered([(s["start"], s["end"]) for s in result["segments"] if s["text"].strip()], chunks, "segment")
    if backend in ONE_SEGMENT_PER_CHUNK:
        starts = [round(start, 2) for start, _ in segments]
        missing = [round(start, 2) for start, _ in chunks if round(start, 2) not in starts]
        assert not missing, f"no segment starts where the chunks at {missing} s do; segments start at {starts}"
    if words:
        timings = [(word["start"], word["end"]) for seg in result["segments"] for word in seg.get("words", [])]
        assert_timeline(timings, clip.duration, "words")
        assert_chunks_covered(timings, chunks, "word")
    assert_wer(run.stdout, clip, backend)
    return result
