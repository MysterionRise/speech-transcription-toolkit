"""Shared pieces of the integration tests: the samples, the packages each test needs, running the commands, and the
checks on their output (word error rate, timestamps)."""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Dict, Iterable, List, Tuple

import pytest

from tests.integration.wer import wer

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
}

# The transcribe options that select each backend and its model.
BACKEND_ARGS: Dict[str, Tuple[str, ...]] = {
    "whisper": ("-m", "tiny"),
    "faster-whisper": ("-b", "faster-whisper", "-m", "tiny"),
    "parakeet": ("-b", "parakeet"),
    "canary": ("-b", "canary", "-l", "en"),  # Canary can't detect the language
}

# The highest word error rate accepted, per backend and sample. Every backend transcribes the speech sample without
# an error today; the margins absorb small changes between model and library releases (the tiny models get the
# widest), while a broken backend (words missing or glued together, another language, no text) lands well above.
MAX_WER: Dict[str, Dict[str, float]] = {
    "whisper": {"speech": 0.25, "dialogue": 0.3},
    "faster-whisper": {"speech": 0.25, "dialogue": 0.3},
    "parakeet": {"speech": 0.15, "dialogue": 0.15},
    "canary": {"speech": 0.15, "dialogue": 0.15},
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
    fails to import always fails the test.
    """
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


def command_env() -> Dict[str, str]:
    """The environment the commands run in: this checkout first on the import path, UTF-8 output."""
    env = dict(os.environ, PYTHONUTF8="1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))
    return env


def run_module(module: str, *args: object, cwd: pathlib.Path) -> subprocess.CompletedProcess:
    """Run ``python -m <module> <args>`` on this checkout, with the interpreter that runs pytest.

    Fails the test if the command fails; otherwise returns it with its stdout and stderr.
    """
    command = [sys.executable, "-m", module, *map(str, args)]
    done = subprocess.run(
        command,
        cwd=cwd,
        env=command_env(),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=COMMAND_TIMEOUT,
    )
    if done.returncode != 0:
        pytest.fail(f"python {' '.join(command[1:])} exited with {done.returncode}:\n{done.stderr}", pytrace=False)
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
    """The transcript's word error rate on *sample* is within *backend*'s threshold; it is printed for the CI log."""
    score, limit = wer(sample.text, transcript), MAX_WER[backend][sample.name]
    print(f"{backend} on {sample.name}: WER {score:.3f} (at most {limit}): {transcript.strip()!r}")
    assert score <= limit, (
        f"{backend} on {sample.name}: WER {score:.2f} is above {limit}\n"
        f"  reference:  {sample.text}\n  transcript: {transcript.strip()}"
    )
