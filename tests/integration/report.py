"""The integration report: one row per backend, model and device, written to integration-report.md and printed.

Every file a model transcribes is a Run: the instrumented commands record theirs (instrument.py), and the Python API
test records its own. The word error rate a test checks is counted against the test's run. At the end of the session,
conftest.py writes the report to pytest's base temp folder and prints it.
"""

from __future__ import annotations

import dataclasses
import pathlib
import time
from typing import Dict, List, Optional, Sequence, Tuple

MIB = 1024 * 1024
FILE_NAME = "integration-report.md"

COLUMNS = ("Backend", "Model", "Device", "Precision", "Files", "Audio (s)", "Load (s)", "Wall (s)", "RTF", "WER")
NOTES = (
    "Wall: seconds spent transcribing, decoding the audio included. RTF: wall time / audio length (below 1 is faster "
    "than real time). Load: the fastest model load (the first one includes the download). WER: over every transcript "
    "the tests checked. Peak GPU: torch's own count; with ~, the whole GPU's use above the level before the model "
    "loaded, sampled with nvidia-smi (CTranslate2, which faster-whisper runs on, keeps no count)."
)


@dataclasses.dataclass
class Run:
    """One file transcribed by a model."""

    test: str  # the test's node id
    backend: str
    model: str
    devices: Dict[str, str]  # where each model was: {"model": "cuda", "diarization": "cuda"}
    precision: str
    load_seconds: float
    audio_seconds: Optional[float]
    seconds: float
    torch_peak_bytes: Optional[int] = None
    sampled_peak_bytes: Optional[int] = None
    errors: Optional[int] = None  # word errors and reference words, once the test has checked the transcript
    words: Optional[int] = None

    @property
    def key(self) -> Tuple[str, str, str]:
        """The report row the run goes in: backend (with pyannote when diarizing), model and device."""
        device = self.devices.get("model", "unknown")
        if "diarization" not in self.devices:
            return self.backend, self.model, device
        if self.devices["diarization"] != device:
            device += f" (pyannote on {self.devices['diarization']})"
        return f"{self.backend} + pyannote", self.model, device

    @property
    def peak(self) -> Optional[Tuple[int, bool]]:
        """The peak GPU memory in bytes, and whether it was sampled with nvidia-smi rather than counted by torch."""
        if "cuda" not in self.devices.values():
            return None
        # torch counts its own models exactly; CTranslate2 keeps no count, so nvidia-smi's samples stand in for it.
        if self.torch_peak_bytes and self.backend != "faster-whisper":
            return self.torch_peak_bytes, False
        if self.sampled_peak_bytes is not None:
            return self.sampled_peak_bytes, True
        if self.torch_peak_bytes:
            return self.torch_peak_bytes, False
        return None


@dataclasses.dataclass
class Row:
    """The runs of one backend, model and device, added up."""

    backend: str
    model: str
    device: str
    precision: str
    files: int = 0
    audio: float = 0.0  # seconds of audio, over the runs whose length is known
    timed: float = 0.0  # the seconds those runs took
    seconds: float = 0.0  # the seconds all runs took
    load: Optional[float] = None
    errors: int = 0
    words: int = 0
    peak: Optional[int] = None
    sampled: bool = False

    def add(self, run: Run) -> None:
        self.files += 1
        self.seconds += run.seconds
        if run.audio_seconds:
            self.audio += run.audio_seconds
            self.timed += run.seconds
        self.load = run.load_seconds if self.load is None else min(self.load, run.load_seconds)
        if run.errors is not None and run.words:
            self.errors += run.errors
            self.words += run.words
        if run.peak is not None and (self.peak is None or run.peak[0] > self.peak):
            self.peak, self.sampled = run.peak

    def cells(self) -> List[str]:
        peak = "-" if self.peak is None else f"{'~' if self.sampled else ''}{self.peak / MIB:.0f}"
        return [
            self.backend,
            self.model,
            self.device,
            self.precision or "-",
            str(self.files),
            f"{self.audio:.1f}",
            "-" if self.load is None else f"{self.load:.1f}",
            f"{self.seconds:.1f}",
            f"{self.timed / self.audio:.3f}" if self.audio else "-",
            f"{self.errors / self.words:.3f}" if self.words else "-",
            peak,
        ]


class Report:
    """The runs of this session, by test."""

    def __init__(self) -> None:
        self.runs: List[Run] = []
        self.test = ""  # the test running now
        self.folder: Optional[pathlib.Path] = None  # where the report goes: pytest's base temp folder
        self.started: Optional[float] = None  # when the first integration test started (time.monotonic)
        self._pending_wer: Optional[Tuple[str, int, int]] = None

    def start_test(self, test: str, folder: pathlib.Path) -> None:
        self.test, self.folder, self._pending_wer = test, folder, None
        if self.started is None:
            self.started = time.monotonic()

    def add(self, run: Run) -> None:
        """Add a run; a WER the test checked before the run was recorded (as with the server) goes with it."""
        if self._pending_wer and self._pending_wer[0] == run.test and run.errors is None:
            _, run.errors, run.words = self._pending_wer
            self._pending_wer = None
        self.runs.append(run)

    def note_wer(self, errors: int, words: int) -> None:
        """Count a transcript's word errors against the current test's first run without a WER yet.

        If the test has no run yet, they wait for its next one. A run already counted is left alone: a test that
        checks one run's output twice (subtitles and JSON) counts it once.
        """
        runs = [run for run in self.runs if run.test == self.test]
        if not runs:
            self._pending_wer = (self.test, errors, words)
            return
        unscored = next((run for run in runs if run.errors is None), None)
        if unscored is not None:
            unscored.errors, unscored.words = errors, words

    def rows(self) -> List[Row]:
        rows: Dict[Tuple[str, str, str], Row] = {}
        for run in self.runs:
            if run.key not in rows:
                rows[run.key] = Row(*run.key, precision=run.precision)
            rows[run.key].add(run)
        return [rows[key] for key in sorted(rows)]

    def markdown(
        self,
        environment: Sequence[str],
        outcomes: Dict[str, int],
        skipped: Dict[str, int],
        failed: Sequence[str],
    ) -> str:
        """The report: *environment* (Markdown list items), the test *outcomes* ({"passed": 3, ...}), the table, and
        the reasons tests were *skipped* (with how many) and the tests that *failed*."""
        took = f" in {time.monotonic() - self.started:.0f} s" if self.started is not None else ""
        counts = ", ".join(f"{count} {outcome}" for outcome, count in outcomes.items())
        lines = ["## Integration report", "", *environment, f"- Tests: {counts or 'none'}{took}", ""]
        rows = self.rows()
        if rows:
            header = [*COLUMNS, "Peak GPU (MiB)"]
            lines += ["| " + " | ".join(header) + " |", "|" + "---|" * 4 + "--:|" * (len(header) - 4)]
            lines += ["| " + " | ".join(row.cells()) + " |" for row in rows]
            lines += ["", NOTES]
        else:
            lines.append("No model ran.")
        if skipped:
            lines += ["", "### Skipped", ""]
            lines += [f"- {reason} ({count} test{'s' * (count != 1)})" for reason, count in skipped.items()]
        if failed:
            lines += ["", "### Failed", ""] + [f"- `{test}`" for test in failed]
        return "\n".join(lines) + "\n"


REPORT = Report()
