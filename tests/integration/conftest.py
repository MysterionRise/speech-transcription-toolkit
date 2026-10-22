"""Integration tests: real models transcribe the speech samples in tests/data.

Every test in this folder is marked ``integration``, which plain ``pytest`` deselects. Run them with::

    pip install -e ".[all]" openai     # or just the extras to test
    pytest -m integration

A test whose packages aren't installed is skipped; with ``INTEGRATION_REQUIRE_EXTRAS=1``, as in CI, it fails
instead. The diarization tests also need ``HF_TOKEN``. Models download on first use. The commands run as
``python -m speech_toolkit`` on this checkout, with the interpreter that runs pytest, through
tests/integration/instrument.py, which records the device, timings and memory of the models they load.

``INTEGRATION_DEVICE`` (``auto``, ``cpu`` or ``cuda``) picks the device and ``INTEGRATION_TIER=full`` adds the larger
models; see devices.py and docs/manual-testing.md. At the end, the session writes ``integration-report.md`` to
pytest's base temp folder and prints it.
"""

from __future__ import annotations

import pathlib
import subprocess
from collections import Counter
from typing import Callable, Dict, List

import pytest

from tests.integration.devices import check_settings, device_args, environment_lines
from tests.integration.helpers import MANIFEST, EventLog, Sample, build_long_clip, load_samples, run_module
from tests.integration.report import FILE_NAME, REPORT

HERE = pathlib.Path(__file__).resolve().parent


@pytest.hookimpl(tryfirst=True)  # mark before -m deselects
def pytest_collection_modifyitems(items: List[pytest.Item]) -> None:
    for item in items:
        if HERE in item.path.resolve().parents:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def integration_test(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Every test here: check INTEGRATION_DEVICE and INTEGRATION_TIER, and count the test's runs in the report."""
    check_settings()
    REPORT.start_test(request.node.nodeid, tmp_path_factory.getbasetemp())


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Write integration-report.md to the base temp folder and print it, if any integration test ran."""
    if REPORT.started is None or REPORT.folder is None:
        return
    stats = terminalreporter.stats
    outcomes = {name: len(stats.get(key, [])) for name, key in OUTCOMES if stats.get(key)}
    skipped = Counter(skip_reason(report) for report in stats.get("skipped", []))
    failed = list(dict.fromkeys(report.nodeid for key in ("failed", "error") for report in stats.get(key, [])))
    text = REPORT.markdown(environment_lines(), outcomes, dict(skipped), failed)
    path = REPORT.folder / FILE_NAME
    path.write_text(text, encoding="utf-8")
    terminalreporter.write_sep("=", "integration report")
    for line in [*text.splitlines(), f"Written to {path}"]:
        # ASCII only: on Windows, output redirected to a file uses the ANSI code page, which can't encode everything
        terminalreporter.write_line(line.encode("ascii", "replace").decode("ascii"))


OUTCOMES = (("passed", "passed"), ("failed", "failed"), ("errors", "error"), ("skipped", "skipped"))


def skip_reason(report: pytest.TestReport) -> str:
    """Why a test was skipped, as pytest shows it."""
    longrepr = report.longrepr
    reason = str(longrepr[2]) if isinstance(longrepr, tuple) else str(longrepr)
    return reason.removeprefix("Skipped: ")


@pytest.fixture(scope="session")
def samples() -> Dict[str, Sample]:
    """The speech samples by name ("speech", "dialogue"), with their reference transcripts and turn times."""
    if not MANIFEST.is_file():
        pytest.skip("needs a source checkout: the sdist has no tests/data")
    return load_samples()


@pytest.fixture(scope="session")
def speech(samples: Dict[str, Sample]) -> Sample:
    """One voice saying "Hello world. This is a speech recognition test." (3.4 s)."""
    return samples["speech"]


@pytest.fixture(scope="session")
def dialogue(samples: Dict[str, Sample]) -> Sample:
    """Two voices taking turns (12 s)."""
    return samples["dialogue"]


@pytest.fixture(scope="session")
def long_clip(samples: Dict[str, Sample], tmp_path_factory: pytest.TempPathFactory) -> Sample:
    """The dialogue and the speech sample six times over, a second of silence between parts (104 s, WAV)."""
    return build_long_clip(samples, tmp_path_factory.mktemp("long-clip"))


@pytest.fixture
def transcribe(tmp_path: pathlib.Path) -> Callable[..., subprocess.CompletedProcess]:
    """Run the transcribe command in the test's temporary folder: ``transcribe(path, "-q")``; fails on an error.

    It gets ``--device`` from INTEGRATION_DEVICE; the test fails unless its models ran there, and the files it
    transcribed go in the report.
    """
    events = EventLog(tmp_path / "instrument-events.jsonl")

    def run(*args: object) -> subprocess.CompletedProcess:
        return run_module("speech_toolkit", *args, *device_args(), cwd=tmp_path, events=events)

    return run
