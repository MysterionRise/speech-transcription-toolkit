"""Integration tests: real models transcribe the speech samples in tests/data.

Every test in this folder is marked ``integration``, which plain ``pytest`` deselects. Run them with::

    pip install -e ".[all]" openai     # or just the extras to test
    pytest -m integration

A test whose packages aren't installed is skipped; with ``INTEGRATION_REQUIRE_EXTRAS=1``, as in CI, it fails
instead. The diarization test also needs ``HF_TOKEN``. Models download on first use. The commands run as
``python -m speech_toolkit`` on this checkout, with the interpreter that runs pytest.
"""

from __future__ import annotations

import pathlib
import subprocess
from typing import Callable, Dict, List

import pytest

from tests.integration.helpers import MANIFEST, Sample, load_samples, run_module

HERE = pathlib.Path(__file__).resolve().parent


@pytest.hookimpl(tryfirst=True)  # mark before -m deselects
def pytest_collection_modifyitems(items: List[pytest.Item]) -> None:
    for item in items:
        if HERE in item.path.resolve().parents:
            item.add_marker(pytest.mark.integration)


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


@pytest.fixture
def transcribe(tmp_path: pathlib.Path) -> Callable[..., subprocess.CompletedProcess]:
    """Run the transcribe command in the test's temporary folder: ``transcribe(path, "-q")``; fails on an error."""

    def run(*args: object) -> subprocess.CompletedProcess:
        return run_module("speech_toolkit", *args, cwd=tmp_path)

    return run
