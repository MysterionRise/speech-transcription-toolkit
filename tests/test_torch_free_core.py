"""A plain install (no extras) is torch-free: faster-whisper is the default backend and a core dependency, openai-whisper
comes with the ``whisper`` extra, and ogg2wav needs no pydub."""

from __future__ import annotations

import pathlib
import re
import sys
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from speech_toolkit import Transcriber, cli
from speech_toolkit.backends import DEFAULT_BACKEND

PYPROJECT = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"

# Packages that pull in PyTorch, or that the core install dropped.
NOT_IN_CORE = {"torch", "torchaudio", "openai-whisper", "transformers", "pyannote.audio", "pydub", "audioop-lts"}


def requirement_names(requirements: List[str]) -> List[str]:
    """The distribution names of PEP 508 *requirements*, lower-cased: ``"av>=17,<19"`` -> ``"av"``."""
    return [re.match(r"[A-Za-z0-9._-]+", requirement).group(0).lower() for requirement in requirements]


@pytest.fixture(scope="module")
def project() -> Dict:
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


class TestPackageMetadata:
    def test_core_dependencies_have_no_torch(self, project):
        names = requirement_names(project["dependencies"])

        assert "faster-whisper" in names
        assert not NOT_IN_CORE & set(names)

    def test_whisper_extra(self, project):
        extras = project["optional-dependencies"]

        assert requirement_names(extras["whisper"]) == ["openai-whisper"]
        assert "faster-whisper" in extras  # kept, so `pip install "...[faster-whisper]"` still works
        assert re.search(r"\[(.+,)?whisper(,.+)?\]", extras["all"][0])

    @pytest.mark.parametrize("extra", ["voxtral", "nvidia"])
    def test_pytorch_extras_declare_torch(self, project, extra):
        """These no longer get torch through openai-whisper."""
        assert "torch" in requirement_names(project["optional-dependencies"][extra])


class TestDefaultBackend:
    def test_list_backends_marks_faster_whisper(self, capsys):
        cli.show_backends()
        lines = capsys.readouterr().out.splitlines()

        assert [line.strip() for line in lines if line.endswith("(default)")] == ["faster-whisper (default)"]

    def test_help_names_the_default(self, capsys, monkeypatch):
        monkeypatch.setenv("COLUMNS", "500")
        monkeypatch.setenv("PYTHON_COLORS", "0")
        with pytest.raises(SystemExit):
            cli.parse_args(["--help"])

        assert "Transcription backend (default: faster-whisper)." in capsys.readouterr().out

    def test_transcriber_loads_faster_whisper_turbo(self):
        """Transcriber() needs faster-whisper only: openai-whisper and torch are never imported."""
        faster_whisper, ctranslate2 = MagicMock(), MagicMock()
        ctranslate2.get_cuda_device_count.return_value = 0
        modules = {"faster_whisper": faster_whisper, "ctranslate2": ctranslate2, "whisper": None, "torch": None}
        with patch.dict(sys.modules, modules):
            transcriber = Transcriber()

        assert (transcriber.backend_name, transcriber.model_name) == (DEFAULT_BACKEND, "turbo")
        faster_whisper.WhisperModel.assert_called_once_with("turbo", device="cpu", compute_type="int8")
