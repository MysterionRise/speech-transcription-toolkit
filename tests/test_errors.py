"""Tests for the exception hierarchy (speech_toolkit.errors): the raise sites, the warnings, and per-file CLI errors.

Heavy libraries are replaced in ``sys.modules`` (None makes an import fail), as in the backend tests.
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any, Callable, Dict, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import speech_toolkit
from speech_toolkit import (
    AudioDecodeError,
    BackendNotFoundError,
    BackendUnavailableError,
    DiarizationError,
    ModelLoadError,
    ModelNotFoundError,
    SpeechToolkitError,
    SpeechToolkitWarning,
    Transcriber,
    UnsupportedOptionError,
    backends,
    cli,
    errors,
)
from speech_toolkit.backends import get_backend, get_backend_class
from speech_toolkit.backends.faster_whisper_backend import FasterWhisperBackend
from speech_toolkit.backends.nvidia_backend import CanaryBackend, ParakeetBackend
from speech_toolkit.backends.voxtral_backend import VoxtralBackend
from speech_toolkit.backends.whisper_backend import WhisperBackend
from speech_toolkit.diarization import load_diarization_pipeline
from tests.conftest import FakeBackend

AUDIO = pathlib.Path("talk.wav")  # never read: every raise site below fails before that
BACKENDS = [WhisperBackend, FasterWhisperBackend, VoxtralBackend, ParakeetBackend, CanaryBackend]
TRANSFORMERS_BACKENDS = [VoxtralBackend, ParakeetBackend, CanaryBackend]
BUILTINS = [
    (BackendNotFoundError, ValueError),
    (BackendUnavailableError, ImportError),
    (ModelNotFoundError, ValueError),
    (ModelLoadError, RuntimeError),
    (AudioDecodeError, RuntimeError),
    (UnsupportedOptionError, ValueError),
    (DiarizationError, RuntimeError),
]


class TestHierarchy:
    @pytest.mark.parametrize("error, builtin", BUILTINS, ids=[error.__name__ for error, _ in BUILTINS])
    def test_each_error_is_also_the_builtin_it_replaces(self, error, builtin):
        assert issubclass(error, SpeechToolkitError)
        with pytest.raises(builtin, match="^boom$"):
            raise error("boom")

    def test_exported_from_the_package(self):
        for name in errors.__all__:
            assert name in speech_toolkit.__all__
            assert getattr(speech_toolkit, name) is getattr(errors, name)
        assert issubclass(SpeechToolkitWarning, UserWarning)


###############################################################################
# Every raise site raises a SpeechToolkitError subclass
###############################################################################


def hidden(*modules: str) -> Dict[str, Any]:
    """sys.modules entries that make importing *modules* fail."""
    return dict.fromkeys(modules)


def torch_mock() -> MagicMock:
    torch = MagicMock()
    torch.cuda.is_available.return_value = False
    return torch


def failing_faster_whisper() -> Dict[str, Any]:
    """faster_whisper and ctranslate2 whose model download fails."""
    faster_whisper, ctranslate2 = MagicMock(), MagicMock()
    ctranslate2.get_cuda_device_count.return_value = 0
    faster_whisper.WhisperModel.side_effect = OSError("network down")
    return {"faster_whisper": faster_whisper, "ctranslate2": ctranslate2}


def failing_transformers() -> Dict[str, Any]:
    """torch and transformers whose model download fails."""
    transformers = MagicMock()
    transformers.AutoProcessor.from_pretrained.side_effect = OSError("401 gated repo")
    return {"torch": torch_mock(), "transformers": transformers}


def load(
    backend_class: Any, modules: Callable[[], Dict[str, Any]] = dict, model: Optional[str] = None
) -> Callable[[], None]:
    """A call that loads *model* (default: the backend's default) with ``modules()`` patched into sys.modules."""

    def call() -> None:
        with patch.dict(sys.modules, modules()):
            backend_class().load_model(model or backend_class.default_model())

    return call


def voxtral_translate() -> None:
    backend = VoxtralBackend()
    backend._model = backend._processor = object()  # loaded, as far as transcribe() can tell
    backend.transcribe(AUDIO, task="translate")


def pyannote_missing() -> None:
    with patch.dict(sys.modules, {"torch": torch_mock(), **hidden("pyannote.audio")}):
        load_diarization_pipeline()


def diarization_model_gated() -> None:
    pyannote_audio = MagicMock()
    pyannote_audio.Pipeline.from_pretrained.return_value = None  # pyannote's answer when the download is refused
    with patch.dict(sys.modules, {"torch": torch_mock(), "pyannote": MagicMock(), "pyannote.audio": pyannote_audio}):
        load_diarization_pipeline()


RAISE_SITES = [
    # Unknown backend: backends/__init__.py, reached by Transcriber too.
    pytest.param(lambda: get_backend("nope"), BackendNotFoundError, id="get_backend"),
    pytest.param(lambda: get_backend_class("nope"), BackendNotFoundError, id="get_backend_class"),
    pytest.param(lambda: Transcriber("nope"), BackendNotFoundError, id="Transcriber"),
    # load_model(): unknown model, missing extra, failed download.
    *(pytest.param(load(cls, model="huge"), ModelNotFoundError, id=f"{cls.name}-unknown-model") for cls in BACKENDS),
    pytest.param(load(WhisperBackend, lambda: hidden("whisper")), BackendUnavailableError, id="whisper-missing"),
    pytest.param(
        load(FasterWhisperBackend, lambda: hidden("faster_whisper", "ctranslate2")),
        BackendUnavailableError,
        id="faster-whisper-missing",
    ),
    *(
        pytest.param(
            load(cls, lambda: hidden("torch", "transformers")), BackendUnavailableError, id=f"{cls.name}-missing"
        )
        for cls in TRANSFORMERS_BACKENDS
    ),
    pytest.param(
        load(ParakeetBackend, lambda: {"torch": torch_mock(), "transformers": MagicMock(spec=[])}),
        BackendUnavailableError,
        id="parakeet-transformers-too-old",
    ),
    pytest.param(load(FasterWhisperBackend, failing_faster_whisper), ModelLoadError, id="faster-whisper-load-fails"),
    *(
        pytest.param(load(cls, failing_transformers), ModelLoadError, id=f"{cls.name}-load-fails")
        for cls in TRANSFORMERS_BACKENDS
    ),
    # transcribe(): no model loaded, a task or option that can't be honoured.
    *(
        pytest.param(lambda cls=cls: cls().transcribe(AUDIO), ModelLoadError, id=f"{cls.name}-not-loaded")
        for cls in BACKENDS
    ),
    pytest.param(voxtral_translate, UnsupportedOptionError, id="voxtral-translate"),
    pytest.param(
        lambda: ParakeetBackend().transcribe(AUDIO, task="translate"), UnsupportedOptionError, id="parakeet-translate"
    ),
    pytest.param(
        lambda: Transcriber("fake").transcribe(AUDIO, num_speakers=2),
        UnsupportedOptionError,
        id="speaker-hints-without-diarize",
    ),
    # diarization.py: the pipeline loader.
    pytest.param(pyannote_missing, BackendUnavailableError, id="pyannote-missing"),
    pytest.param(diarization_model_gated, DiarizationError, id="diarization-model-gated"),
]


@pytest.mark.parametrize("call, error", RAISE_SITES)
def test_raise_site_raises_a_speech_toolkit_error(fake_backend, call, error):
    with patch.dict(os.environ), pytest.raises(error) as caught:  # the diarization loader sets an environment variable
        call()

    assert type(caught.value) is error
    assert isinstance(caught.value, SpeechToolkitError)


###############################################################################
# Warnings
###############################################################################


def canary_without_language(audio: pathlib.Path) -> None:
    backend = CanaryBackend()
    backend._model = backend._processor = object()  # loaded, as far as transcribe() can tell
    with patch("speech_toolkit.backends.nvidia_backend.load_audio", return_value=np.zeros(0, dtype=np.float32)):
        backend.transcribe(audio, verbose=False)


WARNING_SITES = [
    pytest.param(
        lambda audio: Transcriber("fake").transcribe(audio, vad=True),
        "the fake backend doesn't support vad",
        id="unsupported-option",
    ),
    pytest.param(
        lambda audio: Transcriber("fake", "turbo").transcribe(audio, task="translate"),
        "'turbo' isn't trained for translation",
        id="turbo-translation",
    ),
    pytest.param(canary_without_language, "Canary can't detect the language", id="canary-language"),
]


@pytest.mark.parametrize("call, message", WARNING_SITES)
def test_library_warnings_are_speech_toolkit_warnings(fake_backend, tmp_path, call, message):
    audio = tmp_path / "talk.wav"
    audio.touch()

    with pytest.warns(SpeechToolkitWarning, match=message) as caught:
        call(audio)

    assert [warning.category for warning in caught] == [SpeechToolkitWarning]


# Only SpeechToolkitWarning is let through: pyproject.toml turns any other warning from speech_toolkit into an error.
@pytest.mark.filterwarnings("default::speech_toolkit.SpeechToolkitWarning")
def test_cli_prints_a_library_warning_as_one_line(fake_backend, tmp_path, capsys):
    for name in ("a.wav", "b.wav"):
        (tmp_path / name).touch()

    cli.main([str(tmp_path), "-b", "fake", "-q", "--vad"])

    err = capsys.readouterr().err
    assert [line for line in err.splitlines() if "Warning" in line] == [
        "Warning: the fake backend doesn't support vad; ignoring it."
    ]


###############################################################################
# CLI: a failing file fails alone
###############################################################################


class FailingBackend(FakeBackend):
    """FakeBackend that raises ``errors[stem]`` for the files named in it."""

    name = "failing"
    errors: Dict[str, BaseException] = {}

    def transcribe(self, audio_path, language=None, task="transcribe", verbose=True):
        if audio_path.stem in self.errors:
            raise self.errors[audio_path.stem]
        return super().transcribe(audio_path, language, task, verbose)


@pytest.fixture
def failing(fake_backend, monkeypatch):
    """Register FailingBackend as 'failing'; returns a function that sets which files fail and how."""
    monkeypatch.setitem(backends._BACKENDS, "failing", FailingBackend)
    return lambda **errors: monkeypatch.setattr(FailingBackend, "errors", errors)


def error_lines(capsys) -> list:
    return [line for line in capsys.readouterr().err.splitlines() if line.startswith("Error:")]


class TestCliFileErrors:
    @pytest.mark.parametrize(
        "error, shown",
        [
            pytest.param(KeyError("segments"), "KeyError: 'segments'", id="KeyError"),
            pytest.param(AssertionError(), "AssertionError", id="no-message"),
            pytest.param(ModelLoadError("CUDA out of memory"), "CUDA out of memory", id="library-error"),
            pytest.param(SpeechToolkitError("no speech found"), "no speech found", id="base-class"),
        ],
    )
    def test_a_failing_file_does_not_stop_the_batch(self, failing, tmp_path, capsys, error, shown):
        for name in ("a.wav", "bad.wav", "c.wav"):
            (tmp_path / name).touch()
        failing(bad=error)

        with pytest.raises(SystemExit, match="1 of 3 files failed"):
            cli.main([str(tmp_path), "-b", "failing", "-q"])

        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "Hello from a.\n"
        assert (tmp_path / "c.txt").read_text(encoding="utf-8") == "Hello from c.\n"  # transcribed after the failure
        assert not (tmp_path / "bad.txt").exists()
        assert error_lines(capsys) == [f"Error: {tmp_path / 'bad.wav'}: {shown}"]

    def test_keyboard_interrupt_stops_the_batch(self, failing, tmp_path):
        for name in ("a.wav", "b.wav"):
            (tmp_path / name).touch()
        failing(a=KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            cli.main([str(tmp_path), "-b", "failing", "-q"])

        assert not (tmp_path / "b.txt").exists()

    def test_single_file_with_an_unexpected_error(self, failing, tmp_path, capsys):
        audio = tmp_path / "bad.wav"
        audio.touch()
        failing(bad=KeyError("segments"))

        with pytest.raises(SystemExit) as exc_info:
            cli.main([str(audio), "-b", "failing", "-q"])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert f"Error: {audio}: KeyError: 'segments'" in captured.err.splitlines()

    def test_an_output_that_cannot_be_written_fails_its_file_only(self, fake_backend, tmp_path, capsys):
        src, out = tmp_path / "in", tmp_path / "out"
        src.mkdir()
        for name in ("a.wav", "b.wav"):
            (src / name).touch()
        (out / "a.txt").mkdir(parents=True)  # a folder where a.txt should go, so writing it fails

        with pytest.raises(SystemExit, match="1 of 2 files failed"):
            cli.main([str(src), "--outdir", str(out), "-b", "fake", "-q"])

        assert (out / "b.txt").read_text(encoding="utf-8") == "Hello from b.\n"
        lines = error_lines(capsys)
        assert len(lines) == 1 and lines[0].startswith(f"Error: {src / 'a.wav'}: ")

    def test_model_load_error_is_one_line(self, fake_backend, monkeypatch, tmp_path):
        audio = tmp_path / "talk.wav"
        audio.touch()

        def load_model(self, model_name, device=None):
            raise SpeechToolkitError("no GPU memory left")

        monkeypatch.setattr(FakeBackend, "load_model", load_model)

        with pytest.raises(SystemExit, match="^Error: no GPU memory left$"):
            cli.main([str(audio), "-b", "fake", "-q"])
