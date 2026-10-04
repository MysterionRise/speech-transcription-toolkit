"""Tests for the Python API (speech_toolkit.Transcriber and speech_toolkit.transcribe)."""

from __future__ import annotations

import os
import pathlib

import pytest

import speech_toolkit
from speech_toolkit import Transcriber, TranscriptionResult, transcribe
from speech_toolkit.api import configure_hf_token


@pytest.fixture
def audio(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "talk.mp3"
    path.touch()
    return path


class TestPublicApi:
    def test_exports(self):
        for name in ("Transcriber", "transcribe", "TranscriptionResult", "list_backends", "register_backend"):
            assert name in speech_toolkit.__all__
            assert hasattr(speech_toolkit, name)

    def test_version_is_set(self):
        assert speech_toolkit.__version__.count(".") == 2


class TestTranscriber:
    def test_loads_model_once_for_many_files(self, fake_backend, tmp_path):
        transcriber = Transcriber("fake", "small")
        for name in ("a.wav", "b.wav"):
            (tmp_path / name).touch()
            result = transcriber.transcribe(tmp_path / name)
            assert isinstance(result, TranscriptionResult)

        assert fake_backend.loaded == 1
        assert result.text == " Hello from b."
        assert result.speaker_segments is None

    def test_default_model_and_str_paths(self, fake_backend, audio):
        transcriber = Transcriber("fake")

        assert transcriber.model_name == "small"  # the backend's default (first listed) model
        assert transcriber.backend.model_name == "small"
        assert transcriber.transcribe(str(audio)).render() == "Hello from talk."

    def test_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            Transcriber("nope")

    def test_unknown_model(self, fake_backend):
        with pytest.raises(ValueError, match="Unknown fake model"):
            Transcriber("fake", "huge")

    def test_backend_errors_propagate(self, fake_backend, tmp_path):
        bad = tmp_path / "bad.wav"
        bad.touch()

        with pytest.raises(RuntimeError, match="Failed to load audio"):
            Transcriber("fake").transcribe(bad)

    def test_diarization_labels_speakers(self, fake_backend, audio, monkeypatch):
        calls = {}
        monkeypatch.setattr("speech_toolkit.api.load_diarization_pipeline", lambda device: "pipeline")

        def fake_diarize(path, pipeline, **hints):
            calls.update(path=path, pipeline=pipeline, **hints)
            return [(0.0, 2.0, "SPEAKER_01")]

        monkeypatch.setattr("speech_toolkit.api.diarize_audio", fake_diarize)

        transcriber = Transcriber("fake", diarize=True)
        result = transcriber.transcribe(audio, num_speakers=2)

        assert transcriber.diarize
        assert calls == {
            "path": audio,
            "pipeline": "pipeline",
            "num_speakers": 2,
            "min_speakers": None,
            "max_speakers": None,
        }
        assert result.speaker_segments[0]["speaker"] == "SPEAKER_01"
        assert result.render("txt") == "[SPEAKER_01] Hello from talk."

    def test_diarization_loads_before_the_model(self, fake_backend, monkeypatch):
        """A missing pyannote or HF token fails before the (possibly large) model download."""

        def missing(device):
            raise ImportError("speaker diarization needs pyannote.audio")

        monkeypatch.setattr("speech_toolkit.api.load_diarization_pipeline", missing)

        with pytest.raises(ImportError, match="pyannote"):
            Transcriber("fake", diarize=True)
        assert fake_backend.loaded == 0

    def test_speaker_hints_need_diarization(self, fake_backend, audio):
        with pytest.raises(ValueError, match="need diarize=True"):
            Transcriber("fake").transcribe(audio, num_speakers=2)

    def test_translate_with_turbo_warns(self, fake_backend, audio):
        transcriber = Transcriber("fake", "turbo")

        with pytest.warns(UserWarning, match="isn't trained for translation"):
            transcriber.transcribe(audio, task="translate")

    def test_hf_token_is_exported(self, fake_backend, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        Transcriber("fake", hf_token="hf_test")
        assert os.environ["HF_TOKEN"] == "hf_test"


class TestTranscribeFunction:
    def test_one_shot(self, fake_backend, audio, tmp_path):
        result = transcribe(audio, backend="fake", model="small", language="en")

        assert result.language == "en"
        result.save(tmp_path / "talk.srt")
        assert (tmp_path / "talk.srt").read_text(encoding="utf-8").startswith("1\n00:00:00,000 --> 00:00:01,500\n")


class TestConfigureHfToken:
    """Test Hugging Face token handling."""

    def test_explicit_token_wins(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "env_token")
        configure_hf_token("cli_token")
        assert os.environ["HF_TOKEN"] == "cli_token"

    def test_legacy_env_var_is_exported_as_hf_token(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "legacy_token")
        configure_hf_token(None)
        assert os.environ["HF_TOKEN"] == "legacy_token"

    def test_no_token_leaves_env_alone(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
        configure_hf_token(None)
        assert "HF_TOKEN" not in os.environ
