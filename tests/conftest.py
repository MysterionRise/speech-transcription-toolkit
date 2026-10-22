"""Shared fixtures: a fake backend so CLI and API tests run without any model library."""

from __future__ import annotations

import hashlib
import pathlib

import pytest

from speech_toolkit.backends import TranscriptionBackend, TranscriptionResult


class FakeBackend(TranscriptionBackend):
    """Backend that 'transcribes' instantly and, like Whisper, prints to stdout while working.

    It translates and reports the language, but takes none of the optional transcribe() keywords.
    """

    name = "fake"
    description = "Fake backend for tests"
    capabilities = frozenset({"translate", "language_detection"})
    loaded = 0

    @classmethod
    def available_models(cls):
        return ["small", "turbo"]

    def load_model(self, model_name, device=None):
        if model_name not in self.available_models():
            raise ValueError(f"Unknown fake model: {model_name}")
        FakeBackend.loaded += 1
        self._model = object()
        self._model_name = model_name

    def transcribe(self, audio_path, language=None, task="transcribe", verbose=True):
        print("library noise on stdout")
        if audio_path.name.startswith("bad"):
            raise RuntimeError("Failed to load audio")
        return TranscriptionResult(
            text=f" Hello from {audio_path.stem}.",
            segments=[{"start": 0.0, "end": 1.5, "text": f" Hello from {audio_path.stem}."}],
            language="en",
        )


class FakeWordsBackend(FakeBackend):
    """FakeBackend that also takes the optional transcribe() keywords and records what it got."""

    name = "fake-words"
    capabilities = frozenset({"translate", "language_detection", "prompt", "vad", "word_timestamps"})
    calls: list = []
    tasks: list = []

    def transcribe(self, audio_path, language=None, task="transcribe", verbose=True, **options):
        FakeWordsBackend.calls.append(options)
        FakeWordsBackend.tasks.append(task)
        segment = {"start": 0.0, "end": 1.5, "text": f" Hello from {audio_path.stem}."}
        if options.get("word_timestamps"):
            segment["words"] = [
                {"word": " Hello", "start": 0.0, "end": 0.5},
                {"word": " from", "start": 0.5, "end": 1.0},
                {"word": f" {audio_path.stem}.", "start": 1.0, "end": 1.5},
            ]
        return TranscriptionResult(text=segment["text"], segments=[segment], language="en")


class FakeWaveformBackend(FakeBackend):
    """FakeBackend with the "waveform" capability: it takes samples as they are and, like Parakeet, decodes a path
    with load_audio() itself. Its transcript names the samples it heard (their number and a digest), and it records
    what it was given in ``received``.
    """

    name = "fake-waveform"
    capabilities = frozenset({"language_detection", "word_timestamps", "waveform"})
    received: list = []

    def transcribe(self, audio_path, language=None, task="transcribe", verbose=True, **options):
        from speech_toolkit.media import SAMPLE_RATE, load_audio

        FakeWaveformBackend.received.append(audio_path)
        samples = load_audio(audio_path) if isinstance(audio_path, pathlib.Path) else audio_path
        text = f" {len(samples)} samples {hashlib.sha256(samples.tobytes()).hexdigest()[:16]}."
        segment = {"id": 0, "start": 0.0, "end": len(samples) / SAMPLE_RATE, "text": text}
        if options.get("word_timestamps"):
            segment["words"] = [{"word": text, "start": 0.0, "end": segment["end"]}]
        return TranscriptionResult(text=text, segments=[segment], language="en")


@pytest.fixture
def fake_waveform_backend(monkeypatch):
    """Register FakeWaveformBackend as 'fake-waveform' (and FakeBackend as 'fake') for the duration of a test."""
    from speech_toolkit.backends import _BACKENDS

    monkeypatch.setattr(
        "speech_toolkit.backends._BACKENDS", {**_BACKENDS, "fake": FakeBackend, "fake-waveform": FakeWaveformBackend}
    )
    FakeBackend.loaded = 0
    FakeWaveformBackend.received = []
    return FakeWaveformBackend


@pytest.fixture
def fake_backend(monkeypatch):
    """Register FakeBackend as 'fake' (and FakeWordsBackend as 'fake-words') for the duration of a test."""
    from speech_toolkit.backends import _BACKENDS

    monkeypatch.setattr(
        "speech_toolkit.backends._BACKENDS", {**_BACKENDS, "fake": FakeBackend, "fake-words": FakeWordsBackend}
    )
    FakeBackend.loaded = 0
    FakeWordsBackend.calls = []
    FakeWordsBackend.tasks = []
    return FakeBackend
