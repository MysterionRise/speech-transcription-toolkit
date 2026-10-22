"""In-memory audio with the real ffmpeg and real models: bytes, file objects and arrays decode like the file, an M4A
with its index at the end decodes from memory too, and every backend transcribes audio it gets as samples."""

from __future__ import annotations

import io
import pathlib
import subprocess
from typing import Optional

import numpy as np
import pytest

from tests.conftest import FakeWaveformBackend
from tests.integration.helpers import assert_wer, require

MODELS = {"whisper": "tiny", "faster-whisper": "tiny"}  # the others use their default model, as in test_cli.py


class Unseekable(io.RawIOBase):
    """A binary stream that can only be read, like a pipe or an upload."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def readable(self) -> bool:
        return True

    def read(self, size: Optional[int] = -1) -> bytes:
        size = len(self._data) if size is None or size < 0 else size
        chunk, self._data = self._data[:size], self._data[size:]
        return chunk


def test_in_memory_input_decodes_like_the_file(fake_waveform_backend, speech):
    """FakeWaveformBackend's transcript is the number of samples it got and their digest."""
    from speech_toolkit import Transcriber
    from speech_toolkit.media import load_audio

    transcriber = Transcriber("fake-waveform")
    data = speech.path.read_bytes()
    from_path = transcriber.transcribe(speech.path).text  # the backend decodes the file itself

    assert from_path.startswith(f" {round(speech.duration * 16000)} samples ")
    assert transcriber.transcribe(data).text == from_path
    assert transcriber.transcribe(io.BytesIO(data)).text == from_path
    assert transcriber.transcribe(Unseekable(data)).text == from_path
    with speech.path.open("rb") as file:
        assert transcriber.transcribe(file).text == from_path
    assert transcriber.transcribe(load_audio(speech.path)).text == from_path
    assert isinstance(FakeWaveformBackend.received[0], pathlib.Path)
    assert all(isinstance(received, np.ndarray) for received in FakeWaveformBackend.received[1:])


@pytest.mark.parametrize("index_first", [False, True], ids=["index-at-end", "index-first"])
def test_m4a_from_memory(fake_waveform_backend, speech, tmp_path: pathlib.Path, index_first):
    """ffmpeg writes an M4A's index after its audio, as phones do, unless asked for "faststart"."""
    from speech_toolkit import Transcriber
    from speech_toolkit.api import _index_at_end

    m4a = tmp_path / "speech.m4a"
    command = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(speech.path), "-c:a", "aac", "-b:a", "64k"]
    subprocess.run(command + (["-movflags", "+faststart"] if index_first else []) + [str(m4a)], check=True)
    data = m4a.read_bytes()
    assert _index_at_end(data) is not index_first

    transcriber = Transcriber("fake-waveform")
    from_path = transcriber.transcribe(m4a).text

    assert transcriber.transcribe(data).text == from_path
    assert transcriber.transcribe(Unseekable(data)).text == from_path


@pytest.mark.parametrize("backend", ["whisper", "faster-whisper", "parakeet", "canary"])
def test_backends_transcribe_audio_from_memory(backend, speech):
    """Each backend gets the samples that Transcriber decoded from the file's content; the result has their length."""
    require(backend)
    from speech_toolkit import Transcriber

    transcriber = Transcriber(backend, MODELS.get(backend))
    result = transcriber.transcribe(speech.path.read_bytes(), language="en" if backend == "canary" else None)

    assert_wer(result.text, speech, backend)
    assert result.duration == pytest.approx(speech.duration, abs=0.01)
