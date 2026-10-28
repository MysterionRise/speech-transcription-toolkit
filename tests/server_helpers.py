"""Shared by the server's tests: a stand-in for the ffmpeg decoding that transcribe-server does (#31)."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from speech_toolkit.media import SAMPLE_RATE


@dataclasses.dataclass
class Decoding:
    """One call of the fake decoder: the file, whether it existed then, and load_audio()'s keyword arguments."""

    path: pathlib.Path
    existed: bool
    options: Dict[str, Any]


@dataclasses.dataclass
class FakeDecoder:
    """Decodes any upload to *seconds* of silence, or raises *error*; records its calls."""

    seconds: float = 1.5  # as long as the fake backends' transcripts
    error: Optional[Exception] = None
    calls: List[Decoding] = dataclasses.field(default_factory=list)

    def __call__(self, source: Any, **options: Any) -> np.ndarray:
        path = pathlib.Path(source)
        self.calls.append(Decoding(path, path.is_file(), options))
        if self.error is not None:
            raise self.error
        return np.zeros(round(self.seconds * SAMPLE_RATE), np.float32)


@pytest.fixture
def decoded(monkeypatch: pytest.MonkeyPatch) -> FakeDecoder:
    """The server decodes uploads with the fake decoder, so no ffmpeg is needed and any bytes are "audio"."""
    decoder = FakeDecoder()
    monkeypatch.setattr("speech_toolkit.server.load_audio", decoder)
    return decoder
