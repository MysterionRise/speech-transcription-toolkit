"""Offline speech-to-text with Whisper, faster-whisper, Voxtral, Parakeet or Canary, optionally labelling speakers.

    from speech_toolkit import transcribe

    result = transcribe("talk.mp3", backend="faster-whisper", model="small")
    print(result.text)
    result.save("talk.srt")

Heavy libraries (torch, whisper, transformers, pyannote) are imported only when a model loads.

The library doesn't print. It logs to the ``speech_toolkit`` logger, which has a ``NullHandler``, so its messages show
only once you configure logging (``logging.basicConfig(level=logging.INFO)``); ``Transcriber.transcribe()`` reports
progress to a ``progress=`` callback, and with ``verbose=True`` whisper and faster-whisper show their progress bar.
"""

from __future__ import annotations

import logging

from .api import Transcriber, transcribe
from .backends import ProgressEvent, TranscriptionBackend, TranscriptionResult, list_backends, register_backend
from .errors import (
    AudioDecodeError,
    BackendNotFoundError,
    BackendUnavailableError,
    DiarizationError,
    ModelLoadError,
    ModelNotFoundError,
    SpeechToolkitError,
    SpeechToolkitWarning,
    UnsupportedOptionError,
)
from .types import Segment, Word

# A library leaves showing its messages to the application: without this, Python would print warnings and errors.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.5.0"

__all__ = [
    "Transcriber",
    "transcribe",
    "TranscriptionBackend",
    "TranscriptionResult",
    "ProgressEvent",
    "Segment",
    "Word",
    "list_backends",
    "register_backend",
    "SpeechToolkitError",
    "BackendNotFoundError",
    "BackendUnavailableError",
    "ModelNotFoundError",
    "ModelLoadError",
    "AudioDecodeError",
    "UnsupportedOptionError",
    "DiarizationError",
    "SpeechToolkitWarning",
    "__version__",
]
