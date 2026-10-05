"""Offline speech-to-text with Whisper, faster-whisper, Voxtral, Parakeet or Canary, optionally labelling speakers.

    from speech_toolkit import transcribe

    result = transcribe("talk.mp3", backend="faster-whisper", model="small")
    print(result.text)
    result.save("talk.srt")

Heavy libraries (torch, whisper, transformers, pyannote) are imported only when a model loads.
"""

from __future__ import annotations

from .api import Transcriber, transcribe
from .backends import TranscriptionBackend, TranscriptionResult, list_backends, register_backend

__version__ = "0.3.0"

__all__ = [
    "Transcriber",
    "transcribe",
    "TranscriptionBackend",
    "TranscriptionResult",
    "list_backends",
    "register_backend",
    "__version__",
]
