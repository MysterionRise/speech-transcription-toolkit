"""Exceptions and warnings raised by speech_toolkit.

Each exception is a :class:`SpeechToolkitError` and also the built-in error the library raised before, so code that
catches ``ValueError``, ``ImportError`` or ``RuntimeError`` keeps working:

    from speech_toolkit import ModelNotFoundError, SpeechToolkitError, transcribe

    try:
        result = transcribe("talk.mp3", model="huge")
    except ModelNotFoundError as e:  # also a ValueError
        print(e)  # lists the backend's models
    except SpeechToolkitError as e:  # any other failure the library reports
        print(f"transcription failed: {e}")

A missing input file raises the built-in ``FileNotFoundError``. Library warnings are :class:`SpeechToolkitWarning`.
"""

from __future__ import annotations

__all__ = [
    "SpeechToolkitError",
    "BackendNotFoundError",
    "BackendUnavailableError",
    "ModelNotFoundError",
    "ModelLoadError",
    "AudioDecodeError",
    "UnsupportedOptionError",
    "DiarizationError",
    "SpeechToolkitWarning",
]


class SpeechToolkitError(Exception):
    """Base class of speech_toolkit's errors: catch it to handle any failure the library reports."""


class BackendNotFoundError(SpeechToolkitError, ValueError):
    """No backend is registered under that name."""


class BackendUnavailableError(SpeechToolkitError, ImportError):
    """An optional package is missing, for a backend or for diarization; the message names the extra to install."""


class ModelNotFoundError(SpeechToolkitError, ValueError):
    """The backend has no model of that name."""


class ModelLoadError(SpeechToolkitError, RuntimeError):
    """The model couldn't be downloaded or loaded, or ``transcribe()`` ran before ``load_model()``."""


class AudioDecodeError(SpeechToolkitError, RuntimeError):
    """The audio couldn't be decoded."""


class UnsupportedOptionError(SpeechToolkitError, ValueError):
    """The request can't be honoured, such as translating with a backend that only transcribes."""


class DiarizationError(SpeechToolkitError, RuntimeError):
    """Speaker diarization failed, for example because its model couldn't be downloaded."""


class SpeechToolkitWarning(UserWarning):
    """Category of speech_toolkit's warnings, such as an option the backend doesn't support."""
