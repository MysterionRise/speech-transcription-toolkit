"""Pluggable transcription backend system.

This module provides a registry of available transcription backends and utilities
for loading and using them. Supported backends:

- **whisper**: OpenAI Whisper (default) - fast, accurate, multiple model sizes
- **faster-whisper**: CTranslate2 Whisper - faster and lighter, no torch needed
- **voxtral**: Mistral Voxtral Mini/Small - strong multilingual support
- **parakeet**: NVIDIA Parakeet TDT - fast and accurate, with word timestamps
- **canary**: NVIDIA Canary - multilingual transcription and translation

Heavy libraries (torch, transformers, ...) are imported only when a model is loaded,
so importing this package and listing backends/models stays fast.

Each backend class declares what it can do in ``capabilities`` (see ``CAPABILITIES``), and
``backends_with()`` names the backends that declare one, e.g. the ones that translate.

Usage:
    from speech_toolkit.backends import get_backend, list_backends

    # Get available backends
    backends = list_backends()

    # Load a backend and transcribe
    backend = get_backend("whisper")
    backend.load_model("turbo")
    result = backend.transcribe(Path("audio.mp3"))
"""

from __future__ import annotations

from typing import Dict, List, Type

from .base import CAPABILITIES, TranscriptionBackend, TranscriptionResult
from .faster_whisper_backend import FasterWhisperBackend
from .nvidia_backend import CanaryBackend, ParakeetBackend
from .voxtral_backend import VoxtralBackend
from .whisper_backend import WhisperBackend

# Registry of available backends
_BACKENDS: Dict[str, Type[TranscriptionBackend]] = {
    "whisper": WhisperBackend,
    "faster-whisper": FasterWhisperBackend,
    "voxtral": VoxtralBackend,
    "parakeet": ParakeetBackend,
    "canary": CanaryBackend,
}

# Default backend
DEFAULT_BACKEND = "whisper"


def list_backends() -> List[str]:
    """Return list of registered backend names."""
    return list(_BACKENDS.keys())


def backends_with(capability: str) -> List[str]:
    """Names of the registered backends that declare *capability*, such as ``"translate"``, in registry order."""
    return [name for name, backend_class in _BACKENDS.items() if capability in backend_class.capabilities]


def get_backend(name: str) -> TranscriptionBackend:
    """Get an instance of the specified backend.

    Args:
        name: Backend name ('whisper', 'faster-whisper', 'voxtral', etc.)

    Returns:
        An instance of the requested backend.

    Raises:
        ValueError: If backend name is not recognized.
    """
    if name not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys())
        raise ValueError(f"Unknown backend: '{name}'. Available backends: {available}")

    return _BACKENDS[name]()


def get_backend_class(name: str) -> Type[TranscriptionBackend]:
    """Get the class (not instance) of the specified backend.

    Args:
        name: Backend name.

    Returns:
        The backend class.

    Raises:
        ValueError: If backend name is not recognized.
    """
    if name not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys())
        raise ValueError(f"Unknown backend: '{name}'. Available backends: {available}")

    return _BACKENDS[name]


def register_backend(name: str, backend_class: Type[TranscriptionBackend], *, force: bool = False) -> None:
    """Register a custom backend.

    Args:
        name: Name to register the backend under.
        backend_class: The backend class to register (must subclass TranscriptionBackend).
        force: If True, allow overwriting an existing backend.

    Raises:
        TypeError: If backend_class is not a subclass of TranscriptionBackend.
        ValueError: If name is already registered and force is False.

    Example:
        from speech_toolkit.backends import register_backend
        from my_custom_backend import MyBackend

        register_backend("custom", MyBackend)
    """
    if not (isinstance(backend_class, type) and issubclass(backend_class, TranscriptionBackend)):
        raise TypeError(f"backend_class must be a subclass of TranscriptionBackend, got {backend_class}")
    if name in _BACKENDS and not force:
        raise ValueError(f"Backend '{name}' is already registered. Use force=True to overwrite.")
    _BACKENDS[name] = backend_class


__all__ = [
    "TranscriptionBackend",
    "TranscriptionResult",
    "WhisperBackend",
    "FasterWhisperBackend",
    "VoxtralBackend",
    "ParakeetBackend",
    "CanaryBackend",
    "list_backends",
    "backends_with",
    "get_backend",
    "get_backend_class",
    "register_backend",
    "CAPABILITIES",
    "DEFAULT_BACKEND",
]
