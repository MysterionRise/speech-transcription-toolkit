"""Pluggable transcription backend system.

This module provides a registry of available transcription backends and utilities
for loading and using them. Supported backends:

- **whisper**: OpenAI Whisper (default) - fast, accurate, multiple model sizes
- **voxtral**: Mistral Voxtral Mini/Small - strong multilingual support

Usage:
    from backends import get_backend, list_backends

    # Get available backends
    backends = list_backends()

    # Load a backend and transcribe
    backend = get_backend("whisper")
    backend.load_model("turbo")
    result = backend.transcribe(Path("audio.mp3"))
"""

from __future__ import annotations

from typing import Dict, List, Type

from .base import TranscriptionBackend, TranscriptionResult
from .voxtral_backend import VoxtralBackend
from .whisper_backend import WhisperBackend

# Registry of available backends
_BACKENDS: Dict[str, Type[TranscriptionBackend]] = {
    "whisper": WhisperBackend,
    "voxtral": VoxtralBackend,
}

# Default backend
DEFAULT_BACKEND = "whisper"


def list_backends() -> List[str]:
    """Return list of registered backend names."""
    return list(_BACKENDS.keys())


def get_backend(name: str) -> TranscriptionBackend:
    """Get an instance of the specified backend.

    Args:
        name: Backend name ('whisper', 'voxtral', etc.)

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
        from backends import register_backend
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
    "VoxtralBackend",
    "list_backends",
    "get_backend",
    "get_backend_class",
    "register_backend",
    "DEFAULT_BACKEND",
]
