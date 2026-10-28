"""Pluggable transcription backend system.

This module provides a registry of available transcription backends and utilities
for loading and using them. Supported backends:

- **faster-whisper**: CTranslate2 Whisper (default) - fast and light, no torch needed
- **whisper**: OpenAI Whisper on PyTorch - the ``whisper`` extra
- **voxtral**: Mistral Voxtral Mini/Small - strong multilingual support
- **parakeet**: NVIDIA Parakeet TDT - fast and accurate, with word timestamps
- **canary**: NVIDIA Canary - multilingual transcription and translation

Heavy libraries (torch, transformers, ...) are imported only when a model is loaded,
so importing this package and listing backends/models stays fast.

Each backend class declares what it can do in ``capabilities`` (see ``CAPABILITIES``), and
``backends_with()`` names the backends that declare one, e.g. the ones that translate. It also
declares what loads: the custom models it takes (``MODEL_SOURCES``), the devices it runs on
(``DEVICES``, parsed by ``parse_device()``) and its settings (``OPTIONS``, ``BackendOption`` specs).
A backend reports its progress with ``report_progress()``.

Usage:
    from speech_toolkit.backends import get_backend, list_backends

    # Get available backends
    backends = list_backends()

    # Load a backend and transcribe
    backend = get_backend("faster-whisper")
    backend.load_model("turbo")
    result = backend.transcribe(Path("audio.mp3"))
"""

from __future__ import annotations

from typing import Dict, List, Type

from ..errors import BackendNotFoundError
from .base import (
    CAPABILITIES,
    AudioInput,
    ProgressCallback,
    ProgressEvent,
    TranscriptionBackend,
    TranscriptionResult,
    report_progress,
)
from .config import BackendOption, Device, parse_device
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

# Default backend: faster-whisper is the one a plain install includes (no torch).
DEFAULT_BACKEND = "faster-whisper"


def list_backends() -> List[str]:
    """Return list of registered backend names."""
    return list(_BACKENDS.keys())


def backends_with(capability: str) -> List[str]:
    """Names of the registered backends that declare *capability*, such as ``"translate"``, in registry order."""
    return [name for name, backend_class in _BACKENDS.items() if capability in backend_class.capabilities]


def backends_with_option(option: str) -> List[str]:
    """Names of the registered backends that declare *option* in ``OPTIONS``, such as ``"compute_type"``."""
    return [name for name, backend_class in _BACKENDS.items() if option in backend_class.OPTIONS]


def backends_on(device: str) -> List[str]:
    """Names of the registered backends that list the device kind *device*, such as ``"mps"``, in ``DEVICES``."""
    return [name for name, backend_class in _BACKENDS.items() if device in (backend_class.DEVICES or ())]


def get_backend(name: str) -> TranscriptionBackend:
    """Get an instance of the specified backend.

    Args:
        name: Backend name ('whisper', 'faster-whisper', 'voxtral', etc.)

    Returns:
        An instance of the requested backend.

    Raises:
        BackendNotFoundError: If backend name is not recognized.
    """
    if name not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys())
        raise BackendNotFoundError(f"Unknown backend: '{name}'. Available backends: {available}")

    return _BACKENDS[name]()


def get_backend_class(name: str) -> Type[TranscriptionBackend]:
    """Get the class (not instance) of the specified backend.

    Args:
        name: Backend name.

    Returns:
        The backend class.

    Raises:
        BackendNotFoundError: If backend name is not recognized.
    """
    if name not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys())
        raise BackendNotFoundError(f"Unknown backend: '{name}'. Available backends: {available}")

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
    "backends_with_option",
    "backends_on",
    "get_backend",
    "get_backend_class",
    "register_backend",
    "CAPABILITIES",
    "DEFAULT_BACKEND",
    "AudioInput",
    "ProgressEvent",
    "ProgressCallback",
    "report_progress",
    "BackendOption",
    "Device",
    "parse_device",
]
