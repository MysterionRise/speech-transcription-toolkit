"""Base protocol for transcription backends.

This module defines the interface that all transcription backends must implement,
enabling pluggable support for different speech-to-text models (Whisper, Voxtral, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class TranscriptionResult:
    """Standardized transcription result across all backends.

    Attributes:
        text: Full transcript text.
        segments: List of segments, each with start, end, text, and optional metadata.
        language: Detected or specified language code.
        raw: Raw result from the underlying model (backend-specific).
    """

    def __init__(
        self,
        text: str,
        segments: List[Dict[str, Any]],
        language: Optional[str] = None,
        raw: Optional[Dict[str, Any]] = None,
    ):
        self.text = text
        self.segments = segments
        self.language = language
        self.raw = raw or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            **self.raw,  # Backend-specific fields first
            "text": self.text,  # Standardized fields override raw
            "segments": self.segments,
            "language": self.language,
        }


class TranscriptionBackend(ABC):
    """Abstract base class for transcription backends.

    All transcription backends (Whisper, Voxtral, etc.) must inherit from this
    class and implement the required methods.
    """

    name: str = "base"
    description: str = "Base transcription backend"

    def __init__(self):
        self._model = None
        self._model_name: Optional[str] = None
        self._device: Optional[str] = None

    @classmethod
    @abstractmethod
    def available_models(cls) -> List[str]:
        """Return list of available model names/sizes for this backend."""
        pass

    @classmethod
    def default_model(cls) -> str:
        """Return the default model name for this backend."""
        models = cls.available_models()
        return models[0] if models else ""

    @abstractmethod
    def load_model(self, model_name: str, device: Optional[str] = None) -> None:
        """Load the specified model.

        Args:
            model_name: Name or size of the model to load.
            device: Device to run on ('cpu', 'cuda', or None for auto-detect).
        """
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = True,
    ) -> TranscriptionResult:
        """Transcribe an audio file.

        Args:
            audio_path: Path to the audio file.
            language: Language code (e.g., 'en', 'es'). None for auto-detect.
            task: 'transcribe' or 'translate' (to English).
            verbose: Whether to show progress output.

        Returns:
            TranscriptionResult with text, segments, and metadata.
        """
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model is not None

    @property
    def model_name(self) -> Optional[str]:
        """Return the name of the currently loaded model."""
        return self._model_name

    @property
    def device(self) -> Optional[str]:
        """Return the device the model is running on."""
        return self._device
