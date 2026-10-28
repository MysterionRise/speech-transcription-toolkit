"""OpenAI Whisper transcription backend.

This backend wraps the openai-whisper library for local speech-to-text transcription. openai-whisper runs on
PyTorch, so it comes with the ``whisper`` extra: ``pip install "speech-transcription-toolkit[whisper]"``.

Besides the names in ``MODELS``, it loads local checkpoint files (``.pt``), the only custom models openai-whisper
takes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Union

from ..errors import BackendUnavailableError, ModelLoadError
from .base import TranscriptionBackend, TranscriptionResult

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = logging.getLogger(__name__)


class WhisperBackend(TranscriptionBackend):
    """OpenAI Whisper transcription backend.

    Supports all Whisper model sizes from tiny to large, plus the optimized
    turbo model for faster inference.
    """

    name = "whisper"
    description = "OpenAI Whisper - offline speech-to-text with multiple model sizes"
    capabilities = frozenset({"translate", "language_detection", "prompt", "word_timestamps", "waveform"})
    requires = ("whisper",)
    MODEL_SOURCES = {"file": "a local checkpoint file (.pt)"}  # openai-whisper loads no repo ids or folders
    DEVICES = frozenset({"cpu", "cuda", "mps"})

    # Mirrors whisper.available_models(); kept static so listing models doesn't import torch.
    MODELS = [
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large-v1",
        "large-v2",
        "large-v3",
        "large",
        "large-v3-turbo",
        "turbo",
    ]

    @classmethod
    def available_models(cls) -> List[str]:
        """Return list of available Whisper model names."""
        return cls.MODELS.copy()

    @classmethod
    def default_model(cls) -> str:
        """Return 'turbo' as the default - best balance of speed and accuracy."""
        return "turbo"

    def load_model(
        self, model_name: str, device: Optional[str] = None, *, options: Optional[Mapping[str, Any]] = None
    ) -> None:
        """Load a Whisper model.

        Args:
            model_name: One of the available model names (tiny ... large-v3, turbo), or a local checkpoint file.
            device: 'cpu', 'cuda', 'cuda:N' (the N-th GPU), 'mps', or None or 'auto' for the GPU when torch sees one.
            options: The backend has no options, so only an empty mapping is accepted.

        Raises:
            ModelNotFoundError: If model_name is not recognized.
            UnsupportedOptionError: If the device isn't a device, or an option is given.
            BackendUnavailableError: If openai-whisper is not installed.
            ModelLoadError: If the model cannot be downloaded or loaded.
        """
        model_id = self.resolve_model(model_name)
        target = self._parse_device(device)
        settings = self._resolve_options(options)

        try:
            import whisper  # imported lazily: it pulls in torch, which is slow to import
        except ImportError as e:
            raise BackendUnavailableError(
                f"Whisper backend needs openai-whisper ({e}). "
                'Install with: pip install "speech-transcription-toolkit[whisper]"'
            ) from e

        on = None if target.auto else str(target)  # None: openai-whisper picks CUDA when torch sees a GPU
        logger.debug("Loading whisper model %r on %s", model_id, on or "the device whisper picks")
        try:
            self._model = whisper.load_model(model_id, device=on)
        except Exception as e:  # download, checksum, unpickling and out-of-memory errors come in many types
            self._model = None
            raise ModelLoadError(f"Failed to load whisper model '{model_name}': {e}") from e
        self._model_name = model_name
        self._device = on or str(next(self._model.parameters()).device)
        self._options = settings

    def transcribe(
        self,
        audio_path: Union[Path, "np.ndarray"],
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = True,
        *,
        prompt: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio using Whisper.

        Args:
            audio_path: Path to the audio file, or its samples (mono float32 at 16 kHz), which Whisper takes as is.
            language: Language code or None for auto-detection.
            task: 'transcribe' or 'translate'.
            verbose: Show a progress bar during transcription.
            prompt: Names, terms or a sample sentence that guide spelling (Whisper's initial_prompt).
            word_timestamps: Add per-word timings ("words") to each segment.

        Returns:
            TranscriptionResult with transcript text and segments.

        Raises:
            ModelLoadError: If no model is loaded.
            FileNotFoundError: If audio file doesn't exist.
        """
        if self._model is None:
            raise ModelLoadError("No model loaded. Call load_model() first.")

        if isinstance(audio_path, Path) and not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Build transcription kwargs
        kwargs: Dict[str, Any] = {
            "task": task,
            # Whisper prints every segment when verbose=True; False shows only a progress bar, None nothing.
            "verbose": False if verbose else None,
            # FP16 isn't supported on CPU; saying so up front avoids Whisper's warning.
            "fp16": not str(self._device).startswith("cpu"),
        }
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["initial_prompt"] = prompt
        if word_timestamps:
            kwargs["word_timestamps"] = True

        # Run transcription
        audio = str(audio_path) if isinstance(audio_path, Path) else audio_path
        raw_result: Dict[str, Any] = self._model.transcribe(audio, **kwargs)

        # Convert to standardized format
        segments = [_segment(seg) for seg in raw_result.get("segments", [])]

        return TranscriptionResult(
            text=raw_result.get("text", ""),
            segments=segments,
            language=raw_result.get("language"),
            raw=raw_result,
        )


def _segment(seg: Dict[str, Any]) -> Dict[str, Any]:
    """Whisper's segment dict in the standard format (with "words" when word timestamps were asked for)."""
    segment = {
        "start": seg["start"],
        "end": seg["end"],
        "text": seg["text"],
        "id": seg.get("id"),
        "tokens": seg.get("tokens"),
        "temperature": seg.get("temperature"),
        "avg_logprob": seg.get("avg_logprob"),
        "compression_ratio": seg.get("compression_ratio"),
        "no_speech_prob": seg.get("no_speech_prob"),
    }
    if seg.get("words"):
        segment["words"] = [
            {"word": w["word"], "start": w["start"], "end": w["end"], "probability": w.get("probability")}
            for w in seg["words"]
        ]
    return segment
