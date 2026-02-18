"""Mistral Voxtral transcription backend.

This backend supports Voxtral Mini (3B) and Voxtral Small for speech-to-text transcription.
Voxtral models are Mistral's open-weight speech models with strong multilingual support.

Note: Voxtral requires the transformers library and a Hugging Face token for model access.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import TranscriptionBackend, TranscriptionResult

# Optional imports - only fail when actually trying to use the backend
_HAS_TRANSFORMERS = False
_HAS_TORCH = False
_HAS_LIBROSA = False

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    pass

try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    _HAS_TRANSFORMERS = True
except ImportError:
    pass

try:
    import librosa

    _HAS_LIBROSA = True
except ImportError:
    pass


class VoxtralBackend(TranscriptionBackend):
    """Mistral Voxtral transcription backend.

    Supports Voxtral Mini (3B parameters) and Voxtral Small models for
    high-quality multilingual speech-to-text transcription.

    Requirements:
        - torch
        - transformers>=4.36.0
        - librosa (for audio loading)
        - Hugging Face token with access to Mistral models
    """

    name = "voxtral"
    description = "Mistral Voxtral - open-weight multilingual speech-to-text"

    # Available Voxtral models
    MODELS = {
        "voxtral-mini": "mistralai/Voxtral-Mini-3B-2507",
        "voxtral-small": "mistralai/Voxtral-Small-2507",
    }

    def __init__(self):
        super().__init__()
        self._processor = None
        self._pipe = None

    @classmethod
    def available_models(cls) -> List[str]:
        """Return list of available Voxtral model names."""
        return list(cls.MODELS.keys())

    @classmethod
    def default_model(cls) -> str:
        """Return 'voxtral-mini' as the default - good balance of speed and quality."""
        return "voxtral-mini"

    @classmethod
    def _check_dependencies(cls) -> None:
        """Check that all required dependencies are installed."""
        missing = []
        if not _HAS_TORCH:
            missing.append("torch")
        if not _HAS_TRANSFORMERS:
            missing.append("transformers>=4.36.0")
        if not _HAS_LIBROSA:
            missing.append("librosa")

        if missing:
            raise ImportError(
                f"Voxtral backend requires additional packages: {', '.join(missing)}. "
                f"Install with: pip install {' '.join(missing)}"
            )

    def load_model(self, model_name: str, device: Optional[str] = None) -> None:
        """Load a Voxtral model.

        Args:
            model_name: 'voxtral-mini' or 'voxtral-small'.
            device: 'cpu', 'cuda', or None for auto-detection.

        Raises:
            ImportError: If required dependencies are missing.
            ValueError: If model_name is not recognized.
            RuntimeError: If Hugging Face authentication fails.
        """
        self._check_dependencies()

        if model_name not in self.MODELS:
            raise ValueError(f"Unknown Voxtral model: {model_name}. Available: {', '.join(self.MODELS.keys())}")

        model_id = self.MODELS[model_name]

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set up dtype based on device
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        # Check for HF token
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        if not hf_token:
            warnings.warn(
                "No Hugging Face token found. Set HUGGINGFACE_TOKEN or HF_TOKEN environment variable. "
                "Voxtral models may require authentication.",
                UserWarning,
            )

        try:
            # Load model and processor
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                token=hf_token,
            )
            self._model.to(device)

            self._processor = AutoProcessor.from_pretrained(model_id, token=hf_token)

            # Create pipeline for easier inference
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self._model,
                tokenizer=self._processor.tokenizer,
                feature_extractor=self._processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device,
            )

            self._model_name = model_name
            self._device = device

        except Exception as e:
            self._model = None
            self._processor = None
            self._pipe = None
            raise RuntimeError(f"Failed to load Voxtral model '{model_name}': {e}") from e

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = True,
    ) -> TranscriptionResult:
        """Transcribe audio using Voxtral.

        Args:
            audio_path: Path to the audio file.
            language: Language code or None for auto-detection.
            task: 'transcribe' or 'translate' (translate to English).
            verbose: Show progress during transcription (currently ignored).

        Returns:
            TranscriptionResult with transcript text and segments.

        Raises:
            RuntimeError: If no model is loaded.
            FileNotFoundError: If audio file doesn't exist.
        """
        if self._pipe is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio with librosa (resamples to 16kHz)
        audio, _sr = librosa.load(str(audio_path), sr=16000)

        # Build generation kwargs
        generate_kwargs: Dict[str, Any] = {}
        if language:
            generate_kwargs["language"] = language
        if task == "translate":
            generate_kwargs["task"] = "translate"

        # Run transcription with timestamps
        result = self._pipe(
            audio,
            return_timestamps=True,
            generate_kwargs=generate_kwargs if generate_kwargs else None,
        )

        # Parse result into standardized format
        text = result.get("text", "")
        chunks = result.get("chunks", [])

        segments = []
        for i, chunk in enumerate(chunks):
            timestamp = chunk.get("timestamp", (None, None))
            segments.append(
                {
                    "id": i,
                    "start": timestamp[0] if timestamp[0] is not None else 0.0,
                    "end": timestamp[1] if timestamp[1] is not None else 0.0,
                    "text": chunk.get("text", ""),
                }
            )

        return TranscriptionResult(
            text=text,
            segments=segments,
            language=language,  # Voxtral doesn't return detected language in same way
            raw={"chunks": chunks},
        )
