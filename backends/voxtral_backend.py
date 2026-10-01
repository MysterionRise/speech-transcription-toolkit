"""Mistral Voxtral transcription backend.

Runs Voxtral Mini (3B) or Voxtral Small (24B) locally through Hugging Face transformers.
Needs the extra packages in requirements-voxtral.txt; models download from the Hugging Face
Hub on first use (set HF_TOKEN if the download asks for authentication).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .base import TranscriptionBackend, TranscriptionResult

SAMPLE_RATE = 16000
# Voxtral's transcription mode returns text without timestamps, so audio is split into
# fixed chunks and each chunk becomes one (coarsely) timestamped segment.
CHUNK_SECONDS = 30
BATCH_SIZE = 8
MAX_NEW_TOKENS = 500


class VoxtralBackend(TranscriptionBackend):
    """Mistral Voxtral transcription backend.

    Supports Voxtral Mini (3B parameters) and Voxtral Small (24B parameters)
    for multilingual speech-to-text transcription.

    Requirements:
        - torch
        - transformers>=4.56
        - mistral-common[audio]>=1.8.1
    """

    name = "voxtral"
    description = "Mistral Voxtral - open-weight multilingual speech-to-text"

    # Available Voxtral models
    MODELS = {
        "voxtral-mini": "mistralai/Voxtral-Mini-3B-2507",
        "voxtral-small": "mistralai/Voxtral-Small-24B-2507",
    }

    def __init__(self) -> None:
        super().__init__()
        self._processor: Any = None
        self._repo_id: Optional[str] = None

    @classmethod
    def available_models(cls) -> List[str]:
        """Return list of available Voxtral model names."""
        return list(cls.MODELS.keys())

    @classmethod
    def default_model(cls) -> str:
        """Return 'voxtral-mini' as the default - good balance of speed and quality."""
        return "voxtral-mini"

    def load_model(self, model_name: str, device: Optional[str] = None) -> None:
        """Load a Voxtral model.

        Args:
            model_name: 'voxtral-mini' or 'voxtral-small'.
            device: 'cpu', 'cuda', or None for auto-detection.

        Raises:
            ImportError: If required dependencies are missing.
            ValueError: If model_name is not recognized.
            RuntimeError: If the model cannot be downloaded or loaded.
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown Voxtral model: {model_name}. Available: {', '.join(self.MODELS.keys())}")

        try:
            import torch
            from transformers import AutoProcessor, VoxtralForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                f"Voxtral backend requires extra packages ({e}). Install with: pip install -r requirements-voxtral.txt"
            ) from e

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        repo_id = self.MODELS[model_name]
        try:
            self._processor = AutoProcessor.from_pretrained(repo_id)
            self._model = VoxtralForConditionalGeneration.from_pretrained(repo_id, dtype=dtype).to(device)
        except Exception as e:  # download, auth and out-of-memory errors come in many types
            self._model = None
            self._processor = None
            raise RuntimeError(f"Failed to load Voxtral model '{model_name}': {e}") from e

        self._repo_id = repo_id
        self._model_name = model_name
        self._device = device

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
            task: Only 'transcribe' is supported.
            verbose: Print progress to stderr.

        Returns:
            TranscriptionResult with one segment per ~30 s chunk of audio.

        Raises:
            RuntimeError: If no model is loaded.
            ValueError: If task is not 'transcribe'.
            FileNotFoundError: If audio file doesn't exist.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        if task != "transcribe":
            raise ValueError("The Voxtral backend only supports --task transcribe.")

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        from whisper import load_audio  # ffmpeg decode to 16 kHz mono float32

        audio = load_audio(str(audio_path), sr=SAMPLE_RATE)
        step = CHUNK_SECONDS * SAMPLE_RATE
        chunks = [audio[i : i + step] for i in range(0, len(audio), step)]

        segments: List[Dict[str, Any]] = []
        for first in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[first : first + BATCH_SIZE]
            for index, (chunk, text) in enumerate(zip(batch, self._transcribe_batch(batch, language)), start=first):
                start = float(index * CHUNK_SECONDS)
                segments.append({"id": index, "start": start, "end": start + len(chunk) / SAMPLE_RATE, "text": text})
            if verbose:
                print(f"Voxtral: transcribed {len(segments)}/{len(chunks)} chunks", file=sys.stderr)

        return TranscriptionResult(
            text=" ".join(seg["text"] for seg in segments if seg["text"]),
            segments=segments,
            language=language,  # Voxtral doesn't report the detected language
        )

    def _transcribe_batch(self, chunks: Sequence[Any], language: Optional[str]) -> List[str]:
        """Transcribe a batch of audio chunks, returning one string per chunk."""
        inputs = self._processor.apply_transcription_request(
            audio=list(chunks),
            model_id=self._repo_id,
            language=language,
            sampling_rate=SAMPLE_RATE,
            format="wav",
        )
        inputs = inputs.to(self._device, dtype=self._model.dtype)
        outputs = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        texts = self._processor.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        return [text.strip() for text in texts]
