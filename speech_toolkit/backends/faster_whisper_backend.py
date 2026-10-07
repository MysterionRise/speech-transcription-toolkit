"""faster-whisper transcription backend.

faster-whisper re-implements Whisper on CTranslate2: it is several times faster than
openai-whisper, uses less memory and does not need torch. Needs the ``faster-whisper`` extra;
models download from the Hugging Face Hub on first use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..errors import BackendUnavailableError, ModelLoadError, ModelNotFoundError
from .base import TranscriptionBackend, TranscriptionResult


class FasterWhisperBackend(TranscriptionBackend):
    """faster-whisper (CTranslate2) transcription backend."""

    name = "faster-whisper"
    description = "faster-whisper - CTranslate2 Whisper: faster, lighter, no torch needed"
    capabilities = frozenset({"prompt", "vad", "word_timestamps"})

    # Mirrors faster_whisper.available_models(); kept static so listing models needs no import.
    MODELS = [
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "distil-small.en",
        "medium",
        "medium.en",
        "distil-medium.en",
        "large-v1",
        "large-v2",
        "large-v3",
        "large",
        "distil-large-v2",
        "distil-large-v3",
        "distil-large-v3.5",
        "large-v3-turbo",
        "turbo",
    ]

    @classmethod
    def available_models(cls) -> List[str]:
        """Return list of available faster-whisper model names."""
        return cls.MODELS.copy()

    @classmethod
    def default_model(cls) -> str:
        """Return 'turbo' as the default - best balance of speed and accuracy."""
        return "turbo"

    def load_model(self, model_name: str, device: Optional[str] = None) -> None:
        """Load a faster-whisper model (int8 on CPU, float16 on CUDA).

        Args:
            model_name: One of the available model names.
            device: 'cpu', 'cuda', or None for auto-detection.

        Raises:
            BackendUnavailableError: If faster-whisper is not installed.
            ModelNotFoundError: If model_name is not recognized.
            ModelLoadError: If the model cannot be downloaded or loaded.
        """
        if model_name not in self.MODELS:
            raise ModelNotFoundError(f"Unknown faster-whisper model: {model_name}. Available: {', '.join(self.MODELS)}")

        try:
            import ctranslate2
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise BackendUnavailableError(
                "faster-whisper backend requires extra packages. "
                'Install with: pip install "speech-transcription-toolkit[faster-whisper]"'
            ) from e

        if device is None:
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        try:
            self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception as e:  # download and CTranslate2 errors come in many types
            self._model = None
            raise ModelLoadError(f"Failed to load faster-whisper model '{model_name}': {e}") from e

        self._model_name = model_name
        self._device = device

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = True,
        *,
        prompt: Optional[str] = None,
        vad: bool = False,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio using faster-whisper.

        Args:
            audio_path: Path to the audio file.
            language: Language code or None for auto-detection.
            task: 'transcribe' or 'translate'.
            verbose: Show a progress bar during transcription.
            prompt: Names, terms or a sample sentence that guide spelling (initial_prompt).
            vad: Skip silence with the built-in Silero VAD filter.
            word_timestamps: Add per-word timings ("words") to each segment.

        Returns:
            TranscriptionResult with transcript text and segments.

        Raises:
            ModelLoadError: If no model is loaded.
            FileNotFoundError: If audio file doesn't exist.
        """
        if self._model is None:
            raise ModelLoadError("No model loaded. Call load_model() first.")

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        segment_iter, info = self._model.transcribe(
            str(audio_path),
            language=language,
            task=task,
            log_progress=verbose,
            initial_prompt=prompt,
            vad_filter=vad,
            word_timestamps=word_timestamps,
        )
        segments = [_segment(seg) for seg in segment_iter]  # a generator: decoding happens while iterating

        return TranscriptionResult(
            text="".join(seg["text"] for seg in segments).strip(),
            segments=segments,
            language=info.language,
            raw={"duration": info.duration, "language_probability": info.language_probability},
        )


def _segment(seg: Any) -> Dict[str, Any]:
    """A faster-whisper Segment in the standard format (with "words" when word timestamps were asked for)."""
    segment: Dict[str, Any] = {
        "id": seg.id,
        "start": seg.start,
        "end": seg.end,
        "text": seg.text,
        "tokens": seg.tokens,
        "temperature": seg.temperature,
        "avg_logprob": seg.avg_logprob,
        "compression_ratio": seg.compression_ratio,
        "no_speech_prob": seg.no_speech_prob,
    }
    if getattr(seg, "words", None):
        segment["words"] = [
            {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in seg.words
        ]
    return segment
