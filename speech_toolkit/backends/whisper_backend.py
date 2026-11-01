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
from .config import BackendOption

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = logging.getLogger(__name__)

# Whisper's controls against made-up text, which openai-whisper and faster-whisper share, with the same names and
# defaults. Both decode 30 s windows one after another; faster-whisper's batched pipeline ignores these (it skips
# silence with voice activity detection instead).
HALLUCINATION_OPTIONS: Mapping[str, BackendOption] = {
    "condition_on_previous_text": BackendOption(
        bool,
        True,
        "Prompt each 30 s window with the text before it; false is less prone to repetition loops, if less consistent",
    ),
    "no_speech_threshold": BackendOption(
        float,
        0.6,
        "Skip a 30 s window as silence when its no-speech probability is above this and its text is improbable",
        minimum=0,
    ),
    "hallucination_silence_threshold": BackendOption(
        float,
        0.0,
        "Skip silences longer than this many seconds around text that looks made up; turns on word timings (0: off)",
        minimum=0,
    ),
}


def decoding_arguments(options: Mapping[str, Any], declared: Mapping[str, BackendOption]) -> Dict[str, Any]:
    """The keyword arguments of the model's transcribe() for the *options* in effect: the ones that differ from their
    defaults in *declared*, which are the library's own (a hallucination_silence_threshold of 0, off, is its None).

    Leaving the others out keeps the call as it was before the options existed.
    """
    return {name: options[name] for name in declared if options[name] != declared[name].default}


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
    # Keyword arguments of the model's transcribe(), all with openai-whisper's defaults.
    OPTIONS: Mapping[str, BackendOption] = {
        "beam_size": BackendOption(int, 1, "Beams in beam search; 1 decodes greedily, the fastest", minimum=1),
        **HALLUCINATION_OPTIONS,
    }

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
            options: Values for ``OPTIONS``: ``beam_size``, ``condition_on_previous_text``, ``no_speech_threshold``
                and ``hallucination_silence_threshold``, which transcribe() passes on to openai-whisper.

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
        # The options set; beam_size goes on to the decoder, which searches greedily without it.
        kwargs.update(decoding_arguments(self._options, self.OPTIONS))
        # hallucination_silence_threshold works on word timings: they are computed for it, and kept if asked for.
        if word_timestamps or kwargs.get("hallucination_silence_threshold"):
            kwargs["word_timestamps"] = True

        # Run transcription
        audio = str(audio_path) if isinstance(audio_path, Path) else audio_path
        raw_result: Dict[str, Any] = self._model.transcribe(audio, **kwargs)

        # Convert to standardized format
        segments = [_segment(seg, words=word_timestamps) for seg in raw_result.get("segments", [])]

        return TranscriptionResult(
            text=raw_result.get("text", ""),
            segments=segments,
            language=raw_result.get("language"),
            raw=raw_result,
        )


def _segment(seg: Dict[str, Any], words: bool = True) -> Dict[str, Any]:
    """Whisper's segment dict in the standard format, with "words" when it has them and *words* is true."""
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
    if words and seg.get("words"):
        segment["words"] = [
            {"word": w["word"], "start": w["start"], "end": w["end"], "probability": w.get("probability")}
            for w in seg["words"]
        ]
    return segment
