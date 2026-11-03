"""OpenAI Whisper transcription backend.

This backend wraps the openai-whisper library for local speech-to-text transcription. openai-whisper runs on
PyTorch, so it comes with the ``whisper`` extra: ``pip install "speech-transcription-toolkit[whisper]"``.

Besides the names in ``MODELS``, it loads local checkpoint files (``.pt``), the only custom models openai-whisper
takes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union

from ..errors import BackendUnavailableError, ModelLoadError
from .base import TranscriptionBackend, TranscriptionResult

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = logging.getLogger(__name__)


class WhisperBackend(TranscriptionBackend):
    """OpenAI Whisper transcription backend.

    Supports all Whisper model sizes from tiny to large, plus the optimized
    turbo model for faster inference.

    Results carry the probability of the language openai-whisper detects (``language_probability``), which it
    doesn't report itself: the backend records what the model's ``detect_language()`` finds while transcribing. A
    language given, or the English of an English-only model, has probability 1, as faster-whisper reports it.
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

    def detect_language(self, audio: Union[Path, "np.ndarray"]) -> List[Tuple[str, float]]:
        """The languages spoken in the first 30 seconds of *audio*, with their probabilities, most likely first.

        The model looks at the same 30 seconds as when it detects the language to transcribe, the same way.

        Args:
            audio: A path, of which only the first 30 seconds are decoded, or the samples (mono float32 at 16 kHz).

        Returns:
            Every language the model knows, with its probability: ``[("de", 0.97), ("en", 0.01), ...]``. An
            English-only model gives ``[("en", 1.0)]``.

        Raises:
            ModelLoadError: If no model is loaded.
            FileNotFoundError: If the audio file doesn't exist.
        """
        if self._model is None:
            raise ModelLoadError("No model loaded. Call load_model() first.")
        if not self._model.is_multilingual:
            return [("en", 1.0)]
        import whisper  # imported by load_model() already

        samples = _first_seconds(audio, whisper.audio.CHUNK_LENGTH)
        # As openai-whisper's transcribe() does: the log-Mel spectrogram, padded with 30 s of silence, then 30 s of it.
        mel = whisper.log_mel_spectrogram(samples, self._model.dims.n_mels, padding=whisper.audio.N_SAMPLES)
        window = whisper.pad_or_trim(mel, whisper.audio.N_FRAMES).to(self._model.device)
        if not str(self._device).startswith("cpu"):
            window = window.half()  # transcribe() decodes in float16 off the CPU (see its fp16 argument)
        _, probabilities = self._model.detect_language(window)
        return sorted(((code, float(chance)) for code, chance in probabilities.items()), key=_chance, reverse=True)

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
        # openai-whisper keeps the language it detects but not its probability: this records the probability.
        detection = _DetectionRecorder.on(self._model)

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
            language_probability=detection.probability(raw_result.get("language"), given=language),
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


def _chance(candidate: Tuple[str, float]) -> float:
    """The probability of a ``(language, probability)`` pair, to sort them by."""
    return candidate[1]


def _first_seconds(audio: Union[Path, "np.ndarray"], seconds: float) -> "np.ndarray":
    """The samples of the first *seconds* of *audio*: a path decoded only that far, or the samples given, cut."""
    from ..media import SAMPLE_RATE, load_audio

    if isinstance(audio, Path):
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio}")
        return load_audio(audio, max_seconds=seconds)
    return audio[: int(seconds * SAMPLE_RATE)]


class _DetectionRecorder:
    """Stands in for a Whisper model's ``detect_language()``, to record the probability of each language it finds.

    openai-whisper's ``transcribe()`` calls that method to detect the language, keeps the most likely one and drops
    the probabilities. The recorder passes each call on unchanged and keeps the probabilities of one recording's.
    """

    def __init__(self, model: Any) -> None:
        self._detect = model.detect_language
        self._multilingual = bool(getattr(model, "is_multilingual", True))
        self.probabilities: Optional[Mapping[str, float]] = None

    @classmethod
    def on(cls, model: Any) -> _DetectionRecorder:
        """The recorder of *model*, put in place the first time, and emptied for a new transcription."""
        recorder = vars(model).get("detect_language")
        if not isinstance(recorder, cls):
            recorder = cls(model)
            model.detect_language = recorder  # an attribute of the model object: its class stays as it is
        recorder.probabilities = None
        return recorder

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        tokens, probabilities = self._detect(*args, **kwargs)
        if isinstance(probabilities, Mapping):  # one recording's; a batch of them gives a list
            self.probabilities = probabilities
        return tokens, probabilities

    def probability(self, detected: Optional[str], given: Optional[str]) -> Optional[float]:
        """The probability of the language *detected* in a transcription: 1 for a language *given*, and for the
        English of an English-only model, as faster-whisper reports them; None if the model didn't detect it."""
        if given or not self._multilingual:
            return 1.0
        if self.probabilities is None or detected is None or detected not in self.probabilities:
            return None
        return float(self.probabilities[detected])
