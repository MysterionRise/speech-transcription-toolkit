"""NVIDIA Parakeet and Canary backends, run through Hugging Face transformers.

Parakeet TDT (0.6B parameters) is a fast, accurate transcriber for English and 24 other European
languages that also gives word timestamps; Canary (1B) transcribes and translates between English
and the same languages. Needs the ``nvidia`` extra; models download from the Hugging Face Hub on
first use.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..media import SAMPLE_RATE, load_audio, split_audio
from .base import TranscriptionBackend, TranscriptionResult

INSTALL_HINT = 'Install with: pip install "speech-transcription-toolkit[nvidia]"'
# Parakeet word timings are grouped into segments that end at a sentence, a pause or this length.
MAX_SEGMENT_SECONDS = 30.0
PAUSE_SECONDS = 1.0


class _TransformersBackend(TranscriptionBackend):
    """Loading shared by backends that run a transformers speech model (imported lazily)."""

    MODELS: Dict[str, str] = {}
    MODEL_CLASS = ""  # transformers class name

    def __init__(self) -> None:
        super().__init__()
        self._processor: Any = None

    @classmethod
    def available_models(cls) -> List[str]:
        """Return the model names; the first is the default."""
        return list(cls.MODELS)

    def load_model(self, model_name: str, device: Optional[str] = None) -> None:
        """Load a model (bfloat16/float16 on CUDA, float32 on CPU).

        Raises:
            ImportError: If torch or a recent enough transformers is not installed.
            ValueError: If model_name is not recognized.
            RuntimeError: If the model cannot be downloaded or loaded.
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown {self.name} model: {model_name}. Available: {', '.join(self.MODELS)}")

        try:
            import torch
            import transformers

            model_class = getattr(transformers, self.MODEL_CLASS)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"The {self.name} backend needs torch and transformers>=5.18 ({e}). {INSTALL_HINT}"
            ) from e

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        repo_id = self.MODELS[model_name]
        try:
            self._processor = transformers.AutoProcessor.from_pretrained(repo_id)
            self._model = model_class.from_pretrained(repo_id, dtype=dtype).to(device)
        except Exception as e:  # download, auth and out-of-memory errors come in many types
            self._model = None
            self._processor = None
            raise RuntimeError(f"Failed to load {self.name} model '{model_name}': {e}") from e

        self._model_name = model_name
        self._device = device

    def _check_ready(self, audio_path: Path) -> None:
        if self._model is None or self._processor is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")


class ParakeetBackend(_TransformersBackend):
    """NVIDIA Parakeet TDT: fast, accurate transcription with word timestamps."""

    name = "parakeet"
    description = "NVIDIA Parakeet TDT - fast, accurate English/European speech-to-text with word timestamps"
    capabilities = frozenset({"word_timestamps"})

    MODELS = {
        "parakeet-tdt-0.6b-v3": "nvidia/parakeet-tdt-0.6b-v3",  # 25 European languages, detected automatically
        "parakeet-tdt-0.6b-v2": "nvidia/parakeet-tdt-0.6b-v2",  # English
    }
    MODEL_CLASS = "ParakeetForTDT"
    # The encoder attends over the whole chunk, so memory grows quickly with chunk length.
    CHUNK_SECONDS = 120

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = True,
        *,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio with Parakeet; segments end at sentences or pauses.

        Args:
            audio_path: Path to the audio file.
            language: Ignored: the model detects the language itself.
            task: Only 'transcribe' is supported.
            verbose: Print progress to stderr.
            word_timestamps: Keep each segment's per-word timings ("words").

        Raises:
            RuntimeError: If no model is loaded.
            ValueError: If task is not 'transcribe'.
            FileNotFoundError: If audio file doesn't exist.
        """
        if task != "transcribe":
            raise ValueError("The parakeet backend only transcribes; translate with -b canary or a Whisper model.")
        self._check_ready(audio_path)

        chunks = split_audio(load_audio(audio_path), self.CHUNK_SECONDS)
        words: List[Dict[str, Any]] = []
        for number, (offset, samples) in enumerate(chunks, start=1):
            words += self._transcribe_chunk(samples, offset)
            if verbose:
                print(f"Parakeet: transcribed {number}/{len(chunks)} chunks", file=sys.stderr)

        segments = _segments_from_words(words)
        if not word_timestamps:
            for segment in segments:
                del segment["words"]
        return TranscriptionResult(
            text="".join(segment["text"] for segment in segments).strip(),
            segments=segments,
            language=language,
        )

    def _transcribe_chunk(self, samples: Any, offset: float) -> List[Dict[str, Any]]:
        """Words of one chunk, with times shifted by the chunk's *offset* in seconds."""
        inputs = self._processor(samples, sampling_rate=SAMPLE_RATE).to(self._device, dtype=self._model.dtype)
        with warnings.catch_warnings():
            # Parakeet sizes generation from the audio length, but transformers still warns about max_length.
            warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")
            outputs = self._model.generate(**inputs)
        # The TDT decoder predicts how many frames each token lasts; decode() turns that into token times.
        _, timestamps = self._processor.decode(outputs.sequences, durations=outputs.durations)
        return _words_from_tokens(timestamps[0], offset)


class CanaryBackend(_TransformersBackend):
    """NVIDIA Canary: transcription and translation for English and 24 European languages."""

    name = "canary"
    description = "NVIDIA Canary - multilingual speech-to-text and translation to English (25 European languages)"

    MODELS = {"canary-1b-v2": "nvidia/canary-1b-v2"}
    MODEL_CLASS = "CanaryForConditionalGeneration"
    # Canary returns text without timestamps, so each chunk becomes one coarsely timestamped segment.
    CHUNK_SECONDS = 30
    BATCH_SIZE = 8
    MAX_NEW_TOKENS = 500

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = True,
    ) -> TranscriptionResult:
        """Transcribe (or translate to English) audio with Canary.

        Args:
            audio_path: Path to the audio file.
            language: Language code of the speech; Canary can't detect it, so None means English.
            task: 'transcribe', or 'translate' to English.
            verbose: Print progress to stderr.

        Returns:
            TranscriptionResult with one segment per chunk of up to 30 s.
        """
        self._check_ready(audio_path)
        if language is None:
            warnings.warn("Canary can't detect the language and assumes English; pass a language code otherwise.")

        chunks = split_audio(load_audio(audio_path), self.CHUNK_SECONDS)
        segments: List[Dict[str, Any]] = []
        for first in range(0, len(chunks), self.BATCH_SIZE):
            batch = chunks[first : first + self.BATCH_SIZE]
            texts = self._transcribe_batch([samples for _, samples in batch], language or "en", task)
            for index, ((start, samples), text) in enumerate(zip(batch, texts), start=first):
                segments.append({"id": index, "start": start, "end": start + len(samples) / SAMPLE_RATE, "text": text})
            if verbose:
                print(f"Canary: transcribed {len(segments)}/{len(chunks)} chunks", file=sys.stderr)

        return TranscriptionResult(
            text=" ".join(segment["text"] for segment in segments if segment["text"]),
            segments=segments,
            language="en" if task == "translate" else (language or "en"),
        )

    def _transcribe_batch(self, chunks: Sequence[Any], language: str, task: str) -> List[str]:
        """Transcribe (or translate) a batch of audio chunks, returning one string per chunk."""
        inputs = self._processor.apply_transcription_request(
            audio=list(chunks),
            source_language=language,
            target_language="en" if task == "translate" else None,
        )
        inputs = inputs.to(self._device, dtype=self._model.dtype)
        outputs = self._model.generate(**inputs, max_new_tokens=self.MAX_NEW_TOKENS)
        # The output starts with the prompt (language and task tokens); keep only the generated text.
        prompt_length = inputs["decoder_input_ids"].shape[1] if "decoder_input_ids" in inputs else 0
        texts = self._processor.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)
        return [text.strip() for text in texts]


def _words_from_tokens(tokens: Sequence[Dict[str, Any]], offset: float) -> List[Dict[str, Any]]:
    """Join subword tokens (``{"token", "start", "end"}``) into words; a leading space starts a new word."""
    words: List[Dict[str, Any]] = []
    for token in tokens:
        text = token["token"]
        if not text.strip():
            continue
        start, end = round(offset + token["start"], 3), round(offset + token["end"], 3)
        if words and not text[0].isspace():
            words[-1]["word"] += text
            words[-1]["end"] = max(words[-1]["end"], end)
        else:
            words.append({"word": " " + text.lstrip(), "start": start, "end": end})
    return words


def _segments_from_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group words into segments that end after a sentence, before a pause, or at MAX_SEGMENT_SECONDS."""
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for word in words:
        if current and (
            word["start"] - current[-1]["end"] > PAUSE_SECONDS
            or word["end"] - current[0]["start"] > MAX_SEGMENT_SECONDS
        ):
            groups.append(current)
            current = []
        current.append(word)
        if word["word"].rstrip().endswith((".", "?", "!")):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return [
        {"id": i, "start": g[0]["start"], "end": g[-1]["end"], "text": "".join(w["word"] for w in g), "words": g}
        for i, g in enumerate(groups)
    ]
