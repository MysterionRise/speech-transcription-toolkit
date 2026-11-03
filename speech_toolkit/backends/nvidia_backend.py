"""NVIDIA Parakeet and Canary backends, run through Hugging Face transformers.

Parakeet TDT (0.6B parameters) is a fast, accurate transcriber for English and 24 other European
languages that also gives word timestamps; Canary (1B) transcribes and translates between English
and the same languages. Needs the ``nvidia`` extra; models download from the Hugging Face Hub on
first use. Loading is shared with Voxtral, in ``_transformers.py``.
"""

from __future__ import annotations

import dataclasses
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..errors import SpeechToolkitWarning, UnsupportedOptionError
from ..media import SAMPLE_RATE, load_audio, split_audio
from ._transformers import _clock  # noqa: F401  (moved to _transformers.py; still importable from here)
from ._transformers import (
    BATCH_SIZE_OPTION,
    MAX_NEW_TOKENS,
    MAX_NEW_TOKENS_OPTION,
    _cut_off_warning,
    _rows_at_limit,
    _segments_from_words,
    _TransformersBackend,
)
from .base import TranscriptionResult, report_progress
from .config import BackendOption

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = logging.getLogger(__name__)


class _NvidiaBackend(_TransformersBackend):
    """What Parakeet and Canary share: the nvidia extra."""

    requires = ("torch", "transformers", "librosa")  # the nvidia extra, plus torch
    EXTRA = "nvidia"
    NEEDS = "torch and transformers>=5.18"


class ParakeetBackend(_NvidiaBackend):
    """NVIDIA Parakeet TDT: fast, accurate transcription with word timestamps."""

    name = "parakeet"
    description = "NVIDIA Parakeet TDT - fast, accurate English/European speech-to-text with word timestamps"
    # No "language_detection": v3 detects the language internally but can't report it.
    capabilities = frozenset({"vad", "word_timestamps", "waveform"})

    MODELS = {
        "parakeet-tdt-0.6b-v3": "nvidia/parakeet-tdt-0.6b-v3",  # 25 European languages, detected automatically
        "parakeet-tdt-0.6b-v2": "nvidia/parakeet-tdt-0.6b-v2",  # English
    }
    MODEL_CLASS = "ParakeetForTDT"
    # The encoder attends over the whole chunk, so memory grows quickly with chunk length.
    CHUNK_SECONDS = 120
    OPTIONS: Mapping[str, BackendOption] = {
        **_NvidiaBackend.OPTIONS,
        # Two 120 s chunks at a time by default: the 4 minutes of audio of Canary's and Voxtral's eight 30 s chunks.
        "batch_size": dataclasses.replace(BATCH_SIZE_OPTION, default=2),
    }

    def transcribe(
        self,
        audio_path: Union[Path, "np.ndarray"],
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = True,
        *,
        vad: bool = False,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio with Parakeet; segments end at sentences or pauses.

        Args:
            audio_path: Path to the audio file, or its samples (mono float32 at 16 kHz).
            language: Ignored, with a warning: the model detects the language itself. The result's
                language is None, as Parakeet can't report the one it detected.
            task: Only 'transcribe' is supported.
            verbose: Log progress (chunks transcribed) at INFO rather than DEBUG. How far it got is also passed to
                report_progress() after each batch of chunks, whatever *verbose* is.
            vad: Transcribe only the speech, found with the shared Silero VAD (tuned by the ``vad_`` options): the
                chunks are cut at its edges, and audio without speech reaches no model.
            word_timestamps: Keep each segment's per-word timings ("words").

        The audio is cut into chunks of up to 120 s, which the model transcribes ``batch_size`` (an option, 2 by
        default) at a time.

        Raises:
            ModelLoadError: If no model is loaded.
            UnsupportedOptionError: If task is not 'transcribe'.
            FileNotFoundError: If audio file doesn't exist.
        """
        if task != "transcribe":
            raise UnsupportedOptionError(
                "The parakeet backend only transcribes; translate with -b canary or a Whisper model."
            )
        self._check_ready(audio_path)
        if language:
            warnings.warn(
                f"Parakeet detects the language itself; ignoring language '{language}'.", SpeechToolkitWarning
            )

        audio = load_audio(audio_path) if isinstance(audio_path, Path) else audio_path
        duration = len(audio) / SAMPLE_RATE
        chunks = split_audio(audio, self.CHUNK_SECONDS, regions=self._speech(audio) if vad else None)
        batch_size = self._options["batch_size"]
        words: List[Dict[str, Any]] = []
        for first in range(0, len(chunks), batch_size):
            batch = chunks[first : first + batch_size]
            words += self._transcribe_batch(batch)
            # A segment can go on into the next chunk, so segments are made, and reported, once every chunk is done.
            offset, samples = batch[-1]
            report_progress(offset + len(samples) / SAMPLE_RATE, duration)
            level = logging.INFO if verbose else logging.DEBUG
            logger.log(level, "Parakeet: transcribed %d/%d chunks", first + len(batch), len(chunks))

        segments = _segments_from_words(words)
        if not word_timestamps:
            for segment in segments:
                del segment["words"]
        return TranscriptionResult(
            text="".join(segment["text"] for segment in segments).strip(),
            segments=segments,
            language=None,  # Parakeet doesn't report the language it detected
            duration=duration,
        )

    def _transcribe_batch(self, chunks: Sequence[Tuple[float, Any]]) -> List[Dict[str, Any]]:
        """The words of a batch of *chunks*, (offset in seconds, samples) pairs, in order, with each chunk's times
        shifted by its offset. The processor pads the shorter chunks, and generate() stops at the end of each."""
        inputs = self._processor([samples for _, samples in chunks], sampling_rate=SAMPLE_RATE)
        inputs = inputs.to(self._device, dtype=self._model.dtype)
        longest = max(len(samples) for _, samples in chunks)
        outputs = self._model.generate(**inputs, max_new_tokens=self._max_steps(longest))
        # The TDT decoder predicts how many frames each token lasts; decode() turns that into token times, per chunk.
        _, timestamps = self._processor.decode(outputs.sequences, durations=outputs.durations)
        words: List[Dict[str, Any]] = []
        for (offset, _), tokens in zip(chunks, timestamps):
            words += _words_from_tokens(tokens, offset)
        return words

    def _max_steps(self, samples: int) -> int:
        """The most decoding steps generate() may take for *samples* of audio: ``max_symbols_per_step`` for each
        frame of the encoder, which makes one per ``hop_length`` samples ``subsampling_factor`` times fewer.

        That is the limit Parakeet sets itself when it is given none. Giving it keeps transformers from warning about
        its default ``max_length`` on every call. (Hiding that warning in a ``warnings.catch_warnings()`` block, as
        before, made every other warning show again for each file: leaving the block resets Python's record of the
        warnings shown.)
        """
        config = self._model.config
        symbols = int(getattr(config, "max_symbols_per_step", 10))
        subsampling = int(getattr(getattr(config, "encoder_config", None), "subsampling_factor", 8))
        hop = int(getattr(getattr(self._processor, "feature_extractor", None), "hop_length", 160))
        frames = samples // hop + 1
        return symbols * -(-frames // subsampling)


class CanaryBackend(_NvidiaBackend):
    """NVIDIA Canary: transcription and translation for English and 24 European languages."""

    name = "canary"
    description = "NVIDIA Canary - multilingual speech-to-text and translation to English (25 European languages)"
    # No "language_detection": it needs the language, else assumes English.
    capabilities = frozenset({"translate", "vad", "waveform"})

    MODELS = {"canary-1b-v2": "nvidia/canary-1b-v2"}
    MODEL_CLASS = "CanaryForConditionalGeneration"
    # Canary returns text without timestamps, so each chunk becomes one coarsely timestamped segment.
    CHUNK_SECONDS = 30
    MAX_NEW_TOKENS = MAX_NEW_TOKENS  # the max_new_tokens option's default
    OPTIONS: Mapping[str, BackendOption] = {
        **_NvidiaBackend.OPTIONS,
        "max_new_tokens": MAX_NEW_TOKENS_OPTION,
        "batch_size": BATCH_SIZE_OPTION,
    }

    def transcribe(
        self,
        audio_path: Union[Path, "np.ndarray"],
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = True,
        *,
        vad: bool = False,
    ) -> TranscriptionResult:
        """Transcribe (or translate to English) audio with Canary.

        Args:
            audio_path: Path to the audio file, or its samples (mono float32 at 16 kHz).
            language: Language code of the speech; Canary can't detect it, so None means English.
            task: 'transcribe', or 'translate' to English.
            verbose: Log progress (chunks transcribed) at INFO rather than DEBUG. The segments of each batch of chunks
                are also passed to report_progress(), whatever *verbose* is.
            vad: Transcribe only the speech, found with the shared Silero VAD (tuned by the ``vad_`` options): the
                chunks are cut at its edges, so each stretch of speech gets segments of its own, and audio without
                speech reaches no model.

        Returns:
            TranscriptionResult with one segment per chunk of up to 30 s; the model transcribes ``batch_size`` chunks
            (an option, 8 by default) at a time. A chunk whose text reaches the
            max_new_tokens option's limit may be cut off: one warning per file gives the number of such chunks and the
            first one's times.
        """
        self._check_ready(audio_path)
        if language is None:
            warnings.warn(
                "Canary can't detect the language and assumes English; pass a language code otherwise.",
                SpeechToolkitWarning,
            )

        audio = load_audio(audio_path) if isinstance(audio_path, Path) else audio_path
        duration = len(audio) / SAMPLE_RATE
        chunks = split_audio(audio, self.CHUNK_SECONDS, regions=self._speech(audio) if vad else None)
        segments: List[Dict[str, Any]] = []
        cut_off: List[Dict[str, Any]] = []  # the segments of chunks that reached max_new_tokens
        batch_size = self._options["batch_size"]
        for first in range(0, len(chunks), batch_size):
            batch = chunks[first : first + batch_size]
            texts, at_limit = self._transcribe_batch([samples for _, samples in batch], language or "en", task)
            for index, ((start, samples), text) in enumerate(zip(batch, texts), start=first):
                segments.append({"id": index, "start": start, "end": start + len(samples) / SAMPLE_RATE, "text": text})
            cut_off += [segments[first + row] for row in at_limit]
            for segment in segments[first:]:
                report_progress(segment["end"], duration, segment)
            level = logging.INFO if verbose else logging.DEBUG
            logger.log(level, "Canary: transcribed %d/%d chunks", len(segments), len(chunks))
        if cut_off:
            limit = self._options["max_new_tokens"]
            warnings.warn(_cut_off_warning(audio_path, "Canary", limit, cut_off), SpeechToolkitWarning)

        return TranscriptionResult(
            text=" ".join(segment["text"] for segment in segments if segment["text"]),
            segments=segments,
            language="en" if task == "translate" else (language or "en"),
            duration=duration,
        )

    def _transcribe_batch(self, chunks: Sequence[Any], language: str, task: str) -> Tuple[List[str], List[int]]:
        """Transcribe (or translate) a batch of audio chunks.

        Returns:
            One string per chunk, and the positions in the batch of the chunks whose text reached max_new_tokens.
        """
        inputs = self._processor.apply_transcription_request(
            audio=list(chunks),
            source_language=language,
            target_language="en" if task == "translate" else None,
        )
        inputs = inputs.to(self._device, dtype=self._model.dtype)
        outputs = self._model.generate(**inputs, max_new_tokens=self._options["max_new_tokens"])
        # The output starts with the prompt (language and task tokens); keep only the generated text.
        prompt_length = inputs["decoder_input_ids"].shape[1] if "decoder_input_ids" in inputs else 0
        generated = outputs[:, prompt_length:]
        texts = self._processor.batch_decode(generated, skip_special_tokens=True)
        at_limit = _rows_at_limit(generated, self._model.generation_config, self._options["max_new_tokens"])
        return [text.strip() for text in texts], at_limit


def _words_from_tokens(tokens: Sequence[Dict[str, Any]], offset: float) -> List[Dict[str, Any]]:
    """Join subword tokens (``{"token", "start", "end"}``) into words.

    A token with a leading space starts a new word, and so does one after a whitespace-only token: "Hello", " ",
    "world" are two words.
    """
    words: List[Dict[str, Any]] = []
    new_word = True  # whether the next token starts a word
    for token in tokens:
        text = token["token"]
        if not text.strip():
            new_word = True
            continue
        start, end = round(offset + token["start"], 3), round(offset + token["end"], 3)
        if new_word or text[0].isspace():
            words.append({"word": " " + text.lstrip(), "start": start, "end": end})
        else:
            words[-1]["word"] += text
            words[-1]["end"] = max(words[-1]["end"], end)
        new_word = False
    return words
