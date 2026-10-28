"""NVIDIA Parakeet and Canary backends, run through Hugging Face transformers.

Parakeet TDT (0.6B parameters) is a fast, accurate transcriber for English and 24 other European
languages that also gives word timestamps; Canary (1B) transcribes and translates between English
and the same languages. Needs the ``nvidia`` extra; models download from the Hugging Face Hub on
first use. Loading is shared with Voxtral, in ``_transformers.py``.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..errors import SpeechToolkitWarning, UnsupportedOptionError
from ..media import SAMPLE_RATE, load_audio, split_audio
from ._transformers import MAX_NEW_TOKENS, MAX_NEW_TOKENS_OPTION, _TransformersBackend
from .base import TranscriptionResult
from .config import BackendOption

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

# Parakeet word timings are grouped into segments that end at a sentence, a pause or this length.
MAX_SEGMENT_SECONDS = 30.0
PAUSE_SECONDS = 1.0
# A "." after these words (lowercased) doesn't end a sentence when more text follows: "Dr. Smith", "Madrid vs. Milan".
# _ends_sentence() also skips initials ("J.", "U.S.", "e.g.", "i.e."), decimal points ("3." then "5") and a "."
# followed by a lowercase word ("etc. and so on"), so "etc." still ends a sentence before a capital letter.
ABBREVIATIONS = frozenset({"dr", "mr", "mrs", "ms", "st", "vs"})


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
    capabilities = frozenset({"word_timestamps", "waveform"})

    MODELS = {
        "parakeet-tdt-0.6b-v3": "nvidia/parakeet-tdt-0.6b-v3",  # 25 European languages, detected automatically
        "parakeet-tdt-0.6b-v2": "nvidia/parakeet-tdt-0.6b-v2",  # English
    }
    MODEL_CLASS = "ParakeetForTDT"
    # The encoder attends over the whole chunk, so memory grows quickly with chunk length.
    CHUNK_SECONDS = 120

    def transcribe(
        self,
        audio_path: Union[Path, "np.ndarray"],
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = True,
        *,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio with Parakeet; segments end at sentences or pauses.

        Args:
            audio_path: Path to the audio file, or its samples (mono float32 at 16 kHz).
            language: Ignored, with a warning: the model detects the language itself. The result's
                language is None, as Parakeet can't report the one it detected.
            task: Only 'transcribe' is supported.
            verbose: Print progress to stderr.
            word_timestamps: Keep each segment's per-word timings ("words").

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
        chunks = split_audio(audio, self.CHUNK_SECONDS)
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
            language=None,  # Parakeet doesn't report the language it detected
            duration=len(audio) / SAMPLE_RATE,
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


class CanaryBackend(_NvidiaBackend):
    """NVIDIA Canary: transcription and translation for English and 24 European languages."""

    name = "canary"
    description = "NVIDIA Canary - multilingual speech-to-text and translation to English (25 European languages)"
    # No "language_detection": it needs the language, else assumes English.
    capabilities = frozenset({"translate", "waveform"})

    MODELS = {"canary-1b-v2": "nvidia/canary-1b-v2"}
    MODEL_CLASS = "CanaryForConditionalGeneration"
    # Canary returns text without timestamps, so each chunk becomes one coarsely timestamped segment.
    CHUNK_SECONDS = 30
    BATCH_SIZE = 8
    MAX_NEW_TOKENS = MAX_NEW_TOKENS  # the max_new_tokens option's default
    OPTIONS: Mapping[str, BackendOption] = {**_NvidiaBackend.OPTIONS, "max_new_tokens": MAX_NEW_TOKENS_OPTION}

    def transcribe(
        self,
        audio_path: Union[Path, "np.ndarray"],
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = True,
    ) -> TranscriptionResult:
        """Transcribe (or translate to English) audio with Canary.

        Args:
            audio_path: Path to the audio file, or its samples (mono float32 at 16 kHz).
            language: Language code of the speech; Canary can't detect it, so None means English.
            task: 'transcribe', or 'translate' to English.
            verbose: Print progress to stderr.

        Returns:
            TranscriptionResult with one segment per chunk of up to 30 s. A chunk whose text reaches the
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
        chunks = split_audio(audio, self.CHUNK_SECONDS)
        segments: List[Dict[str, Any]] = []
        cut_off: List[Dict[str, Any]] = []  # the segments of chunks that reached max_new_tokens
        for first in range(0, len(chunks), self.BATCH_SIZE):
            batch = chunks[first : first + self.BATCH_SIZE]
            texts, at_limit = self._transcribe_batch([samples for _, samples in batch], language or "en", task)
            for index, ((start, samples), text) in enumerate(zip(batch, texts), start=first):
                segments.append({"id": index, "start": start, "end": start + len(samples) / SAMPLE_RATE, "text": text})
            cut_off += [segments[first + row] for row in at_limit]
            if verbose:
                print(f"Canary: transcribed {len(segments)}/{len(chunks)} chunks", file=sys.stderr)
        if cut_off:
            limit = self._options["max_new_tokens"]
            warnings.warn(_cut_off_warning(audio_path, "Canary", limit, cut_off), SpeechToolkitWarning)

        return TranscriptionResult(
            text=" ".join(segment["text"] for segment in segments if segment["text"]),
            segments=segments,
            language="en" if task == "translate" else (language or "en"),
            duration=len(audio) / SAMPLE_RATE,
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


def _segments_from_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group words into segments that end after a sentence, before a pause, or at MAX_SEGMENT_SECONDS."""
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for index, word in enumerate(words):
        if current and (
            word["start"] - current[-1]["end"] > PAUSE_SECONDS
            or word["end"] - current[0]["start"] > MAX_SEGMENT_SECONDS
        ):
            groups.append(current)
            current = []
        current.append(word)
        following = words[index + 1]["word"] if index + 1 < len(words) else ""
        if _ends_sentence(word["word"], following):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return [
        {"id": i, "start": g[0]["start"], "end": g[-1]["end"], "text": "".join(w["word"] for w in g), "words": g}
        for i, g in enumerate(groups)
    ]


def _ends_sentence(word: str, following: str) -> bool:
    """Whether *word* ends a sentence, given the *following* word ("" after the last one).

    "?" and "!" do, and so does "." unless the following word continues the sentence: it starts in lowercase ("etc.
    and so on"), or *word* is in ABBREVIATIONS ("Dr."), initials ("J.", "U.S.", "e.g.", but not the pronoun "I.") or a
    number cut at its decimal point ("3." then "5").
    """
    word, following = word.strip(), following.strip()
    if not word.endswith("."):
        return word.endswith(("?", "!"))
    if not following:
        return True
    stem = re.sub(r"^\W+", "", word[:-1])  # without the "." and any leading punctuation, as in "¿Dr."
    initials = stem != "I" and all(len(letter) == 1 and letter.isalpha() for letter in stem.split("."))
    if following[0].islower() or stem.lower() in ABBREVIATIONS or initials:
        return False
    return not (stem[-1:].isdigit() and following[0].isdigit())


def _rows_at_limit(generated: Any, generation_config: Any, limit: int) -> List[int]:
    """The rows of a batch of generated tokens (prompt removed) that reached *limit* tokens without ending.

    generate() stops once every row has ended or *limit* tokens are generated, and pads the rows that ended earlier: a
    row that is *limit* tokens long and doesn't end in an end-of-text or padding token was cut off.
    """
    eos = generation_config.eos_token_id
    stops = {generation_config.pad_token_id, *(eos if isinstance(eos, (list, tuple)) else [eos])}
    return [row for row, tokens in enumerate(generated.tolist()) if len(tokens) >= limit and tokens[-1] not in stops]


def _cut_off_warning(
    audio_path: Union[Path, "np.ndarray"], backend: str, limit: int, segments: List[Dict[str, Any]]
) -> str:
    """The warning about the chunks (their *segments*) whose text reached the *limit* of generated tokens."""
    first = f"from {_clock(segments[0]['start'])} to {_clock(segments[0]['end'])}"
    chunks = f"the chunk {first}" if len(segments) == 1 else f"{len(segments)} chunks, the first {first}"
    # The file name also keeps Python from hiding the warning as a repeat when another file has the same chunk times.
    # Samples (the "waveform" capability) have no name.
    source = audio_path if isinstance(audio_path, Path) else "audio"
    return (
        f"{source}: {backend} reached its limit of {limit} tokens in {chunks}, so the text may be cut off. "
        f"Raise the limit with the max_new_tokens option (-O max_new_tokens={2 * limit})."
    )


def _clock(seconds: float) -> str:
    """*seconds* as M:SS, or H:MM:SS from an hour on."""
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
