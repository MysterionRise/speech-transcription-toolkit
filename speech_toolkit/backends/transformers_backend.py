"""Most speech recognition models on the Hugging Face Hub, run through transformers' ASR pipeline.

``-b transformers -m <repo id or folder>`` loads ``transformers.pipeline("automatic-speech-recognition")`` for the
model, with no code per model:

- **Whisper** models and their fine-tunes, distil-whisper included: timestamps per segment, word timestamps when the
  model's generation config has alignment heads, and a language, language detection and translation when it is
  multilingual.
- **CTC** models, such as wav2vec2, HuBERT and MMS: word timestamps, grouped into segments at sentences and pauses.
- The other **sequence-to-sequence** models and speech LLMs the pipeline supports, such as Moonshine and Granite
  Speech: text only, one segment per chunk of up to 30 s.

What a model can do is read from it once it has loaded (see :class:`TransformersBackend`). Long recordings are cut
into chunks of up to 30 s in pauses (``split_audio``), which each model takes whole. Needs the ``transformers`` extra;
models download from the Hub on first use (set HF_TOKEN for gated ones). Loading is shared with the other transformers
backends, in ``_transformers.py``.
"""

from __future__ import annotations

import dataclasses
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple, Union

from ..errors import BackendUnavailableError, ModelLoadError, SpeechToolkitWarning, UnsupportedOptionError
from ..media import SAMPLE_RATE, load_audio, split_audio
from ._transformers import _segments_from_words, _TransformersBackend, torch_device, torch_dtype
from .base import TranscriptionResult, report_progress

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = logging.getLogger(__name__)

# How this backend runs a model, by the kind the pipeline makes of it (its ``type``).
WHISPER = "whisper"  # Whisper and its fine-tunes ("seq2seq_whisper")
CTC = "ctc"  # wav2vec2, HuBERT, MMS, ... ("ctc", and "ctc_with_lm" with a language model)
TEXT = "text"  # everything else: other sequence-to-sequence models, speech LLMs, transducers
_FAMILIES = {"seq2seq_whisper": WHISPER, "ctc": CTC, "ctc_with_lm": CTC}

TASKS = ("transcribe", "translate")


@dataclasses.dataclass(frozen=True)
class ModelTraits:
    """What a loaded model can do, as :func:`read_traits` finds it.

    Attributes:
        family: ``WHISPER``, ``CTC`` or ``TEXT``.
        languages: The language codes it can be told, such as ``"de"``: those of a multilingual Whisper model.
            Empty when it takes none.
        language: The language it always transcribes, when that is known: ``"en"`` for an English-only Whisper model,
            or the language a Whisper model's generation config sets (distil-whisper's English models).
        translate: Whether it translates to English.
        detects_language: Whether it detects and reports the language when given none.
        segment_timestamps: Whether it gives timestamps per segment (Whisper) or per word (CTC). Without them, each
            chunk of up to 30 s is one segment.
        word_timestamps: Whether it gives word timestamps.
    """

    family: str
    languages: FrozenSet[str] = frozenset()
    language: Optional[str] = None
    translate: bool = False
    detects_language: bool = False
    segment_timestamps: bool = False
    word_timestamps: bool = False

    @property
    def capabilities(self) -> FrozenSet[str]:
        """These traits as backend capabilities (see TranscriptionBackend)."""
        found = {"waveform"}  # the pipeline takes the samples
        if self.translate:
            found.add("translate")
        if self.detects_language:
            found.add("language_detection")
        if self.word_timestamps:
            found.add("word_timestamps")
        return frozenset(found)


def read_traits(asr: Any) -> ModelTraits:
    """What the model of the ASR pipeline *asr* can do: from the kind the pipeline made of it, and for Whisper, from
    its generation config."""
    kind = getattr(asr, "type", "")
    family = _FAMILIES.get(kind, TEXT)
    if family == CTC:
        # The word timings come from a language model's decoder, or from the tokenizer: wav2vec2's has them, but not
        # every CTC model's (Parakeet CTC's hasn't), and those then give text only.
        if kind == "ctc_with_lm" or callable(getattr(asr.tokenizer, "_get_word_offsets", None)):
            return ModelTraits(CTC, segment_timestamps=True, word_timestamps=True)
        return ModelTraits(TEXT)
    if family == TEXT:
        return ModelTraits(TEXT)
    config = getattr(asr.model, "generation_config", None)
    lang_to_id = getattr(config, "lang_to_id", None) or {}  # {"<|en|>": 50259, ...}
    multilingual = bool(lang_to_id) and bool(getattr(config, "is_multilingual", True))
    forced = getattr(config, "language", None)  # "<|en|>", "en" or "english"
    timestamps = getattr(config, "no_timestamps_token_id", None) is not None
    return ModelTraits(
        WHISPER,
        languages=frozenset(_language_code(token) for token in lang_to_id) if multilingual else frozenset(),
        language=(_language_code(forced) if forced else None) if multilingual else "en",
        translate=multilingual and "translate" in (getattr(config, "task_to_id", None) or {}),
        detects_language=multilingual and not forced,
        segment_timestamps=timestamps,
        word_timestamps=timestamps and bool(getattr(config, "alignment_heads", None)),
    )


class TransformersBackend(_TransformersBackend):
    """Any speech recognition model the transformers ASR pipeline runs: ``-b transformers -m <repo id or folder>``.

    ``capabilities`` lists what some models can do, which is what the checks made before a model loads see (the CLI's
    ``-t translate`` check, ``--list-backends``). Once a model has loaded, the instance's ``capabilities`` are what that
    model can do (see :func:`read_traits`), so these checks happen:

    - **Before loading:** the model name's shape, the device and the options, as for every backend.
    - **Right after loading,** with the loaded model's capabilities: :class:`speech_toolkit.Transcriber` refuses a
      translation the model can't do with :class:`~speech_toolkit.UnsupportedOptionError` before any audio is
      decoded, and skips word timestamps it can't give with a warning, as for any backend without them; so a
      diarization run or subtitle line splitting doesn't ask for them. The server rejects both requests (400).
    - **At the start of** :meth:`transcribe`, before the audio is read or the model runs: a task, word timestamps or a
      language the model can't take raise UnsupportedOptionError. This is what a direct call of the backend gets.
    """

    name = "transformers"
    description = (
        "Hugging Face transformers - most speech recognition models on the Hub (-m <repo id or folder>); "
        "what each can do depends on the model"
    )
    # What some models can do: multilingual Whisper models translate and detect the language, Whisper models with
    # alignment heads and CTC models give word timestamps. load_model() narrows them to the loaded model's.
    capabilities = frozenset({"translate", "language_detection", "word_timestamps", "waveform"})
    EXTRA = "transformers"
    NEEDS = "torch and transformers"
    # A few models that work well, for --list-models; -m takes any other repo id or folder. The first is the default.
    MODELS = {
        repo: repo
        for repo in (
            "openai/whisper-large-v3-turbo",  # multilingual, fast; transcription only (not trained to translate)
            "openai/whisper-small",  # multilingual; translates
            "distil-whisper/distil-large-v3.5",  # English, fast
            "facebook/wav2vec2-base-960h",  # English, CTC
            "moonshine-ai/moonshine-tiny",  # English, tiny; text only
        )
    }
    MODEL_SOURCES = {"hub": "the Hugging Face repo id (org/name) of a speech recognition model", "directory": ""}
    # The longest chunk each model gets: Whisper's 30 s window, and short enough for the others' memory.
    CHUNK_SECONDS = 30

    def __init__(self) -> None:
        super().__init__()
        self._pipeline: Any = None  # the transformers ASR pipeline
        self._traits = ModelTraits(TEXT)  # what the loaded model can do

    def load_model(
        self, model_name: str, device: Optional[str] = None, *, options: Optional[Mapping[str, Any]] = None
    ) -> None:
        """Load the model's ASR pipeline (bfloat16/float16 on a GPU, float32 on the CPU, unless ``compute_type`` says
        otherwise), then set ``capabilities`` to what the model can do.

        Args:
            model_name: The Hugging Face repo id (org/name) of a speech recognition model, such as one of ``MODELS``,
                or a local model folder.
            device: 'cpu', 'cuda', 'cuda:N' (the N-th GPU), 'mps' (an Apple GPU), or None or 'auto' for CUDA when
                torch sees a GPU, else the CPU.
            options: Values for ``OPTIONS``, such as ``{"compute_type": "float16"}``.

        Raises:
            ModelNotFoundError: If model_name is neither a repo id nor a folder.
            UnsupportedOptionError: If the device isn't a device, or an option is unknown or invalid.
            BackendUnavailableError: If torch or transformers is not installed.
            ModelLoadError: If the model can't be downloaded or loaded, or the pipeline doesn't take it (it isn't a
                speech recognition model).
        """
        repo_id = self.resolve_model(model_name)
        target = self._parse_device(device)
        settings = self._resolve_options(options)

        try:
            import torch
            from transformers import pipeline
        except ImportError as e:
            raise BackendUnavailableError(
                f"The {self.name} backend needs {self.NEEDS} ({e}). "
                f'Install with: pip install "speech-transcription-toolkit[{self.EXTRA}]"'
            ) from e

        try:
            on = torch_device(torch, target)
            dtype = torch_dtype(torch, on, settings["compute_type"])
            logger.debug("Loading %s model %r on %s in %s", self.name, repo_id, on, dtype)
            asr = pipeline("automatic-speech-recognition", model=repo_id, device=on, dtype=dtype)
            traits = read_traits(asr)
        except Exception as e:  # download, auth, unknown-architecture, CUDA and out-of-memory errors come in many types
            self._pipeline = self._model = None
            self.capabilities = type(self).capabilities
            raise ModelLoadError(f"Failed to load {self.name} model '{model_name}': {e}") from e

        self._pipeline = asr
        self._model = asr.model
        self._traits = traits
        self.capabilities = traits.capabilities
        self._repo_id = repo_id
        self._model_name = model_name
        self._device = on
        self._options = settings
        logger.debug("%s is a %s model; capabilities: %s", model_name, traits.family, sorted(self.capabilities))

    @property
    def traits(self) -> ModelTraits:
        """What the loaded model can do (a text-only model's traits before one loads)."""
        return self._traits

    def transcribe(
        self,
        audio_path: Union[Path, "np.ndarray"],
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = True,
        *,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        """Transcribe (or with a multilingual Whisper model, translate to English) audio, in chunks of up to 30 s.

        Args:
            audio_path: Path to the audio file, or its samples (mono float32 at 16 kHz).
            language: Language code such as 'de', for a multilingual Whisper model; None detects it in the first 30 s,
                as Whisper does, and keeps it for the rest. Other models take none: it is ignored with a warning.
            task: 'transcribe', or 'translate' to English (multilingual Whisper models).
            verbose: Log progress (chunks transcribed) at INFO rather than DEBUG. Each chunk's segments are also passed
                to report_progress(), whatever *verbose* is.
            word_timestamps: Give each segment its words with their timings (Whisper models with alignment heads, and
                CTC models). Whisper's segments are then made from the words, ending at sentences and pauses.

        Returns:
            TranscriptionResult: Whisper's segments, the words of a CTC model grouped into segments at sentences and
            pauses, or for other models one segment per chunk. The language is the one given, detected or that the
            model always transcribes; None if unknown.

        Raises:
            ModelLoadError: If no model is loaded.
            UnsupportedOptionError: If the model can't do the task, give word timestamps, or know the language;
                raised before the audio is read.
            FileNotFoundError: If audio file doesn't exist.
        """
        if self._pipeline is None:
            raise ModelLoadError("No model loaded. Call load_model() first.")
        language = self._check_request(language, task, word_timestamps)
        if isinstance(audio_path, Path) and not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio = load_audio(audio_path) if isinstance(audio_path, Path) else audio_path
        duration = len(audio) / SAMPLE_RATE
        chunks = split_audio(audio, self.CHUNK_SECONDS)
        segments: List[Dict[str, Any]] = []
        level = logging.INFO if verbose else logging.DEBUG
        for number, (offset, samples) in enumerate(chunks, start=1):
            if language is None and self._traits.detects_language:
                language = self._detect_language(samples)  # in the first chunk, then kept, as Whisper does
            new = self._transcribe_chunk(samples, offset, language, task, word_timestamps)
            for segment in new:
                segments.append({"id": len(segments), **segment})
                report_progress(segment["end"], duration, segments[-1])
            report_progress(offset + len(samples) / SAMPLE_RATE, duration)
            logger.log(level, "Transformers: transcribed %d/%d chunks", number, len(chunks))

        return TranscriptionResult(
            text=" ".join(segment["text"].strip() for segment in segments),
            segments=segments,
            language=language or self._traits.language,
            duration=duration,
        )

    def _check_request(self, language: Optional[str], task: str, word_timestamps: bool) -> Optional[str]:
        """Check what transcribe() was asked for against the loaded model, before any audio is read.

        Returns:
            The language code to give the model, or None (detect it, or the model takes none).

        Raises:
            UnsupportedOptionError: For a task, word timestamps or a language the model can't take.
        """
        traits, model = self._traits, self._model_name
        if task not in TASKS:
            raise UnsupportedOptionError(f"task must be 'transcribe' or 'translate', not {task!r}")
        if task == "translate" and not traits.translate:
            raise UnsupportedOptionError(
                f"the model {model} can't translate; multilingual Whisper models can, such as openai/whisper-small"
            )
        if word_timestamps and not traits.word_timestamps:
            raise UnsupportedOptionError(
                f"the model {model} can't give word timestamps; Whisper models with alignment heads and CTC models "
                "can, such as openai/whisper-small or facebook/wav2vec2-base-960h"
            )
        if task == "translate" and "turbo" in str(model).lower():
            warnings.warn(
                f"'{model}' isn't trained for translation and keeps the original language; "
                "use a model such as openai/whisper-medium or openai/whisper-large-v3.",
                SpeechToolkitWarning,
                stacklevel=3,
            )
        if not language:
            return None
        code = _language_code(language)
        if traits.languages:
            if code not in traits.languages:
                raise UnsupportedOptionError(
                    f"the model {model} doesn't know the language {language!r}; give a language code such as en, de "
                    "or fr"
                )
            return code
        if code != traits.language:  # an English-only Whisper model told "en" does just that
            warnings.warn(
                f"The model {model} can't be told the language; ignoring language '{language}'.",
                SpeechToolkitWarning,
                stacklevel=3,
            )
        return None

    def _detect_language(self, samples: "np.ndarray") -> Optional[str]:
        """The language a multilingual Whisper model hears in the first 30 s of *samples*, as a code such as "en".

        It is what Whisper's generate() does when given no language. Detected here, it is given to every chunk, which
        keeps the language the same throughout, and spares the pipeline's return_language, which doesn't move the
        word timings it gives along with the language token it adds.
        """
        asr = self._pipeline
        features = asr.feature_extractor(samples, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        found = asr.model.detect_language(input_features=features.to(asr.model.device, dtype=asr.model.dtype))
        tokens = {token_id: token for token, token_id in asr.model.generation_config.lang_to_id.items()}
        token = tokens.get(int(found[0]))
        logger.debug("Detected the language %s", token)
        return _language_code(token) if token else None

    def _transcribe_chunk(
        self, samples: "np.ndarray", offset: float, language: Optional[str], task: str, word_timestamps: bool
    ) -> List[Dict[str, Any]]:
        """Transcribe one chunk of audio that starts *offset* seconds into the recording.

        Returns:
            Its segments with text: ``start``, ``end``, ``text``, and with *word_timestamps* ``words``.
        """
        traits = self._traits
        end = offset + len(samples) / SAMPLE_RATE
        options: Dict[str, Any] = {}
        if traits.family == WHISPER:
            generate: Dict[str, Any] = {}
            if traits.languages:  # a multilingual model: it takes the task, and a language (else its own, or detected)
                generate["task"] = task
                if language:
                    generate["language"] = language
            if generate:
                options["generate_kwargs"] = generate
            if traits.segment_timestamps:
                options["return_timestamps"] = "word" if word_timestamps else True
        elif traits.family == CTC:
            options["return_timestamps"] = "word"  # the words make the segments
        # A fresh dict each time: the pipeline takes its keys out. It resamples for models that need another rate.
        output = self._pipeline({"raw": samples, "sampling_rate": SAMPLE_RATE}, **options)

        items: Sequence[Dict[str, Any]] = output.get("chunks") or []
        timestamps = options.get("return_timestamps")
        if timestamps == "word" and items:
            words = [
                {"word": text if traits.family == WHISPER else " " + text.strip(), "start": start, "end": stop}
                for start, stop, text in _timed(items, offset, end)
                if text.strip()
            ]
            segments = _segments_from_words(words)
            for segment in segments:
                del segment["id"]  # numbered across the chunks by transcribe()
                if not word_timestamps:
                    del segment["words"]
        elif timestamps and items:
            segments = [{"start": start, "end": stop, "text": text} for start, stop, text in _timed(items, offset, end)]
        else:
            segments = [{"start": round(offset, 3), "end": round(end, 3), "text": str(output.get("text", "")).strip()}]
        return [segment for segment in segments if segment["text"].strip()]


def _timed(items: Sequence[Mapping[str, Any]], offset: float, end: float) -> List[Tuple[float, float, str]]:
    """The pipeline's timed *items* (``{"text", "timestamp": (start, end)}``, segments or words, in seconds into the
    chunk) as ``(start, end, text)`` in seconds into the recording, for a chunk that starts at *offset* and ends at
    *end*.

    A missing start is the previous item's end (or the chunk's start), and a missing end, as Whisper's last segment can
    have, is the chunk's end. Times stay inside the chunk and never go back.
    """
    timed: List[Tuple[float, float, str]] = []
    last_start = last_end = offset
    for item in items:
        start, stop = (tuple(item.get("timestamp") or ()) + (None, None))[:2]
        start = last_end if start is None else offset + float(start)
        stop = end if stop is None else offset + float(stop)
        start = min(max(start, last_start), end)
        stop = min(max(stop, start, last_end), end)
        timed.append((round(start, 3), round(stop, 3), str(item.get("text", ""))))
        last_start, last_end = start, stop
    return timed


def _language_code(language: str) -> str:
    """A Whisper language as its code: "en" for "en", "<|en|>" or "english", as generation configs and the pipeline
    name them; other text lowercased."""
    text = language.strip().lower()
    if text.startswith("<|") and text.endswith("|>"):
        text = text[2:-2]
    try:
        from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
    except ImportError:  # moved in another transformers release: codes still work
        return text
    return str(TO_LANGUAGE_CODE.get(text, text))
