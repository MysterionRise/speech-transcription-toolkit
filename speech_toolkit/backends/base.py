"""Base protocol for transcription backends.

This module defines the interface that all transcription backends must implement,
enabling pluggable support for different speech-to-text models (Whisper, Voxtral, etc.).
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import numbers
import os
import sys
from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, Iterator, List, Mapping, Optional, Tuple, Union, cast

from ..errors import ModelNotFoundError
from ..formats import format_for_path, render, write_text
from ..media import AudioSource
from ..types import SCHEMA_VERSION, Segment
from .config import DEVICE_KINDS, MODEL_SOURCE_NAMES, BackendOption, Device, check_device, model_source, parse_options

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

# What a backend can declare in ``capabilities`` (see TranscriptionBackend), in the order they are listed.
CAPABILITIES = ("translate", "language_detection", "prompt", "vad", "word_timestamps", "waveform")

# What speech_toolkit.Transcriber.transcribe() takes: a path, the file's content (bytes), a binary file object, or the
# waveform itself, a 1-D float32 NumPy array of mono samples at 16 kHz.
AudioInput = Union[AudioSource, "np.ndarray"]

# The stages of Transcriber.transcribe() that ProgressEvent reports, in the order they run.
STAGES = ("decode", "transcribe", "diarize")


@dataclass(frozen=True)
class ProgressEvent:
    """How far :meth:`speech_toolkit.Transcriber.transcribe` has got: what its *progress* callback receives.

    Attributes:
        stage: ``"decode"``, ``"transcribe"`` or ``"diarize"``, in that order. ``"decode"`` is reported when the
            Transcriber decodes the audio itself (in-memory input, or a path with speaker labels and a backend that
            takes samples), ``"diarize"`` only with speaker labels.
        done: Seconds of the audio this stage has processed. Within a stage it never decreases and never exceeds
            *total*, and the stage's last event has ``done == total``.
        total: The audio's length in seconds, or None until it is known.
        segment: In the ``"transcribe"`` stage, a new segment of the transcript, the dict ``result.segments`` will
            hold; otherwise None. Every segment is reported once, in order. faster-whisper reports each one as soon
            as it is decoded, Canary and Voxtral after each batch of chunks, and other backends at the end of the
            stage.
    """

    stage: str
    done: float
    total: Optional[float] = None
    segment: Optional[Mapping[str, Any]] = None


# What Transcriber.transcribe(progress=...) takes. An exception it raises propagates, which cancels the transcription.
ProgressCallback = Callable[[ProgressEvent], None]


class _Call:
    """One Transcriber.transcribe() call: the recording's name for messages, and the progress reported so far, which
    it passes on to the caller's callback kept monotonic."""

    def __init__(self, name: Optional[str], callback: Optional[ProgressCallback]) -> None:
        self.name = name
        self.callback = callback
        self.stage: Optional[str] = None
        self.done = 0.0
        self.total: Optional[float] = None
        self.segments = 0  # how many segments the "transcribe" stage has reported

    def report(
        self, stage: str, done: float, total: Optional[float] = None, segment: Optional[Mapping[str, Any]] = None
    ) -> None:
        """Pass an event on: *done* no lower than before in the stage, and no higher than its *total* once known."""
        if stage != self.stage:
            self.stage, self.done, self.total = stage, 0.0, None
        if total is not None:
            self.total = max(float(total), self.done)
        self.done = max(self.done, float(done) if self.total is None else min(float(done), self.total))
        if segment is not None:
            self.segments += 1
        if self.callback is not None:
            self.callback(ProgressEvent(stage, self.done, self.total, segment))


_CALL: ContextVar[Optional[_Call]] = ContextVar("speech_toolkit_call", default=None)


@contextlib.contextmanager
def _transcription_call(name: Optional[str], callback: Optional[ProgressCallback]) -> Iterator[_Call]:
    """Make report_progress() and recording_name() serve this Transcriber.transcribe() call for the ``with`` block."""
    call = _Call(name, callback)
    token = _CALL.set(call)
    try:
        yield call
    finally:
        _CALL.reset(token)


def report_progress(done: float, total: Optional[float] = None, segment: Optional[Mapping[str, Any]] = None) -> None:
    """Report how far a backend's ``transcribe()`` has got, to the *progress* callback of Transcriber.transcribe().

    Args:
        done: Seconds of the audio transcribed so far, such as the end of the newest segment.
        total: The audio's length in seconds, if known.
        segment: A new segment, the dict the result's ``segments`` will hold. Report segments in order, each once;
            Transcriber reports the ones a backend didn't when ``transcribe()`` returns.

    It does nothing outside Transcriber.transcribe(), as when a backend is used on its own. An exception the callback
    raises comes out of this call: let it propagate, as it is how the caller cancels the transcription.
    """
    call = _CALL.get()
    if call is not None:
        call.report("transcribe", done, total, segment)


def report_stage(stage: str, done: float, total: Optional[float] = None) -> None:
    """Report progress in another stage of Transcriber.transcribe(), such as ``"diarize"`` (see report_progress)."""
    call = _CALL.get()
    if call is not None:
        call.report(stage, done, total)


def recording_name(audio: Union[Path, "np.ndarray"]) -> str:
    """How messages name the recording a backend's ``transcribe()`` got as *audio*.

    In Transcriber.transcribe(), the name of its input, which a backend that gets samples or a temporary file can't
    tell: the path given, or a file object's file name, else ``"audio"``. On its own, a path is named as it is.
    """
    call = _CALL.get()
    if call is not None:
        return call.name or "audio"
    return str(audio) if isinstance(audio, (str, os.PathLike)) else "audio"


class TranscriptionResult:
    """Standardized transcription result across all backends.

    Attributes:
        text: Full transcript text.
        segments: The segments, dicts shaped like :class:`~speech_toolkit.Segment`: ``id`` (counting from 0),
            ``start`` and ``end`` in seconds, ``text``, and optional keys such as ``words``.
        language: Detected or specified language code.
        duration: Length of the audio in seconds, if the backend reports it (all built-in backends but whisper do), or
            if Transcriber decoded the audio itself (in-memory input, diarization); otherwise None.
        language_probability: Probability of the detected language, from 0 to 1, if the backend reports it
            (faster-whisper does); otherwise None.
        raw: Raw result from the underlying model (backend-specific).
        speaker_segments: With diarization, the segments with a ``speaker`` label each; otherwise None.

    :meth:`to_dict` gives the result as a dict, as the JSON output has it, and :meth:`from_dict` and :meth:`load`
    turn that back into a result.
    """

    # The keys to_dict() writes itself (speaker_segments only after diarization); raw can't replace them.
    _FIELDS = ("schema_version", "text", "segments", "language", "duration", "language_probability", "speaker_segments")

    def __init__(
        self,
        text: str,
        segments: Union[List[Segment], List[Dict[str, Any]]],
        language: Optional[str] = None,
        raw: Optional[Dict[str, Any]] = None,
        speaker_segments: Union[List[Segment], List[Dict[str, Any]], None] = None,
        *,
        duration: Optional[float] = None,
        language_probability: Optional[float] = None,
    ):
        self.text = text
        # Typed as plain dicts, so backend-specific keys need no cast; at run time a Segment is a dict.
        self.segments = cast(List[Dict[str, Any]], segments)
        self.language = language
        self.raw = raw or {}
        self.speaker_segments = cast(Optional[List[Dict[str, Any]]], speaker_segments)
        self.duration = duration
        self.language_probability = language_probability

    def __repr__(self) -> str:
        details = [f"language={self.language!r}"]
        if self.duration is not None:
            details.append(f"duration={round(self.duration, 2)}")
        details.append(f"segments={len(self.segments)}")
        if self.speaker_segments is not None:
            details.append(f"speaker_segments={len(self.speaker_segments)}")
        text = " ".join(str(self.text).split())
        if len(text) > 40:
            text = text[:40].rstrip() + "..."
        details.append(f"text={text!r}")
        return f"<{type(self).__name__} {' '.join(details)}>"

    def to_dict(self) -> Dict[str, Any]:
        """The result as a dict, as the JSON output has it; :meth:`from_dict` turns it back into a result.

        Its keys: ``schema_version``, ``text``, ``segments``, ``language``, ``duration``, ``language_probability``,
        ``speaker_segments`` after diarization, and the keys of ``raw`` that aren't one of these.
        """
        data: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            **{key: value for key, value in self.raw.items() if key not in self._FIELDS},  # backend-specific fields
            "text": self.text,
            "segments": self.segments,
            "language": self.language,
            "duration": self.duration,
            "language_probability": self.language_probability,
        }
        if self.speaker_segments is not None:
            data["speaker_segments"] = self.speaker_segments
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TranscriptionResult:
        """The result a :meth:`to_dict` dict describes, such as the JSON output read with ``json.load``.

        Dicts without ``schema_version``, written before it existed, load too. Keys that aren't result fields go to
        ``raw``. The result gets copies of the segments, so changing one doesn't change *data*.

        Raises:
            ValueError: If *data* isn't a transcription result (the message says what is wrong, and where), or a
                newer version of speech-transcription-toolkit wrote it (a higher ``schema_version``).
        """
        _object(data, "a transcription result")  # JSON can hold anything
        if "schema_version" in data:
            _check_schema_version(data["schema_version"])
        text = _field(data, "text", "a string")
        segments = _segments(_field(data, "segments", "a list"), "segments")
        speaker_segments = _field(data, "speaker_segments", "a list", optional=True)
        return cls(
            text,
            segments,
            language=_field(data, "language", "a string", optional=True),
            raw={key: value for key, value in data.items() if key not in cls._FIELDS},
            speaker_segments=(
                None if speaker_segments is None else _segments(speaker_segments, "speaker_segments", speakers=True)
            ),
            duration=_field(data, "duration", "a number", optional=True),
            language_probability=_field(data, "language_probability", "a number", optional=True),
        )

    @classmethod
    def load(cls, path: Union[str, "os.PathLike[str]"]) -> TranscriptionResult:
        """Read a result saved as JSON, by ``result.save("talk.json")`` or ``transcribe --json``, to render it again.

        Raises:
            FileNotFoundError: If *path* doesn't exist.
            ValueError: If the file isn't a transcription result in JSON (see :meth:`from_dict`).
        """
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8-sig"))  # utf-8-sig also reads a byte order mark
        except ValueError as e:  # not UTF-8 or not JSON: UnicodeDecodeError and JSONDecodeError are ValueErrors
            raise ValueError(f"{source} isn't a JSON file: {e}") from e
        try:
            return cls.from_dict(data)
        except ValueError as e:
            raise ValueError(f"{source}: {e}") from e

    def render(self, fmt: str = "txt", max_line_width: Optional[int] = None) -> str:
        """The result as ``txt``, ``srt``, ``vtt`` or ``json`` text (speaker-labelled after diarization).

        *max_line_width* splits subtitles into lines of at most that many characters, two per cue.
        """
        return render(self.to_dict(), fmt, max_line_width)

    def save(
        self, path: Union[str, "os.PathLike[str]"], fmt: Optional[str] = None, max_line_width: Optional[int] = None
    ) -> Path:
        """Write the result to *path*, in *fmt* or else the format its extension names (default txt)."""
        dest = Path(path)
        write_text(dest, self.render(fmt or format_for_path(dest), max_line_width))
        return dest


# What TranscriptionResult.from_dict() checks a value is, by the words its error messages use.
_IS = {
    "a string": lambda value: isinstance(value, str),
    "a number": lambda value: isinstance(value, numbers.Real) and not isinstance(value, bool),
    "an integer": lambda value: isinstance(value, numbers.Integral) and not isinstance(value, bool),
    "a list": lambda value: isinstance(value, list),
}


def _object(value: object, what: str) -> Mapping[str, Any]:
    """*value*, checked to be a JSON object (a mapping)."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{what} must be a JSON object, not {_kind(value)}")
    return value


def _field(obj: Mapping[str, Any], key: str, expected: str, where: str = "", *, optional: bool = False) -> Any:
    """``obj[key]``, checked to be *expected* (a key of ``_IS``); an optional key may also be missing or null."""
    name = f"{where}.{key}" if where else key
    value = obj.get(key)
    if value is None and optional:
        return None
    if key not in obj:
        raise ValueError(f"'{name}' is missing")
    if not _IS[expected](value):
        raise ValueError(f"'{name}' must be {expected}{' or null' if optional else ''}, not {_kind(value)}")
    return value


def _segments(value: List[Any], where: str, *, speakers: bool = False) -> List[Dict[str, Any]]:
    """Copies of the segments in *value*, checked for the keys the library reads (and a speaker each if *speakers*)."""
    segments = []
    for index, item in enumerate(value):
        name = f"{where}[{index}]"
        segment = dict(_object(item, f"'{name}'"))
        for key in ("start", "end"):
            _field(segment, key, "a number", name)
        _field(segment, "text", "a string", name)
        for key in ("id", "segment_id"):
            _field(segment, key, "an integer", name, optional=True)
        _field(segment, "speaker", "a string", name, optional=not speakers)
        words = _field(segment, "words", "a list", name, optional=True)
        if words is not None:
            segment["words"] = [_word(word, f"{name}.words[{number}]") for number, word in enumerate(words)]
        segments.append(segment)
    return segments


def _word(value: object, name: str) -> Dict[str, Any]:
    """A copy of the word *value*, checked for its text and times."""
    word = dict(_object(value, f"'{name}'"))
    _field(word, "word", "a string", name)
    for key in ("start", "end"):
        _field(word, key, "a number", name)
    return word


def _check_schema_version(version: Any) -> None:
    if not _IS["an integer"](version) or version < 1:
        raise ValueError(f"'schema_version' must be a whole number from 1 up, not {version!r}")
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"this result has schema_version {version}, and this version of speech-transcription-toolkit reads up to "
            f"{SCHEMA_VERSION}: upgrade it (pip install -U speech-transcription-toolkit) to load the result"
        )


def _kind(value: object) -> str:
    """What JSON calls the type of *value*, for error messages."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, numbers.Real):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, Mapping):
        return "an object"
    return f"a {type(value).__name__}"


class TranscriptionBackend(ABC):
    """Abstract base class for transcription backends.

    All transcription backends (Whisper, Voxtral, etc.) must inherit from this
    class and implement the required methods.

    A backend lists what it can do in ``capabilities`` (all of them are opt-in):

    - ``"translate"``: ``task="translate"`` translates to English. Without it, :class:`speech_toolkit.Transcriber`
      raises :class:`~speech_toolkit.UnsupportedOptionError` for a translation, before any audio is decoded.
    - ``"language_detection"``: with ``language=None``, the backend detects the language and reports it
      in ``TranscriptionResult.language``.
    - ``"waveform"``: ``transcribe()`` also takes the audio itself in place of a path: a 1-D float32 NumPy array of
      mono samples at 16 kHz, as :func:`speech_toolkit.media.load_audio` returns. :class:`speech_toolkit.Transcriber`
      passes one when it has decoded the audio already (in-memory input, or diarization, which needs the samples
      too), so each file is decoded once. A backend without it always gets a path: the file given, or a temporary
      WAV file, deleted afterwards, for in-memory input.

    Optional features, which the backend accepts as keyword-only arguments of ``transcribe()``:

    - ``"prompt"`` (``prompt: str``): names, terms or a sample sentence that guide the transcript.
    - ``"vad"`` (``vad: bool``): skip silence (voice activity detection) before transcribing.
    - ``"word_timestamps"`` (``word_timestamps: bool``): give each segment a ``"words"`` list of
      ``{"word": " Hello", "start": 0.0, "end": 0.4}`` dicts (Whisper's format; ``"probability"`` optional).

    :class:`speech_toolkit.Transcriber` passes a backend only the options it declares.

    **Progress and messages:** ``transcribe()`` can call :func:`report_progress` as it goes, to reach the *progress*
    callback of :meth:`speech_toolkit.Transcriber.transcribe`. Backends never print: they log to
    ``logging.getLogger(__name__)``, under the ``speech_toolkit`` logger.

    ``requires`` holds the top-level import names of the packages the backend needs, such as
    ``("faster_whisper",)``; ``transcribe --list-backends`` checks them without importing anything.

    What :meth:`load_model` takes is declared too, and :class:`speech_toolkit.Transcriber` checks a request against
    it before anything loads. All three are optional, and None or empty by default:

    - ``MODEL_SOURCES``: the custom models it loads besides ``available_models()``, a map from ``"hub"`` (a Hugging
      Face repo id, ``org/name``), ``"directory"`` (a local folder) or ``"file"`` (a local file) to how messages
      describe them (``""`` for the usual words). See :meth:`resolve_model`. None leaves model names to
      ``load_model()`` to check.
    - ``DEVICES``: the kinds of device it runs on, of ``"cpu"``, ``"cuda"`` (which also takes ``"cuda:N"``) and
      ``"mps"``. See :meth:`check_device`. None passes the device on unchecked.
    - ``OPTIONS``: its settings, a map from names to :class:`~speech_toolkit.backends.BackendOption` specs (type,
      default, help), which users set with ``Transcriber(options={...})`` or ``transcribe -O name=value``. A
      backend with options takes them as a keyword-only argument, ``load_model(name, device=None, *,
      options=None)``: the options the user gave, checked and converted. :meth:`_resolve_options` adds the
      defaults, and ``self.options`` holds them all once the model has loaded. A ``compute_type`` option, for the
      precision, is what ``transcribe --compute-type`` sets.

    ``Transcriber`` passes ``options`` only when there are some, so a backend without ``OPTIONS`` keeps the
    signature ``load_model(model_name, device=None)``.
    """

    name: str = "base"
    description: str = "Base transcription backend"
    capabilities: FrozenSet[str] = frozenset()
    requires: Tuple[str, ...] = ()
    MODEL_SOURCES: Optional[Mapping[str, str]] = None
    DEVICES: Optional[FrozenSet[str]] = None
    OPTIONS: Mapping[str, BackendOption] = {}

    def __init__(self) -> None:
        self._model: Any = None  # the backend library's model object
        self._model_name: Optional[str] = None
        self._device: Optional[str] = None
        self._options: Dict[str, Any] = self.default_options()  # what load_model() was given, with the defaults

    @classmethod
    @abstractmethod
    def available_models(cls) -> List[str]:
        """Return list of available model names/sizes for this backend."""
        pass

    @classmethod
    def default_model(cls) -> str:
        """Return the default model name for this backend."""
        models = cls.available_models()
        return models[0] if models else ""

    @classmethod
    def missing_requirements(cls) -> List[str]:
        """The modules in ``requires`` that aren't installed. Nothing is imported, so this is fast."""
        return [module for module in cls.requires if not _installed(module)]

    @classmethod
    def resolve_model(cls, model_name: str) -> str:
        """What the backend's loader gets for *model_name*: the id of a name from ``available_models()``, or a
        custom model of a kind in ``MODEL_SOURCES``, unchanged. Only the name is looked at: nothing is downloaded.

        A backend without ``MODEL_SOURCES`` gets every name unchanged, and its ``load_model()`` checks it.

        Raises:
            ModelNotFoundError: If *model_name* is none of these. The message lists the models and the custom ones
                the backend takes.
        """
        if model_name in cls.available_models():
            return cls._model_id(model_name)
        if cls.MODEL_SOURCES is None:
            return model_name
        source = model_source(model_name)
        if source in cls.MODEL_SOURCES:
            return model_name
        listed = f"Available: {', '.join(cls.available_models())}"
        custom = cls.describe_custom_models()
        if custom:
            listed += f"; or {custom}"
        if source is None or not custom:  # not a custom model, or the backend takes none
            raise ModelNotFoundError(f"Unknown {cls.name} model: {model_name}. {listed}")
        what = {"hub": "a Hugging Face repo id", "directory": "a folder", "file": "a file"}[source]
        raise ModelNotFoundError(f"The {cls.name} backend can't load a model from {what}: {model_name}. {listed}")

    @classmethod
    def _model_id(cls, model_name: str) -> str:
        """What the loader gets for a name from ``available_models()``: the name itself, unless overridden."""
        return model_name

    @classmethod
    def describe_custom_models(cls) -> str:
        """The custom models the backend loads, from ``MODEL_SOURCES``, as words: ``"a local model folder"``, or
        ``""`` for none."""
        return " or ".join(text or MODEL_SOURCE_NAMES[source] for source, text in (cls.MODEL_SOURCES or {}).items())

    @classmethod
    def check_device(cls, device: Optional[str]) -> Optional[str]:
        """*device* as :meth:`load_model` takes it: None to pick one, else such as ``"cpu"`` or ``"cuda:1"``.

        A backend with ``DEVICES`` takes ``None``, ``"auto"`` (which becomes None), ``"cpu"``, ``"cuda"``,
        ``"cuda:N"`` and ``"mps"``, of the kinds it lists. One without gets *device* unchecked, but for ``"auto"``.

        Raises:
            UnsupportedOptionError: If the backend can't run on *device*.
        """
        if cls.DEVICES is None:
            return None if device is None or str(device).strip().lower() == "auto" else device
        parsed = check_device(device, cls.DEVICES, cls.name)
        return None if parsed.auto else str(parsed)

    @classmethod
    def check_options(cls, options: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """The *options* given, checked against ``OPTIONS`` and converted to their types; no defaults are added.

        ``compute_type="auto"``, the precision every backend picks by default, is dropped for a backend without
        a ``compute_type`` option.

        Raises:
            UnsupportedOptionError: For an option the backend doesn't have (the message lists the ones it has), or a
                value that doesn't fit.
        """
        given = {str(key).strip().replace("-", "_"): value for key, value in (options or {}).items()}
        if "compute_type" not in cls.OPTIONS and given.get("compute_type") == "auto":
            del given["compute_type"]
        return parse_options(cls.OPTIONS, given, cls.name)

    @classmethod
    def default_options(cls) -> Dict[str, Any]:
        """Every option in ``OPTIONS`` with its default value."""
        return {name: option.default for name, option in cls.OPTIONS.items()}

    @property
    def options(self) -> Dict[str, Any]:
        """The options in effect: the ones :meth:`load_model` was given, and the defaults of the others."""
        return dict(self._options)

    def _parse_device(self, device: Optional[str]) -> Device:
        """*device* parsed, and checked against ``DEVICES`` (any device kind if the backend lists none)."""
        return check_device(device, DEVICE_KINDS if self.DEVICES is None else self.DEVICES, self.name)

    def _resolve_options(self, options: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """Every option in ``OPTIONS``: *options* checked, then the defaults of the others. For ``load_model()``."""
        return {**self.default_options(), **self.check_options(options)}

    @abstractmethod
    def load_model(self, model_name: str, device: Optional[str] = None) -> None:
        """Load the specified model.

        Args:
            model_name: A name from ``available_models()``, or a custom model the backend declares in
                ``MODEL_SOURCES``, such as a Hugging Face repo id; :meth:`resolve_model` gives what to load.
            device: None (or ``"auto"``) to pick one, else ``"cpu"``, ``"cuda"``, ``"cuda:N"`` or ``"mps"``, of
                the kinds in ``DEVICES``; :meth:`check_device` checks it.

        A backend with ``OPTIONS`` also takes ``options``, keyword-only: the options the user gave, which
        :meth:`_resolve_options` checks and completes with the defaults.
        """
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = True,
    ) -> TranscriptionResult:
        """Transcribe an audio file.

        Args:
            audio_path: Path to the audio file. A backend with the "waveform" capability may get the samples instead,
                as a 1-D float32 NumPy array at 16 kHz. Transcriber passes this argument positionally.
            language: Language code (e.g., 'en', 'es'). None for auto-detect.
            task: 'transcribe', or 'translate' to English (Transcriber asks for it only with the
                "translate" capability).
            verbose: Whether the caller wants to see progress: the built-in backends then log their progress
                messages at INFO rather than DEBUG, and whisper and faster-whisper show their progress bar on stderr.

        Returns:
            TranscriptionResult with text, segments, and metadata.
        """
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model is not None

    @property
    def model_name(self) -> Optional[str]:
        """Return the name of the currently loaded model."""
        return self._model_name

    @property
    def device(self) -> Optional[str]:
        """Return the device the model is running on."""
        return self._device


def _installed(module: str) -> bool:
    """Whether *module* can be imported, found without importing it (only its top-level package is looked up)."""
    top_level = module.partition(".")[0]  # find_spec("a.b") would import package a
    if top_level in sys.modules:
        return sys.modules[top_level] is not None  # None blocks the import
    return importlib.util.find_spec(top_level) is not None
