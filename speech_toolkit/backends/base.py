"""Base protocol for transcription backends.

This module defines the interface that all transcription backends must implement,
enabling pluggable support for different speech-to-text models (Whisper, Voxtral, etc.).
"""

from __future__ import annotations

import importlib.util
import json
import numbers
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple, Union, cast

from ..formats import format_for_path, render, write_text
from ..types import SCHEMA_VERSION, Segment

# What a backend can declare in ``capabilities`` (see TranscriptionBackend), in the order they are listed.
CAPABILITIES = ("translate", "language_detection", "prompt", "vad", "word_timestamps")


class TranscriptionResult:
    """Standardized transcription result across all backends.

    Attributes:
        text: Full transcript text.
        segments: The segments, dicts shaped like :class:`~speech_toolkit.Segment`: ``id`` (counting from 0),
            ``start`` and ``end`` in seconds, ``text``, and optional keys such as ``words``.
        language: Detected or specified language code.
        duration: Length of the audio in seconds, if the backend reports it (faster-whisper does); otherwise None.
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
        _field(segment, "id", "an integer", name, optional=True)
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

    Optional features, which the backend accepts as keyword-only arguments of ``transcribe()``:

    - ``"prompt"`` (``prompt: str``): names, terms or a sample sentence that guide the transcript.
    - ``"vad"`` (``vad: bool``): skip silence (voice activity detection) before transcribing.
    - ``"word_timestamps"`` (``word_timestamps: bool``): give each segment a ``"words"`` list of
      ``{"word": " Hello", "start": 0.0, "end": 0.4}`` dicts (Whisper's format; ``"probability"`` optional).

    :class:`speech_toolkit.Transcriber` passes a backend only the options it declares.

    ``requires`` holds the top-level import names of the packages the backend needs, such as
    ``("faster_whisper",)``; ``transcribe --list-backends`` checks them without importing anything.
    """

    name: str = "base"
    description: str = "Base transcription backend"
    capabilities: FrozenSet[str] = frozenset()
    requires: Tuple[str, ...] = ()

    def __init__(self) -> None:
        self._model: Any = None  # the backend library's model object
        self._model_name: Optional[str] = None
        self._device: Optional[str] = None

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

    @abstractmethod
    def load_model(self, model_name: str, device: Optional[str] = None) -> None:
        """Load the specified model.

        Args:
            model_name: Name or size of the model to load.
            device: Device to run on ('cpu', 'cuda', or None for auto-detect).
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
            audio_path: Path to the audio file.
            language: Language code (e.g., 'en', 'es'). None for auto-detect.
            task: 'transcribe', or 'translate' to English (Transcriber asks for it only with the
                "translate" capability).
            verbose: Whether to show progress output.

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
