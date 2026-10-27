"""Python API: load a model once, then transcribe as many files as you like.

    from speech_toolkit import Transcriber

    transcriber = Transcriber(backend="faster-whisper", model="small", diarize=True)
    for path in ["a.mp3", "b.mp3"]:
        transcriber.transcribe(path, num_speakers=2).save(f"{path}.srt")

Besides a path, ``transcribe()`` takes the file's content as ``bytes``, a binary file object, or the samples as a
NumPy array. Each recording is decoded at most once, also with speaker labels. ``progress=`` gets a
:class:`~speech_toolkit.ProgressEvent` as the work advances, with each segment as it is transcribed.

Nothing is printed: messages go to the ``speech_toolkit`` logger, which shows them only once you configure logging.

Errors are raised, never turned into ``sys.exit``. They are :class:`~speech_toolkit.SpeechToolkitError`
subclasses that are also the built-in error for their kind: ``ValueError`` for bad arguments, ``ImportError``
for a missing optional package, ``RuntimeError`` for audio or model problems. A missing file raises
``FileNotFoundError``. Warnings are :class:`~speech_toolkit.SpeechToolkitWarning`.
"""

from __future__ import annotations

import contextlib
import logging
import os
import pathlib
import shutil
import tempfile
import time
import warnings
import wave
from typing import IO, TYPE_CHECKING, AbstractSet, Any, Callable, Dict, Iterator, Optional, Union, cast

from .backends import (
    DEFAULT_BACKEND,
    AudioInput,
    ProgressCallback,
    TranscriptionBackend,
    TranscriptionResult,
    backends_with,
    get_backend_class,
)
from .backends.base import _Call, _transcription_call, report_stage
from .diarization import check_speaker_hints, diarize_audio, load_diarization_pipeline, merge_diarization
from .errors import SpeechToolkitWarning, UnsupportedOptionError
from .media import SAMPLE_RATE, load_audio

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = logging.getLogger(__name__)

TASKS = ("transcribe", "translate")
# Whisper's turbo weights weren't trained for translation: they return the original language.
TURBO_MODELS = ("turbo", "large-v3-turbo")

AudioPath = Union[str, "os.PathLike[str]"]
# What a backend transcribes, and diarize_audio() labels: a file, or the samples decoded from the input.
_Source = Union[pathlib.Path, "np.ndarray"]

# MP4, M4A, MOV and 3GP files are a series of boxes, each with an 8-byte header: its size, then its type. These types
# can come first.
_MP4_FIRST_BOXES = frozenset({b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide", b"pnot"})
_HEAD_BYTES = 64 * 1024  # how much of a file object is looked at to tell whether it is such a file
_WAV_PART_SAMPLES = 1024 * 1024  # how many samples are converted at a time when writing a temporary WAV file


def configure_hf_token(token: Optional[str] = None) -> None:
    """Expose the Hugging Face token as HF_TOKEN, which huggingface_hub uses for every model download.

    Falls back to an existing HF_TOKEN or the older HUGGINGFACE_TOKEN variable.
    """
    token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token


def _check_task(task: str, backend: str, capabilities: AbstractSet[str]) -> None:
    """Raise UnsupportedOptionError unless *task* is "transcribe", or "translate" for a backend that can translate."""
    if task not in TASKS:
        raise UnsupportedOptionError(f"task must be 'transcribe' or 'translate', not {task!r}")
    if task == "translate" and "translate" not in capabilities:
        translators = ", ".join(backends_with("translate"))
        raise UnsupportedOptionError(f"the {backend} backend can't translate; backends that translate: {translators}")


class Transcriber:
    """A loaded model, plus a speaker-diarization pipeline with ``diarize=True``, reusable across files.

    Args:
        backend: Backend name, see :func:`speech_toolkit.list_backends` (default: ``"faster-whisper"``).
        model: Model name (default: the backend's default, e.g. ``turbo`` for faster-whisper).
        device: ``"cpu"``, ``"cuda"`` or None to pick automatically.
        diarize: Also label speakers with pyannote.audio (needs the ``diarize`` extra and a Hugging Face token).
        hf_token: Hugging Face token for model downloads (default: ``HF_TOKEN`` or ``HUGGINGFACE_TOKEN``).
    """

    def __init__(
        self,
        backend: str = DEFAULT_BACKEND,
        model: Optional[str] = None,
        *,
        device: Optional[str] = None,
        diarize: bool = False,
        hf_token: Optional[str] = None,
    ) -> None:
        configure_hf_token(hf_token)
        backend_class = get_backend_class(backend)
        self.backend_name = backend
        self.model_name = model or backend_class.default_model()

        # Diarization first: a missing package or HF token should fail before a big model download.
        self._pipeline = load_diarization_pipeline(device) if diarize else None
        self.backend: TranscriptionBackend = backend_class()
        self.backend.load_model(self.model_name, device=device)

    @property
    def diarize(self) -> bool:
        """Whether results get speaker labels."""
        return self._pipeline is not None

    def supports(self, option: str) -> bool:
        """Whether the backend declares *option*, one of its capabilities.

        ``"translate"``, ``"language_detection"``, ``"prompt"``, ``"vad"``, ``"word_timestamps"`` or
        ``"waveform"``; see :class:`speech_toolkit.TranscriptionBackend`.
        """
        return option in self.backend.capabilities

    def transcribe(
        self,
        audio: AudioInput,
        language: Optional[str] = None,
        task: str = "transcribe",
        *,
        prompt: Optional[str] = None,
        vad: bool = False,
        word_timestamps: Optional[bool] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        verbose: bool = False,
        progress: Optional[ProgressCallback] = None,
    ) -> TranscriptionResult:
        """Transcribe one audio or video recording.

        Args:
            audio: What to transcribe, decoded at most once, also when diarizing:

                - a path to anything ffmpeg can decode;
                - the file's content as ``bytes`` (or ``bytearray``, ``memoryview``);
                - a binary file object, read from its current position to the end and not closed;
                - the samples: a 1-D float32 NumPy array of mono audio at 16 kHz, such as
                  :func:`speech_toolkit.media.load_audio` returns. Another array raises ``ValueError``.

                Another type raises ``TypeError``. An MP4, M4A or MOV recording in memory whose index comes after its
                audio, as phones record them, is decoded from a temporary file, deleted afterwards.
            language: Language code such as ``"en"`` (default: auto-detect).
            task: ``"transcribe"``, or ``"translate"`` to English with a backend that has the ``"translate"``
                capability. Another task, or a translation the backend can't do, raises
                :class:`~speech_toolkit.UnsupportedOptionError` before any audio is decoded.
            prompt: Names, terms or a sample sentence that guide spelling (whisper, faster-whisper).
            vad: Skip silence first, which avoids made-up text in quiet parts (faster-whisper).
            word_timestamps: Add per-word timings to each segment's ``"words"``. The default (None) turns
                them on when diarizing, so speaker labels can change mid-segment.
            num_speakers, min_speakers, max_speakers: Speaker-count hints for diarization. Hints that can't be honoured,
                such as ``min_speakers`` above ``max_speakers``, raise :class:`~speech_toolkit.UnsupportedOptionError`.
            verbose: Show the backend's progress: whisper's and faster-whisper's progress bar on stderr, and the
                progress messages of the other built-in backends, logged at INFO rather than DEBUG by the
                ``speech_toolkit`` logger (they show once you configure logging, as the ``transcribe`` command does).
                Before 0.5.0 those backends printed them.
            progress: Called with a :class:`~speech_toolkit.ProgressEvent` as each stage advances: ``"decode"`` (when
                the audio is decoded here), ``"transcribe"`` with each new segment, and ``"diarize"``. Within a stage,
                ``done`` never decreases. It runs in the calling thread, and an exception it raises propagates from
                here, which is how to cancel a transcription.

        Options the backend doesn't support are skipped with a warning.

        Returns:
            The transcript; ``result.speaker_segments`` holds speaker-labelled segments when diarizing.
        """
        _check_task(task, self.backend_name, self.backend.capabilities)
        check_speaker_hints(num_speakers, min_speakers, max_speakers, diarize=self.diarize)
        _check_audio(audio)
        if task == "translate" and self.model_name in TURBO_MODELS:
            warnings.warn(
                f"'{self.model_name}' isn't trained for translation and keeps the original language; "
                "use model 'medium' or 'large-v3'.",
                SpeechToolkitWarning,
                stacklevel=2,
            )
        if word_timestamps is None:
            word_timestamps = self.diarize and self.supports("word_timestamps")

        options = self._backend_options(prompt=prompt, vad=vad, word_timestamps=word_timestamps)
        name = _name(audio)
        # The backend and diarization report progress, and name the recording, through this call's context.
        with _transcription_call(name, progress) as call:
            source = self._source(audio)
            total = None if isinstance(source, pathlib.Path) else len(source) / SAMPLE_RATE
            logger.debug("Transcribing %s with %s '%s'", name or "audio", self.backend_name, self.model_name)
            started = time.perf_counter()
            call.report("transcribe", 0.0, total)
            with self._backend_input(source) as backend_audio:
                # Passed positionally, so a custom backend may name its first parameter as it likes. An array reaches
                # only backends with the "waveform" capability.
                result = self.backend.transcribe(
                    cast(pathlib.Path, backend_audio), language=language, task=task, verbose=verbose, **options
                )
            if result.duration is None and total is not None:  # a backend's own duration is kept
                result.duration = total
            logger.debug("Transcribed in %.1f s: %d segments", time.perf_counter() - started, len(result.segments))
            _end_transcribe_stage(call, result)
            if self._pipeline is not None:
                turns = diarize_audio(
                    source,
                    self._pipeline,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
                result.speaker_segments = merge_diarization({"segments": result.segments}, turns)
        return result

    def _source(self, audio: AudioInput) -> _Source:
        """What to transcribe and diarize: the file given, or the samples, decoded here once when they're needed.

        They are needed for in-memory input, and for diarization when the backend takes samples: both then get the
        same array. Otherwise the backend reads the file itself, and so does diarization. Decoding is reported as the
        "decode" stage.
        """
        if not isinstance(audio, (str, os.PathLike)):
            if isinstance(audio, (bytes, bytearray, memoryview)) or callable(getattr(audio, "read", None)):
                return _decoding(lambda: _decode(audio))
            return _decode(audio)  # the samples themselves
        path = pathlib.Path(audio)
        if not (self.diarize and self.supports("waveform")):
            return path
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return _decoding(lambda: load_audio(path))

    @contextlib.contextmanager
    def _backend_input(self, source: _Source) -> Iterator[_Source]:
        """What the backend gets: a path, the samples if it takes them, else a temporary WAV file of them."""
        if isinstance(source, pathlib.Path) or self.supports("waveform"):
            yield source
            return
        with _temporary_file(".wav") as wav:
            _write_wav(wav, source)
            yield wav

    def _backend_options(self, **requested: Any) -> Dict[str, Any]:
        """The requested options the backend supports; the others are dropped with a warning."""
        options: Dict[str, Any] = {}
        for name, value in requested.items():
            if not value:
                continue
            if self.supports(name):
                options[name] = value
            else:
                warnings.warn(
                    f"the {self.backend_name} backend doesn't support {name.replace('_', ' ')}; ignoring it.",
                    SpeechToolkitWarning,
                    stacklevel=3,
                )
        return options


def transcribe(
    audio: AudioInput,
    backend: str = DEFAULT_BACKEND,
    model: Optional[str] = None,
    *,
    language: Optional[str] = None,
    task: str = "transcribe",
    device: Optional[str] = None,
    diarize: bool = False,
    hf_token: Optional[str] = None,
    prompt: Optional[str] = None,
    vad: bool = False,
    word_timestamps: Optional[bool] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    verbose: bool = False,
    progress: Optional[ProgressCallback] = None,
) -> TranscriptionResult:
    """Load a model and transcribe one file. To transcribe several files, create a :class:`Transcriber` once.

    Takes the arguments of :class:`Transcriber` and :meth:`Transcriber.transcribe`. A *task* the backend can't do, or
    speaker hints that can't be honoured, raise :class:`~speech_toolkit.UnsupportedOptionError` before the model loads,
    and *audio* of a type it can't take raises ``TypeError`` (or ``ValueError``) then.
    """
    _check_task(task, backend, get_backend_class(backend).capabilities)
    check_speaker_hints(num_speakers, min_speakers, max_speakers, diarize=diarize)
    _check_audio(audio)
    transcriber = Transcriber(backend, model, device=device, diarize=diarize, hf_token=hf_token)
    return transcriber.transcribe(
        audio,
        language,
        task,
        prompt=prompt,
        vad=vad,
        word_timestamps=word_timestamps,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        verbose=verbose,
        progress=progress,
    )


def _name(audio: AudioInput) -> Optional[str]:
    """What messages call *audio*: a path as given, or a file object's file name; None for other in-memory input."""
    if isinstance(audio, (str, os.PathLike)):
        return str(audio)
    name = getattr(audio, "name", None)  # open("talk.ogg", "rb").name, sys.stdin.buffer.name ("<stdin>")
    return name if isinstance(name, str) else None


def _decoding(decode: Callable[[], "np.ndarray"]) -> "np.ndarray":
    """The samples *decode* returns, its run reported as the "decode" stage and logged with how long it took."""
    report_stage("decode", 0.0)
    started = time.perf_counter()
    samples = decode()
    seconds = len(samples) / SAMPLE_RATE
    logger.debug("Decoded %.1f s of audio in %.1f s", seconds, time.perf_counter() - started)
    report_stage("decode", seconds, seconds)
    return samples


def _end_transcribe_stage(call: _Call, result: TranscriptionResult) -> None:
    """Report the segments the backend didn't report itself, in order, then the end of the "transcribe" stage: done
    and total are the audio's duration, or, if that isn't known, as far as the segments go."""
    for segment in result.segments[call.segments :]:
        call.report("transcribe", segment.get("end") or 0.0, segment=segment)
    end = result.duration if result.duration is not None else call.done
    call.report("transcribe", end, end)


def _check_audio(audio: object) -> None:
    """Raise TypeError unless *audio* is a path, bytes, a binary file object or an array, and ValueError for an array
    that isn't 1-D float32 samples. Cheap: call it before loading anything."""
    if isinstance(audio, (str, os.PathLike, bytes, bytearray, memoryview)) or callable(getattr(audio, "read", None)):
        return
    import numpy as np  # imported lazily, as in media.load_audio(): listing backends shouldn't pay for it

    if not isinstance(audio, np.ndarray):
        raise TypeError(
            f"audio must be a path, bytes, a binary file object or a NumPy array, not {type(audio).__name__}"
        )
    if audio.ndim != 1 or audio.dtype != np.float32:
        raise ValueError(
            "audio as an array must hold 1-D float32 samples (mono, 16 kHz), "
            f"not a {audio.ndim}-D {audio.dtype} array; convert it first, e.g. int16 samples with "
            "samples.astype(np.float32) / 32768"
        )


def _decode(audio: Union[bytes, bytearray, memoryview, IO[bytes], "np.ndarray"]) -> "np.ndarray":
    """The samples of in-memory *audio*: bytes or a file object decoded with ffmpeg, or the array given.

    ffmpeg reads bytes and file objects through a pipe, which can't seek: an MP4 whose index comes after its audio
    can't be decoded that way, so it is copied to a temporary file and decoded from there.
    """
    if isinstance(audio, (bytes, bytearray, memoryview)):
        data = memoryview(audio).cast("B")
        return _load_from_temporary_file(data) if _index_at_end(data) else load_audio(audio)
    if callable(getattr(audio, "read", None)):
        stream = cast(IO[bytes], audio)
        head = _read_head(stream)
        if _index_at_end(head):
            return _load_from_temporary_file(head, stream)
        return load_audio(cast(IO[bytes], _Rewound(head, stream)))  # load_audio() only calls read()
    import numpy as np

    return np.ascontiguousarray(audio)  # checked by _check_audio(); torch can't take negative strides


def _read_head(stream: IO[bytes]) -> bytes:
    """The first _HEAD_BYTES of *stream*, or all of it if it is shorter."""
    head = bytearray()
    while len(head) < _HEAD_BYTES:
        chunk = stream.read(_HEAD_BYTES - len(head))
        if not chunk:
            break
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError(f"transcribe() needs a binary file object, but read() returned {type(chunk).__name__}")
        head += chunk
    return bytes(head)


class _Rewound:
    """A file object whose start was read already: read() gives those bytes again, then the rest of the file."""

    def __init__(self, head: bytes, stream: IO[bytes]) -> None:
        self._head = head
        self._stream = stream

    def read(self, size: Optional[int] = -1) -> bytes:
        if not self._head:
            return self._stream.read(-1 if size is None else size)
        if size is None or size < 0:
            data, self._head = self._head + (self._stream.read() or b""), b""
        else:
            data, self._head = self._head[:size], self._head[size:]
        return data


def _index_at_end(head: Union[bytes, memoryview]) -> bool:
    """Whether *head*, the start of a file, is an MP4, M4A, MOV or 3GP file whose index (``moov`` box) may come after
    its audio (``mdat`` box), so that ffmpeg can decode it only from a file it can seek in, not from a pipe.

    Files with the index first, as streaming tools write them, are False. So is any other format. When *head* ends
    before either box, the index is taken to come last.
    """
    offset = 0
    while offset + 8 <= len(head):
        size, kind = int.from_bytes(head[offset : offset + 4], "big"), bytes(head[offset + 4 : offset + 8])
        if offset == 0 and kind not in _MP4_FIRST_BOXES:
            return False
        if kind in (b"moov", b"mdat"):
            return kind == b"mdat"
        if size == 1 and offset + 16 <= len(head):  # a 64-bit size follows the type
            size = int.from_bytes(head[offset + 8 : offset + 16], "big")
        if size < 8:  # 0: the box runs to the end of the file; less: a damaged file. No index before it either way.
            return True
        offset += size
    return offset > 0


def _load_from_temporary_file(head: Union[bytes, memoryview], rest: Optional[IO[bytes]] = None) -> "np.ndarray":
    """Decode *head* and then the *rest* of a file object from a temporary file, deleted afterwards."""
    with _temporary_file(".mp4") as path:
        with open(path, "wb") as file:
            file.write(head)
            if rest is not None:
                shutil.copyfileobj(rest, file)
        return load_audio(path)


@contextlib.contextmanager
def _temporary_file(suffix: str) -> Iterator[pathlib.Path]:
    """The path of a new, empty temporary file, deleted when the ``with`` block ends."""
    with tempfile.NamedTemporaryFile(prefix="speech_toolkit-", suffix=suffix, delete=False) as file:
        path = pathlib.Path(file.name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def _write_wav(path: pathlib.Path, samples: "np.ndarray") -> None:
    """Write *samples* (float32 at 16 kHz) to *path* as a 16-bit mono WAV file, a part at a time to save memory.

    Samples that load_audio() decoded come from 16-bit audio, so they are written back exactly.
    """
    import numpy as np

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        for start in range(0, len(samples), _WAV_PART_SAMPLES):
            part = np.rint(np.clip(samples[start : start + _WAV_PART_SAMPLES] * 32768.0, -32768, 32767))
            wav.writeframes(part.astype("<i2").tobytes())
