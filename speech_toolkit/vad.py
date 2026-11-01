"""Voice activity detection (VAD): where a recording has speech, so that the backends skip the silence.

With ``--vad`` (``vad=True``), every built-in backend transcribes only the speech, and its timestamps stay those of the
recording:

- **whisper** transcribes the stretches of speech joined end to end, and :class:`SpeechMap` moves its timestamps back.
- **voxtral, parakeet, canary** cut their chunks at the edges of the speech (:func:`speech_toolkit.media.split_audio`
  with *regions*), so no chunk holds the silence between two stretches of speech.
- **faster-whisper** keeps its built-in filter, which runs the same model with the same settings.

Audio without speech never reaches the model: the result is empty.

The detector is Silero VAD, the ONNX model that faster-whisper ships and runs with onnxruntime, so it needs no other
package; it is imported on first use. It runs on the CPU, at about 10 seconds per hour of audio. Three settings tune
it, which every built-in backend declares as options (:data:`VAD_OPTIONS`): ``-O vad_min_silence=0.5`` on the command
line, ``options={"vad_min_silence": 0.5}`` in Python.
"""

from __future__ import annotations

import bisect
import logging
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, MutableMapping, NamedTuple, Optional, Sequence

from .backends.config import BackendOption
from .errors import BackendUnavailableError
from .media import SAMPLE_RATE

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = logging.getLogger(__name__)

# The defaults of the settings: faster-whisper's, so its --vad works as it did before the other backends had one.
THRESHOLD = 0.5
MIN_SILENCE = 2.0  # seconds
PADDING = 0.4  # seconds

VAD_OPTIONS: Mapping[str, BackendOption] = {
    "vad_threshold": BackendOption(
        float,
        THRESHOLD,
        "With --vad: how sure, from 0 to 1, the detector must be that a 32 ms frame is speech; higher skips more noise",
        minimum=0.0,
        maximum=1.0,
    ),
    "vad_min_silence": BackendOption(
        float,
        MIN_SILENCE,
        "With --vad: seconds of silence that end a stretch of speech; shorter pauses stay in it",
        minimum=0.0,
    ),
    "vad_padding": BackendOption(
        float,
        PADDING,
        "With --vad: seconds of audio kept before and after each stretch of speech",
        minimum=0.0,
    ),
}
"""The options every built-in backend declares for ``--vad`` (``-O vad_min_silence=0.5``); a custom backend can add
them to its ``OPTIONS``, and :func:`settings` reads them back as :func:`detect_speech`'s settings."""

# has_speech() looks at this many seconds at a time, so that it stops at the first speech. Each look starts a second
# before the previous one ended, so the model has warmed up when it reaches audio it hasn't seen.
_SCAN_SECONDS = 60.0
_SCAN_OVERLAP_SECONDS = 1.0


class SpeechRegion(NamedTuple):
    """A stretch of speech: where it starts and ends in the recording, in seconds."""

    start: float
    end: float


def settings(options: Mapping[str, Any]) -> Dict[str, float]:
    """The VAD settings in a backend's *options* (``vad_threshold``, ``vad_min_silence``, ``vad_padding``), as the
    keyword arguments of :func:`detect_speech`. A setting the options don't have, as a custom backend's may not, gets
    its default."""
    defaults = {"threshold": THRESHOLD, "min_silence": MIN_SILENCE, "padding": PADDING}
    return {name: float(options.get(f"vad_{name}", default)) for name, default in defaults.items()}


def faster_whisper_parameters(
    *, threshold: float = THRESHOLD, min_silence: float = MIN_SILENCE, padding: float = PADDING
) -> Dict[str, Any]:
    """faster-whisper's ``vad_parameters`` for these settings: the fields of its ``VadOptions``."""
    return {
        "threshold": threshold,
        "min_silence_duration_ms": round(min_silence * 1000),
        "speech_pad_ms": round(padding * 1000),
    }


def detect_speech(
    audio: "np.ndarray", *, threshold: float = THRESHOLD, min_silence: float = MIN_SILENCE, padding: float = PADDING
) -> List[SpeechRegion]:
    """The stretches of speech in *audio*, in order: where faster-whisper's ``--vad`` filter finds speech.

    Args:
        audio: The samples: a 1-D float array of mono audio at 16 kHz, such as
            :func:`speech_toolkit.media.load_audio` returns.
        threshold: How likely, from 0 to 1, a 32 ms frame must be speech to start a stretch of speech. A stretch ends
            where the likelihood stays below ``threshold - 0.15``.
        min_silence: Seconds of silence that end a stretch of speech; shorter pauses stay in it.
        padding: Seconds of audio added before and after each stretch, so that its first and last words stay whole.
            Stretches that the padding would make overlap meet halfway between them instead.

    Returns:
        The stretches of speech, in seconds; none for audio without speech.

    Raises:
        BackendUnavailableError: If faster-whisper, whose Silero model this runs, isn't installed. It is also an
            ``ImportError``.
        ValueError: If *audio* isn't 1-D.
    """
    samples = _samples(audio)
    timestamps = _speech_timestamps(samples, threshold, min_silence, padding)
    regions = [SpeechRegion(chunk["start"] / SAMPLE_RATE, chunk["end"] / SAMPLE_RATE) for chunk in timestamps]
    speech = sum(end - start for start, end in regions)
    total = len(samples) / SAMPLE_RATE
    logger.debug(
        "Voice activity detection: %.1f s of speech in %.1f s of audio, in %d stretches", speech, total, len(regions)
    )
    return regions


def has_speech(audio: "np.ndarray", *, threshold: float = THRESHOLD) -> bool:
    """Whether :func:`detect_speech` would find any speech in *audio* (*threshold* as there).

    It looks at a minute at a time and stops at the first speech, so a recording that starts with speech takes a
    fraction of a second; the other settings don't change whether there is any.
    """
    samples = _samples(audio)
    step, overlap = round(_SCAN_SECONDS * SAMPLE_RATE), round(_SCAN_OVERLAP_SECONDS * SAMPLE_RATE)
    for start in range(0, len(samples), step):
        if _speech_timestamps(samples[max(0, start - overlap) : start + step], threshold, MIN_SILENCE, 0.0):
            return True
    return False


def _samples(audio: "np.ndarray") -> "np.ndarray":
    """*audio* as float32 samples, which the model takes (no copy if it is already)."""
    import numpy as np  # imported lazily, as in media.load_audio(): listing backends shouldn't pay for it

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError(f"voice activity detection takes 1-D samples (mono, 16 kHz), not a {samples.ndim}-D array")
    return samples


def _speech_timestamps(samples: "np.ndarray", threshold: float, min_silence: float, padding: float) -> List[dict]:
    """faster-whisper's get_speech_timestamps(): ``{"start", "end"}`` dicts in samples."""
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError as e:
        raise BackendUnavailableError(
            f"Voice activity detection needs faster-whisper ({e}), which speech-transcription-toolkit depends on. "
            "Install it with: pip install speech-transcription-toolkit"
        ) from e
    options = VadOptions(**faster_whisper_parameters(threshold=threshold, min_silence=min_silence, padding=padding))
    return list(get_speech_timestamps(samples, options, sampling_rate=SAMPLE_RATE))


class SpeechMap:
    """The stretches of speech of a recording joined end to end, without the silence between them, and the way back.

    A backend transcribes the joined audio (:meth:`join`), then moves the times in its result to the recording's
    (:meth:`restore`), as faster-whisper's VAD filter does::

        speech = SpeechMap(detect_speech(samples))
        result = model.transcribe(speech.join(samples))
        speech.restore(result["segments"])

    Args:
        regions: The stretches of speech, as :func:`detect_speech` finds them: ``(start, end)`` pairs in seconds, in
            order and not overlapping. At least one.
    """

    def __init__(self, regions: Sequence[Sequence[float]]) -> None:
        if not regions:
            raise ValueError("SpeechMap needs at least one stretch of speech")
        self._spans = [(round(start * SAMPLE_RATE), round(end * SAMPLE_RATE)) for start, end in regions]
        self._ends: List[float] = []  # where each stretch ends in the joined audio, in seconds
        self._shifts: List[float] = []  # how much of the recording comes before each stretch and isn't joined
        joined = 0
        for start, end in self._spans:
            self._shifts.append((start - joined) / SAMPLE_RATE)
            joined += end - start
            self._ends.append(joined / SAMPLE_RATE)

    @property
    def duration(self) -> float:
        """The length of the joined audio, in seconds."""
        return self._ends[-1]

    def join(self, audio: "np.ndarray") -> "np.ndarray":
        """The stretches of speech of *audio*, the recording's samples, joined end to end."""
        import numpy as np

        return np.concatenate([audio[start:end] for start, end in self._spans])

    def region(self, time: float, *, end: bool = False) -> int:
        """Which stretch of speech (its index) the time *time* of the joined audio falls in.

        A time where two stretches meet belongs to the later one, or with *end* (the end of a segment) to the earlier
        one. Times past the end belong to the last one.
        """
        index = bisect.bisect_left(self._ends, time) if end else bisect.bisect_right(self._ends, time)
        return min(index, len(self._ends) - 1)

    def original_time(self, time: float, region: Optional[int] = None, *, end: bool = False) -> float:
        """Where the time *time* of the joined audio is in the recording, in seconds (rounded to milliseconds).

        Args:
            time: A time in the joined audio, in seconds.
            region: The stretch of speech it belongs to; by default, the one it falls in (see :meth:`region`).
            end: Whether it is the end of a segment, for a time where two stretches meet.
        """
        index = self.region(time, end=end) if region is None else region
        return round(time + self._shifts[index], 3)

    def restore(self, segments: Iterable[MutableMapping[str, Any]]) -> None:
        """Move the times of *segments*, and of their ``"words"``, from the joined audio to the recording, in place.

        A word moves with the stretch of speech its middle falls in, and a segment with words then runs from its first
        word's start to its last word's end. A segment without words moves its start and its end on their own, so it
        spans the silence left out between them if they fall in different stretches.
        """
        for segment in segments:
            words = segment.get("words")
            if words:
                for word in words:
                    index = self.region((word["start"] + word["end"]) / 2)
                    word["start"] = self.original_time(word["start"], index)
                    word["end"] = self.original_time(word["end"], index)
                segment["start"], segment["end"] = words[0]["start"], words[-1]["end"]
            else:
                segment["start"] = self.original_time(segment["start"])
                segment["end"] = max(segment["start"], self.original_time(segment["end"], end=True))
