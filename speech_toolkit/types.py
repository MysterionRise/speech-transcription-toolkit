"""The shape of a transcription result: :class:`Segment` and :class:`Word` dicts, and the JSON schema version.

``TranscriptionResult.segments`` (and ``speaker_segments``) hold :class:`Segment` dicts, and a segment's ``"words"``
holds :class:`Word` dicts. Both are ``TypedDict`` classes: plain dicts at run time, whose keys type checkers know.
Backends can build their segments with them::

    from speech_toolkit import Segment

    segment: Segment = {"id": 0, "start": 0.0, "end": 1.5, "text": " Hello."}

``TranscriptionResult.segments`` itself stays typed as ``List[Dict[str, Any]]`` for now, so code that reads
backend-specific keys, or passes segments on as plain dicts, keeps type-checking.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict

# The version of the layout of TranscriptionResult.to_dict(), and so of the JSON output (its "schema_version").
# It goes up only when a change would make older versions misread a result, not when a key is added.
# TranscriptionResult.from_dict() reads this version and the JSON written before the key existed.
SCHEMA_VERSION = 1


class _WordTiming(TypedDict):
    word: str
    start: float
    end: float


class Word(_WordTiming, total=False):
    """One word of a segment, with its times in seconds (Whisper's format)::

        {"word": " Hello", "start": 0.0, "end": 0.42, "probability": 0.98}

    ``word`` keeps the space in front of it. ``probability``, the model's confidence from 0 to 1, is optional.
    """

    probability: Optional[float]


class _SegmentTiming(TypedDict):
    id: int
    start: float
    end: float
    text: str


class Segment(_SegmentTiming, total=False):
    """A stretch of the transcript, with its times in seconds::

        {"id": 0, "start": 0.0, "end": 3.2, "text": " Hello world."}

    ``id`` numbers a result's segments from 0. The other keys are optional:

    - ``words``: the segment's words with their times, when word timestamps are on.
    - ``speaker``: the speaker label, such as ``"SPEAKER_00"``, in ``TranscriptionResult.speaker_segments``.
    - ``tokens``, ``temperature``, ``avg_logprob``, ``compression_ratio`` and ``no_speech_prob``: Whisper's decoding
      details, from the whisper and faster-whisper backends.

    A backend can add keys of its own; to type them, subclass :class:`Segment` with ``total=False``.
    """

    words: List[Word]
    speaker: str
    tokens: Optional[List[int]]
    temperature: Optional[float]
    avg_logprob: Optional[float]
    compression_ratio: Optional[float]
    no_speech_prob: Optional[float]
