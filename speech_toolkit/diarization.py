"""Speaker diarization with pyannote.audio, and merging speaker turns into transcript segments."""

from __future__ import annotations

import bisect
import itertools
import operator
import os
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import BackendUnavailableError, DiarizationError, UnsupportedOptionError
from .media import SAMPLE_RATE, load_audio

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

# Audio shorter than this gets no speaker turns, and pyannote isn't run on it: there is nothing to tell apart, and
# pyannote can fail on near-empty audio. Its segments are labelled "unknown".
MIN_DIARIZATION_SECONDS = 0.5

# Smoothing: inside a segment, a speaker run (consecutive words with the same speaker) of fewer than MIN_RUN_WORDS
# words, or shorter than MIN_RUN_SECONDS, is taken for a diarization error when the same speaker talks right before
# and after it, and its words get that speaker. Set both to 0 to keep every speaker change.
MIN_RUN_WORDS = 2
MIN_RUN_SECONDS = 0.5

# Whisper's decoding details describe a whole segment, so the pieces of a split segment leave them out.
WHOLE_SEGMENT_FIELDS = frozenset(
    {"tokens", "avg_logprob", "compression_ratio", "no_speech_prob", "seek", "temperature"}
)

_Turn = Tuple[float, float, str]  # (start, end, speaker)


def load_diarization_pipeline(device: Optional[str] = None) -> Any:
    """Load the pyannote speaker-diarization pipeline (imported lazily: it is slow to import).

    Raises:
        BackendUnavailableError: If pyannote.audio is not installed.
        DiarizationError: If the gated model can't be downloaded (terms not accepted, or no token).
    """
    # pyannote.audio 4 sends usage metrics to pyannote.ai by default; stay offline unless the user opted in.
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise BackendUnavailableError(
            f"speaker diarization needs pyannote.audio ({e}). "
            'Install with: pip install "speech-transcription-toolkit[diarize]"'
        ) from e

    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=os.getenv("HF_TOKEN"))
    if pipeline is None:  # pyannote returns None when the gated model can't be downloaded
        raise DiarizationError(
            f"could not download '{DIARIZATION_MODEL}'. Accept its terms at https://hf.co/{DIARIZATION_MODEL} "
            "and set HF_TOKEN (or pass --hf-token)."
        )
    pipeline.to(torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")))
    return pipeline


def check_speaker_hints(
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    *,
    diarize: bool,
    flags: bool = False,
) -> None:
    """Raise UnsupportedOptionError unless the speaker-count hints can be honoured (cheap: call it before loading).

    The hints need diarization and must be positive integers. ``min_speakers`` can't be more than ``max_speakers``,
    and ``num_speakers``, given with either, must lie between them. With *flags*, messages name the command-line
    options (``--num-speakers``) instead of the keyword arguments (``num_speakers``).
    """

    def name(hint: str) -> str:
        return f"--{hint.replace('_', '-')}" if flags else hint

    hints = {"num_speakers": num_speakers, "min_speakers": min_speakers, "max_speakers": max_speakers}
    given = {hint: value for hint, value in hints.items() if value is not None}
    if given and not diarize:
        names = f"{name('num_speakers')}, {name('min_speakers')} and {name('max_speakers')}"
        raise UnsupportedOptionError(f"{names} need {'--diarize' if flags else 'diarize=True'}")
    for hint, value in given.items():
        try:  # operator.index takes ints and NumPy integers, but not 2.5 or "2"
            positive = not isinstance(value, bool) and operator.index(value) > 0
        except TypeError:
            positive = False
        if not positive:
            raise UnsupportedOptionError(f"{name(hint)} must be a positive integer, got {value!r}")
    bounds = (("min_speakers", "max_speakers"), ("min_speakers", "num_speakers"), ("num_speakers", "max_speakers"))
    for low, high in bounds:  # low can't be more than high
        low_value, high_value = hints[low], hints[high]
        if low_value is not None and high_value is not None and low_value > high_value:
            raise UnsupportedOptionError(f"{name(low)} ({low_value}) can't be more than {name(high)} ({high_value})")


def diarize_audio(
    audio_path: pathlib.Path,
    pipeline: Any,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[Tuple[float, float, str]]:
    """Return list of (start, end, speaker_label): none for audio shorter than MIN_DIARIZATION_SECONDS."""
    import torch

    # Decode with ffmpeg here so pyannote needs no audio I/O backend of its own, whatever the format.
    audio = load_audio(audio_path)
    if len(audio) < MIN_DIARIZATION_SECONDS * SAMPLE_RATE:  # too short to tell speakers apart
        return []
    waveform = torch.from_numpy(audio).unsqueeze(0)
    output = pipeline(
        {"waveform": waveform, "sample_rate": SAMPLE_RATE},
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    # pyannote 4 returns both variants; the "exclusive" one has no overlapping turns, which suits transcripts.
    annotation = getattr(output, "exclusive_speaker_diarization", output)
    return [(turn.start, turn.end, speaker) for turn, _, speaker in annotation.itertracks(yield_label=True)]


def merge_diarization(
    transcription_result: Dict[str, Any],
    spk_segments: List[Tuple[float, float, str]],
) -> List[Dict[str, Any]]:
    """Attach to each segment the speaker who talks the most during it.

    Segments with word timings (``"words"``) are labelled word by word instead and split wherever
    the speaker changes, so a quick reply inside one segment gets its own line:

    - a word in a pause between turns gets the speaker of the nearer turn;
    - a speaker change too short to be real between two stretches of one speaker is smoothed out (see
      ``MIN_RUN_WORDS`` and ``MIN_RUN_SECONDS``), so one mislabelled word doesn't split a line.

    Each returned segment's ``id`` is its position in the list, and ``segment_id`` is the ``id`` of the transcript
    segment it comes from (its position, if it has none). The pieces of a split segment leave out the fields that
    describe the whole segment, such as ``tokens`` (``WHOLE_SEGMENT_FIELDS``); other segments keep every field.
    """
    turns = _TurnIndex(spk_segments)
    output: List[Dict[str, Any]] = []
    for index, seg in enumerate(transcription_result.get("segments", [])):
        segment_id = index if seg.get("id") is None else seg["id"]
        for piece in _speaker_pieces(seg, turns):
            output.append({**piece, "id": len(output), "segment_id": segment_id})
    return output


def _speaker_pieces(seg: Dict[str, Any], turns: _TurnIndex) -> List[Dict[str, Any]]:
    """*seg* with a ``speaker`` label, or the pieces it splits into where the speaker changes."""
    if "start" not in seg or "end" not in seg:
        return [{**seg, "speaker": "unknown"}]
    speaker = turns.speaker(seg["start"], seg["end"]) or "unknown"
    words = seg.get("words") or []
    if not words or not all("start" in w and "end" in w for w in words):
        return [{**seg, "speaker": speaker}]
    # A word in a pause between turns gets the nearer turn's speaker.
    labels = [turns.speaker(w["start"], w["end"]) or turns.nearest(w["start"], w["end"], speaker) for w in words]
    labels = _smooth(words, labels)
    runs = _runs(labels)
    if len(runs) == 1:
        return [{**seg, "speaker": labels[0]}]
    whole = {key: value for key, value in seg.items() if key not in WHOLE_SEGMENT_FIELDS}
    pieces = []
    for label, first, stop in runs:
        part = words[first:stop]
        start, end, text = part[0]["start"], part[-1]["end"], "".join(w["word"] for w in part)
        pieces.append({**whole, "start": start, "end": end, "text": text, "words": part, "speaker": label})
    return pieces


def _runs(labels: Sequence[str]) -> List[Tuple[str, int, int]]:
    """The runs of equal labels, as ``(label, first index, index after the last)``."""
    runs: List[Tuple[str, int, int]] = []
    first = 0
    for label, group in itertools.groupby(labels):
        stop = first + sum(1 for _ in group)
        runs.append((label, first, stop))
        first = stop
    return runs


def _smooth(words: Sequence[Dict[str, Any]], labels: Sequence[str]) -> List[str]:
    """*labels* with each short run between two runs of one speaker given that speaker (see MIN_RUN_WORDS)."""
    smoothed = list(labels)
    runs = _runs(labels)
    for (_, first, stop), (next_label, _, _) in zip(runs[1:-1], runs[2:]):
        short = stop - first < MIN_RUN_WORDS or words[stop - 1]["end"] - words[first]["start"] < MIN_RUN_SECONDS
        if short and smoothed[first - 1] == next_label:
            smoothed[first:stop] = [next_label] * (stop - first)
    return smoothed


class _TurnIndex:
    """Speaker turns sorted by start, with lookups that bisect to the turns near a time instead of trying them all."""

    def __init__(self, turns: Sequence[_Turn]) -> None:
        self.turns = sorted(turns, key=lambda turn: turn[0])
        self.starts = [start for start, _, _ in self.turns]
        # reach[i] is the latest end among turns[: i + 1], and latest[i] the turn that ends then. reach never
        # decreases, so the turns before the first i with reach[i] > t all end by t, even when turns overlap.
        self.reach: List[float] = []
        self.latest: List[int] = []
        for i, (_, end, _) in enumerate(self.turns):
            if not self.reach or end > self.reach[-1]:
                self.reach.append(end)
                self.latest.append(i)
            else:
                self.reach.append(self.reach[-1])
                self.latest.append(self.latest[-1])

    def talk_time(self, start: float, end: float) -> Dict[str, float]:
        """Seconds each speaker talks during [start, end]: only speakers who do, in the order of their first turn."""
        talk: Dict[str, float] = {}
        for i in range(bisect.bisect_right(self.reach, start), bisect.bisect_left(self.starts, end)):
            turn_start, turn_end, speaker = self.turns[i]
            shared = min(end, turn_end) - max(start, turn_start)
            if shared > 0:
                talk[speaker] = talk.get(speaker, 0.0) + shared
        return talk

    def speaker(self, start: float, end: float) -> Optional[str]:
        """Speaker with the most overlap with [start, end]; else the one whose turn contains its midpoint; else None."""
        talk = self.talk_time(start, end)
        if talk:
            return max(talk, key=lambda label: talk[label])
        middle = (start + end) / 2.0
        first = bisect.bisect_left(self.reach, middle)  # the first turn that ends at or after the midpoint
        if first < len(self.turns) and self.starts[first] <= middle:
            return self.turns[first][2]
        return None

    def nearest(self, start: float, end: float, preferred: str) -> str:
        """Speaker of the turn nearest to [start, end], by the length of the gap between them.

        A tie between the nearest turns before and after goes to *preferred* if it is one of them, else to the one
        before. Without any turns, *preferred* is returned.
        """
        after = bisect.bisect_left(self.starts, end)  # the first turn that starts once [start, end] is over
        nearest: List[Tuple[float, str]] = []  # (gap, speaker) of the nearest turn before and the one after
        if after > 0:
            before = self.turns[self.latest[after - 1]]
            nearest.append((max(start - before[1], 0.0), before[2]))
        if after < len(self.turns):
            nearest.append((self.starts[after] - end, self.turns[after][2]))
        if not nearest:
            return preferred
        closest = min(gap for gap, _ in nearest)
        speakers = [speaker for gap, speaker in nearest if gap == closest]
        return preferred if preferred in speakers else speakers[0]
