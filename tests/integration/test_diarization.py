"""Speaker labels with pyannote.audio on the two-voice dialogue.

pyannote's model is gated, so the test needs HF_TOKEN from an account that accepted the terms of
https://hf.co/pyannote/speaker-diarization-community-1; CI passes the HF_TOKEN secret when it is set.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, List, Sequence, Set

import pytest

from tests.integration.helpers import BACKEND_ARGS, Turn, assert_timeline, require


def labels_by_voice(turns: Sequence[Turn], segments: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """The speaker labels each voice got: per turn, the label of the segments that cover most of it."""
    labels: Dict[str, Set[str]] = {}
    for turn in turns:
        overlap: Counter = Counter()
        for seg in segments:
            shared = min(seg["end"], turn.end) - max(seg["start"], turn.start)
            if shared > 0:
                overlap[seg["speaker"]] += shared
        assert overlap, f"no speaker segment during {turn.text!r} ({turn.start}-{turn.end} s)"
        labels.setdefault(turn.voice, set()).add(overlap.most_common(1)[0][0])
    return labels


def test_speaker_labels_follow_the_turns(dialogue, transcribe, tmp_path):
    if not os.getenv("HF_TOKEN"):
        pytest.skip("needs HF_TOKEN with access to pyannote/speaker-diarization-community-1 (a gated model)")
    require("diarize", "faster-whisper")
    txt, result_file = tmp_path / "dialogue.txt", tmp_path / "dialogue.json"
    options = ["-q", "--diarize", "--num-speakers", "2", "-o", txt, "--json", result_file]

    run = transcribe(dialogue.path, *BACKEND_ARGS["faster-whisper"], *options)

    assert run.stdout == ""
    segments = json.loads(result_file.read_text(encoding="utf-8"))["speaker_segments"]
    assert_timeline(((seg["start"], seg["end"]) for seg in segments), dialogue.duration, "speaker segments")
    labels = labels_by_voice(dialogue.turns, segments)
    print(f"speaker labels by voice: {labels}")
    assert all(len(found) == 1 for found in labels.values()), f"a voice got several speaker labels: {labels}"
    speakers = set().union(*labels.values())
    assert len(speakers) == 2, f"both voices got the same speaker label: {labels}"
    transcript = txt.read_text(encoding="utf-8")  # one "[SPEAKER_00] text" line per speaker turn
    assert all(f"[{speaker}] " in transcript for speaker in speakers), transcript
