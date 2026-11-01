"""--vad with real models: silence before or after the speech gives no text, and the timestamps stay the recording's.

The clips are the speech sample with 10 s of silence (zeros) added before or after it at test time, rather than more
committed audio. Without --vad, Whisper models tend to fill such silence with made-up text (#55).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import List, Tuple

import pytest

from tests.integration.helpers import BACKEND_ARGS, Sample, assert_timeline, assert_wer, require, write_wav

SILENCE = 10.0  # seconds of silence added before or after the speech
FIRST_WORD_TOLERANCE = 0.3  # how far from the start of the speech its first word (or segment) may start
END_TOLERANCE = 0.3  # how far past the end of the speech and the padding --vad keeps the last word may end


def with_silence(sample: Sample, folder: pathlib.Path, name: str, before: float = 0.0, after: float = 0.0) -> Sample:
    """*sample* with *before* and *after* seconds of silence, written to *folder* as <name>.wav."""
    import numpy as np

    from speech_toolkit.media import SAMPLE_RATE, load_audio

    audio = load_audio(sample.path)
    silence = [np.zeros(round(seconds * SAMPLE_RATE), np.float32) for seconds in (before, after)]
    path = folder / f"{name}.wav"
    write_wav(path, np.concatenate([silence[0], audio, silence[1]]), SAMPLE_RATE)
    turns = tuple(dataclasses.replace(turn, start=turn.start + before, end=turn.end + before) for turn in sample.turns)
    return Sample(name, path, sample.duration + before + after, turns)


def test_silero_from_faster_whisper(speech, tmp_path):
    """The installed faster-whisper's Silero model, and the API speech_toolkit.vad uses, find the speech."""
    require("faster-whisper")
    from speech_toolkit.media import load_audio
    from speech_toolkit.vad import PADDING, detect_speech, has_speech

    clip = with_silence(speech, tmp_path, "lead", before=SILENCE, after=SILENCE)
    audio = load_audio(clip.path)

    regions = detect_speech(audio)

    start, end = clip.turns[0].start, clip.turns[-1].end
    assert len(regions) == 1, regions
    assert start - PADDING - FIRST_WORD_TOLERANCE <= regions[0].start <= start + FIRST_WORD_TOLERANCE, regions
    assert end <= regions[0].end <= end + PADDING + END_TOLERANCE, regions
    assert has_speech(audio) and not has_speech(audio[: round(SILENCE * 16000) - 16000])


@pytest.mark.parametrize("backend", list(BACKEND_ARGS))
def test_silence_gives_no_text(backend, speech, transcribe, tmp_path):
    """Both clips in one run (one model load): no word in the silence, the first one where the speech starts, and
    the transcript as accurate as without the silence."""
    require(backend)
    from speech_toolkit.backends import get_backend_class
    from speech_toolkit.vad import PADDING

    folder, out = tmp_path / "clips", tmp_path / "out"
    folder.mkdir()
    clips = [
        with_silence(speech, folder, "leading", before=SILENCE),
        with_silence(speech, folder, "trailing", after=SILENCE),
    ]
    words = "word_timestamps" in get_backend_class(backend).capabilities

    run = transcribe(
        folder,
        *BACKEND_ARGS[backend],
        "-q",
        "--vad",
        "--outdir",
        out,
        "-f",
        "json",
        *(["--word-timestamps"] if words else []),
    )

    assert run.stdout == ""
    assert "doesn't support" not in run.stderr
    for clip in clips:
        result = json.loads((out / f"{clip.name}.json").read_text(encoding="utf-8"))
        segments = [segment for segment in result["segments"] if segment["text"].strip()]
        spans: List[Tuple[float, float]] = (
            [(word["start"], word["end"]) for segment in segments for word in segment.get("words", [])]
            if words
            else [(segment["start"], segment["end"]) for segment in segments]
        )
        what = f"{clip.name}: {'words' if words else 'segments'}"
        assert_timeline(spans, clip.duration, what)
        start, end = clip.turns[0].start, clip.turns[-1].end
        print(
            f"{backend} on {clip.name}: speech {start:.2f}-{end:.2f} s, {what} {spans[0][0]:.2f}-{spans[-1][1]:.2f} s"
        )
        assert abs(spans[0][0] - start) <= FIRST_WORD_TOLERANCE, f"{what} start at {spans[0][0]:.2f} s, not {start} s"
        in_silence = [span for span in spans if span[0] < start - PADDING or span[1] > end + PADDING + END_TOLERANCE]
        assert not in_silence, f"{what} in the silence: {in_silence} (speech {start:.2f}-{end:.2f} s)"
        assert result["duration"] == pytest.approx(clip.duration, abs=0.01)
        assert_wer(result["text"], clip, backend)
