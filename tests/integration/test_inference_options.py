"""Inference options with real models (#36): faster-whisper's batched pipeline and greedy decoding, Whisper's controls
against made-up text, and Parakeet transcribing its chunks in batches.

Each run gets the checks of test_long_audio.py (timestamps in order and inside the audio, every chunk's text, the word
error rate) or the word error rate of the speech sample: the options must not cost accuracy. The full tier
(INTEGRATION_TIER=full) also checks that faster-whisper turbo writes nothing in silence at the end of the long clip
with condition_on_previous_text=false.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess
from typing import Callable

import numpy as np
import pytest

from tests.integration.devices import TIER, require_memory
from tests.integration.helpers import (
    BACKEND_ARGS,
    Sample,
    assert_wer,
    check_long_audio,
    join_audio,
    require,
    write_wav,
)

Transcribe = Callable[..., subprocess.CompletedProcess]
# Seconds of silence after the long clip in which faster-whisper turbo made up text (#55).
TRAILING_SILENCE = (1.0, 3.0)


def test_faster_whisper_batched_and_greedy(long_clip: Sample, transcribe: Transcribe, tmp_path: pathlib.Path) -> None:
    """--batch-size runs the batched pipeline, which cuts the clip at its pauses; word timings stay in order."""
    require("faster-whisper")

    args = (*BACKEND_ARGS["faster-whisper"], "--batch-size", "4", "--beam-size", "1")
    result = check_long_audio("faster-whisper", args, long_clip, transcribe, tmp_path)

    assert all(segment.get("words") for segment in result["segments"] if segment["text"].strip())


@pytest.mark.parametrize("backend", ["whisper", "faster-whisper"])
def test_controls_against_made_up_text(backend: str, speech: Sample, transcribe: Transcribe) -> None:
    """Beam search, no conditioning on the text before, and skipping silences around likely made-up text (which
    computes word timings) still transcribe the sample."""
    require(backend)

    run = transcribe(
        speech.path,
        *BACKEND_ARGS[backend],
        "-q",
        "--beam-size",
        "2",
        "-O",
        "condition_on_previous_text=false",
        "-O",
        "no_speech_threshold=0.5",
        "-O",
        "hallucination_silence_threshold=2",
    )

    assert_wer(run.stdout, speech, backend)


@pytest.fixture(scope="module")
def three_parakeet_chunks(long_clip: Sample, tmp_path_factory: pytest.TempPathFactory) -> Sample:
    """The long clip three times over, a second of silence between (314 s): three of Parakeet's 120 s chunks."""
    from speech_toolkit.media import SAMPLE_RATE, load_audio

    audio = load_audio(long_clip.path)
    joined, turns = join_audio([(long_clip, audio)] * 3, 1.0, SAMPLE_RATE)
    path = tmp_path_factory.mktemp("longer-clip") / "longer.wav"
    write_wav(path, joined, SAMPLE_RATE)
    return Sample("longer", path, len(joined) / SAMPLE_RATE, tuple(turns))


def test_parakeet_in_batches(three_parakeet_chunks: Sample, transcribe: Transcribe, tmp_path: pathlib.Path) -> None:
    """Two chunks in the first batch, padded to the longer one, and the third alone: each chunk's words land at its
    own place in the timeline, and none is cut off."""
    require("parakeet")

    args = (*BACKEND_ARGS["parakeet"], "--batch-size", "2")
    check_long_audio("parakeet", args, three_parakeet_chunks, transcribe, tmp_path)


@pytest.mark.skipif(TIER != "full", reason="full tier: set INTEGRATION_TIER=full")
@pytest.mark.parametrize("silence", TRAILING_SILENCE)
def test_turbo_writes_nothing_in_trailing_silence(
    silence: float, long_clip: Sample, transcribe: Transcribe, tmp_path: pathlib.Path
) -> None:
    """faster-whisper turbo made up text in silence after the long clip (#55), such as "All right, thank you to the
    text test. Thank you." after 3 s. Without the text before as a prompt (condition_on_previous_text=false), it
    writes none."""
    require_memory(("faster-whisper", "turbo"))
    require("faster-whisper")
    from speech_toolkit.media import SAMPLE_RATE, load_audio

    audio = load_audio(long_clip.path)
    padded = np.concatenate([audio, np.zeros(int(silence * SAMPLE_RATE), np.float32)])
    path = tmp_path / "trailing-silence.wav"
    write_wav(path, padded, SAMPLE_RATE)
    clip = dataclasses.replace(long_clip, path=path, duration=len(padded) / SAMPLE_RATE)
    result_file = tmp_path / "trailing-silence.json"

    run = transcribe(
        clip.path,
        "-b",
        "faster-whisper",
        "-m",
        "turbo",
        "-q",
        "--json",
        result_file,
        "-O",
        "condition_on_previous_text=false",
    )

    segments = json.loads(result_file.read_text(encoding="utf-8"))["segments"]
    made_up = [seg["text"] for seg in segments if seg["start"] >= long_clip.duration - 0.1 and seg["text"].strip()]
    assert not made_up, f"made-up text in the {silence} s of silence at the end: {made_up}"
    assert_wer(run.stdout, clip, "faster-whisper")
