"""Tests for speech_toolkit.media: silence-aware chunking (ffmpeg decoding is in test_load_audio.py)."""

from __future__ import annotations

import numpy as np

from speech_toolkit.media import SAMPLE_RATE, split_audio


def _seconds(chunks):
    return [(round(start, 2), round(len(samples) / SAMPLE_RATE, 2)) for start, samples in chunks]


class TestSplitAudio:
    def test_short_audio_is_one_chunk(self):
        audio = np.ones(10 * SAMPLE_RATE, dtype=np.float32)
        assert _seconds(split_audio(audio, 30)) == [(0.0, 10.0)]

    def test_empty_audio(self):
        assert split_audio(np.zeros(0, dtype=np.float32), 30) == []

    def test_cuts_in_the_quietest_moment_before_the_limit(self):
        audio = np.full(70 * SAMPLE_RATE, 0.5, dtype=np.float32)
        audio[int(27.0 * SAMPLE_RATE) : int(27.1 * SAMPLE_RATE)] = 0.0  # a pause 3 s before the 30 s limit
        audio[int(54.0 * SAMPLE_RATE) : int(54.1 * SAMPLE_RATE)] = 0.0

        chunks = split_audio(audio, 30)

        starts = [start for start, _ in chunks]
        assert len(starts) == 3 and starts[0] == 0.0
        assert 27.0 <= starts[1] <= 27.1 and 54.0 <= starts[2] <= 54.1  # each cut falls inside its pause
        assert all(len(samples) <= 30 * SAMPLE_RATE for _, samples in chunks)
        assert sum(len(samples) for _, samples in chunks) == len(audio)  # nothing lost or repeated

    def test_long_audio_without_pauses_still_respects_the_limit(self):
        rng = np.random.default_rng(0)
        audio = rng.uniform(-0.5, 0.5, 125 * SAMPLE_RATE).astype(np.float32)

        chunks = split_audio(audio, 30)

        assert all(len(samples) <= 30 * SAMPLE_RATE for _, samples in chunks)
        assert np.array_equal(np.concatenate([samples for _, samples in chunks]), audio)
