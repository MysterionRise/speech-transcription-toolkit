"""load_audio(channels=...): mono by default, or several channels as a (samples, channels) array (ogg2wav uses it).

subprocess.run is mocked, so these tests need no ffmpeg; tests/test_convert.py converts real stereo files.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import numpy as np
import pytest

from speech_toolkit.media import load_audio

# What ffmpeg writes for two stereo frames: (0.0, 0.5) and (-1.0, 0.25), interleaved.
STEREO_PCM = np.array([0, 16384, -32768, 8192], dtype=np.int16).tobytes()


def ffmpeg_writes(stdout: bytes):
    return patch(
        "speech_toolkit.media.subprocess.run",
        return_value=subprocess.CompletedProcess(args=["ffmpeg"], returncode=0, stdout=stdout, stderr=b""),
    )


def test_mono_by_default():
    with ffmpeg_writes(STEREO_PCM) as run:
        audio = load_audio("talk.ogg")

    cmd = run.call_args.args[0]
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert audio.shape == (4,)


def test_stereo():
    with ffmpeg_writes(STEREO_PCM) as run:
        audio = load_audio("talk.ogg", sr=48000, channels=2)

    cmd = run.call_args.args[0]
    assert cmd[cmd.index("-ac") + 1] == "2"
    assert cmd[cmd.index("-ar") + 1] == "48000"
    assert audio.dtype == np.float32
    np.testing.assert_array_equal(audio, [[0.0, 0.5], [-1.0, 0.25]])


@pytest.mark.parametrize("channels", [0, -1, 1.5, "2", None])
def test_rejects_bad_channels(channels):
    with ffmpeg_writes(STEREO_PCM) as run:
        with pytest.raises(ValueError, match="channels must be a positive whole number"):
            load_audio("talk.ogg", channels=channels)
    run.assert_not_called()
