"""Speaker labels with pyannote.audio on the two-voice dialogue.

pyannote's model is gated, so the test needs HF_TOKEN from an account that accepted the terms of
https://hf.co/pyannote/speaker-diarization-community-1; CI passes the HF_TOKEN secret when it is set.
With INTEGRATION_DEVICE=cuda, faster-whisper and pyannote share the GPU in one process.
"""

from __future__ import annotations

from tests.integration.helpers import BACKEND_ARGS, check_speaker_labels, require, require_hf_token


def test_speaker_labels_follow_the_turns(dialogue, transcribe, tmp_path):
    require_hf_token()
    require("diarize", "faster-whisper")

    check_speaker_labels(dialogue, BACKEND_ARGS["faster-whisper"], transcribe, tmp_path)
