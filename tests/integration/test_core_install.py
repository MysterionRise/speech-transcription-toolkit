"""transcribe with no backend chosen: the default, faster-whisper, needs nothing beyond a plain install (no PyTorch).

CI's Core Install job runs this file in an environment made by ``pip install .`` alone, plus pytest.
"""

from __future__ import annotations

from tests.integration.helpers import assert_wer, require


def test_default_backend(speech, transcribe):
    require("faster-whisper")

    run = transcribe(speech.path, "-m", "tiny")  # no -b; tiny rather than the default turbo, to download less

    assert "Loading faster-whisper model 'tiny'" in run.stderr
    assert_wer(run.stdout, speech, "faster-whisper")
