"""Messages with real models: -q hides the progress bars of Hugging Face and transformers, -v adds debug messages, and
stdout carries only the transcript either way."""

from __future__ import annotations

from tests.integration.helpers import BACKEND_ARGS, assert_wer, require

# What progress bars leave on stderr: tqdm's bar ("45%|####  |") and transformers' "Loading weights" bar.
BAR_MARKS = ("%|", "Loading weights")


def test_quiet_hides_progress_bars(speech, transcribe):
    """Parakeet loads through transformers, which shows a "Loading weights" bar on every load unless -q hides it."""
    require("parakeet")

    run = transcribe(speech.path, *BACKEND_ARGS["parakeet"], "-q")

    assert_wer(run.stdout, speech, "parakeet")
    assert not [mark for mark in BAR_MARKS if mark in run.stderr], run.stderr
    assert "Parakeet: transcribed" not in run.stderr  # its progress, an info message


def test_verbose_shows_debug_messages(speech, transcribe):
    require("faster-whisper")

    run = transcribe(speech.path, *BACKEND_ARGS["faster-whisper"], "-v")

    assert_wer(run.stdout, speech, "faster-whisper")  # stdout is still only the transcript
    assert f"Debug: Transcribing {speech.path} with faster-whisper 'tiny'" in run.stderr
    assert "Debug: Transcribed in " in run.stderr
