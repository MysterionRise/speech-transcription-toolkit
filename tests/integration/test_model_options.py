"""Custom models and backend options with a real model: faster-whisper loaded by Hugging Face repo id and from a local
folder, and with -O and --compute-type.

Tiny models only: the repo id is the one faster-whisper's ``tiny`` downloads, so the model cache has it already.
"""

from __future__ import annotations

from tests.integration.helpers import BACKEND_ARGS, assert_wer, require

TINY_REPO = "Systran/faster-whisper-tiny"  # what faster-whisper downloads for "-m tiny"


def test_faster_whisper_by_repo_id(speech, transcribe):
    require("faster-whisper")

    run = transcribe(speech.path, "-b", "faster-whisper", "-m", TINY_REPO)

    assert f"Loading faster-whisper model '{TINY_REPO}'" in run.stderr
    assert_wer(run.stdout, speech, "faster-whisper")


def test_faster_whisper_from_a_local_folder(speech, transcribe):
    require("faster-whisper")
    from faster_whisper import download_model

    folder = download_model("tiny")  # the folder in the Hugging Face cache that holds TINY_REPO

    run = transcribe(speech.path, "-b", "faster-whisper", "-m", folder, "-q")

    assert_wer(run.stdout, speech, "faster-whisper")


def test_backend_options(speech, transcribe):
    """-O and --compute-type reach faster-whisper, which transcribes as well as without them."""
    require("faster-whisper")

    run = transcribe(speech.path, *BACKEND_ARGS["faster-whisper"], "-q", "-O", "cpu_threads=2", "--compute-type", "auto")

    assert_wer(run.stdout, speech, "faster-whisper")
