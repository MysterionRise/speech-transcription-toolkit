"""The transcribe command's batch features with a real model: audio piped into standard input, several formats from
one transcription, --skip-existing, and folder scans that leave out the ._ files macOS writes."""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

import pytest

from tests.integration.devices import device_args
from tests.integration.helpers import (
    BACKEND_ARGS,
    COMMAND_TIMEOUT,
    EventLog,
    assert_timeline,
    assert_wer,
    instrumented_env,
    parse_cues,
    require,
    run_module,
)

BACKEND = "faster-whisper"  # a core dependency, so every install has it


def transcribe_stdin(audio: pathlib.Path, *args: object, cwd: pathlib.Path) -> subprocess.CompletedProcess:
    """Run ``transcribe - <args>`` with *audio* piped into its standard input, as the ``transcribe`` fixture runs it:
    instrumented, with INTEGRATION_DEVICE's --device. Fails the test if the command fails."""
    events = EventLog(cwd / "instrument-events.jsonl")
    command = [sys.executable, "-m", "tests.integration.instrument", "speech_toolkit", "-", *map(str, args)]
    with audio.open("rb") as stdin:
        done = subprocess.run(
            [*command, *device_args()],
            stdin=stdin,
            cwd=cwd,
            env=instrumented_env(events.path),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=COMMAND_TIMEOUT,
        )
    if done.returncode != 0:
        pytest.fail(f"transcribe - exited with {done.returncode}:\n{done.stderr}", pytrace=False)
    assert events.record(), "transcribe - loaded no model, or the instrumentation recorded none"
    return done


def test_audio_from_stdin(speech, tmp_path):
    """cat speech.flac | transcribe - : the transcript, and only the transcript, on stdout."""
    require(BACKEND)

    run = transcribe_stdin(speech.path, *BACKEND_ARGS[BACKEND], cwd=tmp_path)

    assert f"Loading {BACKEND} model" in run.stderr
    assert run.stdout.endswith("\n") and run.stdout.count("\n") == 1, run.stdout
    assert_wer(run.stdout, speech, BACKEND)


def test_stdin_to_outdir_in_several_formats(speech, tmp_path):
    """transcribe - --outdir subs -f srt,json writes subs/stdin.srt and subs/stdin.json."""
    require(BACKEND)
    subs = tmp_path / "subs"

    run = transcribe_stdin(speech.path, *BACKEND_ARGS[BACKEND], "-q", "--outdir", subs, "-f", "srt,json", cwd=tmp_path)

    assert run.stdout == ""
    assert sorted(path.name for path in subs.iterdir()) == ["stdin.json", "stdin.srt"]
    cues = parse_cues((subs / "stdin.srt").read_text(encoding="utf-8"))
    assert_timeline(((cue.start, cue.end) for cue in cues), speech.duration, "cues")
    assert_wer(" ".join(cue.text for cue in cues), speech, BACKEND)
    assert_wer(json.loads((subs / "stdin.json").read_text(encoding="utf-8"))["text"], speech, BACKEND)


def test_folder_in_several_formats_then_skip_existing(speech, transcribe, tmp_path):
    """A folder with two recordings and a macOS ._ file: two formats each, [i/N] lines and a summary; run again with
    --skip-existing, nothing is transcribed and no model loads."""
    require(BACKEND)
    folder, subs = tmp_path / "batch", tmp_path / "subs"
    folder.mkdir()
    for name in ("one", "two"):
        shutil.copy(speech.path, folder / f"{name}{speech.path.suffix}")
    # What macOS writes beside each file on a FAT or exFAT disk: metadata, not audio, so ffmpeg would fail on it.
    (folder / f"._one{speech.path.suffix}").write_bytes(b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        ")

    run = transcribe(folder, *BACKEND_ARGS[BACKEND], "--outdir", subs, "-f", "srt,json")

    assert run.stdout == ""
    assert sorted(path.name for path in subs.iterdir()) == ["one.json", "one.srt", "two.json", "two.srt"]
    assert "[1/2] " in run.stderr and "[2/2] " in run.stderr
    assert ": 2 done, 0 skipped, 0 failed." in run.stderr
    for name in ("one", "two"):
        assert_wer(json.loads((subs / f"{name}.json").read_text(encoding="utf-8"))["text"], speech, BACKEND)

    again = run_module(
        "speech_toolkit",
        folder,
        *BACKEND_ARGS[BACKEND],
        "--outdir",
        subs,
        "-f",
        "srt,json",
        "--skip-existing",
        cwd=tmp_path,
    )

    assert again.stdout == ""
    assert "Loading" not in again.stderr
    assert ": 0 done, 2 skipped, 0 failed." in again.stderr
