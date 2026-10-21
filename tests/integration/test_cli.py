"""The transcribe command with real models: accuracy, timestamps, output formats, batch mode and accuracy options."""

from __future__ import annotations

import json
import shutil

import pytest

from tests.integration.helpers import BACKEND_ARGS, assert_timeline, assert_wer, parse_cues, require

SAMPLE_NAMES = ["speech", "dialogue"]


@pytest.mark.parametrize("backend", list(BACKEND_ARGS))
@pytest.mark.parametrize("sample_name", SAMPLE_NAMES)
def test_transcript_on_stdout(backend, sample_name, samples, transcribe, tmp_path):
    """The transcript is accurate, and it is all that goes to stdout: progress goes to stderr."""
    require(backend)
    sample = samples[sample_name]
    result_file = tmp_path / "result.json"

    run = transcribe(sample.path, *BACKEND_ARGS[backend], "--json", result_file)  # not --quiet: progress is shown

    result = json.loads(result_file.read_text(encoding="utf-8"))
    assert f"Loading {backend} model" in run.stderr
    assert run.stdout == result["text"].strip() + "\n"
    assert_wer(run.stdout, sample, backend)
    assert_timeline(((seg["start"], seg["end"]) for seg in result["segments"]), sample.duration, "segments")


@pytest.mark.parametrize("backend", ["whisper", "faster-whisper", "parakeet"])
@pytest.mark.parametrize("sample_name", SAMPLE_NAMES)
def test_word_timestamps(backend, sample_name, samples, transcribe, tmp_path):
    """--word-timestamps gives every segment its words, in order, inside the audio, and spelling the transcript."""
    require(backend)
    sample = samples[sample_name]
    result_file = tmp_path / "words.json"

    run = transcribe(sample.path, *BACKEND_ARGS[backend], "-q", "--word-timestamps", "-o", result_file)

    assert run.stdout == ""
    segments = json.loads(result_file.read_text(encoding="utf-8"))["segments"]
    assert segments and all(seg.get("words") for seg in segments if seg["text"].strip()), segments
    words = [word for seg in segments for word in seg.get("words", [])]
    assert_timeline(((word["start"], word["end"]) for word in words), sample.duration, "words")
    assert_wer("".join(word["word"] for word in words), sample, backend)  # words glued together (#27) show here


def test_subtitles_and_json(speech, transcribe, tmp_path):
    """-o picks SRT from the extension, and --json writes the full result from the same run."""
    require("whisper")
    srt, result_file = tmp_path / "out.srt", tmp_path / "out.json"

    run = transcribe(speech.path, *BACKEND_ARGS["whisper"], "-q", "-o", srt, "--json", result_file)

    assert run.stdout == ""
    cues = parse_cues(srt.read_text(encoding="utf-8"))
    assert_timeline(((cue.start, cue.end) for cue in cues), speech.duration, "cues")
    assert_wer(" ".join(cue.text for cue in cues), speech, "whisper")
    result = json.loads(result_file.read_text(encoding="utf-8"))
    assert result["language"] == "en"
    assert_wer(result["text"], speech, "whisper")


def test_batch_mode(speech, transcribe, tmp_path):
    """A folder in, one WebVTT file per audio file out."""
    require("whisper")
    folder, subs = tmp_path / "batch", tmp_path / "subs"
    folder.mkdir()
    for name in ("one", "two"):
        shutil.copy(speech.path, folder / f"{name}{speech.path.suffix}")

    run = transcribe(folder, *BACKEND_ARGS["whisper"], "-q", "--outdir", subs, "-f", "vtt")

    assert run.stdout == ""
    assert sorted(path.name for path in subs.iterdir()) == ["one.vtt", "two.vtt"]
    for path in sorted(subs.iterdir()):
        vtt = path.read_text(encoding="utf-8")
        assert vtt.startswith("WEBVTT\n")
        cues = parse_cues(vtt)
        assert_timeline(((cue.start, cue.end) for cue in cues), speech.duration, f"{path.name} cues")
        assert_wer(" ".join(cue.text for cue in cues), speech, "whisper")


def test_accuracy_options_and_subtitle_lines(speech, transcribe, tmp_path):
    """--vad and --prompt work with faster-whisper, and --max-line-width 20 splits the cues into short lines."""
    require("faster-whisper")
    srt = tmp_path / "lines.srt"

    run = transcribe(
        speech.path,
        *BACKEND_ARGS["faster-whisper"],
        "-q",
        "--vad",
        "--prompt",
        "Speech recognition.",
        "--max-line-width",
        "20",
        "-o",
        srt,
    )

    assert run.stdout == ""
    assert "doesn't support" not in run.stderr  # faster-whisper takes all three options
    cues = parse_cues(srt.read_text(encoding="utf-8"))
    assert len(cues) >= 2  # 47 characters at 20 per line, 2 lines per cue
    for cue in cues:
        lines = cue.text.splitlines()
        assert 1 <= len(lines) <= 2 and all(len(line) <= 20 for line in lines), cue.text
    assert_timeline(((cue.start, cue.end) for cue in cues), speech.duration, "cues")
    assert_wer(" ".join(cue.text for cue in cues), speech, "faster-whisper")
