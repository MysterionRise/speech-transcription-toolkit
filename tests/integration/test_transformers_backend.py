"""The transformers backend with real Hub models, one per kind of model it tells apart: Whisper (openai/whisper-tiny,
also in test_cli.py and test_long_audio.py), a CTC model (wav2vec2) and a text-only sequence-to-sequence model
(Moonshine). All are small enough for the CPU."""

from __future__ import annotations

import json

import pytest

from tests.integration.helpers import BACKEND_ARGS, Sample, assert_timeline, assert_wer, require
from tests.integration.report import REPORT
from tests.integration.wer import edit_distance, normalize

SAMPLE_NAMES = ["speech", "dialogue"]
CTC_MODEL = "facebook/wav2vec2-base-960h"  # 95M parameters, English; upper case, no punctuation
TEXT_MODEL = "moonshine-ai/moonshine-tiny"  # 27M parameters, English; text only
# The highest word error rate accepted for these models; openai/whisper-tiny's is MAX_WER["transformers"]. wav2vec2
# mishears two of the speech sample's 8 words in flite's synthetic voice ("HILLO WHIRLED") and 4 of the dialogue's 27
# (WER 0.25 and 0.15): 0.4 leaves room for one more error in the short sample. Moonshine gets the speech sample right.
MODEL_MAX_WER = {CTC_MODEL: 0.4, TEXT_MODEL: 0.25}


def assert_model_wer(transcript: str, sample: Sample, model: str) -> None:
    """assert_wer() for a model other than BACKEND_ARGS's, against MODEL_MAX_WER; printed for the CI log."""
    reference = normalize(sample.text)
    errors = edit_distance(reference, normalize(transcript))
    score, limit = errors / len(reference), MODEL_MAX_WER[model]
    REPORT.note_wer(errors, len(reference))
    print(f"transformers {model} on {sample.name}: WER {score:.3f} (at most {limit}): {transcript.strip()!r}")
    assert score <= limit, (
        f"transformers {model} on {sample.name}: WER {score:.2f} is above {limit}\n"
        f"  reference:  {sample.text}\n  transcript: {transcript.strip()}"
    )


@pytest.mark.parametrize("sample_name", SAMPLE_NAMES)
def test_whisper_word_timestamps_and_language(sample_name, samples, transcribe, tmp_path):
    """Whisper's words, from its alignment heads, spell the transcript in order; the language is detected."""
    require("transformers")
    sample = samples[sample_name]
    result_file = tmp_path / "words.json"

    run = transcribe(sample.path, *BACKEND_ARGS["transformers"], "-q", "--word-timestamps", "-o", result_file)

    assert run.stdout == ""
    result = json.loads(result_file.read_text(encoding="utf-8"))
    assert result["language"] == "en"
    segments = result["segments"]
    assert segments and all(seg.get("words") for seg in segments), segments
    assert_timeline(((seg["start"], seg["end"]) for seg in segments), sample.duration, "segments")
    words = [word for seg in segments for word in seg["words"]]
    assert_timeline(((word["start"], word["end"]) for word in words), sample.duration, "words")
    assert_wer("".join(word["word"] for word in words), sample, "transformers")


@pytest.mark.parametrize("sample_name", SAMPLE_NAMES)
def test_ctc_model(sample_name, samples, transcribe, tmp_path):
    """A CTC model: its word timings make the segments, and --word-timestamps keeps them."""
    require("transformers")
    sample = samples[sample_name]
    result_file = tmp_path / "result.json"

    run = transcribe(sample.path, "-b", "transformers", "-m", CTC_MODEL, "--word-timestamps", "--json", result_file)

    assert "doesn't support" not in run.stderr  # no option was skipped
    result = json.loads(result_file.read_text(encoding="utf-8"))
    assert run.stdout == result["text"].strip() + "\n"
    segments = result["segments"]
    assert segments and all(seg.get("words") for seg in segments), segments
    assert_timeline(((seg["start"], seg["end"]) for seg in segments), sample.duration, "segments")
    words = [word for seg in segments for word in seg["words"]]
    assert_timeline(((word["start"], word["end"]) for word in words), sample.duration, "words")
    assert_model_wer(run.stdout, sample, CTC_MODEL)
    assert_model_wer("".join(word["word"] for word in words), sample, CTC_MODEL)


def test_text_only_model(speech, transcribe, tmp_path):
    """A model without timestamps gives one segment per chunk, and word timestamps are skipped with a warning."""
    require("transformers")
    result_file = tmp_path / "result.json"

    run = transcribe(speech.path, "-b", "transformers", "-m", TEXT_MODEL, "--word-timestamps", "--json", result_file)

    assert "Warning: the transformers backend doesn't support word timestamps; ignoring it." in run.stderr
    result = json.loads(result_file.read_text(encoding="utf-8"))
    [segment] = result["segments"]  # the clip is one chunk
    assert segment["start"] == 0.0 and segment["end"] == pytest.approx(speech.duration, abs=0.01)
    assert "words" not in segment
    assert result["language"] is None
    assert_model_wer(run.stdout, speech, TEXT_MODEL)
