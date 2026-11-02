"""Language detection with real models (``--detect-language``, ``Transcriber.detect_language()``), and the fields a JSON
result carries with it: segment ids from 0, the duration and the probability of the language, and a result that loads
back to the same subtitles."""

from __future__ import annotations

import json
import pathlib
from typing import Any, List

import pytest

from tests.integration.devices import api_device
from tests.integration.helpers import BACKEND_ARGS, require

DETECTORS = ["whisper", "faster-whisper"]
SAMPLE_NAMES = ["speech", "dialogue"]


def transcribed_files(folder: pathlib.Path) -> List[str]:
    """The files the instrumented command transcribed (see tests/integration/instrument.py)."""
    events = folder / "instrument-events.jsonl"
    lines = events.read_text(encoding="utf-8").splitlines() if events.exists() else []
    return [event["audio"] for event in map(json.loads, lines) if event["event"] == "transcribe"]


@pytest.mark.parametrize("backend", DETECTORS)
def test_a_line_per_file_without_transcribing(backend, samples, transcribe, tmp_path):
    """file<TAB>language<TAB>probability on stdout for each file, in order, and nothing transcribed."""
    require(backend)
    paths = [samples[name].path for name in SAMPLE_NAMES]

    run = transcribe("--detect-language", *paths, *BACKEND_ARGS[backend])

    assert f"Loading {backend} model" in run.stderr
    lines = [line.split("\t") for line in run.stdout.splitlines()]
    print(f"{backend}: {lines}")
    assert [file for file, _, _ in lines] == [str(path) for path in paths]
    for file, language, probability in lines:
        assert language == "en", file
        assert 0.5 < float(probability) <= 1, file
    assert transcribed_files(tmp_path) == []


def test_json_lines(dialogue, transcribe, tmp_path):
    """-f json: the file, the language, its probability and the five most likely languages."""
    require("faster-whisper")

    run = transcribe("--detect-language", dialogue.path, *BACKEND_ARGS["faster-whisper"], "-q", "-f", "json")

    detection = json.loads(run.stdout)
    assert (detection["file"], detection["language"]) == (str(dialogue.path), "en")
    candidates = detection["candidates"]
    assert len(candidates) == 5
    assert candidates[0] == {"language": "en", "probability": detection["probability"]}
    chances = [candidate["probability"] for candidate in candidates]
    assert chances == sorted(chances, reverse=True) and sum(chances) <= 1.0001
    assert transcribed_files(tmp_path) == []


@pytest.mark.parametrize("backend", DETECTORS)
def test_the_transcriber_detects_what_it_transcribes(backend, speech, long_clip, monkeypatch):
    """detect_language() finds the language transcribe() reports, with about the probability its result carries, from
    the file, its content or its samples; of a long recording, only the first 30 s are decoded."""
    require(backend)
    from speech_toolkit import Transcriber, api
    from speech_toolkit.media import SAMPLE_RATE, load_audio

    transcriber = Transcriber(backend, "tiny", device=api_device())
    result = transcriber.transcribe(speech.path)
    detections = [
        transcriber.detect_language(speech.path),
        transcriber.detect_language(speech.path.read_bytes()),
        transcriber.detect_language(load_audio(speech.path), top=None),
    ]

    print(f"{backend}: transcribed {result.language} ({result.language_probability}), detected {detections[0]}")
    assert result.language == "en"
    assert 0.5 < result.language_probability <= 1
    for detection in detections:
        assert detection["language"] == "en"
        assert detection["probability"] == pytest.approx(result.language_probability, abs=0.05)
    assert len(detections[2]["candidates"]) > 90  # every language the model knows

    decoded: List[float] = []  # the seconds of audio each decoding gave

    def decode(*args: Any, **kwargs: Any) -> Any:
        samples = load_audio(*args, **kwargs)
        decoded.append(len(samples) / SAMPLE_RATE)
        return samples

    monkeypatch.setattr(api, "load_audio", decode)
    assert transcriber.detect_language(long_clip.path)["language"] == "en"
    assert decoded == [30.0]  # of 104 s


@pytest.mark.parametrize("backend", DETECTORS)
def test_result_fields(backend, dialogue, transcribe, tmp_path):
    """The JSON result: segment ids from 0, the duration and the language's probability, and it loads back with
    TranscriptionResult.load() to the same subtitles."""
    require(backend)
    from speech_toolkit import TranscriptionResult
    from speech_toolkit.types import SCHEMA_VERSION

    srt, result_file = tmp_path / "dialogue.srt", tmp_path / "dialogue.json"

    run = transcribe(dialogue.path, *BACKEND_ARGS[backend], "-q", "-o", srt, "--json", result_file)

    assert run.stdout == ""
    data = json.loads(result_file.read_text(encoding="utf-8"))
    print(f"{backend}: duration {data['duration']}, language {data['language']} ({data['language_probability']})")
    assert data["schema_version"] == SCHEMA_VERSION
    assert [segment["id"] for segment in data["segments"]] == list(range(len(data["segments"])))
    assert data["language"] == "en"
    assert 0.5 < data["language_probability"] <= 1
    if data["duration"] is not None or backend != "whisper":  # openai-whisper doesn't report it for a file (#23)
        assert data["duration"] == pytest.approx(dialogue.duration, abs=0.1)
    assert TranscriptionResult.load(result_file).render("srt") == srt.read_text(encoding="utf-8")
