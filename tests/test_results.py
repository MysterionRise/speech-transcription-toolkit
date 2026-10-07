"""Typed results: Segment and Word, duration and language_probability, schema_version and the JSON round trip.

The tests that run a backend replace its model libraries in ``sys.modules``, as test_backends.py does.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from types import SimpleNamespace
from typing import Any, Callable, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import speech_toolkit
from speech_toolkit import Segment, TranscriptionResult, Word
from speech_toolkit.backends import get_backend, list_backends
from speech_toolkit.server import verbose_json
from speech_toolkit.types import SCHEMA_VERSION

RATE = 16000
SEGMENT = {"id": 0, "start": 0.0, "end": 1.0, "text": " Hi."}
MISSING = object()


def segment(**changes: Any) -> Dict[str, Any]:
    """SEGMENT with *changes*; a key changed to MISSING is left out."""
    return {key: value for key, value in {**SEGMENT, **changes}.items() if value is not MISSING}


def diarized_result() -> TranscriptionResult:
    """A result with every field set: word timings, speaker segments, duration, language probability and raw."""
    words = [
        {"word": " Hello", "start": 0.0, "end": 0.4, "probability": 0.9},
        {"word": " there.", "start": 0.4, "end": 1.0, "probability": 0.8},
        {"word": " Hi.", "start": 1.2, "end": 1.6, "probability": 0.95},
    ]
    first = {"id": 0, "start": 0.0, "end": 1.6, "text": " Hello there. Hi.", "avg_logprob": -0.2, "words": words}
    second = {"id": 1, "start": 2.0, "end": 3.0, "text": " Bye.", "avg_logprob": -0.4}
    return TranscriptionResult(
        text=" Hello there. Hi. Bye.",
        segments=[first, second],
        language="en",
        raw={"model": "tiny"},  # a backend-specific field
        speaker_segments=[
            {**first, "end": 1.0, "text": " Hello there.", "words": words[:2], "speaker": "SPEAKER_00"},
            {**first, "start": 1.2, "text": " Hi.", "words": words[2:], "speaker": "SPEAKER_01"},
            {**second, "speaker": "SPEAKER_00"},
        ],
        duration=3.5,
        language_probability=0.97,
    )


class TestTypes:
    def test_exported_from_the_package(self):
        assert (speech_toolkit.Segment, speech_toolkit.Word) == (
            speech_toolkit.types.Segment,
            speech_toolkit.types.Word,
        )
        assert {"Segment", "Word"} <= set(speech_toolkit.__all__)

    def test_required_and_optional_keys(self):
        assert Segment.__required_keys__ == {"id", "start", "end", "text"}
        assert {"words", "speaker", "tokens", "avg_logprob", "no_speech_prob"} <= Segment.__optional_keys__
        assert Word.__required_keys__ == {"word", "start", "end"}
        assert Word.__optional_keys__ == {"probability"}

    def test_typed_segments_are_plain_dicts(self):
        word: Word = {"word": " Hi.", "start": 0.0, "end": 1.0}
        typed: Segment = {"id": 0, "start": 0.0, "end": 1.0, "text": " Hi.", "words": [word]}
        segments: List[Segment] = [typed]

        result = TranscriptionResult(" Hi.", segments)

        assert result.segments is segments and type(result.segments[0]) is dict
        assert TranscriptionResult.from_dict(result.to_dict()).segments == [typed]


class TestToDict:
    def test_schema_version_comes_first(self):
        data = TranscriptionResult(" Hi.", [SEGMENT]).to_dict()

        assert list(data)[0] == "schema_version"
        assert data["schema_version"] == SCHEMA_VERSION == 1
        assert json.loads(TranscriptionResult(" Hi.", [SEGMENT]).render("json"))["schema_version"] == 1

    def test_duration_and_language_probability_are_always_written(self):
        assert TranscriptionResult(" Hi.", [SEGMENT]).to_dict() == {
            "schema_version": 1,
            "text": " Hi.",
            "segments": [SEGMENT],
            "language": None,
            "duration": None,
            "language_probability": None,
        }
        data = TranscriptionResult(" Hi.", [SEGMENT], duration=1.5, language_probability=0.9).to_dict()
        assert (data["duration"], data["language_probability"]) == (1.5, 0.9)

    def test_raw_never_replaces_a_standard_field(self):
        raw = {"schema_version": 99, "text": "RAW", "duration": 5.0, "speaker_segments": [], "model": "tiny"}

        data = TranscriptionResult(" Hi.", [SEGMENT], raw=raw, duration=2.0).to_dict()

        assert (data["schema_version"], data["text"], data["duration"], data["model"]) == (1, " Hi.", 2.0, "tiny")
        assert "speaker_segments" not in data  # only diarization adds it
        assert list(data)[:2] == ["schema_version", "model"]  # backend-specific fields before the standard ones


class TestRoundTrip:
    def test_from_dict_reproduces_the_result(self):
        result = diarized_result()

        loaded = TranscriptionResult.from_dict(result.to_dict())

        assert loaded.to_dict() == result.to_dict()
        assert (loaded.text, loaded.language, loaded.duration, loaded.language_probability) == (
            " Hello there. Hi. Bye.",
            "en",
            3.5,
            0.97,
        )
        assert loaded.segments == result.segments
        assert loaded.speaker_segments == result.speaker_segments
        assert loaded.raw == {"model": "tiny"}

    @pytest.mark.parametrize("max_line_width", [None, 12])
    @pytest.mark.parametrize("fmt", ["txt", "srt", "vtt", "json"])
    def test_renders_the_same(self, fmt, max_line_width):
        result = diarized_result()
        assert TranscriptionResult.from_dict(result.to_dict()).render(fmt, max_line_width) == result.render(
            fmt, max_line_width
        )

    def test_through_a_json_file(self, tmp_path):
        result = diarized_result()

        loaded = TranscriptionResult.load(result.save(tmp_path / "talk.json"))

        assert loaded.to_dict() == result.to_dict()
        assert TranscriptionResult.load(str(tmp_path / "talk.json")).render("srt") == result.render("srt")

    def test_without_diarization(self):
        result = TranscriptionResult(" Hi.", [SEGMENT], "en")

        loaded = TranscriptionResult.from_dict(result.to_dict())

        assert loaded.speaker_segments is None
        assert loaded.to_dict() == result.to_dict()
        assert "speaker_segments" not in loaded.to_dict()

    def test_the_loaded_result_has_its_own_segments(self):
        result = diarized_result()
        loaded = TranscriptionResult.from_dict(result.to_dict())

        loaded.segments[0]["words"][0]["word"] = " Changed"
        loaded.segments[1]["text"] = " Changed."
        loaded.segments.append(segment(id=2))
        loaded.speaker_segments[0]["speaker"] = "SPEAKER_09"

        assert result.to_dict() == diarized_result().to_dict()

    def test_raw_keeps_the_backend_specific_fields(self):
        """faster-whisper also keeps duration and language_probability in raw; loaded, they are attributes only."""
        raw = {"duration": 2.0, "language_probability": 0.5, "model": "tiny"}
        result = TranscriptionResult(" Hi.", [SEGMENT], raw=raw, duration=2.0, language_probability=0.5)

        loaded = TranscriptionResult.from_dict(result.to_dict())

        assert loaded.raw == {"model": "tiny"}
        assert (loaded.duration, loaded.language_probability) == (2.0, 0.5)
        assert loaded.to_dict() == result.to_dict()

    def test_numpy_numbers_count_as_numbers(self):
        """Timings computed with numpy, before they're written to JSON."""
        timed = {"id": np.int64(0), "start": np.float32(0.0), "end": np.float64(1.0), "text": " Hi."}
        result = TranscriptionResult(" Hi.", [timed], duration=np.float32(1.0))

        assert TranscriptionResult.from_dict(result.to_dict()).segments == [timed]


class TestOtherVersions:
    def test_json_saved_before_schema_version(self, tmp_path):
        """faster-whisper's JSON from before: no schema_version, ids from 1, duration and probability at the top."""
        old = {
            "duration": 3.0,
            "language_probability": 0.98,
            "text": "Hallo Welt",
            "segments": [
                {"id": 1, "start": 0.0, "end": 1.5, "text": " Hallo", "tokens": [1], "avg_logprob": -0.2},
                {"id": 2, "start": 1.5, "end": 3.0, "text": " Welt", "tokens": [2], "avg_logprob": -0.3},
            ],
            "language": "de",
        }
        path = tmp_path / "old.json"
        path.write_text(json.dumps(old), encoding="utf-8")

        result = TranscriptionResult.load(path)

        assert (result.duration, result.language_probability, result.raw) == (3.0, 0.98, {})
        assert [segment["id"] for segment in result.segments] == [1, 2]  # kept as saved
        assert result.to_dict() == {**old, "schema_version": SCHEMA_VERSION}
        assert result.render("srt").startswith("1\n00:00:00,000 --> 00:00:01,500\nHallo\n")

    def test_whisper_json_saved_before_schema_version(self):
        result = TranscriptionResult.from_dict({"text": " Hi.", "segments": [SEGMENT], "language": "en"})
        assert (result.duration, result.language_probability, result.speaker_segments) == (None, None, None)

    def test_a_newer_schema_version_is_refused(self):
        data = {**diarized_result().to_dict(), "schema_version": SCHEMA_VERSION + 1}

        with pytest.raises(ValueError, match=f"has schema_version {SCHEMA_VERSION + 1}.*upgrade it"):
            TranscriptionResult.from_dict(data)

    @pytest.mark.parametrize("version", [0, -1, "1", 1.5, True, None])
    def test_an_invalid_schema_version(self, version):
        with pytest.raises(ValueError, match="'schema_version' must be a whole number from 1 up"):
            TranscriptionResult.from_dict({"schema_version": version, "text": "", "segments": []})


MALFORMED = [
    pytest.param([], "a transcription result must be a JSON object, not a list", id="a-list"),
    pytest.param(" Hi.", "a transcription result must be a JSON object, not a string", id="a-string"),
    pytest.param({"segments": []}, "'text' is missing", id="no-text"),
    pytest.param({"text": None, "segments": []}, "'text' must be a string, not null", id="null-text"),
    pytest.param({"text": ""}, "'segments' is missing", id="no-segments"),
    pytest.param({"text": "", "segments": {}}, "'segments' must be a list, not an object", id="segments-object"),
    pytest.param({"text": "", "segments": ()}, "'segments' must be a list, not a tuple", id="segments-tuple"),
    pytest.param({"text": "", "segments": ["Hi"]}, "'segments[0]' must be a JSON object, not a string", id="segment"),
    pytest.param({"text": "", "segments": [segment(start=MISSING)]}, "'segments[0].start' is missing", id="no-start"),
    pytest.param(
        {"text": "", "segments": [segment(), segment(end="1.0")]},
        "'segments[1].end' must be a number, not a string",
        id="end-string",
    ),
    pytest.param(
        {"text": "", "segments": [segment(start=True)]},
        "'segments[0].start' must be a number, not a boolean",
        id="start-boolean",
    ),
    pytest.param(
        {"text": "", "segments": [segment(text=None)]},
        "'segments[0].text' must be a string, not null",
        id="segment-null-text",
    ),
    pytest.param(
        {"text": "", "segments": [segment(id=0.5)]},
        "'segments[0].id' must be an integer or null, not a number",
        id="id-float",
    ),
    pytest.param(
        {"text": "", "segments": [segment(speaker=0)]},
        "'segments[0].speaker' must be a string or null, not a number",
        id="speaker-number",
    ),
    pytest.param(
        {"text": "", "segments": [segment(words={"word": " Hi."})]},
        "'segments[0].words' must be a list or null, not an object",
        id="words-object",
    ),
    pytest.param(
        {"text": "", "segments": [segment(words=[" Hi."])]},
        "'segments[0].words[0]' must be a JSON object, not a string",
        id="word-string",
    ),
    pytest.param(
        {"text": "", "segments": [segment(words=[{"start": 0.0, "end": 1.0}])]},
        "'segments[0].words[0].word' is missing",
        id="no-word",
    ),
    pytest.param(
        {"text": "", "segments": [segment(words=[{"word": " Hi.", "start": "0", "end": 1.0}])]},
        "'segments[0].words[0].start' must be a number, not a string",
        id="word-start-string",
    ),
    pytest.param(
        {"text": "", "segments": [], "language": 1}, "'language' must be a string or null, not a number", id="language"
    ),
    pytest.param(
        {"text": "", "segments": [], "duration": "3 s"},
        "'duration' must be a number or null, not a string",
        id="duration",
    ),
    pytest.param(
        {"text": "", "segments": [], "language_probability": [0.9]},
        "'language_probability' must be a number or null, not a list",
        id="language-probability",
    ),
    pytest.param(
        {"text": "", "segments": [], "speaker_segments": {}},
        "'speaker_segments' must be a list or null, not an object",
        id="speaker-segments-object",
    ),
    pytest.param(
        {"text": "", "segments": [], "speaker_segments": [segment()]},
        "'speaker_segments[0].speaker' is missing",
        id="no-speaker",
    ),
    pytest.param(
        {"text": "", "segments": [], "speaker_segments": [segment(speaker=None)]},
        "'speaker_segments[0].speaker' must be a string, not null",
        id="null-speaker",
    ),
]


@pytest.mark.parametrize("data, message", MALFORMED)
def test_malformed_input_raises_value_error(data, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        TranscriptionResult.from_dict(data)


class TestLoad:
    def test_a_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TranscriptionResult.load(tmp_path / "missing.json")

    def test_a_file_that_is_not_json(self, tmp_path):
        path = TranscriptionResult(" Hi.", [SEGMENT]).save(tmp_path / "talk.vtt")

        with pytest.raises(ValueError) as caught:
            TranscriptionResult.load(path)

        assert str(caught.value).startswith(f"{path} isn't a JSON file: Expecting value")

    def test_a_file_that_is_not_utf8(self, tmp_path):
        path = tmp_path / "talk.json"
        path.write_bytes(b'{"text": "\xff", "segments": []}')

        with pytest.raises(ValueError, match="isn't a JSON file: 'utf-8' codec can't decode"):
            TranscriptionResult.load(path)

    def test_a_byte_order_mark_is_skipped(self, tmp_path):
        path = tmp_path / "talk.json"
        path.write_text(json.dumps({"text": " Hi.", "segments": [SEGMENT]}), encoding="utf-8-sig")

        assert TranscriptionResult.load(path).segments == [SEGMENT]

    def test_errors_name_the_file(self, tmp_path):
        path = tmp_path / "talk.json"
        path.write_text('{"text": " Hi.", "segments": [{"start": 0.0, "text": " Hi."}]}', encoding="utf-8")

        with pytest.raises(ValueError) as caught:
            TranscriptionResult.load(path)

        assert str(caught.value) == f"{path}: 'segments[0].end' is missing"


class TestRepr:
    def test_summarises_the_result(self):
        result = TranscriptionResult(" Hallo\n Welt. ", [SEGMENT, SEGMENT], "de", duration=3.14159)
        assert repr(result) == "<TranscriptionResult language='de' duration=3.14 segments=2 text='Hallo Welt.'>"

    def test_cuts_long_text_and_counts_speaker_segments(self):
        result = TranscriptionResult("word " * 20, [], speaker_segments=[])
        assert repr(result) == (
            "<TranscriptionResult language=None segments=0 speaker_segments=0 "
            "text='word word word word word word word word...'>"
        )


###############################################################################
# Backends: segment ids from 0, faster-whisper's duration and language probability
###############################################################################


@pytest.fixture
def audio_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """An (empty) audio file; the mocked models never read it."""
    path = tmp_path / "talk.wav"
    path.touch()
    return path


def run_whisper(audio: pathlib.Path) -> TranscriptionResult:
    """openai-whisper numbers its segments from 0 (whisper.transcribe enumerates them), and the backend keeps that."""
    whisper = MagicMock()
    model = whisper.load_model.return_value
    model.parameters.return_value = iter([SimpleNamespace(device="cpu")])
    model.transcribe.return_value = {
        "text": " One. Two.",
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.0, "text": " One."},
            {"id": 1, "start": 1.0, "end": 2.0, "text": " Two."},
        ],
        "language": "en",
    }
    with patch.dict(sys.modules, {"whisper": whisper}):
        backend = get_backend("whisper")
        backend.load_model("tiny")
        return backend.transcribe(audio, verbose=False)


def faster_whisper_segment(number: int, text: str) -> SimpleNamespace:
    """A faster_whisper Segment: faster-whisper numbers them from 1."""
    return SimpleNamespace(
        id=number,
        start=number - 1.0,
        end=float(number),
        text=text,
        tokens=[number],
        temperature=0.0,
        avg_logprob=-0.2,
        compression_ratio=1.1,
        no_speech_prob=0.01,
    )


def run_faster_whisper(audio: pathlib.Path) -> TranscriptionResult:
    faster_whisper, ctranslate2 = MagicMock(), MagicMock()
    ctranslate2.get_cuda_device_count.return_value = 0
    info = SimpleNamespace(language="en", duration=2.5, language_probability=0.9)
    segments = [faster_whisper_segment(1, " One."), faster_whisper_segment(2, " Two.")]
    faster_whisper.WhisperModel.return_value.transcribe.return_value = (iter(segments), info)
    with patch.dict(sys.modules, {"faster_whisper": faster_whisper, "ctranslate2": ctranslate2}):
        backend = get_backend("faster-whisper")
        backend.load_model("tiny")
        return backend.transcribe(audio, verbose=False)


def run_transformers_backend(
    name: str, audio: pathlib.Path, seconds: int, prepare: Callable[[Any], None], **options: Any
) -> TranscriptionResult:
    """Run a backend that loads a transformers model, on *seconds* of silence; *prepare* sets up its processor."""
    torch, transformers = MagicMock(), MagicMock()
    torch.cuda.is_available.return_value = False
    prepare(transformers.AutoProcessor.from_pretrained.return_value)
    module = "voxtral_backend" if name == "voxtral" else "nvidia_backend"
    silence = np.zeros(seconds * RATE, dtype=np.float32)
    with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
        with patch(f"speech_toolkit.backends.{module}.load_audio", return_value=silence):
            backend = get_backend(name)
            backend.load_model(backend.default_model())
            return backend.transcribe(audio, verbose=False, **options)


def chunk_texts(processor: Any) -> None:
    """Voxtral and Canary: 40 s of audio is two chunks, and each chunk's text becomes a segment."""
    processor.batch_decode.return_value = ["One.", "Two."]


def sentence_tokens(processor: Any) -> None:
    """Parakeet: timed tokens, grouped into a segment per sentence."""
    tokens = [{"token": " One.", "start": 0.0, "end": 0.5}, {"token": " Two.", "start": 1.0, "end": 1.5}]
    processor.decode.return_value = ("One. Two.", [tokens])


BACKEND_RUNS: Dict[str, Callable[[pathlib.Path], TranscriptionResult]] = {
    "whisper": run_whisper,
    "faster-whisper": run_faster_whisper,
    "voxtral": lambda audio: run_transformers_backend("voxtral", audio, 40, chunk_texts, language="en"),
    "parakeet": lambda audio: run_transformers_backend("parakeet", audio, 5, sentence_tokens),
    "canary": lambda audio: run_transformers_backend("canary", audio, 40, chunk_texts, language="en"),
}


def test_every_backend_has_an_id_check():
    assert sorted(BACKEND_RUNS) == sorted(list_backends())


@pytest.mark.parametrize("backend", sorted(BACKEND_RUNS))
def test_segment_ids_start_at_0(backend, audio_file):
    result = BACKEND_RUNS[backend](audio_file)

    assert [(segment["id"], segment["text"].strip()) for segment in result.segments] == [(0, "One."), (1, "Two.")]


def test_faster_whisper_reports_duration_and_language_probability(audio_file):
    result = run_faster_whisper(audio_file)

    assert (result.duration, result.language_probability) == (2.5, 0.9)
    assert result.raw == {"duration": 2.5, "language_probability": 0.9}  # where earlier versions put them
    data = result.to_dict()
    assert (data["duration"], data["language_probability"]) == (2.5, 0.9)
    assert TranscriptionResult.from_dict(data).to_dict() == data


###############################################################################
# Server: verbose_json's duration
###############################################################################


class TestVerboseJson:
    def test_uses_the_duration_the_backend_reports(self):
        result = TranscriptionResult(" Hi.", [SEGMENT], "en", raw={"duration": 9.0}, duration=2.5)
        assert verbose_json(result, "transcribe")["duration"] == 2.5

    @pytest.mark.parametrize(
        "raw, segments, duration",
        [({"duration": 3.0}, [SEGMENT], 3.0), ({}, [SEGMENT, segment(start=1.0, end=1.75)], 1.75), ({}, [], 0.0)],
        ids=["from-raw", "last-segment-end", "no-segments"],
    )
    def test_without_a_reported_duration(self, raw, segments, duration):
        result = TranscriptionResult(" Hi.", segments, "en", raw=raw)
        assert verbose_json(result, "transcribe")["duration"] == duration
