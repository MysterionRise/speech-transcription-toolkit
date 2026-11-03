"""Voice activity detection: speech_toolkit.vad, split_audio(regions=...), the vad_ options, and --vad in every backend
and in Transcriber.

Silero is replaced by a stand-in for faster_whisper.vad (patch.dict(sys.modules, ...)) that takes every sample at
least as loud as the threshold for speech, so the tests build their speech regions from silence and loud samples. The
models are stand-ins too: the backends get a mocked model, or their chunk methods are replaced.
"""

from __future__ import annotations

import logging
import pathlib
import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from speech_toolkit import (
    BackendUnavailableError,
    ProgressEvent,
    Transcriber,
    TranscriptionResult,
    UnsupportedOptionError,
    cli,
)
from speech_toolkit.backends import (
    CanaryBackend,
    FasterWhisperBackend,
    ParakeetBackend,
    TransformersBackend,
    VoxtralBackend,
    WhisperBackend,
    backends_with,
    get_backend_class,
    list_backends,
)
from speech_toolkit.backends.transformers_backend import ModelTraits
from speech_toolkit.media import SAMPLE_RATE, split_audio
from speech_toolkit.vad import (
    MIN_SILENCE,
    PADDING,
    THRESHOLD,
    VAD_OPTIONS,
    SpeechMap,
    SpeechRegion,
    detect_speech,
    faster_whisper_parameters,
    has_speech,
    settings,
)
from tests.conftest import FakeBackend, FakeWordsBackend

RATE = SAMPLE_RATE


def audio_with_speech(seconds: float, *spans: Tuple[float, float], level: float = 1.0) -> np.ndarray:
    """*seconds* of silence, with samples at *level* during each ``(start, end)`` span: the fake Silero's speech."""
    audio = np.zeros(round(seconds * RATE), np.float32)
    for start, end in spans:
        audio[round(start * RATE) : round(end * RATE)] = level
    return audio


class FakeSilero(types.ModuleType):
    """faster_whisper.vad with a stand-in for Silero: samples at least as loud as the threshold are speech.

    Like Silero's, its stretches of speech end after *min_silence_duration_ms* of silence and get *speech_pad_ms* of
    padding on each side. ``calls`` records each call: how many samples it got, their type, and the options.
    """

    def __init__(self) -> None:
        super().__init__("faster_whisper.vad")
        self.calls: List[Dict[str, Any]] = []

    @staticmethod
    def VadOptions(**fields: Any) -> SimpleNamespace:  # noqa: N802 (faster-whisper's name)
        return SimpleNamespace(**fields)

    def get_speech_timestamps(self, audio: np.ndarray, vad_options: Any, sampling_rate: int = 16000) -> List[dict]:
        self.calls.append({"samples": len(audio), "dtype": audio.dtype, **vars(vad_options)})
        loud = np.abs(audio) >= vad_options.threshold
        edges = np.flatnonzero(np.diff(np.concatenate([[0], loud.astype(np.int8), [0]])))
        min_gap = vad_options.min_silence_duration_ms * sampling_rate / 1000
        pad = round(vad_options.speech_pad_ms * sampling_rate / 1000)
        spans: List[List[int]] = []
        for start, end in edges.reshape(-1, 2).tolist():
            if spans and start - spans[-1][1] < min_gap:
                spans[-1][1] = end
            else:
                spans.append([start, end])
        return [{"start": max(0, start - pad), "end": min(len(audio), end + pad)} for start, end in spans]


@pytest.fixture
def silero():
    """The fake Silero in place of faster_whisper.vad (faster_whisper itself is a mock: no model loads here)."""
    vad = FakeSilero()
    package = MagicMock()
    package.vad = vad
    with patch.dict(sys.modules, {"faster_whisper": package, "faster_whisper.vad": vad}):
        yield vad


###############################################################################
# speech_toolkit.vad
###############################################################################


class TestDetectSpeech:
    def test_regions_in_seconds_with_padding(self, silero):
        audio = audio_with_speech(30, (5, 7), (20, 21.5))

        regions = detect_speech(audio)

        assert regions == [SpeechRegion(4.6, 7.4), SpeechRegion(19.6, 21.9)]
        assert regions[0].start == 4.6 and regions[0].end == 7.4
        assert silero.calls == [
            {
                "samples": len(audio),
                "dtype": np.float32,
                "threshold": THRESHOLD,
                "min_silence_duration_ms": 2000,
                "speech_pad_ms": 400,
            }
        ]

    def test_settings_reach_silero(self, silero):
        audio = audio_with_speech(30, (5, 7), (8, 9), level=0.6)

        assert detect_speech(audio, threshold=0.7) == []  # too quiet
        assert detect_speech(audio, min_silence=0.5, padding=0.25) == [(4.75, 7.25), (7.75, 9.25)]
        assert detect_speech(audio, min_silence=1.5, padding=0.0) == [(5.0, 9.0)]  # the 1 s pause stays inside
        assert silero.calls[1]["min_silence_duration_ms"] == 500 and silero.calls[1]["speech_pad_ms"] == 250

    def test_silence_and_empty_audio(self, silero):
        assert detect_speech(np.zeros(10 * RATE, np.float32)) == []
        assert detect_speech(np.zeros(0, np.float32)) == []

    def test_other_float_types_are_converted(self, silero):
        assert detect_speech(audio_with_speech(3, (1, 2)).astype(np.float64)) == [(0.6, 2.4)]
        assert silero.calls[0]["dtype"] == np.float32  # what the ONNX model takes

    def test_needs_mono_samples(self, silero):
        with pytest.raises(ValueError, match="1-D samples"):
            detect_speech(np.zeros((RATE, 2), np.float32))

    def test_without_faster_whisper(self):
        with patch.dict(sys.modules, {"faster_whisper": None, "faster_whisper.vad": None}):
            with pytest.raises(BackendUnavailableError, match="needs faster-whisper") as error:
                detect_speech(np.zeros(RATE, np.float32))

        assert isinstance(error.value, ImportError)
        assert "pip install speech-transcription-toolkit" in str(error.value)

    def test_logs_what_it_found(self, silero, caplog):
        caplog.set_level(logging.DEBUG, logger="speech_toolkit")

        detect_speech(audio_with_speech(30, (5, 7)))

        assert "Voice activity detection: 2.8 s of speech in 30.0 s of audio, in 1 stretches" in caplog.messages


class TestHasSpeech:
    def test_stops_at_the_first_speech(self, silero):
        audio = audio_with_speech(600, (30, 31), (500, 501))

        assert has_speech(audio)
        assert [call["samples"] for call in silero.calls] == [60 * RATE]  # only the first minute

    def test_looks_a_minute_at_a_time_with_some_overlap(self, silero):
        audio = audio_with_speech(150, (130, 131))

        assert has_speech(audio)
        assert [call["samples"] for call in silero.calls] == [60 * RATE, 61 * RATE, 31 * RATE]

    def test_speech_where_two_looks_meet(self, silero):
        assert has_speech(audio_with_speech(120, (60.0, 60.05)))
        assert len(silero.calls) == 2

    def test_silence(self, silero):
        assert not has_speech(np.zeros(150 * RATE, np.float32))
        assert len(silero.calls) == 3
        assert not has_speech(np.zeros(0, np.float32))

    def test_threshold(self, silero):
        audio = audio_with_speech(10, (2, 3), level=0.6)

        assert has_speech(audio, threshold=0.5)
        assert not has_speech(audio, threshold=0.7)
        assert silero.calls[-1]["threshold"] == 0.7


class TestSettings:
    def test_defaults_are_faster_whispers(self):
        assert faster_whisper_parameters() == {"threshold": 0.5, "min_silence_duration_ms": 2000, "speech_pad_ms": 400}
        assert {name: option.default for name, option in VAD_OPTIONS.items()} == {
            "vad_threshold": THRESHOLD,
            "vad_min_silence": MIN_SILENCE,
            "vad_padding": PADDING,
        }

    def test_from_a_backends_options(self):
        assert settings(WhisperBackend().options) == {"threshold": 0.5, "min_silence": 2.0, "padding": 0.4}
        assert settings({"vad_min_silence": 1, "compute_type": "auto"}) == {
            "threshold": 0.5,
            "min_silence": 1.0,
            "padding": 0.4,
        }
        assert faster_whisper_parameters(**settings({"vad_padding": 0.25, "vad_min_silence": 0.5})) == {
            "threshold": 0.5,
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 250,
        }

    @pytest.mark.parametrize("name", list_backends())
    def test_every_backend_has_vad_and_its_options(self, name):
        backend_class = get_backend_class(name)

        assert "vad" in backend_class.capabilities
        assert {key: backend_class.OPTIONS[key] for key in VAD_OPTIONS} == dict(VAD_OPTIONS)
        assert name in backends_with("vad")

    def test_options_are_checked_before_loading(self):
        assert WhisperBackend.check_options({"vad-min-silence": "0.5", "vad_threshold": 1}) == {
            "vad_min_silence": 0.5,
            "vad_threshold": 1.0,
        }
        with pytest.raises(UnsupportedOptionError, match="^option vad_threshold must be at most 1, not 1.5$"):
            ParakeetBackend.check_options({"vad_threshold": "1.5"})
        with pytest.raises(UnsupportedOptionError, match="^option vad_padding must be at least 0, not -0.1$"):
            CanaryBackend.check_options({"vad_padding": -0.1})
        with pytest.raises(UnsupportedOptionError, match="takes NUMBER, not 'long'"):
            Transcriber("voxtral", options={"vad_min_silence": "long"})

    def test_command_line(self, tmp_path):
        audio = tmp_path / "talk.wav"
        audio.touch()

        with pytest.raises(SystemExit, match="^Error: option vad_threshold must be at most 1, not 2.0$"):
            cli.main([str(audio), "-q", "-O", "vad_threshold=2"])


###############################################################################
# SpeechMap: transcribing the speech joined end to end, then moving the times back
###############################################################################

# Three stretches of speech: 1-2 s, 5-6.5 s and 10-11 s, joined into 0-1, 1-2.5 and 2.5-3.5 s.
REGIONS = [SpeechRegion(1.0, 2.0), SpeechRegion(5.0, 6.5), SpeechRegion(10.0, 11.0)]


class TestSpeechMap:
    def test_join(self):
        audio = np.arange(12 * RATE, dtype=np.float32)
        speech = SpeechMap(REGIONS)

        joined = speech.join(audio)

        assert speech.duration == 3.5
        assert len(joined) == 3.5 * RATE
        assert np.array_equal(
            joined,
            np.concatenate(
                [audio[RATE : 2 * RATE], audio[5 * RATE : 6 * RATE + RATE // 2], audio[10 * RATE : 11 * RATE]]
            ),
        )

    @pytest.mark.parametrize(
        "joined, original",
        [(0.0, 1.0), (0.5, 1.5), (1.25, 5.25), (2.0, 6.0), (2.75, 10.25), (3.5, 11.0), (4.0, 11.5)],
    )
    def test_times_move_back(self, joined, original):
        assert SpeechMap(REGIONS).original_time(joined) == original

    def test_every_time_maps_back_to_where_it_was(self):
        """Synthetic speech regions: any sample of the joined audio comes from the recording at the mapped time."""
        rng = np.random.default_rng(39)
        regions = sorted(rng.choice(np.arange(0, 600, 2.5), 40, replace=False))
        regions = [SpeechRegion(start, start + float(rng.uniform(0.1, 2.4))) for start in regions]
        audio = np.arange(605 * RATE, dtype=np.float32) / RATE  # each sample holds its own time
        speech = SpeechMap(regions)
        joined = speech.join(audio)

        for index in rng.integers(0, len(joined), 500):
            assert speech.original_time(index / RATE) == pytest.approx(joined[index], abs=1e-3)

    def test_where_two_stretches_meet(self):
        speech = SpeechMap(REGIONS)

        assert speech.region(1.0) == 1 and speech.original_time(1.0) == 5.0  # a start: the later stretch
        assert speech.region(1.0, end=True) == 0 and speech.original_time(1.0, end=True) == 2.0  # an end
        assert speech.original_time(1.0, 0) == 2.0  # or the stretch given

    def test_restore_segments_with_words(self):
        segments: List[Dict[str, Any]] = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": " One two three",
                "words": [
                    {"word": " One", "start": 0.2, "end": 0.8},
                    {"word": " two", "start": 0.9, "end": 1.2},  # its middle is in the second stretch
                    {"word": " three", "start": 1.5, "end": 2.0, "probability": 0.9},
                ],
            }
        ]

        SpeechMap(REGIONS).restore(segments)

        assert segments[0]["words"] == [
            {"word": " One", "start": 1.2, "end": 1.8},
            {"word": " two", "start": 4.9, "end": 5.2},
            {"word": " three", "start": 5.5, "end": 6.0, "probability": 0.9},
        ]
        assert (segments[0]["start"], segments[0]["end"]) == (1.2, 6.0)  # from its first word to its last

    def test_restore_segments_without_words(self):
        segments = [
            {"start": 0.25, "end": 1.0, "text": " One"},
            {"start": 1.0, "end": 2.75, "text": " two, three"},  # spans the silence between two stretches
            {"start": 2.5, "end": 2.5, "text": ""},  # nothing, where two stretches meet
        ]

        SpeechMap(REGIONS).restore(segments)

        assert [(s["start"], s["end"]) for s in segments] == [(1.25, 2.0), (5.0, 10.25), (10.0, 10.0)]

    def test_needs_speech(self):
        with pytest.raises(ValueError, match="at least one"):
            SpeechMap([])


###############################################################################
# split_audio(regions=...): chunks cut at the edges of the speech
###############################################################################


def _chunks(chunks: Sequence[Tuple[float, np.ndarray]]) -> List[Tuple[float, float]]:
    return [(round(start, 2), round(start + len(samples) / RATE, 2)) for start, samples in chunks]


class TestSplitAudioRegions:
    def test_only_the_regions(self):
        audio = np.arange(60 * RATE, dtype=np.float32)

        chunks = split_audio(audio, 30, regions=[(2.0, 5.5), (40.0, 41.0)])

        assert _chunks(chunks) == [(2.0, 5.5), (40.0, 41.0)]
        assert np.array_equal(chunks[0][1], audio[2 * RATE : int(5.5 * RATE)])  # the samples, in place

    def test_a_long_region_is_cut_in_its_quietest_moment(self):
        audio = np.full(100 * RATE, 0.5, np.float32)
        audio[int(33.0 * RATE) : int(33.1 * RATE)] = 0.0  # a pause 27 s into the region that starts at 6 s

        chunks = split_audio(audio, 30, regions=[(6.0, 70.0), (80.0, 90.0)])

        starts = [start for start, _ in chunks]
        assert len(chunks) == 4 and starts[0] == 6.0 and 33.0 <= starts[1] <= 33.1 and starts[3] == 80.0
        assert all(len(samples) <= 30 * RATE for _, samples in chunks)
        assert sum(len(samples) for _, samples in chunks[:3]) == 64 * RATE  # all of the region, and only it

    def test_no_regions_no_chunks(self):
        assert split_audio(np.ones(10 * RATE, np.float32), 30, regions=[]) == []

    def test_regions_are_kept_inside_the_audio(self):
        assert _chunks(split_audio(np.ones(10 * RATE, np.float32), 30, regions=[(-1.0, 2.0), (9.5, 12.0)])) == [
            (0.0, 2.0),
            (9.5, 10.0),
        ]

    def test_without_regions_all_of_the_audio(self):
        audio = np.ones(10 * RATE, np.float32)
        assert _chunks(split_audio(audio, 30)) == _chunks(split_audio(audio, 30, regions=None)) == [(0.0, 10.0)]


###############################################################################
# The backends: --vad
###############################################################################

# 30 s with speech from 5 to 7 s and from 20 to 21.5 s. With the default padding (0.4 s) the stretches of speech are
# 4.6-7.4 s and 19.6-21.9 s, which join into 0-2.8 s and 2.8-5.1 s.
SPEECH = [(5.0, 7.0), (20.0, 21.5)]


def speech_audio() -> np.ndarray:
    return audio_with_speech(30, *SPEECH)


SILENCE = np.zeros(30 * RATE, np.float32)


class TestWhisper:
    @pytest.fixture
    def backend(self) -> WhisperBackend:
        backend = WhisperBackend()
        backend._model = MagicMock()
        backend._device = "cpu"
        return backend

    def test_transcribes_the_speech_joined_and_maps_the_times_back(self, silero, backend):
        backend._model.transcribe.return_value = {
            "text": " Hello. World.",
            "language": "en",
            "segments": [
                {"id": 0, "start": 0.4, "end": 2.4, "text": " Hello."},
                {
                    "id": 1,
                    "start": 3.0,
                    "end": 4.0,
                    "text": " World.",
                    "words": [{"word": " World.", "start": 3.0, "end": 4.0, "probability": 0.9}],
                },
            ],
        }

        result = backend.transcribe(speech_audio(), vad=True, word_timestamps=True)

        audio, kwargs = backend._model.transcribe.call_args
        assert len(audio[0]) == round(5.1 * RATE)  # only the speech
        assert kwargs["word_timestamps"] is True
        assert [(s["start"], s["end"]) for s in result.segments] == [(5.0, 7.0), (19.8, 20.8)]
        assert result.segments[1]["words"] == [{"word": " World.", "start": 19.8, "end": 20.8, "probability": 0.9}]
        assert result.raw["segments"][0]["start"] == 5.0  # raw holds the recording's times too
        assert (result.text, result.language, result.duration) == (" Hello. World.", "en", 30.0)

    def test_silence_never_reaches_the_model(self, silero, backend):
        result = backend.transcribe(SILENCE, language="de", vad=True)

        backend._model.transcribe.assert_not_called()
        assert (result.text, result.segments, result.language, result.duration) == ("", [], "de", 30.0)

    def test_a_path_is_decoded_with_load_audio(self, silero, backend, tmp_path, monkeypatch):
        path = tmp_path / "talk.mp3"
        path.touch()
        decoded: List[pathlib.Path] = []
        monkeypatch.setattr(
            "speech_toolkit.backends.whisper_backend.load_audio", lambda p: decoded.append(p) or speech_audio()
        )
        backend._model.transcribe.return_value = {"text": "", "segments": [], "language": "en"}

        backend.transcribe(path, vad=True)

        assert decoded == [path]
        assert isinstance(backend._model.transcribe.call_args[0][0], np.ndarray)

    def test_options_tune_it(self, silero, backend):
        backend._options = {**backend.options, "vad_padding": 0.0, "vad_min_silence": 0.5}
        backend._model.transcribe.return_value = {"text": "", "segments": []}

        backend.transcribe(speech_audio(), vad=True)

        assert len(backend._model.transcribe.call_args[0][0]) == round(3.5 * RATE)
        assert silero.calls[0]["speech_pad_ms"] == 0 and silero.calls[0]["min_silence_duration_ms"] == 500

    def test_without_vad(self, backend):
        backend._model.transcribe.return_value = {"text": "", "segments": []}

        result = backend.transcribe(SILENCE)

        assert backend._model.transcribe.call_args[0][0] is SILENCE
        assert result.duration is None


class TestFasterWhisper:
    @pytest.fixture
    def backend(self) -> FasterWhisperBackend:
        backend = FasterWhisperBackend()
        backend._model = MagicMock()
        info = SimpleNamespace(language="en", duration=30.0, language_probability=0.9)
        backend._model.transcribe.return_value = (iter([]), info)
        return backend

    def test_keeps_its_built_in_filter(self, silero, backend):
        audio = speech_audio()

        backend.transcribe(audio, vad=True)

        args, kwargs = backend._model.transcribe.call_args
        assert args[0] is audio
        assert kwargs["vad_filter"] is True
        assert kwargs["vad_parameters"] == {"threshold": 0.5, "min_silence_duration_ms": 2000, "speech_pad_ms": 400}
        assert len(silero.calls) == 1  # whether there is any speech: the first minute

    def test_options_reach_the_filter(self, silero, backend):
        backend._options = {**backend.options, "vad_threshold": 0.6, "vad_min_silence": 0.5, "vad_padding": 0.2}

        backend.transcribe(speech_audio(), vad=True)

        parameters = backend._model.transcribe.call_args[1]["vad_parameters"]
        assert parameters == {"threshold": 0.6, "min_silence_duration_ms": 500, "speech_pad_ms": 200}

    def test_silence_never_reaches_the_model(self, silero, backend):
        result = backend.transcribe(SILENCE, vad=True)

        backend._model.transcribe.assert_not_called()
        assert (result.text, result.segments, result.language, result.duration) == ("", [], None, 30.0)
        assert result.to_dict()["language_probability"] is None

    def test_the_batched_pipeline_filters_with_the_settings(self, silero, backend):
        """With a batch_size above 1 the pipeline always filters, with the vad_ settings; --vad still keeps silent
        audio from it."""
        pipeline = MagicMock()
        pipeline.transcribe.return_value = backend._model.transcribe.return_value
        backend._batched_pipeline = lambda: pipeline  # in place of faster-whisper's BatchedInferencePipeline
        backend._options = {**backend.options, "batch_size": 4, "vad_min_silence": 0.5}

        backend.transcribe(speech_audio())  # without --vad

        kwargs = pipeline.transcribe.call_args[1]
        assert kwargs["vad_filter"] is True
        assert kwargs["vad_parameters"] == {"threshold": 0.5, "min_silence_duration_ms": 500, "speech_pad_ms": 400}
        backend._model.transcribe.assert_not_called()

        pipeline.reset_mock()
        result = backend.transcribe(SILENCE, vad=True)

        pipeline.transcribe.assert_not_called()
        assert (result.text, result.segments) == ("", [])

    def test_a_path_is_decoded_with_load_audio(self, silero, backend, tmp_path, monkeypatch):
        """Through load_audio's hardened ffmpeg call rather than PyAV."""
        path = tmp_path / "talk.mp3"
        path.touch()
        monkeypatch.setattr("speech_toolkit.backends.faster_whisper_backend.load_audio", lambda p: speech_audio())

        backend.transcribe(path, vad=True)

        assert isinstance(backend._model.transcribe.call_args[0][0], np.ndarray)

    def test_without_vad(self, backend, tmp_path):
        path = tmp_path / "talk.mp3"
        path.touch()

        backend.transcribe(path)

        args, kwargs = backend._model.transcribe.call_args
        assert args[0] == str(path)  # faster-whisper decodes the file itself, as before
        assert (kwargs["vad_filter"], kwargs["vad_parameters"]) == (False, None)


class TestChunkedBackends:
    """Voxtral, Parakeet, Canary and transformers: chunks cut at the edges of the speech, silent audio in none."""

    @staticmethod
    def loaded(backend_class: type) -> Any:
        backend = backend_class()
        backend._model, backend._processor = MagicMock(), MagicMock()
        return backend

    @pytest.fixture
    def batches(self, monkeypatch) -> List[List[Tuple[int, float]]]:
        """Replace the batch methods of Canary and Voxtral: each chunk's text names its length; returns the
        batches, as the chunks' lengths in samples."""
        batches: List[List[Tuple[int, float]]] = []

        def transcribe_batch(self: Any, chunks: Sequence[np.ndarray], *args: Any) -> Tuple[List[str], List[int]]:
            batches.append([len(chunk) for chunk in chunks])
            return [f"{len(chunk) / RATE:g} s" for chunk in chunks], []

        monkeypatch.setattr(CanaryBackend, "_transcribe_batch", transcribe_batch)
        monkeypatch.setattr(VoxtralBackend, "_transcribe_batch", transcribe_batch)
        return batches

    @pytest.mark.parametrize("backend_class", [CanaryBackend, VoxtralBackend])
    def test_one_segment_per_stretch_of_speech(self, silero, batches, backend_class):
        result = self.loaded(backend_class).transcribe(speech_audio(), "en", vad=True, verbose=False)

        assert batches == [[round(2.8 * RATE), round(2.3 * RATE)]]
        assert [(s["id"], s["start"], round(s["end"], 3), s["text"]) for s in result.segments] == [
            (0, 4.6, 7.4, "2.8 s"),
            (1, 19.6, 21.9, "2.3 s"),
        ]
        assert result.duration == 30.0

    @pytest.mark.parametrize("backend_class", [CanaryBackend, VoxtralBackend])
    def test_silence_never_reaches_the_model(self, silero, batches, backend_class):
        result = self.loaded(backend_class).transcribe(SILENCE, "en", vad=True, verbose=False)

        assert batches == []
        assert (result.text, result.segments, result.duration) == ("", [], 30.0)

    @pytest.mark.parametrize("backend_class", [CanaryBackend, VoxtralBackend])
    def test_without_vad(self, silero, batches, backend_class):
        result = self.loaded(backend_class).transcribe(SILENCE, "en", verbose=False)

        assert batches == [[30 * RATE]]
        assert len(result.segments) == 1 and silero.calls == []

    def test_parakeet(self, silero, monkeypatch):
        chunks: List[Tuple[float, int]] = []

        def transcribe_batch(self: Any, batch: List[Tuple[float, np.ndarray]]) -> List[Dict[str, Any]]:
            words = []
            for offset, samples in batch:
                chunks.append((offset, len(samples)))
                words.append({"word": " Hi.", "start": round(offset + 0.4, 3), "end": round(offset + 1.0, 3)})
            return words

        monkeypatch.setattr(ParakeetBackend, "_transcribe_batch", transcribe_batch)
        backend = self.loaded(ParakeetBackend)
        backend._options = {**backend.options, "vad_padding": 0.1}

        result = backend.transcribe(speech_audio(), vad=True, word_timestamps=True, verbose=False)

        assert chunks == [(4.9, round(2.2 * RATE)), (19.9, round(1.7 * RATE))]
        assert [(s["start"], s["end"], s["text"]) for s in result.segments] == [
            (5.3, 5.9, " Hi."),
            (20.3, 20.9, " Hi."),
        ]
        assert result.segments[1]["words"] == [{"word": " Hi.", "start": 20.3, "end": 20.9}]

        chunks.clear()
        result = backend.transcribe(SILENCE, vad=True, verbose=False)

        assert chunks == [] and result.segments == [] and result.text == ""

    def test_long_speech_is_still_cut_into_chunks(self, silero, batches):
        """A stretch of speech longer than a chunk is cut as audio without --vad is: in its quietest moment."""
        audio = audio_with_speech(100, (10, 80), level=0.5)

        result = self.loaded(CanaryBackend).transcribe(audio, "en", vad=True, verbose=False)

        assert [round(s["start"], 2) for s in result.segments][0] == 9.6
        assert all(length <= 30 * RATE for length in batches[0])
        assert sum(batches[0]) == round(70.8 * RATE)

    ###############################################################################
    # Transcriber: the VAD step
    ###############################################################################

    def test_transformers(self, silero, monkeypatch):
        chunks: List[Tuple[float, int]] = []

        def transcribe_chunk(self: Any, samples: np.ndarray, offset: float, *args: Any) -> List[Dict[str, Any]]:
            chunks.append((offset, len(samples)))
            return [{"start": offset, "end": offset + len(samples) / RATE, "text": " Hi."}]

        detect = MagicMock(return_value="en")
        monkeypatch.setattr(TransformersBackend, "_transcribe_chunk", transcribe_chunk)
        monkeypatch.setattr(TransformersBackend, "_detect_language", detect)
        backend = TransformersBackend()
        backend._pipeline, backend._traits = MagicMock(), ModelTraits("whisper", detects_language=True)

        result = backend.transcribe(speech_audio(), vad=True, verbose=False)

        assert chunks == [(4.6, round(2.8 * RATE)), (19.6, round(2.3 * RATE))]
        assert [(s["id"], s["start"]) for s in result.segments] == [(0, 4.6), (1, 19.6)]
        assert (result.language, detect.call_count) == ("en", 1)  # detected in the first stretch of speech

        chunks.clear()
        detect.reset_mock()
        result = backend.transcribe(SILENCE, vad=True, verbose=False)

        assert (chunks, result.text, result.segments, result.duration) == ([], "", [], 30.0)
        detect.assert_not_called()  # nor does its language detection


class FakeVadBackend(FakeBackend):
    """A backend with the "vad" capability that takes samples, as the built-in ones do: a segment for each stretch of
    speech it finds (for all of the samples without vad), or one for a file. ``calls`` records what it got."""

    name = "fake-vad"
    capabilities = frozenset({"vad", "waveform", "word_timestamps"})
    calls: List[Dict[str, Any]] = []

    def transcribe(self, audio_path, language=None, task="transcribe", verbose=True, **options):
        FakeVadBackend.calls.append({"audio": audio_path, **options})
        if isinstance(audio_path, pathlib.Path):
            regions = [SpeechRegion(0.0, 1.0)]
        elif options.get("vad"):
            regions = detect_speech(audio_path)
        else:
            regions = [SpeechRegion(0.0, len(audio_path) / RATE)]
        segments = [
            {"id": i, "start": start, "end": end, "text": f" Speech {i}."} for i, (start, end) in enumerate(regions)
        ]
        return TranscriptionResult("".join(segment["text"] for segment in segments), segments, language or "en")


@pytest.fixture
def fake_vad(monkeypatch, fake_backend):
    """Register FakeVadBackend as 'fake-vad', besides the fake_backend fixture's 'fake' and 'fake-words'."""
    from speech_toolkit.backends import _BACKENDS

    monkeypatch.setitem(_BACKENDS, "fake-vad", FakeVadBackend)
    FakeVadBackend.calls = []
    return FakeVadBackend


@pytest.fixture
def talk(tmp_path, monkeypatch) -> SimpleNamespace:
    """A file (``path``), which Transcriber's load_audio() 'decodes' to speech_audio(); ``decoded`` records the paths
    it got."""
    path = tmp_path / "talk.wav"
    path.touch()
    decoded: List[pathlib.Path] = []
    monkeypatch.setattr("speech_toolkit.api.load_audio", lambda p: decoded.append(p) or speech_audio())
    return SimpleNamespace(path=path, decoded=decoded)


class TestTranscriber:
    def test_decodes_the_file_once_for_a_backend_that_takes_samples(self, silero, fake_vad, talk):
        events: List[ProgressEvent] = []

        result = Transcriber("fake-vad").transcribe(talk.path, vad=True, progress=events.append)

        assert talk.decoded == [talk.path]
        (call,) = FakeVadBackend.calls
        assert isinstance(call["audio"], np.ndarray) and call["vad"] is True
        assert [event.stage for event in events][:2] == ["decode", "decode"]
        assert [(s["start"], s["end"]) for s in result.segments] == [(4.6, 7.4), (19.6, 21.9)]
        assert result.duration == 30.0

    def test_without_vad_the_backend_reads_the_file(self, fake_vad, talk):
        Transcriber("fake-vad").transcribe(talk.path)

        assert talk.decoded == []
        assert FakeVadBackend.calls == [{"audio": talk.path}]

    def test_a_backend_that_reads_files_still_gets_the_file(self, fake_backend, talk):
        Transcriber("fake-words").transcribe(talk.path, vad=True)

        assert talk.decoded == []
        assert FakeWordsBackend.calls == [{"vad": True}]

    def test_no_speech_is_logged(self, silero, fake_vad, caplog):
        caplog.set_level(logging.INFO, logger="speech_toolkit")

        result = Transcriber("fake-vad").transcribe(SILENCE, vad=True)

        assert result.segments == [] and result.duration == 30.0
        assert "No speech found in the audio" in caplog.messages

    def test_speaker_labels_share_the_decoded_samples(self, silero, fake_vad, talk, monkeypatch):
        labelled: List[Any] = []
        monkeypatch.setattr("speech_toolkit.api.load_diarization_pipeline", lambda device: object())
        monkeypatch.setattr(
            "speech_toolkit.api.diarize_audio", lambda source, pipeline, **hints: labelled.append(source) or []
        )

        result = Transcriber("fake-vad", diarize=True).transcribe(talk.path, vad=True)

        assert talk.decoded == [talk.path]
        assert labelled[0] is FakeVadBackend.calls[0]["audio"]
        assert FakeVadBackend.calls[0]["word_timestamps"] is True
        assert result.speaker_segments is not None

    def test_command_line(self, silero, fake_vad, talk, capsys):
        cli.main([str(talk.path), "-b", "fake-vad", "--vad"])

        captured = capsys.readouterr()
        assert captured.out == "Speech 0. Speech 1.\n"
        assert "doesn't support" not in captured.err

    def test_command_line_silence(self, silero, fake_vad, tmp_path, monkeypatch, capsys):
        path = tmp_path / "quiet.wav"
        path.touch()
        monkeypatch.setattr("speech_toolkit.api.load_audio", lambda p: SILENCE)

        cli.main([str(path), "-b", "fake-vad", "--vad"])

        captured = capsys.readouterr()
        assert captured.out.strip() == ""
        assert f"No speech found in {path}" in captured.err
