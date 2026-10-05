"""Tests for speaker diarization: loading pyannote, running it and merging speakers into segments."""

from __future__ import annotations

import os
import pathlib
import sys
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from speech_toolkit.diarization import DIARIZATION_MODEL, diarize_audio, load_diarization_pipeline, merge_diarization


class TestMergeDiarization:
    """Test merging transcription segments with speaker diarization."""

    def test_merge_single_segment_single_speaker(self):
        """Test merging with one segment and one speaker."""
        transcription_result = {"segments": [{"start": 0.0, "end": 5.0, "text": "Hello world"}]}
        spk_segments = [(0.0, 5.0, "SPEAKER_00")]

        result = merge_diarization(transcription_result, spk_segments)

        assert len(result) == 1
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[0]["text"] == "Hello world"

    def test_merge_multiple_segments_single_speaker(self):
        """Test merging multiple segments with one speaker."""
        transcription_result = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hello"},
                {"start": 2.0, "end": 4.0, "text": "world"},
                {"start": 4.0, "end": 6.0, "text": "today"},
            ]
        }
        spk_segments = [(0.0, 6.0, "SPEAKER_00")]

        result = merge_diarization(transcription_result, spk_segments)

        assert len(result) == 3
        assert all(seg["speaker"] == "SPEAKER_00" for seg in result)

    def test_merge_speaker_changes(self):
        """Test merging with speaker changes."""
        transcription_result = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hello"},
                {"start": 2.5, "end": 4.5, "text": "Hi there"},
                {"start": 5.0, "end": 7.0, "text": "How are you?"},
            ]
        }
        spk_segments = [(0.0, 2.5, "SPEAKER_00"), (2.5, 5.0, "SPEAKER_01"), (5.0, 7.0, "SPEAKER_00")]

        result = merge_diarization(transcription_result, spk_segments)

        assert len(result) == 3
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_01"
        assert result[2]["speaker"] == "SPEAKER_00"

    def test_merge_segment_without_speaker(self):
        """Test segment that doesn't overlap any speaker turn."""
        transcription_result = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hello"},
                {"start": 10.0, "end": 11.0, "text": "World"},  # Gap in speaker timeline
            ]
        }
        spk_segments = [(0.0, 2.0, "SPEAKER_00")]

        result = merge_diarization(transcription_result, spk_segments)

        assert len(result) == 2
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "unknown"

    def test_merge_overlapping_speakers(self):
        """Test with overlapping speaker segments: the larger overlap wins."""
        transcription_result = {"segments": [{"start": 0.0, "end": 3.0, "text": "First segment"}]}
        spk_segments = [(0.0, 2.0, "SPEAKER_00"), (1.5, 4.0, "SPEAKER_01")]

        result = merge_diarization(transcription_result, spk_segments)

        assert len(result) == 1
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_merge_midpoint_in_pause(self):
        """A segment whose midpoint falls in a pause still gets the speaker who said it."""
        transcription_result = {"segments": [{"start": 0.0, "end": 4.0, "text": "Long sentence"}]}
        # Midpoint 2.0 is in the pause between the two turns of SPEAKER_00
        spk_segments = [(0.0, 1.8, "SPEAKER_00"), (2.2, 4.0, "SPEAKER_00")]

        result = merge_diarization(transcription_result, spk_segments)

        assert result[0]["speaker"] == "SPEAKER_00"

    def test_merge_sums_overlap_per_speaker(self):
        """Overlap is summed per speaker, not taken from the single longest turn."""
        transcription_result = {"segments": [{"start": 0.0, "end": 3.0, "text": "Back and forth"}]}
        spk_segments = [(0.0, 0.8, "SPEAKER_00"), (0.8, 1.8, "SPEAKER_01"), (1.8, 2.6, "SPEAKER_00")]

        result = merge_diarization(transcription_result, spk_segments)

        assert result[0]["speaker"] == "SPEAKER_00"  # 1.6 s in total vs 1.0 s

    def test_merge_zero_length_segment_uses_midpoint(self):
        """A zero-length segment falls back to the turn containing it."""
        transcription_result = {"segments": [{"start": 1.0, "end": 1.0, "text": "Hm"}]}

        result = merge_diarization(transcription_result, [(0.0, 2.0, "SPEAKER_01")])

        assert result[0]["speaker"] == "SPEAKER_01"

    def test_merge_preserves_metadata(self):
        """Test that original segment metadata is preserved."""
        transcription_result = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hello", "id": 1, "seek": 0, "tokens": [1, 2, 3], "temperature": 0.0}
            ]
        }
        spk_segments = [(0.0, 2.0, "SPEAKER_00")]

        result = merge_diarization(transcription_result, spk_segments)

        assert result[0]["id"] == 1
        assert result[0]["seek"] == 0
        assert result[0]["tokens"] == [1, 2, 3]
        assert result[0]["temperature"] == 0.0
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_merge_empty_segments(self):
        """Test merging with empty segments."""
        transcription_result: Dict[str, Any] = {"segments": []}
        spk_segments = [(0.0, 5.0, "SPEAKER_00")]

        result = merge_diarization(transcription_result, spk_segments)

        assert len(result) == 0

    def test_merge_unordered_speaker_segments(self):
        """Test that speaker segments are sorted before merging."""
        transcription_result = {
            "segments": [{"start": 0.0, "end": 2.0, "text": "Hello"}, {"start": 5.0, "end": 7.0, "text": "World"}]
        }
        # Deliberately unordered
        spk_segments = [(5.0, 8.0, "SPEAKER_01"), (0.0, 3.0, "SPEAKER_00")]

        result = merge_diarization(transcription_result, spk_segments)

        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_01"

    def test_merge_missing_segments_key(self):
        """Test merging with missing segments key in result."""
        transcription_result: Dict[str, Any] = {"text": "Just text"}  # No segments key

        result = merge_diarization(transcription_result, [(0.0, 1.0, "SPEAKER_00")])

        assert len(result) == 0

    def test_merge_malformed_segment_missing_timestamps(self):
        """Test that segments missing start/end are labeled 'unknown' instead of crashing."""
        transcription_result: Dict[str, Any] = {
            "segments": [
                {"text": "no timestamps"},
                {"start": 0.0, "end": 2.0, "text": "has timestamps"},
            ]
        }
        spk_segments = [(0.0, 5.0, "SPEAKER_00")]

        result = merge_diarization(transcription_result, spk_segments)

        assert len(result) == 2
        assert result[0]["speaker"] == "unknown"
        assert result[0]["text"] == "no timestamps"
        assert result[1]["speaker"] == "SPEAKER_00"

    def test_merge_splits_segments_where_the_speaker_changes(self):
        """With word timings, a reply inside one segment becomes its own speaker segment."""
        words = [
            {"word": " Are", "start": 0.0, "end": 0.3},
            {"word": " you", "start": 0.3, "end": 0.5},
            {"word": " sure?", "start": 0.5, "end": 0.9},
            {"word": " Yes.", "start": 1.2, "end": 1.5},
        ]
        segment = {"id": 3, "start": 0.0, "end": 1.5, "text": " Are you sure? Yes.", "words": words}

        result = merge_diarization({"segments": [segment]}, [(0.0, 1.0, "SPEAKER_00"), (1.1, 1.6, "SPEAKER_01")])

        assert [(s["speaker"], s["text"], s["start"], s["end"]) for s in result] == [
            ("SPEAKER_00", " Are you sure?", 0.0, 0.9),
            ("SPEAKER_01", " Yes.", 1.2, 1.5),
        ]
        assert result[1]["words"] == words[3:]
        assert result[0]["id"] == 3  # other segment fields are kept

    def test_merge_words_in_pauses_keep_the_previous_speaker(self):
        words = [
            {"word": " Um,", "start": 0.0, "end": 0.2},  # before the first turn: takes the first known speaker
            {"word": " so", "start": 0.5, "end": 0.7},
            {"word": " yeah", "start": 1.05, "end": 1.1},  # in a gap between turns
            {"word": " right.", "start": 1.6, "end": 1.9},
        ]
        segment = {"start": 0.0, "end": 1.9, "text": " Um, so yeah right.", "words": words}

        result = merge_diarization({"segments": [segment]}, [(0.4, 1.0, "A"), (1.5, 2.0, "B")])

        assert [(s["speaker"], s["text"]) for s in result] == [("A", " Um, so yeah"), ("B", " right.")]

    def test_merge_words_outside_every_turn(self):
        segment = {"start": 5.0, "end": 6.0, "text": " Hi.", "words": [{"word": " Hi.", "start": 5.0, "end": 6.0}]}

        result = merge_diarization({"segments": [segment]}, [(0.0, 1.0, "A")])

        assert [(s["speaker"], s["text"]) for s in result] == [("unknown", " Hi.")]

    def test_merge_words_without_times_label_the_whole_segment(self):
        segment = {"start": 0.0, "end": 1.0, "text": " Hi.", "words": [{"word": " Hi."}]}

        result = merge_diarization({"segments": [segment]}, [(0.0, 1.0, "A")])

        assert result == [{**segment, "speaker": "A"}]


@pytest.fixture
def pyannote_modules(monkeypatch):
    """Stand-ins for torch and pyannote.audio (no GPU, metrics setting untouched)."""
    monkeypatch.delenv("PYANNOTE_METRICS_ENABLED", raising=False)
    torch = MagicMock()
    torch.cuda.is_available.return_value = False
    pyannote_audio = MagicMock()
    with patch.dict(sys.modules, {"torch": torch, "pyannote": MagicMock(), "pyannote.audio": pyannote_audio}):
        yield SimpleNamespace(torch=torch, Pipeline=pyannote_audio.Pipeline)


class TestLoadDiarizationPipeline:
    """Test diarization pipeline loading."""

    def test_load_pipeline_with_env_token(self, pyannote_modules, monkeypatch):
        """Test the pipeline is downloaded with the HF_TOKEN environment variable."""
        monkeypatch.setenv("HF_TOKEN", "env_token")

        pipeline = load_diarization_pipeline()

        pyannote_modules.Pipeline.from_pretrained.assert_called_once_with(DIARIZATION_MODEL, token="env_token")
        assert pipeline is pyannote_modules.Pipeline.from_pretrained.return_value

    def test_load_pipeline_moves_to_device(self, pyannote_modules):
        """Test the pipeline runs on the requested device (auto: CPU without CUDA)."""
        pipeline = load_diarization_pipeline()
        pyannote_modules.torch.device.assert_called_with("cpu")
        pipeline.to.assert_called_once_with(pyannote_modules.torch.device.return_value)

        load_diarization_pipeline("cuda")
        pyannote_modules.torch.device.assert_called_with("cuda")

    def test_load_pipeline_gated_model(self, pyannote_modules):
        """pyannote returns None when the gated model can't be downloaded: explain what to do."""
        pyannote_modules.Pipeline.from_pretrained.return_value = None

        with pytest.raises(RuntimeError, match="Accept its terms"):
            load_diarization_pipeline()

    def test_load_pipeline_pyannote_not_installed(self):
        """Test error when pyannote.audio is not installed."""
        with patch.dict(sys.modules, {"torch": MagicMock(), "pyannote.audio": None}):
            with pytest.raises(ImportError, match=r"speech-transcription-toolkit\[diarize\]"):
                load_diarization_pipeline()

    def test_load_pipeline_disables_telemetry(self, pyannote_modules):
        """pyannote.audio 4 would send usage metrics by default; the tool stays offline."""
        load_diarization_pipeline()
        assert os.environ["PYANNOTE_METRICS_ENABLED"] == "false"

    def test_load_pipeline_respects_telemetry_opt_in(self, pyannote_modules, monkeypatch):
        """An explicit PYANNOTE_METRICS_ENABLED setting is left alone."""
        monkeypatch.setenv("PYANNOTE_METRICS_ENABLED", "true")
        load_diarization_pipeline()
        assert os.environ["PYANNOTE_METRICS_ENABLED"] == "true"


def _turn(start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(start=start, end=end)


class TestDiarizeAudio:
    """Test audio diarization functionality."""

    @pytest.fixture
    def audio_modules(self):
        """Stand-ins for torch and whisper's ffmpeg audio loader."""
        torch = MagicMock()
        whisper = MagicMock()
        with patch.dict(sys.modules, {"torch": torch, "whisper": whisper}):
            yield SimpleNamespace(torch=torch, whisper=whisper)

    def test_diarize_audio_uses_exclusive_diarization(self, audio_modules, tmp_path: pathlib.Path):
        """pyannote 4 output: the exclusive (non-overlapping) diarization is used."""
        output = MagicMock()
        output.exclusive_speaker_diarization.itertracks.return_value = [
            (_turn(0.0, 2.5), None, "SPEAKER_00"),
            (_turn(2.5, 5.0), None, "SPEAKER_01"),
        ]
        pipeline = MagicMock(return_value=output)

        result = diarize_audio(tmp_path / "audio.mp3", pipeline, num_speakers=2)

        assert result == [(0.0, 2.5, "SPEAKER_00"), (2.5, 5.0, "SPEAKER_01")]
        audio_modules.whisper.load_audio.assert_called_once_with(str(tmp_path / "audio.mp3"), sr=16000)
        (audio_input,), hints = pipeline.call_args
        assert audio_input["sample_rate"] == 16000
        assert audio_input["waveform"] is audio_modules.torch.from_numpy.return_value.unsqueeze.return_value
        assert hints == {"num_speakers": 2, "min_speakers": None, "max_speakers": None}

    def test_diarize_audio_plain_annotation(self, audio_modules, tmp_path: pathlib.Path):
        """Older pyannote output (a bare Annotation) is supported too."""
        annotation = MagicMock(spec=["itertracks"])
        annotation.itertracks.return_value = [
            (_turn(0.0, 1.5), None, "SPEAKER_00"),
            (_turn(1.5, 3.0), None, "SPEAKER_01"),
            (_turn(3.0, 4.5), None, "SPEAKER_00"),
        ]
        pipeline = MagicMock(return_value=annotation)

        result = diarize_audio(tmp_path / "conversation.mp3", pipeline, min_speakers=2, max_speakers=4)

        assert [speaker for _, _, speaker in result] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
        assert pipeline.call_args[1] == {"num_speakers": None, "min_speakers": 2, "max_speakers": 4}
