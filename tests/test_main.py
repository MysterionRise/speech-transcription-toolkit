#!/usr/bin/env python3
"""Tests for main.py speech-to-text functionality."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from main import (
    diarize_audio,
    load_diarization_pipeline,
    merge_diarization,
    parse_args,
    show_backends,
    show_models,
    write_outputs,
)


class TestParseArgs:
    """Test command-line argument parsing for main.py."""

    def test_parse_args_minimal(self):
        """Test parsing with only required audio argument."""
        args = parse_args(["audio.mp3"])
        assert args.audio == pathlib.Path("audio.mp3")
        assert args.model is None  # Uses backend default
        assert args.task == "transcribe"
        assert args.diarize is False
        assert args.quiet is False
        assert args.backend == "whisper"

    def test_parse_args_custom_model(self):
        """Test parsing with custom model."""
        args = parse_args(["audio.mp3", "--model", "large"])
        assert args.model == "large"

    def test_parse_args_language(self):
        """Test parsing with language specification."""
        args = parse_args(["audio.mp3", "--language", "en"])
        assert args.language == "en"

    def test_parse_args_translate_task(self):
        """Test parsing with translate task."""
        args = parse_args(["audio.mp3", "--task", "translate"])
        assert args.task == "translate"

    def test_parse_args_device(self):
        """Test parsing with device specification."""
        args = parse_args(["audio.mp3", "--device", "cuda"])
        assert args.device == "cuda"

    def test_parse_args_quiet(self):
        """Test parsing with quiet flag."""
        args = parse_args(["audio.mp3", "--quiet"])
        assert args.quiet is True

    def test_parse_args_diarization(self):
        """Test parsing with diarization enabled."""
        args = parse_args(["audio.mp3", "--diarize", "--hf-token", "test_token"])
        assert args.diarize is True
        assert args.hf_token == "test_token"

    def test_parse_args_output_files(self):
        """Test parsing with output file specifications."""
        args = parse_args(["audio.mp3", "-o", "transcript.txt", "--json", "result.json"])
        assert args.output == pathlib.Path("transcript.txt")
        assert args.json == pathlib.Path("result.json")

    def test_parse_args_backend_selection(self):
        """Test parsing with backend selection."""
        args = parse_args(["audio.mp3", "--backend", "voxtral"])
        assert args.backend == "voxtral"

    def test_parse_args_short_backend_flag(self):
        """Test parsing with short backend flag."""
        args = parse_args(["audio.mp3", "-b", "voxtral"])
        assert args.backend == "voxtral"

    def test_parse_args_list_backends(self):
        """Test parsing with list-backends flag."""
        args = parse_args(["--list-backends"])
        assert args.list_backends is True
        assert args.audio is None  # Audio not required for info flags

    def test_parse_args_list_models(self):
        """Test parsing with list-models flag."""
        args = parse_args(["--list-models"])
        assert args.list_models is True
        assert args.audio is None

    def test_parse_args_list_models_with_backend(self):
        """Test parsing with list-models and backend."""
        args = parse_args(["--list-models", "--backend", "voxtral"])
        assert args.list_models is True
        assert args.backend == "voxtral"

    def test_parse_args_no_audio_raises_error(self):
        """Test that missing audio argument (without --list-* flags) raises SystemExit."""
        with pytest.raises(SystemExit):
            parse_args([])


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
        """Test segment that doesn't fall within any speaker interval."""
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
        """Test with overlapping speaker segments."""
        transcription_result = {"segments": [{"start": 0.0, "end": 3.0, "text": "First segment"}]}
        # Midpoint is 1.5, which falls in SPEAKER_00's range
        spk_segments = [(0.0, 2.0, "SPEAKER_00"), (1.5, 4.0, "SPEAKER_01")]

        result = merge_diarization(transcription_result, spk_segments)

        assert len(result) == 1
        # Should match first speaker that contains the midpoint
        assert result[0]["speaker"] == "SPEAKER_00"

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
        transcription_result = {"segments": []}
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


class TestWriteOutputs:
    """Test output writing functionality."""

    def test_write_plain_transcript_to_stdout(self, capsys):
        """Test writing plain transcript to stdout."""
        result = {"text": "This is a test transcript.", "segments": []}
        args = MagicMock()
        args.diarize = False
        args.output = None
        args.json = None

        write_outputs(result, args)

        captured = capsys.readouterr()
        assert "This is a test transcript." in captured.out

    def test_write_transcript_to_file(self, tmp_path: pathlib.Path):
        """Test writing transcript to a file."""
        result = {"text": "File transcript test.", "segments": []}
        output_file = tmp_path / "transcript.txt"

        args = MagicMock()
        args.diarize = False
        args.output = output_file
        args.json = None

        write_outputs(result, args)

        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "File transcript test." in content

    def test_write_diarized_transcript(self, tmp_path: pathlib.Path):
        """Test writing transcript with speaker labels."""
        result = {
            "text": "Original text",
            "speaker_segments": [
                {"speaker": "SPEAKER_00", "text": "Hello there."},
                {"speaker": "SPEAKER_01", "text": "Hi!"},
                {"speaker": "SPEAKER_00", "text": "How are you?"},
            ],
        }
        output_file = tmp_path / "diarized.txt"

        args = MagicMock()
        args.diarize = True
        args.output = output_file
        args.json = None

        write_outputs(result, args)

        content = output_file.read_text(encoding="utf-8")
        assert "[SPEAKER_00]" in content
        assert "[SPEAKER_01]" in content
        assert "Hello there." in content
        assert "Hi!" in content

    def test_write_json_output(self, tmp_path: pathlib.Path):
        """Test writing JSON output."""
        result = {"text": "JSON test", "segments": [{"start": 0.0, "end": 2.0, "text": "JSON test"}]}
        json_file = tmp_path / "result.json"

        args = MagicMock()
        args.diarize = False
        args.output = None
        args.json = json_file

        write_outputs(result, args)

        assert json_file.exists()
        loaded = json.loads(json_file.read_text(encoding="utf-8"))
        assert loaded["text"] == "JSON test"
        assert len(loaded["segments"]) == 1

    def test_write_both_transcript_and_json(self, tmp_path: pathlib.Path):
        """Test writing both transcript and JSON files."""
        result = {"text": "Both outputs test", "segments": []}
        txt_file = tmp_path / "transcript.txt"
        json_file = tmp_path / "result.json"

        args = MagicMock()
        args.diarize = False
        args.output = txt_file
        args.json = json_file

        write_outputs(result, args)

        assert txt_file.exists()
        assert json_file.exists()

    def test_write_speaker_changes_formatted(self, capsys):
        """Test speaker change formatting in output."""
        result = {
            "speaker_segments": [
                {"speaker": "SPEAKER_00", "text": "First sentence."},
                {"speaker": "SPEAKER_00", "text": "Second sentence."},
                {"speaker": "SPEAKER_01", "text": "Different speaker."},
            ]
        }
        args = MagicMock()
        args.diarize = True
        args.output = None
        args.json = None

        write_outputs(result, args)

        captured = capsys.readouterr()
        # Should only show speaker label when it changes
        assert captured.out.count("[SPEAKER_00]") == 1
        assert captured.out.count("[SPEAKER_01]") == 1


class TestShowBackends:
    """Test backend listing functionality."""

    def test_show_backends_output(self, capsys):
        """Test that show_backends displays backend info."""
        show_backends()

        captured = capsys.readouterr()
        assert "whisper" in captured.out
        assert "voxtral" in captured.out
        assert "(default)" in captured.out

    def test_show_models_whisper(self, capsys):
        """Test that show_models displays Whisper models."""
        show_models("whisper")

        captured = capsys.readouterr()
        assert "tiny" in captured.out
        assert "turbo" in captured.out
        assert "(default)" in captured.out

    def test_show_models_voxtral(self, capsys):
        """Test that show_models displays Voxtral models."""
        show_models("voxtral")

        captured = capsys.readouterr()
        assert "voxtral-mini" in captured.out
        assert "voxtral-small" in captured.out

    @patch("main.sys.exit")
    def test_show_models_invalid_backend(self, mock_exit):
        """Test show_models with invalid backend."""
        mock_exit.side_effect = SystemExit("Error")

        with pytest.raises(SystemExit):
            show_models("invalid_backend")

        mock_exit.assert_called_once()


class TestLoadDiarizationPipeline:
    """Test diarization pipeline loading."""

    @patch("main.Pipeline")
    @patch.dict("os.environ", {"HUGGINGFACE_TOKEN": "env_token"})
    def test_load_pipeline_with_env_token(self, mock_pipeline):
        """Test loading pipeline with environment variable token."""
        mock_pipeline.from_pretrained.return_value = MagicMock()

        load_diarization_pipeline(None)

        mock_pipeline.from_pretrained.assert_called_once()
        call_kwargs = mock_pipeline.from_pretrained.call_args[1]
        assert call_kwargs["use_auth_token"] == "env_token"

    @patch("main.Pipeline")
    def test_load_pipeline_with_explicit_token(self, mock_pipeline):
        """Test loading pipeline with explicitly provided token."""
        mock_pipeline.from_pretrained.return_value = MagicMock()

        load_diarization_pipeline("explicit_token")

        call_kwargs = mock_pipeline.from_pretrained.call_args[1]
        assert call_kwargs["use_auth_token"] == "explicit_token"

    @patch("main.sys.exit")
    def test_load_pipeline_pyannote_not_installed(self, mock_exit):
        """Test error when pyannote.audio is not installed."""
        # Make sys.exit raise SystemExit as it normally would
        mock_exit.side_effect = SystemExit("pyannote.audio is not installed")

        # Import main and temporarily set Pipeline to None
        import main

        original_pipeline = main.Pipeline
        try:
            main.Pipeline = None
            with pytest.raises(SystemExit):
                load_diarization_pipeline("token")
        finally:
            main.Pipeline = original_pipeline

        mock_exit.assert_called_once()
        assert "not installed" in str(mock_exit.call_args[0][0])

    @patch("main.Pipeline")
    @patch.dict("os.environ", {}, clear=True)
    def test_load_pipeline_no_token_warning(self, mock_pipeline, capsys):
        """Test warning when no token is provided."""
        mock_pipeline.from_pretrained.return_value = MagicMock()

        load_diarization_pipeline(None)

        captured = capsys.readouterr()
        assert "No HF token provided" in captured.err


class TestDiarizeAudio:
    """Test audio diarization functionality."""

    @patch("main.Pipeline")
    def test_diarize_audio_basic(self, mock_pipeline, tmp_path: pathlib.Path):
        """Test basic audio diarization."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.touch()

        # Mock diarization annotation
        mock_turn_1 = MagicMock()
        mock_turn_1.start = 0.0
        mock_turn_1.end = 2.5

        mock_turn_2 = MagicMock()
        mock_turn_2.start = 2.5
        mock_turn_2.end = 5.0

        mock_annotation = MagicMock()
        mock_annotation.itertracks.return_value = [(mock_turn_1, None, "SPEAKER_00"), (mock_turn_2, None, "SPEAKER_01")]

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = mock_annotation

        result = diarize_audio(audio_file, mock_pipeline_instance)

        assert len(result) == 2
        assert result[0] == (0.0, 2.5, "SPEAKER_00")
        assert result[1] == (2.5, 5.0, "SPEAKER_01")

    @patch("main.Pipeline")
    def test_diarize_audio_multiple_speakers(self, mock_pipeline, tmp_path: pathlib.Path):
        """Test diarization with multiple speakers."""
        audio_file = tmp_path / "conversation.mp3"
        audio_file.touch()

        # Mock multiple speaker turns
        turns = [(0.0, 1.5, "SPEAKER_00"), (1.5, 3.0, "SPEAKER_01"), (3.0, 4.5, "SPEAKER_00"), (4.5, 6.0, "SPEAKER_02")]

        mock_turns = []
        for start, end, speaker in turns:
            mock_turn = MagicMock()
            mock_turn.start = start
            mock_turn.end = end
            mock_turns.append((mock_turn, None, speaker))

        mock_annotation = MagicMock()
        mock_annotation.itertracks.return_value = mock_turns

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = mock_annotation

        result = diarize_audio(audio_file, mock_pipeline_instance)

        assert len(result) == 4
        assert result[0][2] == "SPEAKER_00"
        assert result[1][2] == "SPEAKER_01"
        assert result[2][2] == "SPEAKER_00"
        assert result[3][2] == "SPEAKER_02"
