#!/usr/bin/env python3
"""Tests for convert.py audio conversion functionality."""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock, patch

import pytest

from convert import collect_ogg_files, convert_file, main, parse_args


class TestParseArgs:
    """Test command-line argument parsing for convert.py."""

    def test_parse_args_single_file(self):
        """Test parsing a single input file."""
        args = parse_args(["test.ogg"])
        assert len(args.inputs) == 1
        assert args.inputs[0] == pathlib.Path("test.ogg")
        assert args.rate == 16000
        assert args.channels == 1
        assert args.overwrite is False

    def test_parse_args_multiple_files(self):
        """Test parsing multiple input files."""
        args = parse_args(["file1.ogg", "file2.ogg", "dir/"])
        assert len(args.inputs) == 3

    def test_parse_args_custom_rate(self):
        """Test custom sample rate."""
        args = parse_args(["test.ogg", "--rate", "48000"])
        assert args.rate == 48000

    def test_parse_args_stereo(self):
        """Test stereo output."""
        args = parse_args(["test.ogg", "--channels", "2"])
        assert args.channels == 2

    def test_parse_args_overwrite(self):
        """Test overwrite flag."""
        args = parse_args(["test.ogg", "--overwrite"])
        assert args.overwrite is True

    def test_parse_args_outdir(self):
        """Test custom output directory."""
        args = parse_args(["test.ogg", "--outdir", "/tmp/output"])
        assert args.outdir == pathlib.Path("/tmp/output")


class TestCollectOggFiles:
    """Test OGG file collection from various sources."""

    def test_collect_single_ogg_file(self, tmp_path: pathlib.Path):
        """Test collecting a single OGG file."""
        ogg_file = tmp_path / "test.ogg"
        ogg_file.touch()

        result = collect_ogg_files([ogg_file])
        assert result == [(ogg_file, pathlib.Path("test.ogg"))]

    def test_collect_from_directory(self, tmp_path: pathlib.Path):
        """Test collecting OGG files from a directory."""
        # Create test files
        (tmp_path / "file1.ogg").touch()
        (tmp_path / "file2.ogg").touch()
        (tmp_path / "ignore.mp3").touch()

        result = collect_ogg_files([tmp_path])
        assert len(result) == 2
        assert all(f.suffix == ".ogg" for f, _ in result)

    def test_collect_recursive(self, tmp_path: pathlib.Path):
        """Test recursive collection from nested directories."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        (tmp_path / "root.ogg").touch()
        (subdir / "nested.ogg").touch()

        result = collect_ogg_files([tmp_path])
        assert [relative for _, relative in result] == [pathlib.Path("root.ogg"), pathlib.Path("subdir/nested.ogg")]

    def test_collect_mixed_inputs(self, tmp_path: pathlib.Path):
        """Test collecting from both files and directories."""
        file1 = tmp_path / "file1.ogg"
        file1.touch()

        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file2.ogg").touch()

        result = collect_ogg_files([file1, subdir])
        assert len(result) == 2

    def test_collect_case_insensitive(self, tmp_path: pathlib.Path):
        """Test OGG file collection with any extension case, each file listed once."""
        for name in ("upper.OGG", "mixed.oGg", "title.Ogg"):
            (tmp_path / name).touch()

        result = collect_ogg_files([tmp_path])
        assert sorted(f.name for f, _ in result) == ["mixed.oGg", "title.Ogg", "upper.OGG"]

    def test_collect_opus_and_oga(self, tmp_path: pathlib.Path):
        """Opus and .oga files are OGG containers too."""
        (tmp_path / "voice.opus").touch()
        (tmp_path / "old.oga").touch()
        explicit = tmp_path / "note.OPUS"
        explicit.touch()

        result = collect_ogg_files([tmp_path, explicit])
        assert sorted(f.name for f, _ in result) == ["note.OPUS", "old.oga", "voice.opus"]

    def test_collect_non_ogg_file_warning(self, tmp_path: pathlib.Path, capsys):
        """Test warning message for non-OGG files."""
        mp3_file = tmp_path / "audio.mp3"
        mp3_file.touch()

        result = collect_ogg_files([mp3_file])
        assert len(result) == 0

        captured = capsys.readouterr()
        assert "Skipping unsupported file" in captured.err


class TestConvertFile:
    """Test audio file conversion functionality."""

    @patch("convert.AudioSegment")
    def test_convert_file_success(self, mock_audio_segment, tmp_path: pathlib.Path):
        """Test successful file conversion."""
        src = tmp_path / "input.ogg"
        src.touch()
        outdir = tmp_path / "output"

        # Mock AudioSegment chain
        mock_audio = MagicMock()
        mock_audio_segment.from_file.return_value = mock_audio
        mock_audio.set_frame_rate.return_value = mock_audio
        mock_audio.set_channels.return_value = mock_audio
        mock_audio.set_sample_width.return_value = mock_audio

        convert_file(src, outdir, 16000, 1, overwrite=False)

        # Verify the conversion chain
        mock_audio_segment.from_file.assert_called_once_with(src)
        mock_audio.set_frame_rate.assert_called_once_with(16000)
        mock_audio.set_channels.assert_called_once_with(1)
        mock_audio.set_sample_width.assert_called_once_with(2)  # 16-bit

        # Verify export was called
        assert mock_audio.export.called

    @patch("convert.AudioSegment")
    def test_convert_file_creates_outdir(self, mock_audio_segment, tmp_path: pathlib.Path):
        """Test that output directory is created if it doesn't exist."""
        src = tmp_path / "input.ogg"
        src.touch()
        outdir = tmp_path / "new_output_dir"

        # Mock AudioSegment
        mock_audio = MagicMock()
        mock_audio_segment.from_file.return_value = mock_audio
        mock_audio.set_frame_rate.return_value = mock_audio
        mock_audio.set_channels.return_value = mock_audio
        mock_audio.set_sample_width.return_value = mock_audio

        convert_file(src, outdir, 16000, 1)

        assert outdir.exists()
        assert outdir.is_dir()

    @patch("convert.AudioSegment")
    def test_convert_file_no_outdir_uses_parent(self, mock_audio_segment, tmp_path: pathlib.Path):
        """Test conversion without outdir uses source file's parent directory."""
        src = tmp_path / "input.ogg"
        src.touch()

        # Mock AudioSegment
        mock_audio = MagicMock()
        mock_audio_segment.from_file.return_value = mock_audio
        mock_audio.set_frame_rate.return_value = mock_audio
        mock_audio.set_channels.return_value = mock_audio
        mock_audio.set_sample_width.return_value = mock_audio

        convert_file(src, None, 16000, 1)

        # Verify export was called with correct path
        expected_dest = tmp_path / "input.wav"
        call_args = mock_audio.export.call_args
        assert call_args[0][0] == expected_dest

    def test_convert_file_skip_existing(self, tmp_path: pathlib.Path, capsys):
        """Test that existing files are skipped without --overwrite."""
        src = tmp_path / "input.ogg"
        src.touch()

        dest = tmp_path / "input.wav"
        dest.touch()  # Pre-existing WAV

        convert_file(src, None, 16000, 1, overwrite=False)

        captured = capsys.readouterr()
        assert "exists; skipping" in captured.out

    @patch("convert.AudioSegment")
    def test_convert_file_overwrite_existing(self, mock_audio_segment, tmp_path: pathlib.Path):
        """Test that existing files are overwritten with --overwrite flag."""
        src = tmp_path / "input.ogg"
        src.touch()

        dest = tmp_path / "input.wav"
        dest.write_text("old content")

        # Mock AudioSegment
        mock_audio = MagicMock()
        mock_audio_segment.from_file.return_value = mock_audio
        mock_audio.set_frame_rate.return_value = mock_audio
        mock_audio.set_channels.return_value = mock_audio
        mock_audio.set_sample_width.return_value = mock_audio

        convert_file(src, None, 16000, 1, overwrite=True)

        # Verify export was called (would overwrite)
        assert mock_audio.export.called

    @patch("convert.AudioSegment")
    def test_convert_file_custom_params(self, mock_audio_segment, tmp_path: pathlib.Path):
        """Test conversion with custom sample rate and channels."""
        src = tmp_path / "input.ogg"
        src.touch()

        # Mock AudioSegment
        mock_audio = MagicMock()
        mock_audio_segment.from_file.return_value = mock_audio
        mock_audio.set_frame_rate.return_value = mock_audio
        mock_audio.set_channels.return_value = mock_audio
        mock_audio.set_sample_width.return_value = mock_audio

        convert_file(src, None, 48000, 2)

        mock_audio.set_frame_rate.assert_called_once_with(48000)
        mock_audio.set_channels.assert_called_once_with(2)

    @patch("convert.AudioSegment")
    def test_convert_file_handles_exception(self, mock_audio_segment, tmp_path: pathlib.Path, capsys):
        """Test graceful handling of conversion errors."""
        src = tmp_path / "corrupt.ogg"
        src.touch()

        # Simulate conversion error
        mock_audio_segment.from_file.side_effect = Exception("Corrupt audio file")

        assert convert_file(src, None, 16000, 1) is False

        captured = capsys.readouterr()
        assert "Failed to convert" in captured.err
        assert "Corrupt audio file" in captured.err

    def test_convert_file_rejects_invalid_rate(self, tmp_path: pathlib.Path):
        """Test that zero or negative sample rate raises ValueError."""
        src = tmp_path / "input.ogg"
        src.touch()

        with pytest.raises(ValueError, match="Sample rate must be positive"):
            convert_file(src, None, 0, 1)

        with pytest.raises(ValueError, match="Sample rate must be positive"):
            convert_file(src, None, -16000, 1)


class TestIntegration:
    """Integration tests for the convert module."""

    @patch("convert.AudioSegment")
    def test_end_to_end_conversion_workflow(self, mock_audio_segment, tmp_path: pathlib.Path):
        """Test complete workflow: collect files and convert them."""
        # Create test OGG files
        file1 = tmp_path / "audio1.ogg"
        file2 = tmp_path / "audio2.ogg"
        file1.touch()
        file2.touch()

        # Mock AudioSegment
        mock_audio = MagicMock()
        mock_audio_segment.from_file.return_value = mock_audio
        mock_audio.set_frame_rate.return_value = mock_audio
        mock_audio.set_channels.return_value = mock_audio
        mock_audio.set_sample_width.return_value = mock_audio

        # Collect files
        ogg_files = collect_ogg_files([tmp_path])
        assert len(ogg_files) == 2

        # Convert all files
        outdir = tmp_path / "output"
        for ogg_file, relative in ogg_files:
            assert convert_file(ogg_file, outdir, 16000, 1, relative=relative) is True

        # Verify conversion was called for each file
        assert mock_audio_segment.from_file.call_count == 2


def _mock_audio(mock_audio_segment: MagicMock) -> MagicMock:
    """Make AudioSegment.from_file(...).set_*(...) chain back to one mock audio object."""
    mock_audio = MagicMock()
    mock_audio_segment.from_file.return_value = mock_audio
    mock_audio.set_frame_rate.return_value = mock_audio
    mock_audio.set_channels.return_value = mock_audio
    mock_audio.set_sample_width.return_value = mock_audio
    return mock_audio


class TestOutputTree:
    """--outdir mirrors sub-folders so same-named files don't overwrite each other."""

    @patch("convert.AudioSegment")
    def test_same_name_in_different_folders(self, mock_audio_segment, tmp_path: pathlib.Path):
        mock_audio = _mock_audio(mock_audio_segment)
        for folder in ("day1", "day2"):
            (tmp_path / "in" / folder).mkdir(parents=True)
            (tmp_path / "in" / folder / "talk.ogg").touch()
        outdir = tmp_path / "out"

        for src, relative in collect_ogg_files([tmp_path / "in"]):
            convert_file(src, outdir, 16000, 1, relative=relative)

        exported = [call.args[0] for call in mock_audio.export.call_args_list]
        assert exported == [outdir / "day1" / "talk.wav", outdir / "day2" / "talk.wav"]


class TestMain:
    """Test the convert.py entry point."""

    @patch("convert.AudioSegment")
    def test_main_converts_files(self, mock_audio_segment, tmp_path: pathlib.Path):
        mock_audio = _mock_audio(mock_audio_segment)
        (tmp_path / "a.opus").touch()

        main([str(tmp_path), "--outdir", str(tmp_path / "wav")])

        mock_audio.export.assert_called_once_with(tmp_path / "wav" / "a.wav", format="wav")

    @patch("convert.AudioSegment")
    def test_main_exits_non_zero_on_failure(self, mock_audio_segment, tmp_path: pathlib.Path):
        mock_audio_segment.from_file.side_effect = Exception("Corrupt audio file")
        (tmp_path / "a.ogg").touch()

        with pytest.raises(SystemExit, match="1 of 1 files failed"):
            main([str(tmp_path)])

    def test_main_no_files(self, tmp_path: pathlib.Path):
        with pytest.raises(SystemExit, match="No .ogg/.opus files found"):
            main([str(tmp_path)])

    @pytest.mark.parametrize("rate", ["0", "-8000", "abc"])
    def test_main_rejects_bad_rate(self, rate):
        with pytest.raises(SystemExit):
            parse_args(["a.ogg", "--rate", rate])
