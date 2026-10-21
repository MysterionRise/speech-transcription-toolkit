#!/usr/bin/env python3
"""Tests for the ogg2wav converter (speech_toolkit.convert)."""

from __future__ import annotations

import array
import pathlib
import shutil
import subprocess
import sys
import wave
from typing import Iterator, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from speech_toolkit.convert import collect_ogg_files, convert_file, decode, main, parse_args, plan_outputs
from speech_toolkit.errors import AudioDecodeError

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PCM = b"\x01\x00\xff\x7f" * 8  # 16 bytes: 8 mono frames, or 4 stereo ones


def read_wav(path: pathlib.Path) -> tuple:
    """(channels, sample width, rate, frames) of a WAV file."""
    with wave.open(str(path), "rb") as wav:
        return wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.readframes(wav.getnframes())


@pytest.fixture
def fake_decode() -> Iterator[MagicMock]:
    """Replace ffmpeg decoding with a stand-in that returns PCM."""
    with patch("speech_toolkit.convert.decode", return_value=PCM) as decode_mock:
        yield decode_mock


class TestParseArgs:
    """Test command-line argument parsing for ogg2wav."""

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

    @pytest.mark.parametrize("rate", ["0", "-8000", "abc"])
    def test_rejects_bad_rate(self, rate):
        with pytest.raises(SystemExit):
            parse_args(["a.ogg", "--rate", rate])


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
        """A file given by name that isn't OGG/Opus is skipped, with a message on stderr only."""
        mp3_file = tmp_path / "audio.mp3"
        mp3_file.touch()

        result = collect_ogg_files([mp3_file])
        assert len(result) == 0

        captured = capsys.readouterr()
        assert "Skipping unsupported file" in captured.err
        assert captured.out == ""


class TestPlanOutputs:
    """Each input gets its own WAV: next to it, or under --outdir mirroring sub-folders."""

    def test_next_to_the_input(self, tmp_path: pathlib.Path):
        src = tmp_path / "talk.ogg"
        assert plan_outputs([(src, pathlib.Path("talk.ogg"))], None) == [(src, tmp_path / "talk.wav")]

    def test_outdir_mirrors_sub_folders(self, tmp_path: pathlib.Path):
        """Same-named files in different folders don't overwrite each other."""
        for folder in ("day1", "day2"):
            (tmp_path / "in" / folder).mkdir(parents=True)
            (tmp_path / "in" / folder / "talk.ogg").touch()
        outdir = tmp_path / "out"

        jobs = plan_outputs(collect_ogg_files([tmp_path / "in"]), outdir)

        assert [dest for _, dest in jobs] == [outdir / "day1" / "talk.wav", outdir / "day2" / "talk.wav"]

    @pytest.mark.parametrize("outdir", [None, "out"])
    def test_same_stem_keeps_the_full_name(self, tmp_path: pathlib.Path, outdir):
        """a.ogg and a.opus would both become a.wav; they become a.ogg.wav and a.opus.wav, other files keep x.wav."""
        for name in ("a.ogg", "a.opus", "b.oga"):
            (tmp_path / name).touch()
        out = tmp_path / outdir if outdir else tmp_path

        jobs = plan_outputs(collect_ogg_files([tmp_path]), out)

        assert [(src.name, dest) for src, dest in jobs] == [
            ("a.ogg", out / "a.ogg.wav"),
            ("a.opus", out / "a.opus.wav"),
            ("b.oga", out / "b.wav"),
        ]

    def test_inputs_that_still_clash(self, tmp_path: pathlib.Path):
        """Two folders with the same file, into one --outdir: even the full names clash."""
        files = [(tmp_path / folder / "a.ogg", pathlib.Path("a.ogg")) for folder in ("x", "y")]

        with pytest.raises(ValueError, match="same WAV file"):
            plan_outputs(files, tmp_path / "out")


class TestDecode:
    """decode() decodes with media.load_audio and turns its samples back into 16-bit PCM."""

    def test_through_load_audio(self, tmp_path: pathlib.Path):
        src = tmp_path / "talk.opus"
        stereo = np.array([[0.0, 0.5], [-1.0, 32767 / 32768]], dtype=np.float32)  # what ffmpeg's samples become
        with patch("speech_toolkit.convert.load_audio", return_value=stereo) as load_audio:
            pcm = decode(src, 22050, 2)

        load_audio.assert_called_once_with(src, sr=22050, channels=2)
        assert pcm == np.array([0, 16384, -32768, 32767], dtype=np.int16).tobytes()  # exact, and interleaved

    def test_errors_pass_through(self, tmp_path: pathlib.Path):
        error = AudioDecodeError("Failed to load audio: Invalid data found when processing input")
        with patch("speech_toolkit.convert.load_audio", side_effect=error):
            with pytest.raises(AudioDecodeError, match="Invalid data found"):
                decode(tmp_path / "a.ogg", 16000, 1)


class TestConvertFile:
    """convert_file() writes a 16-bit WAV and prints only its path on stdout."""

    @pytest.mark.parametrize("rate, channels", [(16000, 1), (48000, 2)])
    def test_writes_a_16_bit_wav(self, tmp_path: pathlib.Path, fake_decode, capsys, rate, channels):
        src, dest = tmp_path / "input.ogg", tmp_path / "input.wav"

        assert convert_file(src, dest, rate, channels) is True

        fake_decode.assert_called_once_with(src, rate, channels)
        assert read_wav(dest) == (channels, 2, rate, PCM)
        assert capsys.readouterr().out == f"{dest}\n"

    def test_creates_the_output_folder(self, tmp_path: pathlib.Path, fake_decode):
        dest = tmp_path / "new" / "nested" / "input.wav"

        assert convert_file(tmp_path / "input.ogg", dest, 16000, 1) is True
        assert dest.is_file()

    def test_skips_an_existing_wav(self, tmp_path: pathlib.Path, fake_decode, capsys):
        dest = tmp_path / "input.wav"
        dest.write_bytes(b"old")

        assert convert_file(tmp_path / "input.ogg", dest, 16000, 1) is True

        fake_decode.assert_not_called()
        assert dest.read_bytes() == b"old"
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "exists; skipping (use --overwrite)" in captured.err

    def test_overwrites_with_overwrite(self, tmp_path: pathlib.Path, fake_decode):
        dest = tmp_path / "input.wav"
        dest.write_bytes(b"old")

        assert convert_file(tmp_path / "input.ogg", dest, 16000, 1, overwrite=True) is True
        assert read_wav(dest)[3] == PCM

    def test_decode_error(self, tmp_path: pathlib.Path, capsys):
        """A file ffmpeg can't decode is reported on stderr, and no WAV is left behind."""
        src, dest = tmp_path / "corrupt.ogg", tmp_path / "out" / "corrupt.wav"
        with patch("speech_toolkit.convert.decode", side_effect=AudioDecodeError("Failed to load audio: junk")):
            assert convert_file(src, dest, 16000, 1) is False

        assert not dest.parent.exists()
        captured = capsys.readouterr()
        assert captured.out == ""
        assert f"Error: failed to convert {src}: Failed to load audio: junk" in captured.err

    def test_write_error(self, tmp_path: pathlib.Path, fake_decode, capsys):
        """An output folder that can't be created fails this file only."""
        (tmp_path / "taken").write_text("a file, not a folder")

        assert convert_file(tmp_path / "a.ogg", tmp_path / "taken" / "a.wav", 16000, 1) is False
        assert "Error: failed to convert" in capsys.readouterr().err

    def test_rejects_invalid_rate(self, tmp_path: pathlib.Path):
        """Test that zero or negative sample rate raises ValueError."""
        for rate in (0, -16000):
            with pytest.raises(ValueError, match="Sample rate must be positive"):
                convert_file(tmp_path / "input.ogg", tmp_path / "input.wav", rate, 1)


class TestMain:
    """Test the ogg2wav entry point."""

    def test_converts_files(self, tmp_path: pathlib.Path, fake_decode, capsys):
        (tmp_path / "a.opus").touch()

        main([str(tmp_path), "--outdir", str(tmp_path / "wav"), "--rate", "8000", "--channels", "2"])

        fake_decode.assert_called_once_with(tmp_path / "a.opus", 8000, 2)
        assert read_wav(tmp_path / "wav" / "a.wav")[:3] == (2, 2, 8000)
        assert capsys.readouterr().out == f"{tmp_path / 'wav' / 'a.wav'}\n"

    def test_same_stem_gives_two_outputs(self, tmp_path: pathlib.Path, fake_decode, capsys):
        """a.ogg and a.opus in one folder no longer overwrite each other; stdout lists the WAVs, nothing else."""
        (tmp_path / "a.ogg").touch()
        (tmp_path / "a.opus").touch()

        main([str(tmp_path)])

        outputs: List[pathlib.Path] = [tmp_path / "a.ogg.wav", tmp_path / "a.opus.wav"]
        assert all(path.is_file() for path in outputs)
        assert not (tmp_path / "a.wav").exists()
        assert capsys.readouterr().out.splitlines() == [str(path) for path in outputs]

    def test_exits_non_zero_on_failure(self, tmp_path: pathlib.Path):
        (tmp_path / "a.ogg").touch()

        with patch("speech_toolkit.convert.decode", side_effect=AudioDecodeError("junk")):
            with pytest.raises(SystemExit, match="1 of 1 files failed"):
                main([str(tmp_path)])

    def test_clashing_outputs(self, tmp_path: pathlib.Path, fake_decode):
        for folder in ("x", "y"):
            (tmp_path / folder).mkdir()
            (tmp_path / folder / "a.ogg").touch()

        with pytest.raises(SystemExit, match="Error: several inputs would write the same WAV file"):
            main([str(tmp_path / "x"), str(tmp_path / "y"), "--outdir", str(tmp_path / "out")])
        fake_decode.assert_not_called()

    def test_no_files(self, tmp_path: pathlib.Path):
        with pytest.raises(SystemExit, match="No .ogg/.opus files found"):
            main([str(tmp_path)])


class TestImport:
    def test_import_has_no_warnings_or_pydub(self):
        """Importing the converter warns about nothing (pydub did, without ffmpeg and on Python 3.12+)."""
        code = "import sys, speech_toolkit.convert; print('pydub' in sys.modules)"
        run = subprocess.run(
            [sys.executable, "-W", "error", "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        )
        assert run.stdout.strip() == "False"
        assert run.stderr == ""


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg on PATH")
class TestWithFfmpeg:
    """ogg2wav converts real OGG and Opus files with the real ffmpeg."""

    @staticmethod
    def encode(path: pathlib.Path, codec: str) -> pathlib.Path:
        """One second of stereo audio, 440 Hz on the left and 880 Hz on the right, at 48 kHz, encoded with *codec*."""
        tones = "sine=frequency=440:duration=1:sample_rate=48000"
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", tones, "-f", "lavfi"]
        cmd += ["-i", tones.replace("440", "880"), "-filter_complex", "amerge=inputs=2", "-c:a", codec, str(path)]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            pytest.skip(f"this ffmpeg can't encode {codec}: {done.stderr.strip()}")
        return path

    @pytest.fixture
    def recordings(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """A folder with a.ogg (Vorbis) and a.opus (Opus): the same name, different extensions."""
        folder = tmp_path / "recordings"
        folder.mkdir()
        self.encode(folder / "a.ogg", "libvorbis")
        self.encode(folder / "a.opus", "libopus")
        return folder

    def test_converts_ogg_and_opus(self, recordings: pathlib.Path, tmp_path: pathlib.Path, capsys):
        outdir = tmp_path / "wav"

        main([str(recordings), "--outdir", str(outdir)])

        outputs = [outdir / "a.ogg.wav", outdir / "a.opus.wav"]
        assert capsys.readouterr().out.splitlines() == [str(path) for path in outputs]
        for path in outputs:
            channels, width, rate, frames = read_wav(path)
            assert (channels, width, rate) == (1, 2, 16000)
            assert len(frames) // 2 == pytest.approx(16000, abs=400)  # one second; codecs pad a little

    def test_stereo(self, recordings: pathlib.Path, tmp_path: pathlib.Path):
        """--channels 2 keeps both channels, which differ here."""
        main([str(recordings / "a.ogg"), "--outdir", str(tmp_path / "stereo"), "--rate", "22050", "--channels", "2"])
        channels, width, rate, frames = read_wav(tmp_path / "stereo" / "a.wav")
        assert (channels, width, rate) == (2, 2, 22050)
        samples = array.array("h", frames)
        if sys.byteorder == "big":
            samples.byteswap()
        assert samples[0::2] != samples[1::2]

    def test_reports_a_file_it_cant_decode(self, tmp_path: pathlib.Path, capsys):
        broken = tmp_path / "broken.ogg"
        broken.write_bytes(b"not audio at all")

        with pytest.raises(SystemExit, match="1 of 1 files failed"):
            main([str(broken)])

        captured = capsys.readouterr()
        assert captured.out == ""
        assert f"Error: failed to convert {broken}: Failed to load audio:" in captured.err
        assert not (tmp_path / "broken.wav").exists()
