"""Batch features of the transcribe command: several formats from one transcription, --skip-existing, the [i/N]
progress lines and the summary, Ctrl-C, audio from standard input, folder scans, and UTF-8 output on Windows."""

from __future__ import annotations

import io
import json
import pathlib
import sys
from typing import List

import pytest

from speech_toolkit import cli
from speech_toolkit.cli import STDIN, elapsed_time, is_stdin, parse_args, utf8_when_redirected
from speech_toolkit.formats import FORMATS
from speech_toolkit.media import MEDIA_EXTENSIONS, collect_files
from tests.conftest import FakeBackend, FakeWordsBackend


class CountingBackend(FakeBackend):
    """FakeBackend that records the file of every transcription; a file named "stop" is interrupted by Ctrl-C."""

    name = "counting"
    calls: List[pathlib.Path] = []

    def transcribe(self, audio_path, language=None, task="transcribe", verbose=True):
        CountingBackend.calls.append(audio_path)
        if audio_path.stem == "stop":
            raise KeyboardInterrupt
        return super().transcribe(audio_path, language, task, verbose)


@pytest.fixture
def counting(fake_backend, monkeypatch):
    """Register CountingBackend as 'counting', next to the fake backends; it starts with no calls."""
    from speech_toolkit.backends import _BACKENDS  # with the fake backends already in

    monkeypatch.setattr("speech_toolkit.backends._BACKENDS", {**_BACKENDS, "counting": CountingBackend})
    CountingBackend.calls = []
    return CountingBackend


def run(*argv: object) -> None:
    cli.main([str(arg) for arg in argv])


def make_files(folder: pathlib.Path, *names: str) -> pathlib.Path:
    """Create empty files *names* (paths inside *folder*, sub-folders made as needed); returns *folder*."""
    for name in names:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return folder


def names(folder: pathlib.Path) -> List[str]:
    """The files under *folder*, hidden ones included, as sorted relative POSIX paths."""
    return sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file())


class TestFormatList:
    """-f takes one format, several separated by commas, or all."""

    def test_several_formats(self):
        assert parse_args(["a.mp3", "-f", "srt,vtt,json"]).format == ("srt", "vtt", "json")

    def test_all(self):
        assert parse_args(["a.mp3", "-f", "all"]).format == tuple(FORMATS)
        assert parse_args(["a.mp3", "-f", "srt,all"]).format == tuple(FORMATS)

    def test_case_spaces_and_repeats(self):
        assert parse_args(["a.mp3", "-f", "SRT, json,srt,"]).format == ("srt", "json")

    @pytest.mark.parametrize("value", ["docx", "srt,docx", ",", ""])
    def test_unknown_formats_are_refused(self, value, capsys):
        with pytest.raises(SystemExit):
            parse_args(["a.mp3", "-f", value])
        assert "invalid format" in capsys.readouterr().err


class TestSeveralFormats:
    """Several formats are rendered from one transcription and written to a file each."""

    def test_one_transcription_per_input(self, counting, tmp_path, capsys):
        folder = make_files(tmp_path / "in", "a.wav", "b.mp3")
        out = tmp_path / "out"

        run(folder, "-b", "counting", "-m", "small", "--outdir", out, "-f", "srt,vtt,json")

        assert counting.calls == [folder / "a.wav", folder / "b.mp3"]  # one call each, for three formats
        assert names(out) == ["a.json", "a.srt", "a.vtt", "b.json", "b.srt", "b.vtt"]
        assert (out / "a.srt").read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,500\nHello from a.\n"
        assert (out / "a.vtt").read_text(encoding="utf-8").startswith("WEBVTT\n")
        assert json.loads((out / "b.json").read_text(encoding="utf-8"))["text"] == " Hello from b."
        assert capsys.readouterr().out == ""

    def test_single_input_writes_next_to_it(self, counting, tmp_path, capsys):
        audio = make_files(tmp_path, "talk.mp3") / "talk.mp3"

        run(audio, "-b", "counting", "-m", "small", "-q", "-f", "txt,json")

        assert counting.calls == [audio]
        assert names(tmp_path) == ["talk.json", "talk.mp3", "talk.txt"]
        assert (tmp_path / "talk.txt").read_text(encoding="utf-8") == "Hello from talk.\n"
        assert capsys.readouterr().out == ""  # several formats go to files, not stdout

    def test_all_formats(self, counting, tmp_path):
        make_files(tmp_path / "in", "talk.wav")

        run(tmp_path / "in", "-b", "counting", "-m", "small", "-q", "--outdir", tmp_path / "out", "-f", "all")

        assert names(tmp_path / "out") == sorted(f"talk.{fmt}" for fmt in FORMATS)
        assert len(counting.calls) == 1

    def test_json_with_outdir(self, counting, tmp_path):
        audio = make_files(tmp_path, "talk.mp3") / "talk.mp3"
        out, result = tmp_path / "out", tmp_path / "result.json"

        run(audio, "-b", "counting", "-m", "small", "-q", "--outdir", out, "-f", "srt", "--json", result)

        assert names(out) == ["talk.srt"]
        assert json.loads(result.read_text(encoding="utf-8"))["language"] == "en"
        assert len(counting.calls) == 1

    def test_json_with_several_inputs_points_to_format(self, tmp_path, capsys):
        make_files(tmp_path, "a.mp3", "b.mp3")

        with pytest.raises(SystemExit):
            parse_args([str(tmp_path), "--outdir", "out", "--json", "all.json"])
        assert "add json to --format (-f srt,json)" in capsys.readouterr().err

    def test_output_takes_one_format(self, capsys):
        with pytest.raises(SystemExit):
            parse_args(["a.mp3", "-o", "a.srt", "-f", "srt,json"])
        assert "--output takes a single input file and format" in capsys.readouterr().err

    def test_line_width_asks_for_word_timings_when_any_format_is_subtitles(self, fake_backend, tmp_path):
        audio = make_files(tmp_path, "talk.mp3") / "talk.mp3"

        run(audio, "-b", "fake-words", "-q", "-f", "json,srt", "--max-line-width", "5")

        assert FakeWordsBackend.calls == [{"word_timestamps": True}]
        assert (tmp_path / "talk.srt").read_text(encoding="utf-8").startswith("1\n00:00:00,000 --> 00:00:01,000\n")


class TestSkipExisting:
    """--skip-existing keeps existing outputs, and inputs with none missing never reach the backend."""

    def test_inputs_whose_outputs_exist_are_skipped(self, counting, tmp_path, capsys):
        folder = make_files(tmp_path / "in", "a.wav", "b.wav")
        out = tmp_path / "out"
        out.mkdir()
        for name in ("a.srt", "a.json"):
            (out / name).write_text("old\n", encoding="utf-8")

        run(folder, "-b", "counting", "-m", "small", "--outdir", out, "-f", "srt,json", "--skip-existing")

        assert counting.calls == [folder / "b.wav"]
        assert (out / "a.srt").read_text(encoding="utf-8") == "old\n"
        assert names(out) == ["a.json", "a.srt", "b.json", "b.srt"]
        captured = capsys.readouterr()
        assert captured.out == ""
        assert f"[1/2] {folder / 'a.wav'} (skipped: its outputs exist)\n" in captured.err
        assert f"[2/2] {folder / 'b.wav'}\n" in captured.err
        assert ": 1 done, 1 skipped, 0 failed." in captured.err

    def test_nothing_left_to_do_loads_no_model(self, counting, tmp_path, capsys):
        folder = make_files(tmp_path, "a.wav", "a.txt", "b.wav", "b.txt")

        run(folder, "-b", "counting", "-m", "small", "--skip-existing")

        assert counting.calls == []
        assert counting.loaded == 0
        err = capsys.readouterr().err
        assert "Loading" not in err
        assert ": 0 done, 2 skipped, 0 failed." in err

    def test_missing_outputs_are_written_and_existing_ones_kept(self, counting, tmp_path):
        folder = make_files(tmp_path, "talk.wav")
        (folder / "talk.srt").write_text("edited by hand\n", encoding="utf-8")

        run(folder, "-b", "counting", "-m", "small", "-q", "-f", "srt,json", "--skip-existing")

        assert counting.calls == [folder / "talk.wav"]  # one output missing: transcribed again
        assert (folder / "talk.srt").read_text(encoding="utf-8") == "edited by hand\n"
        assert json.loads((folder / "talk.json").read_text(encoding="utf-8"))["text"] == " Hello from talk."

    def test_outputs_are_overwritten_by_default(self, counting, tmp_path):
        folder = make_files(tmp_path, "talk.wav")
        (folder / "talk.txt").write_text("old\n", encoding="utf-8")

        run(folder, "-b", "counting", "-m", "small", "-q")

        assert (folder / "talk.txt").read_text(encoding="utf-8") == "Hello from talk.\n"

    def test_stdout_is_always_written(self, counting, tmp_path, capsys):
        audio = make_files(tmp_path, "talk.mp3") / "talk.mp3"
        result = tmp_path / "talk.json"
        result.write_text("old\n", encoding="utf-8")

        run(audio, "-b", "counting", "-m", "small", "-q", "--json", result, "--skip-existing")

        assert capsys.readouterr().out == "Hello from talk.\n"
        assert result.read_text(encoding="utf-8") == "old\n"

    def test_single_file_says_it_was_skipped(self, counting, tmp_path, capsys):
        audio = make_files(tmp_path, "talk.mp3", "talk.srt") / "talk.mp3"

        run(audio, "-b", "counting", "-m", "small", "-o", tmp_path / "talk.srt", "--skip-existing")

        assert counting.calls == []
        err = capsys.readouterr().err
        assert err == f"{audio} (skipped: its outputs exist)\n"  # no [i/N] counter or summary for one file


class TestProgress:
    """[i/N] lines and the summary go to stderr in batch mode, and -q hides them; stdout stays clean."""

    def test_counter_errors_and_summary(self, counting, tmp_path, capsys):
        folder = make_files(tmp_path, "a.wav", "bad.wav", "c.wav")

        with pytest.raises(SystemExit, match="1 of 3 files failed"):
            run(folder, "-b", "counting", "-m", "small")

        captured = capsys.readouterr()
        assert captured.out == ""
        lines = [line for line in captured.err.splitlines() if line.startswith(("[", "Error", "→", "Finished"))]
        assert lines[:-1] == [
            f"[1/3] {folder / 'a.wav'}",
            f"→ {folder / 'a.txt'}",
            f"[2/3] {folder / 'bad.wav'}",
            f"Error: {folder / 'bad.wav'}: Failed to load audio",
            f"[3/3] {folder / 'c.wav'}",
            f"→ {folder / 'c.txt'}",
        ]
        assert lines[-1].startswith("Finished in ")
        assert lines[-1].endswith(" s: 2 done, 0 skipped, 1 failed.")

    def test_quiet_hides_progress_but_not_errors(self, counting, tmp_path, capsys):
        folder = make_files(tmp_path, "a.wav", "bad.wav")

        with pytest.raises(SystemExit):
            run(folder, "-b", "counting", "-m", "small", "-q")

        err = capsys.readouterr().err
        assert "[1/2]" not in err and "Finished" not in err and "→" not in err
        assert f"Error: {folder / 'bad.wav'}: Failed to load audio" in err

    def test_single_file_has_no_counter_or_summary(self, counting, tmp_path, capsys):
        audio = make_files(tmp_path, "talk.mp3") / "talk.mp3"

        run(audio, "-b", "counting", "-m", "small")

        captured = capsys.readouterr()
        assert captured.out == "Hello from talk.\n"
        assert "[1/1]" not in captured.err and "Finished" not in captured.err

    @pytest.mark.parametrize(
        "seconds, shown", [(0.04, "0.0 s"), (12.34, "12.3 s"), (245.9, "4:05"), (3723, "1:02:03"), (36000, "10:00:00")]
    )
    def test_elapsed_time(self, seconds, shown):
        assert elapsed_time(seconds) == shown


class TestInterrupt:
    """Ctrl-C shows the summary, exits with 130 and leaves no half-written file."""

    def test_summary_and_exit_status(self, counting, tmp_path, capsys):
        folder = make_files(tmp_path, "a.wav", "stop.wav", "z.wav")

        with pytest.raises(SystemExit) as exit_info:
            run(folder, "-b", "counting", "-m", "small")

        assert exit_info.value.code == 130
        assert counting.calls == [folder / "a.wav", folder / "stop.wav"]  # z.wav never started
        assert names(folder) == ["a.txt", "a.wav", "stop.wav", "z.wav"]
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.splitlines()[-1].startswith("Interrupted after ")
        assert captured.err.splitlines()[-1].endswith(": 1 done, 0 skipped, 0 failed, 2 left.")

    def test_quiet_hides_the_summary(self, counting, tmp_path, capsys):
        audio = make_files(tmp_path, "stop.wav") / "stop.wav"

        with pytest.raises(SystemExit) as exit_info:
            run(audio, "-b", "counting", "-m", "small", "-q")

        assert exit_info.value.code == 130
        assert "Interrupted" not in capsys.readouterr().err

    def test_single_file_shows_the_summary_too(self, counting, tmp_path, capsys):
        audio = make_files(tmp_path, "stop.wav") / "stop.wav"

        with pytest.raises(SystemExit):
            run(audio, "-b", "counting", "-m", "small")

        assert ": 0 done, 0 skipped, 0 failed, 1 left." in capsys.readouterr().err

    def test_ctrl_c_while_writing_keeps_the_previous_file(self, counting, tmp_path, monkeypatch):
        audio = make_files(tmp_path, "talk.mp3") / "talk.mp3"
        out = tmp_path / "talk.srt"
        out.write_text("previous run\n", encoding="utf-8")

        def interrupted_write(path: pathlib.Path, text: str) -> None:
            path.write_text(text[:5], encoding="utf-8")  # half the file, then Ctrl-C
            raise KeyboardInterrupt

        monkeypatch.setattr("speech_toolkit.cli.write_text", interrupted_write)

        with pytest.raises(SystemExit) as exit_info:
            run(audio, "-b", "counting", "-m", "small", "-q", "-o", out)

        assert exit_info.value.code == 130
        assert out.read_text(encoding="utf-8") == "previous run\n"
        assert names(tmp_path) == ["talk.mp3", "talk.srt"]  # no temporary file left behind

    def test_ctrl_c_while_writing_a_new_file_leaves_none(self, counting, tmp_path, monkeypatch):
        folder = make_files(tmp_path, "talk.wav")

        def interrupted_write(path: pathlib.Path, text: str) -> None:
            path.write_text(text[:5], encoding="utf-8")
            raise KeyboardInterrupt

        monkeypatch.setattr("speech_toolkit.cli.write_text", interrupted_write)

        with pytest.raises(SystemExit):
            run(folder, "-b", "counting", "-m", "small", "-q", "-f", "srt,json")

        assert names(folder) == ["talk.wav"]

    def test_write_error_fails_the_file_and_leaves_no_temporary_file(self, counting, tmp_path, monkeypatch, capsys):
        folder = make_files(tmp_path, "a.wav", "b.wav")

        def failing_write(path: pathlib.Path, text: str) -> None:
            path.write_text(text[:5], encoding="utf-8")
            if path.name.startswith(".a."):
                raise OSError("No space left on device")
            path.write_text(text, encoding="utf-8")

        monkeypatch.setattr("speech_toolkit.cli.write_text", failing_write)

        with pytest.raises(SystemExit, match="1 of 2 files failed"):
            run(folder, "-b", "counting", "-m", "small", "-q")

        assert names(folder) == ["a.wav", "b.txt", "b.wav"]
        assert (folder / "b.txt").read_text(encoding="utf-8") == "Hello from b."
        assert f"Error: {folder / 'a.wav'}: No space left on device" in capsys.readouterr().err


class FakeStdin:
    """sys.stdin with *data* piped into it, or a terminal."""

    def __init__(self, data: bytes = b"", tty: bool = False) -> None:
        self.buffer = io.BytesIO(data)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@pytest.fixture
def decoded(monkeypatch):
    """Decoding in-memory audio without ffmpeg: one second of silence. Lists what it was asked to decode."""
    import numpy as np

    received: list = []

    def decode(audio):
        received.append(audio)
        if audio.read() == b"not audio":
            raise RuntimeError("Failed to load audio: invalid data")
        return np.zeros(16000, dtype=np.float32)

    monkeypatch.setattr("speech_toolkit.api._decode", decode)
    return received


class TestStdin:
    """`transcribe -` reads the audio piped into standard input."""

    def test_transcript_to_stdout(self, fake_waveform_backend, decoded, monkeypatch, capsys):
        stdin = FakeStdin(b"audio")
        monkeypatch.setattr(sys, "stdin", stdin)

        run("-", "-b", "fake-waveform", "-q")

        assert decoded == [stdin.buffer]  # handed over as it is, to be read in chunks
        assert capsys.readouterr().out.startswith("16000 samples ")

    def test_outdir_names_the_outputs_stdin(self, fake_waveform_backend, decoded, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(sys, "stdin", FakeStdin(b"audio"))

        run("-", "-b", "fake-waveform", "--outdir", tmp_path / "out", "-f", "srt,json")

        assert names(tmp_path / "out") == ["stdin.json", "stdin.srt"]
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "[1/1] standard input\n" in captured.err

    def test_output_file(self, fake_waveform_backend, decoded, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "stdin", FakeStdin(b"audio"))

        run("-", "-b", "fake-waveform", "-q", "-o", tmp_path / "talk.vtt")

        assert (tmp_path / "talk.vtt").read_text(encoding="utf-8").startswith("WEBVTT\n")

    def test_errors_name_standard_input(self, fake_waveform_backend, decoded, monkeypatch, capsys):
        monkeypatch.setattr(sys, "stdin", FakeStdin(b"not audio"))

        with pytest.raises(SystemExit) as exit_info:
            run("-", "-b", "fake-waveform", "-q")

        assert exit_info.value.code == 1
        assert "Error: standard input: Failed to load audio: invalid data" in capsys.readouterr().err

    @pytest.mark.parametrize("stdin", [FakeStdin(tty=True), None], ids=["terminal", "none"])
    def test_needs_piped_audio(self, fake_waveform_backend, decoded, monkeypatch, stdin):
        monkeypatch.setattr(sys, "stdin", stdin)

        with pytest.raises(SystemExit, match=r"piped into standard input, as in: cat talk.mp3 \| transcribe -"):
            run("-", "-b", "fake-waveform")
        assert FakeBackend.loaded == 0 and decoded == []

    @pytest.mark.parametrize(
        "argv, message",
        [
            (["-", "a.mp3"], "must be the only input"),
            (["-", "-f", "srt,json"], "several formats from standard input need --outdir"),
        ],
    )
    def test_invalid_combinations(self, argv, message, capsys):
        with pytest.raises(SystemExit):
            parse_args(argv)
        assert message in capsys.readouterr().err

    def test_a_file_named_dash(self, fake_backend, tmp_path, monkeypatch, capsys):
        """``./-`` is a file named "-", not standard input."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "-").touch()

        args = parse_args(["./-"])
        assert not is_stdin(args.audio[0]) and is_stdin(parse_args(["-"]).audio[0])
        assert parse_args(["-"]).audio == [STDIN]

        run("./-", "-b", "fake", "-q")

        assert capsys.readouterr().out == "Hello from -.\n"


class TestFolderScan:
    """Folders are searched for media files, leaving out hidden files, AppleDouble files and TypeScript."""

    def test_hidden_and_appledouble_files_are_left_out(self, tmp_path):
        make_files(
            tmp_path, "a.mp3", "._a.mp3", ".hidden.wav", ".cache/x.mp3", "sub/._b.wav", "sub/b.wav", "sub/.git/c.wav"
        )

        found = collect_files([tmp_path], MEDIA_EXTENSIONS)

        assert found == [
            (tmp_path / "a.mp3", pathlib.Path("a.mp3")),
            (tmp_path / "sub" / "b.wav", pathlib.Path("sub/b.wav")),
        ]

    def test_a_hidden_file_given_by_name_is_kept(self, tmp_path):
        make_files(tmp_path, ".hidden.wav")

        assert collect_files([tmp_path / ".hidden.wav"], MEDIA_EXTENSIONS) == [
            (tmp_path / ".hidden.wav", pathlib.Path(".hidden.wav"))
        ]

    def test_a_hidden_folder_given_by_name_is_searched(self, tmp_path):
        folder = make_files(tmp_path / ".recordings", "talk.wav")

        assert collect_files([folder], MEDIA_EXTENSIONS) == [(folder / "talk.wav", pathlib.Path("talk.wav"))]

    @pytest.mark.parametrize(
        "extension",
        [".aif", ".aiff", ".m4b", ".avi", ".3gp", ".m4v", ".amr", ".mpg", ".mpeg", ".mka", ".caf", ".mts", ".m2ts"],
    )
    def test_more_audio_and_video_extensions(self, tmp_path, extension):
        make_files(tmp_path, f"clip{extension}", f"CLIP2{extension.upper()}")

        found = {rel.name for _, rel in collect_files([tmp_path], MEDIA_EXTENSIONS)}
        assert found == {f"clip{extension}", f"CLIP2{extension.upper()}"}

    def test_typescript_is_not_scanned(self, tmp_path):
        make_files(tmp_path, "index.ts", "talk.mp3")

        assert [rel.name for _, rel in collect_files([tmp_path], MEDIA_EXTENSIONS)] == ["talk.mp3"]

    def test_batch_skips_appledouble_files(self, counting, tmp_path):
        folder = make_files(tmp_path, "talk.mp3", "._talk.mp3")

        run(folder, "-b", "counting", "-m", "small", "-q")

        assert counting.calls == [folder / "talk.mp3"]


class TestUtf8Stdout:
    """On Windows, a redirected stdout writes UTF-8, so any transcript can be saved."""

    @staticmethod
    def stream(tty: bool = False) -> io.TextIOWrapper:
        class Stream(io.TextIOWrapper):
            def isatty(self) -> bool:
                return tty

        return Stream(io.BytesIO(), encoding="cp1252")

    def test_redirected_on_windows(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.delenv("PYTHONIOENCODING", raising=False)
        stream = self.stream()

        utf8_when_redirected(stream)

        assert stream.encoding == "utf-8"
        stream.write("你好，世界")  # cp1252 can't encode this

    @pytest.mark.parametrize(
        "platform, tty, variable", [("win32", True, None), ("win32", False, "cp1252"), ("linux", False, None)]
    )
    def test_left_alone(self, monkeypatch, platform, tty, variable):
        """A console, PYTHONIOENCODING and other systems keep their encoding."""
        monkeypatch.setattr(sys, "platform", platform)
        if variable:
            monkeypatch.setenv("PYTHONIOENCODING", variable)
        else:
            monkeypatch.delenv("PYTHONIOENCODING", raising=False)
        stream = self.stream(tty)

        utf8_when_redirected(stream)

        assert stream.encoding == "cp1252"

    def test_streams_without_reconfigure(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.delenv("PYTHONIOENCODING", raising=False)

        utf8_when_redirected(io.StringIO())  # nothing to do, and no error
