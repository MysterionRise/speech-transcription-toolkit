#!/usr/bin/env python3
"""Tests for the transcribe command line (speech_toolkit.cli)."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import pytest

from speech_toolkit import __version__, cli
from speech_toolkit.cli import parse_args, plan_outputs, positive_int, show_backends, show_models

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class TestParseArgs:
    """Test command-line argument parsing."""

    def test_parse_args_minimal(self):
        """Test parsing with only required audio argument."""
        args = parse_args(["audio.mp3"])
        assert args.audio == [pathlib.Path("audio.mp3")]
        assert args.model is None  # Uses backend default
        assert args.task == "transcribe"
        assert args.diarize is False
        assert args.quiet is False
        assert args.backend == "whisper"
        assert args.format is None

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
        args = parse_args(["audio.mp3", "--diarize", "--hf-token", "test_token", "--num-speakers", "2"])
        assert args.diarize is True
        assert args.hf_token == "test_token"
        assert args.num_speakers == 2

    def test_parse_args_output_files(self):
        """Test parsing with output file specifications."""
        args = parse_args(["audio.mp3", "-o", "transcript.txt", "--json", "result.json"])
        assert args.output == pathlib.Path("transcript.txt")
        assert args.json == pathlib.Path("result.json")

    def test_parse_args_format(self):
        """Test parsing an explicit output format."""
        args = parse_args(["audio.mp3", "-f", "vtt"])
        assert args.format == "vtt"

    def test_parse_args_invalid_format(self):
        """Test that unknown formats are rejected."""
        with pytest.raises(SystemExit):
            parse_args(["audio.mp3", "-f", "docx"])

    def test_parse_args_batch(self):
        """Test parsing several inputs with an output folder."""
        args = parse_args(["a.mp3", "b.wav", "--outdir", "out"])
        assert args.audio == [pathlib.Path("a.mp3"), pathlib.Path("b.wav")]
        assert args.outdir == pathlib.Path("out")

    def test_parse_args_accuracy_options(self):
        """Test parsing --prompt, --vad, --word-timestamps and --max-line-width."""
        args = parse_args(["a.mp3", "--prompt", "Kubernetes", "--vad", "--word-timestamps", "--max-line-width", "42"])
        assert (args.prompt, args.vad, args.word_timestamps, args.max_line_width) == ("Kubernetes", True, True, 42)

        args = parse_args(["a.mp3"])
        assert (args.prompt, args.vad, args.word_timestamps, args.max_line_width) == (None, False, False, None)

    def test_parse_args_backend_selection(self):
        """Test parsing with backend selection."""
        args = parse_args(["audio.mp3", "--backend", "voxtral"])
        assert args.backend == "voxtral"

    def test_parse_args_short_backend_flag(self):
        """Test parsing with short backend flag."""
        args = parse_args(["audio.mp3", "-b", "faster-whisper"])
        assert args.backend == "faster-whisper"

    def test_parse_args_list_backends(self):
        """Test parsing with list-backends flag."""
        args = parse_args(["--list-backends"])
        assert args.list_backends is True
        assert args.audio == []  # Audio not required for info flags

    def test_parse_args_list_models(self):
        """Test parsing with list-models flag."""
        args = parse_args(["--list-models"])
        assert args.list_models is True
        assert args.audio == []

    def test_parse_args_list_models_with_backend(self):
        """Test parsing with list-models and backend."""
        args = parse_args(["--list-models", "--backend", "voxtral"])
        assert args.list_models is True
        assert args.backend == "voxtral"

    def test_parse_args_version(self, capsys):
        """Test --version prints the package version."""
        with pytest.raises(SystemExit):
            parse_args(["--version"])
        assert capsys.readouterr().out.strip() == f"transcribe {__version__}"

    def test_parse_args_no_audio_raises_error(self):
        """Test that missing audio argument (without --list-* flags) raises SystemExit."""
        with pytest.raises(SystemExit):
            parse_args([])

    @pytest.mark.parametrize(
        "argv",
        [
            ["a.mp3", "b.mp3", "-o", "out.txt"],  # --output needs a single input
            ["a.mp3", "b.mp3", "--json", "out.json"],  # so does --json
            ["a.mp3", "--outdir", "out", "-o", "x.txt"],  # --outdir means batch mode
            ["a.mp3", "--num-speakers", "2"],  # speaker hints need --diarize
            ["a.mp3", "--diarize", "--min-speakers", "0"],  # hints must be positive
        ],
    )
    def test_parse_args_rejects_invalid_combinations(self, argv):
        """Test option combinations that can't work are rejected up front."""
        with pytest.raises(SystemExit):
            parse_args(argv)

    def test_parse_args_folder_counts_as_batch(self, tmp_path):
        """Test that a folder input can't be combined with --output."""
        with pytest.raises(SystemExit):
            parse_args([str(tmp_path), "-o", "out.txt"])


class TestPositiveInt:
    """Test the positive_int argparse type."""

    def test_accepts_positive(self):
        assert positive_int("3") == 3

    @pytest.mark.parametrize("value", ["0", "-2"])
    def test_rejects_non_positive(self, value):
        with pytest.raises(argparse.ArgumentTypeError):
            positive_int(value)


class TestShowBackends:
    """Test backend listing functionality."""

    def test_show_backends_output(self, capsys):
        """Test that show_backends displays backend info."""
        show_backends()

        captured = capsys.readouterr()
        assert "whisper" in captured.out
        assert "faster-whisper" in captured.out
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

    def test_show_models_invalid_backend(self):
        """Test show_models with invalid backend."""
        with pytest.raises(SystemExit, match="Unknown backend"):
            show_models("invalid_backend")

    def test_listing_does_not_import_heavy_libraries(self):
        """--help/--list-* must stay fast: no torch/whisper/pyannote import just to print names."""
        heavy = ("torch", "whisper", "pyannote.audio", "transformers", "faster_whisper", "numpy")
        code = (
            "import sys, speech_toolkit.cli as cli; cli.show_backends(); "
            f"print([m for m in {heavy!r} if m in sys.modules])"
        )
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT)
        assert run.stdout.splitlines()[-1] == "[]"


class TestEntryPoints:
    """The checkout shims and ``python -m speech_toolkit`` run the same commands as the installed scripts."""

    def _run(self, *argv: str) -> str:
        run = subprocess.run([sys.executable, *argv], capture_output=True, text=True, check=True, cwd=REPO_ROOT)
        return run.stdout

    def test_main_py_shim(self):
        assert "faster-whisper" in self._run("main.py", "--list-backends")

    def test_python_dash_m(self):
        assert self._run("-m", "speech_toolkit", "--version").strip() == f"transcribe {__version__}"

    def test_convert_py_shim(self):
        assert "usage: ogg2wav" in self._run("convert.py", "--help")


class TestPlanOutputs:
    """Test how inputs map to output files."""

    def test_single_file_goes_to_output_or_stdout(self):
        args = parse_args(["talk.mp3"])
        assert plan_outputs(args, "txt") == [(pathlib.Path("talk.mp3"), None)]

        args = parse_args(["talk.mp3", "-o", "talk.srt"])
        assert plan_outputs(args, "srt") == [(pathlib.Path("talk.mp3"), pathlib.Path("talk.srt"))]

    def test_folder_outputs_next_to_inputs(self, tmp_path: pathlib.Path):
        (tmp_path / "a.MP3").touch()
        (tmp_path / "notes.txt").touch()  # not media: ignored
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.wav").touch()

        jobs = plan_outputs(parse_args([str(tmp_path)]), "vtt")

        assert jobs == [
            (tmp_path / "a.MP3", tmp_path / "a.vtt"),
            (tmp_path / "sub" / "b.wav", tmp_path / "sub" / "b.vtt"),
        ]

    def test_outdir_mirrors_folder_tree(self, tmp_path: pathlib.Path):
        src = tmp_path / "src"
        (src / "day1").mkdir(parents=True)
        (src / "day2").mkdir()
        (src / "day1" / "talk.mp3").touch()
        (src / "day2" / "talk.mp3").touch()  # same name, different folder: no collision
        out = tmp_path / "out"

        jobs = plan_outputs(parse_args([str(src), "--outdir", str(out)]), "srt")

        assert [dest for _, dest in jobs] == [out / "day1" / "talk.srt", out / "day2" / "talk.srt"]

    def test_same_stem_keeps_full_name(self, tmp_path: pathlib.Path):
        (tmp_path / "talk.mp3").touch()
        (tmp_path / "talk.wav").touch()

        jobs = plan_outputs(parse_args([str(tmp_path)]), "txt")

        assert [dest for _, dest in jobs] == [tmp_path / "talk.mp3.txt", tmp_path / "talk.wav.txt"]

    def test_unresolvable_collision_exits(self, tmp_path: pathlib.Path):
        for folder in ("a", "b"):
            (tmp_path / folder).mkdir()
            (tmp_path / folder / "talk.mp3").touch()
        argv = [str(tmp_path / "a" / "talk.mp3"), str(tmp_path / "b" / "talk.mp3"), "--outdir", str(tmp_path / "o")]

        with pytest.raises(SystemExit, match="same output file"):
            plan_outputs(parse_args(argv), "txt")


class TestMain:
    """End-to-end CLI flow with a fake backend."""

    def test_stdout_carries_only_the_transcript(self, fake_backend, tmp_path, capsys):
        audio = tmp_path / "talk.mp3"
        audio.touch()

        cli.main([str(audio), "-b", "fake", "-m", "small"])

        captured = capsys.readouterr()
        assert captured.out == "Hello from talk.\n"
        assert "library noise on stdout" in captured.err
        assert "Loading fake model 'small'" in captured.err

    def test_quiet_hides_progress(self, fake_backend, tmp_path, capsys):
        audio = tmp_path / "talk.mp3"
        audio.touch()

        cli.main([str(audio), "-b", "fake", "-m", "small", "-q"])

        assert "Loading" not in capsys.readouterr().err

    def test_subtitles_from_output_extension(self, fake_backend, tmp_path):
        audio = tmp_path / "talk.mp3"
        audio.touch()
        out = tmp_path / "talk.srt"

        cli.main([str(audio), "-b", "fake", "-m", "small", "-q", "-o", str(out)])

        assert out.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,500\nHello from talk.\n"

    def test_json_side_output(self, fake_backend, tmp_path):
        audio = tmp_path / "talk.mp3"
        audio.touch()
        out = tmp_path / "talk.json"

        cli.main([str(audio), "-b", "fake", "-m", "small", "-q", "--json", str(out)])

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["language"] == "en"
        assert data["segments"][0]["end"] == 1.5

    def test_batch_loads_model_once_and_writes_each_file(self, fake_backend, tmp_path, capsys):
        folder = tmp_path / "in"
        folder.mkdir()
        (folder / "a.wav").touch()
        (folder / "b.mp3").touch()
        (folder / "notes.txt").touch()
        out = tmp_path / "out"

        cli.main([str(folder), "-b", "fake", "-m", "small", "--outdir", str(out), "-f", "vtt"])

        assert fake_backend.loaded == 1
        assert sorted(p.name for p in out.iterdir()) == ["a.vtt", "b.vtt"]
        assert (out / "b.vtt").read_text(encoding="utf-8").startswith("WEBVTT\n")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert f"→ {out / 'a.vtt'}" in captured.err

    def test_batch_continues_after_a_failure(self, fake_backend, tmp_path, capsys):
        (tmp_path / "bad.wav").touch()
        (tmp_path / "good.wav").touch()

        with pytest.raises(SystemExit, match="1 of 2 files failed"):
            cli.main([str(tmp_path), "-b", "fake", "-m", "small", "-q"])

        assert (tmp_path / "good.txt").read_text(encoding="utf-8") == "Hello from good.\n"
        assert not (tmp_path / "bad.txt").exists()
        assert "Failed to load audio" in capsys.readouterr().err

    def test_single_file_failure_exits_non_zero(self, fake_backend, tmp_path):
        audio = tmp_path / "bad.wav"
        audio.touch()

        with pytest.raises(SystemExit) as exc_info:
            cli.main([str(audio), "-b", "fake", "-m", "small", "-q"])
        assert exc_info.value.code == 1

    def test_missing_file_fails_before_loading_model(self, fake_backend, tmp_path):
        with pytest.raises(SystemExit, match="file not found"):
            cli.main([str(tmp_path / "nope.mp3"), "-b", "fake", "-m", "small"])
        assert fake_backend.loaded == 0

    def test_empty_folder(self, fake_backend, tmp_path):
        with pytest.raises(SystemExit, match="no audio files found"):
            cli.main([str(tmp_path), "-b", "fake"])

    def test_model_load_error_exits(self, fake_backend, tmp_path):
        audio = tmp_path / "talk.mp3"
        audio.touch()

        with pytest.raises(SystemExit, match="Unknown fake model"):
            cli.main([str(audio), "-b", "fake", "-m", "huge"])

    def test_unknown_backend_exits(self, tmp_path):
        audio = tmp_path / "talk.mp3"
        audio.touch()

        with pytest.raises(SystemExit, match="Unknown backend"):
            cli.main([str(audio), "-b", "nope"])

    def test_diarized_transcript(self, fake_backend, tmp_path, capsys, monkeypatch):
        audio = tmp_path / "talk.mp3"
        audio.touch()
        pipeline = object()
        monkeypatch.setattr("speech_toolkit.api.load_diarization_pipeline", lambda device: pipeline)
        monkeypatch.setattr("speech_toolkit.api.diarize_audio", lambda path, pipe, **hints: [(0.0, 2.0, "SPEAKER_00")])

        cli.main([str(audio), "-b", "fake", "-m", "small", "-q", "--diarize"])

        assert capsys.readouterr().out == "[SPEAKER_00] Hello from talk.\n"

    def test_accuracy_options_reach_the_backend(self, fake_backend, tmp_path):
        from tests.conftest import FakeWordsBackend

        audio = tmp_path / "talk.mp3"
        audio.touch()
        out = tmp_path / "talk.json"

        cli.main(
            [str(audio), "-b", "fake-words", "-q", "--prompt", "Grafana", "--vad", "--word-timestamps", "-o", str(out)]
        )

        assert FakeWordsBackend.calls == [{"prompt": "Grafana", "vad": True, "word_timestamps": True}]
        assert json.loads(out.read_text(encoding="utf-8"))["segments"][0]["words"][0]["word"] == " Hello"

    def test_unsupported_option_prints_one_warning_line(self, fake_backend, tmp_path, capsys):
        for name in ("a.wav", "b.wav"):
            (tmp_path / name).touch()

        cli.main([str(tmp_path), "-b", "fake", "-q", "--vad"])

        err = capsys.readouterr().err
        assert err.count("Warning: the fake backend doesn't support vad; ignoring it.") == 1

    def test_max_line_width_uses_word_timings(self, fake_backend, tmp_path):
        from tests.conftest import FakeWordsBackend

        audio = tmp_path / "talk.mp3"
        audio.touch()
        out = tmp_path / "talk.srt"

        cli.main(
            [
                str(audio),
                "-b",
                "fake-words",
                "-q",
                "-o",
                str(out),
                "--max-line-width",
                "5",
                "--json",
                str(tmp_path / "t.json"),
            ]
        )

        assert FakeWordsBackend.calls == [{"word_timestamps": True}]  # turned on for the subtitle timings
        assert out.read_text(encoding="utf-8") == (
            "1\n00:00:00,000 --> 00:00:01,000\nHello\nfrom\n\n2\n00:00:01,000 --> 00:00:01,500\ntalk.\n"
        )

    def test_max_line_width_ignored_for_plain_text(self, fake_backend, tmp_path, capsys):
        from tests.conftest import FakeWordsBackend

        audio = tmp_path / "talk.mp3"
        audio.touch()

        cli.main([str(audio), "-b", "fake-words", "-q", "--max-line-width", "5"])

        assert FakeWordsBackend.calls == [{}]  # no word timings needed for txt
        assert capsys.readouterr().out == "Hello from talk.\n"

    def test_max_line_width_without_word_timings(self, fake_backend, tmp_path):
        audio = tmp_path / "talk.mp3"
        audio.touch()
        out = tmp_path / "talk.vtt"

        cli.main([str(audio), "-b", "fake", "-q", "-o", str(out), "--max-line-width", "10"])

        assert out.read_text(encoding="utf-8") == ("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello from\ntalk.\n")

    def test_diarized_transcript_splits_on_word_timings(self, fake_backend, tmp_path, capsys, monkeypatch):
        audio = tmp_path / "talk.mp3"
        audio.touch()
        monkeypatch.setattr("speech_toolkit.api.load_diarization_pipeline", lambda device: object())
        monkeypatch.setattr(
            "speech_toolkit.api.diarize_audio",
            lambda path, pipe, **hints: [(0.0, 0.9, "SPEAKER_00"), (0.9, 1.5, "SPEAKER_01")],
        )

        cli.main([str(audio), "-b", "fake-words", "-q", "--diarize"])

        assert capsys.readouterr().out == "[SPEAKER_00] Hello from\n[SPEAKER_01] talk.\n"

    @pytest.mark.parametrize("model, warned", [("turbo", True), ("small", False)])
    def test_translate_with_turbo_warns(self, fake_backend, tmp_path, capsys, model, warned):
        audio = tmp_path / "talk.mp3"
        audio.touch()

        cli.main([str(audio), "-b", "fake", "-m", model, "-t", "translate", "-q"])

        assert ("isn't trained for translation" in capsys.readouterr().err) is warned

    def test_list_flags(self, capsys):
        cli.main(["--list-backends"])
        assert "faster-whisper" in capsys.readouterr().out

        cli.main(["--list-models", "-b", "faster-whisper"])
        assert "turbo (default)" in capsys.readouterr().out
