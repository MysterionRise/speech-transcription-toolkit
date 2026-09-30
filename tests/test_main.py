#!/usr/bin/env python3
"""Tests for main.py speech-to-text functionality."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

import main
from backends import TranscriptionBackend, TranscriptionResult
from main import (
    DIARIZATION_MODEL,
    configure_hf_token,
    diarize_audio,
    load_diarization_pipeline,
    merge_diarization,
    parse_args,
    plan_outputs,
    positive_int,
    show_backends,
    show_models,
)


class TestParseArgs:
    """Test command-line argument parsing for main.py."""

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
        heavy = ("torch", "whisper", "pyannote.audio", "transformers", "faster_whisper")
        code = f"import sys, main; main.show_backends(); print([m for m in {heavy!r} if m in sys.modules])"
        repo_root = pathlib.Path(__file__).resolve().parent.parent
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=repo_root)
        assert run.stdout.splitlines()[-1] == "[]"


class TestConfigureHfToken:
    """Test Hugging Face token handling."""

    def test_cli_token_wins(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "env_token")
        configure_hf_token("cli_token")
        assert main.os.environ["HF_TOKEN"] == "cli_token"

    def test_legacy_env_var_is_exported_as_hf_token(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "legacy_token")
        configure_hf_token(None)
        assert main.os.environ["HF_TOKEN"] == "legacy_token"

    def test_no_token_leaves_env_alone(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
        configure_hf_token(None)
        assert "HF_TOKEN" not in main.os.environ


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
            with pytest.raises(ImportError, match="requirements-diarize.txt"):
                load_diarization_pipeline()

    def test_load_pipeline_disables_telemetry(self, pyannote_modules):
        """pyannote.audio 4 would send usage metrics by default; the tool stays offline."""
        load_diarization_pipeline()
        assert main.os.environ["PYANNOTE_METRICS_ENABLED"] == "false"

    def test_load_pipeline_respects_telemetry_opt_in(self, pyannote_modules, monkeypatch):
        """An explicit PYANNOTE_METRICS_ENABLED setting is left alone."""
        monkeypatch.setenv("PYANNOTE_METRICS_ENABLED", "true")
        load_diarization_pipeline()
        assert main.os.environ["PYANNOTE_METRICS_ENABLED"] == "true"


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


class FakeBackend(TranscriptionBackend):
    """Backend that 'transcribes' instantly and, like Whisper, prints to stdout while working."""

    name = "fake"
    description = "Fake backend for tests"
    loaded = 0

    @classmethod
    def available_models(cls):
        return ["small", "turbo"]

    def load_model(self, model_name, device=None):
        if model_name not in self.available_models():
            raise ValueError(f"Unknown fake model: {model_name}")
        FakeBackend.loaded += 1
        self._model = object()
        self._model_name = model_name

    def transcribe(self, audio_path, language=None, task="transcribe", verbose=True):
        print("library noise on stdout")
        if audio_path.name.startswith("bad"):
            raise RuntimeError("Failed to load audio")
        return TranscriptionResult(
            text=f" Hello from {audio_path.stem}.",
            segments=[{"start": 0.0, "end": 1.5, "text": f" Hello from {audio_path.stem}."}],
            language="en",
        )


@pytest.fixture
def fake_backend(monkeypatch):
    """Register FakeBackend as 'fake' for the duration of a test."""
    from backends import _BACKENDS

    monkeypatch.setattr("backends._BACKENDS", {**_BACKENDS, "fake": FakeBackend})
    FakeBackend.loaded = 0
    return FakeBackend


class TestMain:
    """End-to-end CLI flow with a fake backend."""

    def test_stdout_carries_only_the_transcript(self, fake_backend, tmp_path, capsys):
        audio = tmp_path / "talk.mp3"
        audio.touch()

        main.main([str(audio), "-b", "fake", "-m", "small"])

        captured = capsys.readouterr()
        assert captured.out == "Hello from talk.\n"
        assert "library noise on stdout" in captured.err
        assert "Loading fake model 'small'" in captured.err

    def test_quiet_hides_progress(self, fake_backend, tmp_path, capsys):
        audio = tmp_path / "talk.mp3"
        audio.touch()

        main.main([str(audio), "-b", "fake", "-m", "small", "-q"])

        assert "Loading" not in capsys.readouterr().err

    def test_subtitles_from_output_extension(self, fake_backend, tmp_path):
        audio = tmp_path / "talk.mp3"
        audio.touch()
        out = tmp_path / "talk.srt"

        main.main([str(audio), "-b", "fake", "-m", "small", "-q", "-o", str(out)])

        assert out.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,500\nHello from talk.\n"

    def test_json_side_output(self, fake_backend, tmp_path):
        audio = tmp_path / "talk.mp3"
        audio.touch()
        out = tmp_path / "talk.json"

        main.main([str(audio), "-b", "fake", "-m", "small", "-q", "--json", str(out)])

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

        main.main([str(folder), "-b", "fake", "-m", "small", "--outdir", str(out), "-f", "vtt"])

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
            main.main([str(tmp_path), "-b", "fake", "-m", "small", "-q"])

        assert (tmp_path / "good.txt").read_text(encoding="utf-8") == "Hello from good.\n"
        assert not (tmp_path / "bad.txt").exists()
        assert "Failed to load audio" in capsys.readouterr().err

    def test_single_file_failure_exits_non_zero(self, fake_backend, tmp_path):
        audio = tmp_path / "bad.wav"
        audio.touch()

        with pytest.raises(SystemExit) as exc_info:
            main.main([str(audio), "-b", "fake", "-m", "small", "-q"])
        assert exc_info.value.code == 1

    def test_missing_file_fails_before_loading_model(self, fake_backend, tmp_path):
        with pytest.raises(SystemExit, match="file not found"):
            main.main([str(tmp_path / "nope.mp3"), "-b", "fake", "-m", "small"])
        assert fake_backend.loaded == 0

    def test_empty_folder(self, fake_backend, tmp_path):
        with pytest.raises(SystemExit, match="no audio files found"):
            main.main([str(tmp_path), "-b", "fake"])

    def test_model_load_error_exits(self, fake_backend, tmp_path):
        audio = tmp_path / "talk.mp3"
        audio.touch()

        with pytest.raises(SystemExit, match="Unknown fake model"):
            main.main([str(audio), "-b", "fake", "-m", "huge"])

    def test_unknown_backend_exits(self, tmp_path):
        audio = tmp_path / "talk.mp3"
        audio.touch()

        with pytest.raises(SystemExit, match="Unknown backend"):
            main.main([str(audio), "-b", "nope"])

    def test_diarized_transcript(self, fake_backend, tmp_path, capsys, monkeypatch):
        audio = tmp_path / "talk.mp3"
        audio.touch()
        pipeline = object()
        monkeypatch.setattr(main, "load_diarization_pipeline", lambda device: pipeline)
        monkeypatch.setattr(main, "diarize_audio", lambda path, pipe, **hints: [(0.0, 2.0, "SPEAKER_00")])

        main.main([str(audio), "-b", "fake", "-m", "small", "-q", "--diarize"])

        assert capsys.readouterr().out == "[SPEAKER_00] Hello from talk.\n"

    @pytest.mark.parametrize("model, warned", [("turbo", True), ("small", False)])
    def test_translate_with_turbo_warns(self, fake_backend, tmp_path, capsys, model, warned):
        audio = tmp_path / "talk.mp3"
        audio.touch()

        main.main([str(audio), "-b", "fake", "-m", model, "-t", "translate", "-q"])

        assert ("isn't trained for translation" in capsys.readouterr().err) is warned

    def test_list_flags(self, capsys):
        main.main(["--list-backends"])
        assert "faster-whisper" in capsys.readouterr().out

        main.main(["--list-models", "-b", "faster-whisper"])
        assert "turbo (default)" in capsys.readouterr().out
