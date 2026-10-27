"""The scripts in examples/ run as their docstrings say, against fake backends: no models and no network."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess  # nosec B404 - runs the example scripts
import sys
import types
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch
from urllib.parse import urlparse

import numpy as np
import pytest

from speech_toolkit import (
    BackendUnavailableError,
    ModelNotFoundError,
    Transcriber,
    TranscriptionResult,
    register_backend,
)
from tests.conftest import FakeBackend

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
SCRIPTS = sorted(EXAMPLES.glob("*.py"))
EXAMPLES_PAGE = ROOT / "docs" / "examples.md"

pytestmark = pytest.mark.skipif(not EXAMPLES.is_dir(), reason="needs a source checkout: the sdist has no examples/")


def load(name: str) -> ModuleType:
    """examples/<name>.py as a module, as running the script imports it, without running main()."""
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def touch(path: pathlib.Path) -> pathlib.Path:
    """A stand-in recording: the fake backends read only its name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not really audio")
    return path


def stdout_lines(capsys) -> List[str]:
    """What the example printed to stdout, without the line FakeBackend prints while it works."""
    return [line for line in capsys.readouterr().out.splitlines() if line != "library noise on stdout"]


class TestEveryExample:
    def test_there_are_examples(self):
        names = {script.stem for script in SCRIPTS}
        assert {"folder_to_srt", "meeting_notes", "openai_client", "custom_backend", "rerender_json"} <= names

    @pytest.mark.parametrize("script", SCRIPTS, ids=lambda script: script.name)
    def test_runs_as_a_script(self, script):
        """As a user runs it, in a process of its own: its imports work and it answers --help."""
        paths = [str(ROOT), *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}
        result = subprocess.run(  # nosec B603 - our own script, with the running Python
            [sys.executable, str(script), "--help"], capture_output=True, text=True, env=env, timeout=60, check=False
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith("usage:")

    @pytest.mark.parametrize("script", SCRIPTS, ids=lambda script: script.name)
    def test_says_how_to_run_it_and_is_on_the_examples_page(self, script):
        module = load(script.stem)

        assert f"python {script.name}" in module.__doc__
        assert f'--8<-- "examples/{script.name}"' in EXAMPLES_PAGE.read_text(encoding="utf-8")


class TestFolderToSrt:
    @pytest.fixture
    def example(self, fake_backend):
        return load("folder_to_srt")

    @pytest.fixture
    def folder(self, tmp_path):
        for name in ("talk.mp3", "sub/interview.wav", "notes.txt", ".hidden/skip.mp3"):
            touch(tmp_path / "recordings" / name)
        return tmp_path / "recordings"

    def test_writes_subtitles_next_to_each_recording(self, example, folder, capsys):
        assert example.main([str(folder), "-b", "fake"]) == 0

        srt = (folder / "talk.srt").read_text(encoding="utf-8")
        assert srt == "1\n00:00:00,000 --> 00:00:01,500\nHello from talk.\n"
        written = [folder / "sub" / "interview.srt", folder / "talk.srt"]  # not for notes.txt or hidden files
        assert sorted(folder.rglob("*.srt")) == written
        assert stdout_lines(capsys) == [str(path) for path in written]

    def test_mirrors_the_folder_in_outdir_and_skips_recordings_with_subtitles(self, example, folder, tmp_path, capsys):
        outdir = tmp_path / "subtitles"
        assert example.main([str(folder), "--outdir", str(outdir), "-b", "fake", "-m", "small"]) == 0
        assert sorted(path.relative_to(outdir).as_posix() for path in outdir.rglob("*")) == [
            "sub",
            "sub/interview.srt",
            "talk.srt",
        ]
        (outdir / "talk.srt").unlink()
        FakeBackend.loaded = 0
        capsys.readouterr()

        assert example.main([str(folder), "--outdir", str(outdir), "-b", "fake"]) == 0
        assert stdout_lines(capsys) == [str(outdir / "talk.srt")]  # only the missing one

        assert example.main([str(folder), "--outdir", str(outdir), "-b", "fake"]) == 0
        assert FakeBackend.loaded == 1  # the model isn't loaded when there's nothing to do
        assert "Every recording has its subtitles already." in capsys.readouterr().err

    def test_a_recording_that_fails_doesnt_stop_the_others(self, example, folder, capsys):
        touch(folder / "bad.mp3")  # FakeBackend fails on it

        assert example.main([str(folder), "-b", "fake"]) == 1

        assert (folder / "talk.srt").is_file()
        assert not (folder / "bad.srt").exists()
        err = capsys.readouterr().err
        assert f"Error: {folder / 'bad.mp3'}: Failed to load audio" in err
        assert "Error: 1 of 3 recordings failed." in err

    def test_recordings_with_the_same_name_keep_their_whole_name(self, example, tmp_path):
        folder = tmp_path / "recordings"
        touch(folder / "talk.mp3")
        touch(folder / "talk.wav")

        assert example.main([str(folder), "-b", "fake", "--max-line-width", "20"]) == 0

        assert sorted(path.name for path in folder.glob("*.srt")) == ["talk.srt", "talk.wav.srt"]

    def test_needs_a_folder(self, example, tmp_path):
        with pytest.raises(SystemExit, match="is not a folder"):
            example.main([str(touch(tmp_path / "talk.mp3"))])


class TestMeetingNotes:
    @pytest.fixture
    def example(self, fake_backend):
        return load("meeting_notes")

    @pytest.fixture
    def hints(self, monkeypatch) -> Dict[str, Any]:
        """Speaker diarization without pyannote: SPEAKER_00 speaks until 0.9 s, then SPEAKER_01."""
        received: Dict[str, Any] = {}

        def diarize(source, pipeline, **hints):
            received.update(hints)
            return [(0.0, 0.9, "SPEAKER_00"), (0.9, 1.5, "SPEAKER_01")]

        monkeypatch.setattr("speech_toolkit.api.load_diarization_pipeline", lambda device: "pipeline")
        monkeypatch.setattr("speech_toolkit.api.diarize_audio", diarize)
        return received

    def test_writes_a_paragraph_per_turn(self, example, hints, tmp_path, capsys):
        notes = tmp_path / "notes.md"
        argv = [str(touch(tmp_path / "standup.wav")), "-b", "fake-words", "--num-speakers", "2", "-o", str(notes)]

        assert example.main([*argv, "--name", "SPEAKER_00=Ana"]) == 0

        assert notes.read_text(encoding="utf-8") == (
            "# Meeting notes: standup.wav\n"
            "\n"
            "Speakers: Ana, SPEAKER_01.\n"
            "\n"
            "**Ana** (0:00:00): Hello from\n"
            "\n"
            "**SPEAKER_01** (0:00:01): standup.\n"
        )
        assert hints["num_speakers"] == 2
        assert stdout_lines(capsys) == [str(notes)]

    def test_prints_the_notes_without_an_output_file(self, example, hints, tmp_path, capsys):
        assert example.main([str(touch(tmp_path / "standup.wav")), "-b", "fake-words"]) == 0

        assert "**SPEAKER_00** (0:00:00): Hello from" in stdout_lines(capsys)

    def test_joins_a_speakers_segments_and_gives_the_length(self, example):
        segments = [
            {"start": 0.0, "end": 2.0, "text": " Hi.", "speaker": "SPEAKER_01"},
            {"start": 2.0, "end": 3.0, "text": " Where were we?", "speaker": "SPEAKER_01"},
            {"start": 3725.0, "end": 3726.0, "text": " ", "speaker": "SPEAKER_00"},  # nothing said: no turn
            {"start": 3726.0, "end": 3727.0, "text": " Here."},  # no speaker
        ]
        result = TranscriptionResult(" Hi. Where were we? Here.", segments, speaker_segments=segments, duration=3727.4)

        assert example.to_markdown(result, "Notes", {"SPEAKER_01": "Ben"}).splitlines()[2:] == [
            "Length: 1:02:07. Speakers: Ben, unknown.",
            "",
            "**Ben** (0:00:00): Hi. Where were we?",
            "",
            "**unknown** (1:02:06): Here.",
        ]

    def test_reports_what_is_missing(self, example, monkeypatch, tmp_path):
        def missing(device):
            raise BackendUnavailableError('speaker diarization needs pyannote.audio. Install with: pip install "..."')

        monkeypatch.setattr("speech_toolkit.api.load_diarization_pipeline", missing)

        with pytest.raises(SystemExit, match="^Error: speaker diarization needs pyannote.audio"):
            example.main([str(touch(tmp_path / "standup.wav")), "-b", "fake"])

    def test_names_need_a_label_and_a_name(self, example, tmp_path, capsys):
        with pytest.raises(SystemExit):
            example.main([str(tmp_path / "standup.wav"), "--name", "Ana"])
        assert "--name takes LABEL=NAME" in capsys.readouterr().err


class StandInOpenAI:
    """The part of OpenAI's Python client the example uses, sending its requests to the app under test.

    Used when the openai package isn't installed, as in the unit test jobs; it sends what the SDK sends.
    """

    app: Any = None  # the TestClient of the app

    def __init__(self, *, base_url: str, api_key: str) -> None:
        endpoint = self._endpoint(urlparse(base_url).path, {"Authorization": f"Bearer {api_key}"})
        self.audio = SimpleNamespace(transcriptions=endpoint("transcriptions"), translations=endpoint("translations"))

    def _endpoint(self, base: str, headers: Dict[str, str]):
        def endpoint(name: str) -> SimpleNamespace:
            def create(*, model: str, file: Any, response_format: str) -> Any:
                response = self.app.post(
                    f"{base}/audio/{name}",
                    headers=headers,
                    files={"file": (pathlib.Path(file.name).name, file)},
                    data={"model": model, "response_format": response_format},
                )
                response.raise_for_status()
                if response_format in ("json", "verbose_json"):
                    return json.loads(response.text, object_hook=lambda fields: SimpleNamespace(**fields))
                return response.text

            return SimpleNamespace(create=create)

        return endpoint


class TestOpenAIClient:
    @pytest.fixture
    def example(self, fake_backend):
        return load("openai_client")

    @pytest.fixture
    def app(self, fake_backend):
        testclient = pytest.importorskip("fastapi.testclient")
        from speech_toolkit.server import create_app

        return testclient.TestClient(create_app(Transcriber("fake-words"), api_key="s3cret"))

    @pytest.fixture(params=["openai", "stand-in"])
    def sdk(self, request, app, monkeypatch):
        """OpenAI's SDK, its requests sent to the app; or, without the openai package, a stand-in that does the same."""
        if request.param == "openai":
            openai = pytest.importorskip("openai")
            client_class = openai.OpenAI
            monkeypatch.setattr(openai, "OpenAI", lambda **options: client_class(http_client=app, **options))
            yield
        else:
            StandInOpenAI.app = app
            stand_in = types.ModuleType("openai")
            stand_in.OpenAI = StandInOpenAI  # type: ignore[attr-defined]
            with patch.dict(sys.modules, {"openai": stand_in}):
                yield

    @pytest.fixture
    def audio(self, tmp_path):
        return touch(tmp_path / "talk.wav")

    def test_prints_the_transcript(self, example, sdk, audio, monkeypatch, capsys):
        monkeypatch.setenv("TRANSCRIBE_API_KEY", "s3cret")  # the server's key, which the example sends

        assert example.main([str(audio)]) == 0

        assert capsys.readouterr().out.startswith("Hello from ")

    @pytest.mark.parametrize("response_format", ["srt", "vtt"])
    def test_prints_subtitles(self, example, sdk, audio, response_format, capsys):
        assert example.main([str(audio), "--format", response_format, "--api-key", "s3cret"]) == 0

        out = capsys.readouterr().out
        assert ("WEBVTT" in out) == (response_format == "vtt")
        assert "00:00:00" in out and "Hello from" in out

    @pytest.mark.parametrize("translate", [False, True])
    def test_prints_each_segment_with_its_times(self, example, sdk, audio, translate, capsys):
        argv = [str(audio), "--format", "verbose_json", "--api-key", "s3cret"] + (["--translate"] if translate else [])

        assert example.main(argv) == 0

        language, segment = capsys.readouterr().out.splitlines()
        assert language.startswith("Language: en, ")
        assert segment.startswith("[   0.00 -    1.50] Hello from ")

    def test_a_wrong_key_is_refused(self, example, sdk, audio):
        with pytest.raises(Exception, match="401|API key"):
            example.main([str(audio), "--api-key", "wrong"])


class TestCustomBackend:
    @pytest.fixture
    def example(self, fake_backend):  # fake_backend: a copy of the registry, which the example registers in
        return load("custom_backend")

    @pytest.fixture
    def samples(self):
        """1 s of silence, 1 s of noise, 1 s of silence, 0.5 s of noise."""
        noise = np.random.default_rng(0).uniform(-0.5, 0.5, 16000).astype(np.float32)
        silence = np.zeros(16000, dtype=np.float32)
        return np.concatenate([silence, noise, silence, noise[:8000]])

    def test_finds_where_someone_speaks(self, example, samples):
        regions = example.speech_regions(samples)

        assert [value for region in regions for value in region] == pytest.approx([1.0, 2.0, 3.0, 3.5], abs=0.03)

    def test_silence_and_short_audio_have_no_speech(self, example):
        assert example.speech_regions(np.zeros(16000, dtype=np.float32)) == []
        assert example.speech_regions(np.ones(100, dtype=np.float32)) == []

    def test_writes_subtitles(self, example, samples, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(example, "load_audio", lambda path: samples)  # decoding a file needs ffmpeg

        assert example.main([str(touch(tmp_path / "talk.wav"))]) == 0

        assert capsys.readouterr().out == (
            "1\n00:00:00,990 --> 00:00:02,010\n[speech]\n\n2\n00:00:03,000 --> 00:00:03,500\n[speech]\n"  # cues last 0.5 s
        )

    def test_works_like_any_backend_once_registered(self, example, samples):
        register_backend("speech-regions", example.SpeechRegionsBackend)
        transcriber = Transcriber("speech-regions")

        result = transcriber.transcribe(samples)  # samples, which a "waveform" backend gets as they are

        assert transcriber.model_name == "loudness"
        assert result.text == " [speech] [speech]"
        assert [segment["id"] for segment in result.segments] == [0, 1]
        assert result.duration == 3.5
        with pytest.raises(ModelNotFoundError, match="available: loudness"):
            Transcriber("speech-regions", "large")


class TestRerenderJson:
    @pytest.fixture
    def example(self, fake_backend):
        return load("rerender_json")

    @pytest.fixture
    def saved(self, example, tmp_path) -> pathlib.Path:
        result = Transcriber("fake-words").transcribe(touch(tmp_path / "talk.mp3"), word_timestamps=True)
        return result.save(tmp_path / "talk.json")

    def test_writes_srt_and_vtt_as_the_result_renders_them(self, example, saved, capsys):
        assert example.main([str(saved)]) == 0

        result = TranscriptionResult.load(saved)
        assert saved.with_suffix(".srt").read_text(encoding="utf-8").rstrip() == result.render("srt").rstrip()
        assert saved.with_suffix(".vtt").read_text(encoding="utf-8").startswith("WEBVTT\n")
        assert capsys.readouterr().out.splitlines() == [str(saved.with_suffix(".srt")), str(saved.with_suffix(".vtt"))]

    def test_splits_lines_and_keeps_speaker_labels(self, example, tmp_path):
        segment = {"id": 0, "start": 0.0, "end": 2.0, "text": " Hello there, how are you today?"}
        speakers = [{**segment, "speaker": "SPEAKER_00", "segment_id": 0}]
        saved = TranscriptionResult(segment["text"], [segment], speaker_segments=speakers).save(tmp_path / "a.json")

        assert example.main([str(saved), "--formats", "srt", "--max-line-width", "16"]) == 0

        srt = saved.with_suffix(".srt").read_text(encoding="utf-8")
        assert "[SPEAKER_00]" in srt
        text = [line for line in srt.splitlines() if "-->" not in line and not line.isdigit()]
        assert text and max(len(line) for line in text) <= 16
        assert not saved.with_suffix(".vtt").exists()

    def test_refuses_an_unknown_format(self, example, saved, capsys):
        with pytest.raises(SystemExit):
            example.main([str(saved), "--formats", "srt,json"])
        assert "unknown format 'json'" in capsys.readouterr().err

    @pytest.mark.parametrize("content", [None, "{}", "not json"])
    def test_reports_a_file_that_isnt_a_saved_result(self, example, tmp_path, content):
        path = tmp_path / "talk.json"
        if content is not None:
            path.write_text(content, encoding="utf-8")

        with pytest.raises(SystemExit, match="^Error: "):
            example.main([str(path)])
