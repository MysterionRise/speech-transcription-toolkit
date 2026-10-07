"""Backend capabilities and the checks built on them.

What each backend declares, ``requires`` and ``--list-backends``, help text generated from the registry, and
translation refused before any work by the Transcriber, the ``transcribe`` command and the server. Heavy libraries
are mocked, as everywhere else.
"""

from __future__ import annotations

import io
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from speech_toolkit import Transcriber, cli, transcribe
from speech_toolkit.backends import (
    CAPABILITIES,
    CanaryBackend,
    FasterWhisperBackend,
    ParakeetBackend,
    VoxtralBackend,
    WhisperBackend,
    backends_with,
    get_backend_class,
    list_backends,
    register_backend,
)
from speech_toolkit.server import create_app
from speech_toolkit.server import parse_args as server_parse_args
from tests.conftest import FakeBackend, FakeWordsBackend

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TRANSLATORS = "whisper, faster-whisper, canary"  # the built-in backends that translate


class TranscribeOnlyBackend(FakeWordsBackend):
    """Like Voxtral: can't translate or report the language. Records its calls in FakeWordsBackend.calls."""

    name = "transcribe-only"
    capabilities = frozenset()


@pytest.fixture(autouse=True)
def no_color(monkeypatch):
    """Python 3.14's argparse can colour help and errors; these tests match the plain text."""
    monkeypatch.setenv("PYTHON_COLORS", "0")


@pytest.fixture
def transcribe_only(fake_backend):
    """Register TranscribeOnlyBackend for one test, in the fake_backend fixture's copy of the registry."""
    register_backend("transcribe-only", TranscribeOnlyBackend)
    return TranscribeOnlyBackend


@pytest.fixture
def audio(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "talk.wav"
    path.touch()
    return path


def help_text(capsys, monkeypatch, parse_args=cli.parse_args) -> str:
    """The --help output, with one line per option."""
    monkeypatch.setenv("COLUMNS", "500")
    with pytest.raises(SystemExit):
        parse_args(["--help"])
    return capsys.readouterr().out


def help_section(text: str, title: str) -> str:
    """One option group of --help output, e.g. "model"."""
    return text.split(f"\n{title}:\n", 1)[1].split("\n\n", 1)[0].rstrip()


def listing(out: str) -> Dict[str, Dict[str, str]]:
    """show_backends() output as {backend: {"Models": ..., "Capabilities": ..., "Installed": ...}}."""
    backends = {}
    for block in out.split("\n\n")[1:]:  # after the header
        lines = block.strip().splitlines()
        if lines:
            backends[lines[0].split()[0]] = dict(line.strip().split(": ", 1) for line in lines[2:])
    return backends


class TestDeclarations:
    @pytest.mark.parametrize(
        "backend_class, capabilities, requires",
        [
            (WhisperBackend, {"translate", "language_detection", "prompt", "word_timestamps"}, ("whisper",)),
            (
                FasterWhisperBackend,
                {"translate", "language_detection", "prompt", "vad", "word_timestamps"},
                ("faster_whisper",),
            ),
            (VoxtralBackend, set(), ("torch", "transformers", "mistral_common")),
            (ParakeetBackend, {"word_timestamps"}, ("torch", "transformers", "librosa")),
            (CanaryBackend, {"translate"}, ("torch", "transformers", "librosa")),
        ],
    )
    def test_builtin_backends(self, backend_class, capabilities, requires):
        assert backend_class.capabilities == capabilities
        assert backend_class.requires == requires

    @pytest.mark.parametrize("name", list_backends())
    def test_known_capabilities_and_top_level_requirements(self, name):
        backend_class = get_backend_class(name)
        assert backend_class.capabilities <= set(CAPABILITIES)
        assert all("." not in module for module in backend_class.requires)  # find_spec("a.b") imports a

    def test_backends_with(self):
        assert backends_with("translate") == ["whisper", "faster-whisper", "canary"]
        assert backends_with("language_detection") == ["whisper", "faster-whisper"]
        assert backends_with("prompt") == ["whisper", "faster-whisper"]
        assert backends_with("vad") == ["faster-whisper"]
        assert backends_with("word_timestamps") == ["whisper", "faster-whisper", "parakeet"]
        assert backends_with("telepathy") == []

    def test_backends_with_includes_registered_backends(self, transcribe_only):
        assert backends_with("vad") == ["faster-whisper", "fake-words"]
        assert "transcribe-only" not in backends_with("translate")

    def test_supports(self, fake_backend):
        transcriber = Transcriber("fake")
        assert transcriber.supports("translate") and transcriber.supports("language_detection")
        assert not transcriber.supports("vad")


class TestRequirements:
    def test_missing_requirements(self):
        class Needy(WhisperBackend):
            requires = ("json", "no_such_module_for_speech_toolkit_tests", "os")

        assert Needy.missing_requirements() == ["no_such_module_for_speech_toolkit_tests"]
        assert WhisperBackend.missing_requirements() in ([], ["whisper"])

    def test_modules_already_in_sys_modules(self):
        """A blocked import (None) is missing; a loaded or stand-in module, which has no spec, is installed."""

        class Mocked(WhisperBackend):
            requires = ("blocked_module_for_tests", "mocked_module_for_tests")

        with patch.dict(sys.modules, {"blocked_module_for_tests": None, "mocked_module_for_tests": MagicMock()}):
            assert Mocked.missing_requirements() == ["blocked_module_for_tests"]

    def test_listing_finds_packages_without_importing_them(self):
        """Installed packages are found without being imported, and dotted names only look up the top level."""
        code = textwrap.dedent("""
            import sys
            from speech_toolkit import cli
            from speech_toolkit.backends import FasterWhisperBackend, register_backend

            class Heavy(FasterWhisperBackend):
                requires = ("numpy", "xmlrpc.client")  # both importable; neither imported by the toolkit

            register_backend("heavy", Heavy)
            cli.show_backends()
            print([module for module in ("numpy", "xmlrpc") if module in sys.modules])
            """)
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT)

        assert run.stdout.splitlines()[-1] == "[]"
        assert listing(run.stdout)["heavy"]["Installed"] == "yes"


class TestListBackends:
    def test_capabilities_of_each_backend(self, capsys):
        cli.show_backends()
        backends = listing(capsys.readouterr().out)

        assert {name: fields["Capabilities"] for name, fields in backends.items()} == {
            "whisper": "translate, language detection, prompt, word timestamps",
            "faster-whisper": "translate, language detection, prompt, vad, word timestamps",
            "voxtral": "transcription only",
            "parakeet": "word timestamps",
            "canary": "translate",
        }
        assert backends["whisper"]["Models"].endswith("large-v3-turbo, turbo")

    def test_installed_status(self, capsys):
        with patch.dict(sys.modules, {"whisper": MagicMock(), "faster_whisper": None, "mistral_common": None}):
            cli.show_backends()
        backends = listing(capsys.readouterr().out)

        assert backends["whisper"]["Installed"] == "yes"
        assert backends["faster-whisper"]["Installed"] == "no (missing faster_whisper)"
        assert backends["voxtral"]["Installed"].startswith("no (missing ")
        assert backends["voxtral"]["Installed"].endswith("mistral_common)")
        for fields in backends.values():
            assert re.fullmatch(r"yes|no \(missing \w+(, \w+)*\)", fields["Installed"])

    def test_custom_backend(self, fake_backend, capsys):
        class Custom(FakeBackend):
            capabilities = frozenset({"vad", "zebra", "translate", "alpha"})
            requires = ("json",)

        register_backend("custom", Custom)
        cli.main(["--list-backends"])
        backends = listing(capsys.readouterr().out)

        # Known capabilities in their usual order, then any others alphabetically.
        assert backends["custom"] == {
            "Models": "small, turbo",
            "Capabilities": "translate, vad, alpha, zebra",
            "Installed": "yes",
        }
        assert backends["fake"]["Capabilities"] == "translate, language detection"


class TestTranscriber:
    def test_unknown_task(self, fake_backend, audio):
        transcriber = Transcriber("fake-words")

        with pytest.raises(ValueError, match="task must be 'transcribe' or 'translate', not 'foo'"):
            transcriber.transcribe(audio, task="foo")
        assert FakeWordsBackend.calls == []

    def test_translation_needs_the_capability(self, transcribe_only, audio):
        transcriber = Transcriber("transcribe-only")

        message = (
            f"the transcribe-only backend can't translate; backends that translate: {TRANSLATORS}, fake, fake-words"
        )
        with pytest.raises(ValueError, match=re.escape(message)):
            transcriber.transcribe(audio, task="translate")
        assert FakeWordsBackend.calls == []  # the backend never ran
        assert transcriber.transcribe(audio).text == " Hello from talk."

    def test_translation_is_checked_before_diarization(self, transcribe_only, audio, monkeypatch):
        diarize = MagicMock()
        monkeypatch.setattr("speech_toolkit.api.load_diarization_pipeline", lambda device: "pipeline")
        monkeypatch.setattr("speech_toolkit.api.diarize_audio", diarize)

        with pytest.raises(ValueError, match="can't translate"):
            Transcriber("transcribe-only", diarize=True).transcribe(audio, task="translate")
        diarize.assert_not_called()

    def test_translation_with_a_backend_that_can(self, fake_backend, audio):
        Transcriber("fake-words").transcribe(audio, task="translate")
        assert FakeWordsBackend.tasks == ["translate"]

    @pytest.mark.parametrize(
        "backend, task, error",
        [("transcribe-only", "translate", "can't translate"), ("fake", "summarize", "task must be")],
    )
    def test_one_shot_transcribe_checks_before_loading(self, transcribe_only, audio, backend, task, error):
        with pytest.raises(ValueError, match=error):
            transcribe(audio, backend=backend, task=task)
        assert FakeBackend.loaded == 0


class TestTranscribeCommand:
    def test_translation_fails_before_the_model_loads(self, transcribe_only, audio, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli.main([str(audio), "-b", "transcribe-only", "-t", "translate"])

        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert f"error: the transcribe-only backend can't translate; backends that translate: {TRANSLATORS}" in err
        assert "Loading" not in err
        assert FakeBackend.loaded == 0

    @pytest.mark.parametrize("backend", ["voxtral", "parakeet"])
    def test_builtin_backends_that_cant_translate(self, backend, capsys):
        with pytest.raises(SystemExit):
            cli.parse_args(["talk.wav", "-b", backend, "-t", "translate"])
        assert f"the {backend} backend can't translate; backends that translate: {TRANSLATORS}\n" in (
            capsys.readouterr().err
        )

    @pytest.mark.parametrize("backend", ["whisper", "faster-whisper", "canary"])
    def test_builtin_backends_that_translate(self, backend):
        assert cli.parse_args(["talk.wav", "-b", backend, "-t", "translate"]).task == "translate"

    def test_unknown_backend_is_reported_when_loading(self, audio):
        with pytest.raises(SystemExit, match="Unknown backend: 'nope'"):
            cli.main([str(audio), "-b", "nope", "-t", "translate"])

    def test_listing_skips_the_check(self, capsys):
        cli.main(["--list-models", "-b", "voxtral", "-t", "translate"])
        assert "voxtral-mini (default)" in capsys.readouterr().out

    def test_transcribing_still_works(self, transcribe_only, audio, capsys):
        cli.main([str(audio), "-b", "transcribe-only", "-q"])
        assert capsys.readouterr().out == "Hello from talk.\n"


class TestHelpText:
    """The --help lines that name backends come from the registry, so custom backends show up too."""

    OPTIONS = [("--prompt", "prompt"), ("--vad", "vad"), ("--word-timestamps", "word_timestamps")]

    @staticmethod
    def backends_in_help(text: str, option: str) -> str:
        """The parenthesised backend list at the end of *option*'s help line."""
        match = re.search(rf"^  {re.escape(option)}[ \w]*  .*\(([^()]*)\)\.$", text, re.MULTILINE)
        assert match, f"no help line for {option}"
        return match.group(1)

    def test_builtin_backends(self, capsys, monkeypatch):
        text = help_text(capsys, monkeypatch)

        assert self.backends_in_help(text, "--prompt") == "whisper, faster-whisper"
        assert self.backends_in_help(text, "--vad") == "faster-whisper"
        assert self.backends_in_help(text, "--word-timestamps") == "whisper, faster-whisper, parakeet"
        assert f"Backends that translate: {TRANSLATORS}." in text

    @pytest.mark.parametrize("option, capability", OPTIONS)
    def test_matches_the_registry(self, transcribe_only, capsys, monkeypatch, option, capability):
        text = help_text(capsys, monkeypatch)

        assert self.backends_in_help(text, option) == ", ".join(backends_with(capability))
        assert "fake-words" in backends_with(capability)  # so the list isn't the built-in one

    def test_no_backend_with_the_capability(self, capsys, monkeypatch):
        monkeypatch.setattr("speech_toolkit.backends._BACKENDS", {"voxtral": VoxtralBackend})

        text = help_text(capsys, monkeypatch)

        assert self.backends_in_help(text, "--vad") == "no backend"
        assert "Backends that translate: no backend." in text

    def test_server_reuses_the_model_options(self, capsys, monkeypatch):
        """transcribe-server takes --backend, --model and --device exactly as transcribe does."""
        cli_model = help_section(help_text(capsys, monkeypatch), "model")
        server_model = help_section(help_text(capsys, monkeypatch, server_parse_args), "model")

        assert server_model == cli_model
        assert all(option in cli_model for option in ("--backend", "--model", "--device"))
        args = server_parse_args(["-b", "faster-whisper", "-m", "small", "--device", "cpu"])
        assert (args.backend, args.model, args.device) == ("faster-whisper", "small", "cpu")


class FakeBatch(dict):
    """Minimal stand-in for a processor's BatchFeature."""

    def to(self, *args, **kwargs):
        return self


@pytest.fixture
def parakeet_modules():
    """Mocked torch and transformers whose Parakeet hears " Hallo." in 5 s of audio."""
    torch, transformers = MagicMock(), MagicMock()
    torch.cuda.is_available.return_value = False
    processor = transformers.AutoProcessor.from_pretrained.return_value
    processor.return_value = FakeBatch()
    processor.decode.return_value = ("Hallo.", [[{"token": " Hallo.", "start": 0.5, "end": 1.0}]])
    with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
        with patch("speech_toolkit.backends.nvidia_backend.load_audio") as load_audio:
            load_audio.return_value = np.zeros(5 * 16000, dtype=np.float32)
            yield


class TestParakeetLanguage:
    """Parakeet detects the language itself: a given one is ignored with a warning, and none is reported."""

    def _backend(self):
        backend = ParakeetBackend()
        backend.load_model("parakeet-tdt-0.6b-v3")
        return backend

    def test_given_language_is_ignored_with_a_warning(self, parakeet_modules, audio):
        with pytest.warns(UserWarning, match="Parakeet detects the language itself; ignoring language 'de'"):
            result = self._backend().transcribe(audio, language="de", verbose=False)

        assert (result.text, result.language) == ("Hallo.", None)

    def test_no_language_no_warning(self, parakeet_modules, audio):
        result = self._backend().transcribe(audio, verbose=False)  # a warning would fail the test
        assert result.language is None

    @pytest.mark.filterwarnings("default:Parakeet detects the language itself:UserWarning")
    def test_command_line(self, parakeet_modules, audio, tmp_path, capsys):
        out = tmp_path / "talk.json"

        cli.main([str(audio), "-b", "parakeet", "-l", "de", "-q", "-o", str(out)])

        assert capsys.readouterr().err == "Warning: Parakeet detects the language itself; ignoring language 'de'.\n"
        assert json.loads(out.read_text(encoding="utf-8"))["language"] is None


class TestServerTranslations:
    @pytest.fixture
    def uploads(self, tmp_path, monkeypatch):
        """Send the server's temporary uploads to a folder the test can inspect."""
        folder = tmp_path / "uploads"
        folder.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(folder))
        return folder

    def _post(self, client: TestClient, url: str):
        return client.post(url, files={"file": ("talk.wav", io.BytesIO(b"RIFF fake audio"), "audio/wav")})

    def test_400_without_running_the_backend(self, transcribe_only, uploads):
        client = TestClient(create_app(Transcriber("transcribe-only")))

        response = self._post(client, "/v1/audio/translations")

        assert response.status_code == 400
        assert response.json()["error"] == {
            "message": "The transcribe-only backend can't translate.",
            "type": "invalid_request_error",
            "param": None,
            "code": None,
        }
        assert FakeWordsBackend.calls == []  # the backend never ran
        assert list(uploads.iterdir()) == []  # and the upload wasn't saved

    def test_transcriptions_still_work(self, transcribe_only, uploads):
        client = TestClient(create_app(Transcriber("transcribe-only")))

        response = self._post(client, "/v1/audio/transcriptions")

        assert response.status_code == 200
        assert FakeWordsBackend.tasks == ["transcribe"]

    def test_backend_that_translates(self, fake_backend, uploads):
        client = TestClient(create_app(Transcriber("fake-words")))

        response = self._post(client, "/v1/audio/translations")

        assert response.status_code == 200
        assert FakeWordsBackend.tasks == ["translate"]
