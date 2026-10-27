"""Messages: the library never prints, and logs to the ``speech_toolkit`` logger, which shows nothing until the
application configures logging. The transcribe command shows those messages on stderr (-q, -v, --log-level), and its
-q also hides Hugging Face's and transformers' progress bars.

tests/test_progress.py checks the progress events, and the messages the built-in backends log as they go.
"""

from __future__ import annotations

import ast
import logging
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import speech_toolkit
from speech_toolkit import DiarizationError, cli
from speech_toolkit.diarization import DIARIZATION_MODEL, diarize_audio, load_diarization_pipeline
from speech_toolkit.media import SAMPLE_RATE
from tests.conftest import FakeBackend

PACKAGE = pathlib.Path(speech_toolkit.__file__).parent

###############################################################################
# The library doesn't print
###############################################################################

# The command modules: what they print is the output of the transcribe and ogg2wav commands.
COMMAND_MODULES = frozenset({"cli.py", "convert.py"})
# Functions elsewhere that are a command's entry point: transcribe-server's main() prints its startup warning.
COMMAND_FUNCTIONS = frozenset({("server.py", "main")})
STD_STREAMS = frozenset({"stdout", "stderr", "__stdout__", "__stderr__"})


class OutputFinder(ast.NodeVisitor):
    """Finds the calls that write to the terminal: print(), and sys.stdout or sys.stderr .write()/.writelines()."""

    def __init__(self) -> None:
        self.functions: List[str] = []  # the functions the visit is in, outermost first
        self.found: List[Tuple[int, Optional[str], str]] = []  # (line, outermost function or None, the call)

    def visit_FunctionDef(self, node: ast.AST) -> None:
        self.functions.append(getattr(node, "name"))
        self.generic_visit(node)
        self.functions.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        call = None
        if isinstance(func, ast.Name) and func.id == "print":
            call = "print()"
        elif isinstance(func, ast.Attribute) and func.attr in ("write", "writelines"):
            stream = func.value
            if isinstance(stream, ast.Attribute) and stream.attr in STD_STREAMS:
                if isinstance(stream.value, ast.Name) and stream.value.id == "sys":
                    call = f"sys.{stream.attr}.{func.attr}()"
        if call:
            self.found.append((node.lineno, self.functions[0] if self.functions else None, call))
        self.generic_visit(node)


def terminal_output(source: str, module: str) -> List[str]:
    """The calls in *source* (the file *module* of the package) that write to the terminal outside a command."""
    finder = OutputFinder()
    finder.visit(ast.parse(source))
    return [
        f"speech_toolkit/{module}:{line} {call}"
        for line, function, call in finder.found
        if module not in COMMAND_MODULES and (module, function) not in COMMAND_FUNCTIONS
    ]


def test_library_modules_dont_print():
    """Library code logs (logging.getLogger(__name__)) or warns (warnings.warn) instead: an application decides what
    its users see. Only the commands' own output is printed."""
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        found += terminal_output(path.read_text(encoding="utf-8"), path.relative_to(PACKAGE).as_posix())

    assert not found, "use logging.getLogger(__name__) or warnings.warn instead: " + ", ".join(found)


def test_the_check_finds_every_way_to_print():
    source = """
import sys

def helper():
    print("progress")
    sys.stderr.write("progress")
    sys.stdout.writelines(["progress"])
    other.stderr.write("not the terminal")
    logger.info("fine")

def main():
    print("a command's own output")
"""
    assert terminal_output(source, "backends/mine.py") == [
        "speech_toolkit/backends/mine.py:5 print()",
        "speech_toolkit/backends/mine.py:6 sys.stderr.write()",
        "speech_toolkit/backends/mine.py:7 sys.stdout.writelines()",
        "speech_toolkit/backends/mine.py:12 print()",
    ]
    assert terminal_output(source, "server.py") == [  # server.main() is a command's entry point
        "speech_toolkit/server.py:5 print()",
        "speech_toolkit/server.py:6 sys.stderr.write()",
        "speech_toolkit/server.py:7 sys.stdout.writelines()",
    ]
    assert terminal_output(source, "cli.py") == []


###############################################################################
# From Python, nothing shows unless logging is configured
###############################################################################


def test_the_package_logger_has_a_null_handler():
    handlers = logging.getLogger("speech_toolkit").handlers
    assert any(isinstance(handler, logging.NullHandler) for handler in handlers)


# Run in a new interpreter: pytest's own log capture would hide what Python prints without a NullHandler.
QUIET_BY_DEFAULT = """
import logging

import numpy as np

from speech_toolkit import Transcriber, TranscriptionResult, register_backend
from speech_toolkit.backends import TranscriptionBackend, report_progress


class Talkative(TranscriptionBackend):
    capabilities = frozenset({"waveform"})

    @classmethod
    def available_models(cls):
        return ["small"]

    def load_model(self, model_name, device=None):
        self._model = object()

    def transcribe(self, audio, language=None, task="transcribe", verbose=True):
        log = logging.getLogger("speech_toolkit.backends.talkative")
        for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR):
            log.log(level, "a message at level %d", level)
        segment = {"start": 0.0, "end": 1.0, "text": " Hi."}
        report_progress(1.0, 1.0, segment)
        return TranscriptionResult(" Hi.", [segment])


register_backend("talkative", Talkative)
result = Transcriber("talkative").transcribe(np.zeros(16000, np.float32), verbose=True)
assert result.text == " Hi.", result.text
"""


def test_nothing_is_printed_by_default():
    """Without a NullHandler, Python would print the library's warnings and errors itself."""
    run = subprocess.run(
        [sys.executable, "-c", QUIET_BY_DEFAULT], capture_output=True, text=True, cwd=PACKAGE.parent, timeout=120
    )

    assert (run.returncode, run.stdout, run.stderr) == (0, "", "")


###############################################################################
# The transcribe command: -q, -v and --log-level
###############################################################################


class ChattyBackend(FakeBackend):
    """FakeBackend that logs as a library module does, and records the -q settings it ran with."""

    name = "chatty"
    environment: dict = {}

    def transcribe(self, audio_path, language=None, task="transcribe", verbose=True):
        log = logging.getLogger("speech_toolkit.backends.chatty")
        log.debug("details from the backend")
        log.log(logging.INFO if verbose else logging.DEBUG, "Chatty: transcribed 1/1 chunks")
        log.warning("something odd in the audio")
        ChattyBackend.environment = {name: os.environ.get(name) for name in cli.QUIET_ENVIRONMENT}
        return super().transcribe(audio_path, language, task, verbose)


@pytest.fixture
def audio(monkeypatch, tmp_path):
    """talk.wav, with ChattyBackend registered as 'chatty' and no -q settings in the environment."""
    from speech_toolkit.backends import _BACKENDS

    monkeypatch.setattr("speech_toolkit.backends._BACKENDS", {**_BACKENDS, "chatty": ChattyBackend})
    for name in cli.QUIET_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    ChattyBackend.environment = {}
    path = tmp_path / "talk.wav"
    path.touch()
    return path


def run(audio: pathlib.Path, capsys, *flags: str) -> Tuple[str, List[str]]:
    """Run transcribe on *audio* with the chatty backend: its stdout, and its stderr's lines."""
    cli.main([str(audio), "-b", "chatty", "-m", "small", *flags])
    captured = capsys.readouterr()
    return captured.out, captured.err.splitlines()


class TestLogLevels:
    def test_info_messages_show_by_default(self, audio, capsys):
        out, err = run(audio, capsys)

        assert out == "Hello from talk.\n"  # stdout carries only the transcript
        assert "Chatty: transcribed 1/1 chunks" in err  # as the backends printed their progress before
        assert "Warning: something odd in the audio" in err
        assert not [line for line in err if "details" in line]

    def test_verbose_shows_debug_messages(self, audio, capsys):
        out, err = run(audio, capsys, "-v")

        assert out == "Hello from talk.\n"
        assert "Debug: details from the backend" in err
        assert f"Debug: Transcribing {audio} with chatty 'small'" in err  # the Transcriber's own messages
        assert "Chatty: transcribed 1/1 chunks" in err

    @pytest.mark.parametrize("level", ["warning", "WARNING"])
    def test_log_level(self, audio, capsys, level):
        out, err = run(audio, capsys, "--log-level", level)

        assert out == "Hello from talk.\n"
        assert "Chatty: transcribed 1/1 chunks" not in err
        assert "Warning: something odd in the audio" in err

    def test_log_level_error_hides_warnings(self, audio, capsys):
        _, err = run(audio, capsys, "--log-level", "error")
        assert not [line for line in err if "odd" in line]

    def test_quiet_shows_warnings_only(self, audio, capsys):
        out, err = run(audio, capsys, "-q")

        assert out == "Hello from talk.\n"
        assert err == ["Warning: something odd in the audio", "library noise on stdout"]  # the noise: FakeBackend's

    def test_quiet_with_a_log_level(self, audio, capsys):
        """-q still hides progress bars, but the log level asked for wins: the backend logs its progress at DEBUG."""
        _, err = run(audio, capsys, "-q", "--log-level", "debug")
        assert "Debug: Chatty: transcribed 1/1 chunks" in err

    def test_quiet_and_verbose_conflict(self, audio, capsys):
        with pytest.raises(SystemExit) as caught:
            cli.parse_args([str(audio), "-q", "-v"])
        assert caught.value.code == 2
        assert "not allowed with argument" in capsys.readouterr().err

    def test_unknown_level(self, audio, capsys):
        with pytest.raises(SystemExit):
            cli.parse_args([str(audio), "--log-level", "loud"])
        assert "invalid choice: 'loud'" in capsys.readouterr().err

    def test_the_logger_is_left_as_it_was(self, audio, capsys):
        logger = logging.getLogger("speech_toolkit")
        before = (list(logger.handlers), logger.level)

        run(audio, capsys, "-v")
        with pytest.raises(SystemExit):  # also when the command fails
            cli.main([str(audio.with_name("missing.wav")), "-b", "chatty", "-v"])

        assert (list(logger.handlers), logger.level) == before

    @pytest.mark.parametrize(
        "level, shown",
        [
            (logging.DEBUG, "Debug: text"),
            (logging.INFO, "text"),
            (logging.WARNING, "Warning: text"),
            (logging.ERROR, "Error: text"),
            (logging.CRITICAL, "Error: text"),
        ],
    )
    def test_format(self, level, shown):
        record = logging.LogRecord("speech_toolkit.api", level, __file__, 1, "%s", ("text",), None)
        assert cli._LogFormatter().format(record) == shown


class TestQuietLibraries:
    """-q hides Hugging Face's progress bars (downloads, and transformers' "Loading weights") and transformers'
    messages: libraries read the environment when imported, so it is set before any model loads."""

    def test_environment_while_transcribing(self, audio, capsys):
        run(audio, capsys, "-q")

        assert ChattyBackend.environment == {"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TRANSFORMERS_VERBOSITY": "error"}
        assert not set(cli.QUIET_ENVIRONMENT) & set(os.environ)  # restored afterwards

    def test_earlier_settings_are_restored(self, audio, capsys, monkeypatch):
        monkeypatch.setenv("TRANSFORMERS_VERBOSITY", "info")

        run(audio, capsys, "-q")

        assert ChattyBackend.environment["TRANSFORMERS_VERBOSITY"] == "error"
        assert os.environ["TRANSFORMERS_VERBOSITY"] == "info"

    def test_without_quiet_nothing_changes(self, audio, capsys):
        transformers_logging, hub_utils = MagicMock(), MagicMock()
        with patch.dict(
            sys.modules, {"transformers.utils.logging": transformers_logging, "huggingface_hub.utils": hub_utils}
        ):
            run(audio, capsys)

        assert ChattyBackend.environment == {"HF_HUB_DISABLE_PROGRESS_BARS": None, "TRANSFORMERS_VERBOSITY": None}
        assert transformers_logging.mock_calls == hub_utils.mock_calls == []

    def test_libraries_imported_already_are_told(self, audio, capsys):
        transformers_logging, hub_utils = MagicMock(), MagicMock()
        with patch.dict(
            sys.modules, {"transformers.utils.logging": transformers_logging, "huggingface_hub.utils": hub_utils}
        ):
            run(audio, capsys, "-q")

        transformers_logging.set_verbosity_error.assert_called_once_with()
        transformers_logging.disable_progress_bar.assert_called_once_with()
        hub_utils.disable_progress_bars.assert_called_once_with()


###############################################################################
# diarization.py: errors and messages
###############################################################################


@pytest.fixture
def pyannote(monkeypatch):
    """Stand-ins for torch and pyannote.audio."""
    monkeypatch.delenv("PYANNOTE_METRICS_ENABLED", raising=False)
    torch, pyannote_audio = MagicMock(), MagicMock()
    torch.cuda.is_available.return_value = False
    with patch.dict(sys.modules, {"torch": torch, "pyannote": MagicMock(), "pyannote.audio": pyannote_audio}):
        yield SimpleNamespace(torch=torch, Pipeline=pyannote_audio.Pipeline)


class TestDiarization:
    @pytest.mark.parametrize("error", [OSError("401 Client Error: gated repo"), ValueError("bad config")])
    def test_loading_errors_are_diarization_errors(self, pyannote, error):
        """Network, authentication and file errors from pyannote come in many types."""
        pyannote.Pipeline.from_pretrained.side_effect = error

        with pytest.raises(DiarizationError, match=f"could not load '{DIARIZATION_MODEL}': {error}") as caught:
            load_diarization_pipeline()

        assert caught.value.__cause__ is error
        assert "Accept its terms" in str(caught.value)
        assert isinstance(caught.value, RuntimeError)  # what the library raised before

    def test_the_command_reports_them(self, pyannote, audio, capsys):
        pyannote.Pipeline.from_pretrained.side_effect = OSError("Connection refused")

        with pytest.raises(SystemExit, match=f"Error: could not load '{DIARIZATION_MODEL}': Connection refused"):
            cli.main([str(audio), "-b", "chatty", "-m", "small", "-q", "--diarize"])

    def test_messages(self, pyannote, caplog):
        caplog.set_level(logging.DEBUG, logger="speech_toolkit")
        pipeline = load_diarization_pipeline()
        turns = [(SimpleNamespace(start=0.0, end=1.0), None, "A"), (SimpleNamespace(start=1.0, end=2.0), None, "B")]
        pipeline.return_value.exclusive_speaker_diarization.itertracks.return_value = turns

        diarize_audio(np.zeros(2 * SAMPLE_RATE, np.float32), pipeline)

        assert caplog.messages[:2] == [
            f"Loading the speaker-diarization pipeline {DIARIZATION_MODEL}",
            "Speaker diarization runs on cpu",
        ]
        assert caplog.messages[2].startswith("Found 2 turns of 2 speakers in 2.0 s of audio in ")
        assert {record.name for record in caplog.records} == {"speech_toolkit.diarization"}
