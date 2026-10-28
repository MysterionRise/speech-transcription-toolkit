"""transcribe-server shows the package's log on stderr while it runs, also once the speech_toolkit logger has a
NullHandler (which keeps Python's last-resort handler from printing anything)."""

from __future__ import annotations

import io
import logging
import sys
from typing import Any, Callable, List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from speech_toolkit import Transcriber
from speech_toolkit.server import main
from tests.server_helpers import decoded  # a fixture

URL = "/v1/audio/transcriptions"
pytestmark = pytest.mark.usefixtures("fake_backend", "decoded")


def upload() -> dict:
    return {"file": ("talk.wav", io.BytesIO(b"RIFF fake audio"), "audio/wav")}


@pytest.fixture
def package_logger(monkeypatch):
    """The speech_toolkit logger; the handlers and level a test sets are gone afterwards."""
    package = logging.getLogger("speech_toolkit")
    monkeypatch.setattr(package, "handlers", list(package.handlers))
    level = package.level
    yield package
    package.setLevel(level)


@pytest.fixture
def serve():
    """Run main() for the fake backend, with uvicorn.run() replaced by a function of the app, run in its place."""
    uvicorn = MagicMock()

    def run(during_the_run: Callable[[Any], None], *argv: str) -> None:
        uvicorn.run.side_effect = lambda app, **kwargs: during_the_run(app)
        main(["-b", "fake-words", *argv])

    with patch.dict(sys.modules, {"uvicorn": uvicorn}):
        yield run


def failing_request(responses: List[Any]) -> Callable[[Any], None]:
    """A request whose transcription fails, which the server logs with its traceback."""

    def during_the_run(app: Any) -> None:
        with patch.object(Transcriber, "transcribe", side_effect=RuntimeError("CUDA out of memory")):
            responses.append(TestClient(app).post(URL, files=upload()))

    return during_the_run


@pytest.mark.parametrize("null_handler", [False, True], ids=["no-handler", "null-handler"])
def test_a_logged_error_reaches_stderr(serve, package_logger, capsys, null_handler):
    if null_handler:
        package_logger.addHandler(logging.NullHandler())
    responses: List[Any] = []

    serve(failing_request(responses))

    err = capsys.readouterr().err
    assert responses[0].status_code == 500
    assert "ERROR: speech_toolkit.server: Transcription failed\nTraceback" in err
    assert "RuntimeError: CUDA out of memory" in err


def test_from_info_and_only_while_the_server_runs(serve, package_logger, capsys):
    handlers, level = list(package_logger.handlers), package_logger.level

    def during_the_run(app: Any) -> None:
        log = logging.getLogger("speech_toolkit.server")
        log.debug("not shown")
        log.info("A client disconnected")

    serve(during_the_run)
    logging.getLogger("speech_toolkit.server").info("after the run")

    err = capsys.readouterr().err
    assert "INFO: speech_toolkit.server: A client disconnected" in err
    assert "not shown" not in err and "after the run" not in err
    assert (package_logger.handlers, package_logger.level) == (handlers, level)  # as it was


def test_a_level_set_already_is_kept(serve, package_logger, capsys):
    package_logger.setLevel(logging.WARNING)
    responses: List[Any] = []

    def during_the_run(app: Any) -> None:
        logging.getLogger("speech_toolkit.server").info("A client disconnected")
        failing_request(responses)(app)

    serve(during_the_run)

    err = capsys.readouterr().err
    assert "ERROR: speech_toolkit.server: Transcription failed" in err
    assert "A client disconnected" not in err
    assert package_logger.level == logging.WARNING


def test_a_handler_set_up_already_is_used_instead(serve, package_logger, capsys):
    own = io.StringIO()
    package_logger.addHandler(logging.StreamHandler(own))
    handlers = list(package_logger.handlers)  # also the package's NullHandler
    responses: List[Any] = []

    serve(failing_request(responses))

    assert "Transcription failed" in own.getvalue()
    assert "Transcription failed" not in capsys.readouterr().err  # not twice
    assert package_logger.handlers == handlers


def test_the_handler_goes_when_the_model_fails_to_load(serve, package_logger):
    handlers = list(package_logger.handlers)

    with pytest.raises(SystemExit, match="Unknown fake model"):
        serve(lambda app: None, "-m", "huge")

    assert package_logger.handlers == handlers
