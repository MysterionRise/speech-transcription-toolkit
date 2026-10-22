"""transcribe-server with faster-whisper, driven by the official OpenAI SDK."""

from __future__ import annotations

import dataclasses
import pathlib
import socket
import subprocess
import sys
import time
import urllib.request
from typing import Any, Iterator

import pytest

from tests.integration.devices import device_args
from tests.integration.helpers import (
    BACKEND_ARGS,
    EventLog,
    assert_timeline,
    assert_wer,
    instrumented_env,
    parse_cues,
    require,
)

STARTUP_TIMEOUT = 300  # seconds, model download included


@dataclasses.dataclass(frozen=True)
class Server:
    url: str  # the API's base URL
    events: EventLog  # what the server's instrumentation records


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_until_healthy(url: str, process: subprocess.Popen, log: pathlib.Path) -> None:
    """Wait until GET *url* answers, failing the test if the server exits or takes longer than STARTUP_TIMEOUT."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # localhost, never through a proxy
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"transcribe-server exited with {process.returncode}:\n{read_log(log)}")
        try:
            with opener.open(url, timeout=5):
                return
        except OSError:  # not listening yet
            time.sleep(1)
    pytest.fail(f"transcribe-server didn't answer within {STARTUP_TIMEOUT} s:\n{read_log(log)}")


def read_log(log: pathlib.Path) -> str:
    return log.read_text(encoding="utf-8", errors="replace")


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Server]:
    """A transcribe-server with faster-whisper tiny on a free port, with --device from INTEGRATION_DEVICE.

    It runs through the instrumentation, like the transcribe command: the tests fail unless its model is on
    INTEGRATION_DEVICE. Its log is server.log.
    """
    require("server", "faster-whisper")
    folder = tmp_path_factory.mktemp("server")
    log, port, events = folder / "server.log", free_port(), EventLog(folder / "instrument-events.jsonl")
    module = ["tests.integration.instrument", "speech_toolkit.server"]  # python -m speech_toolkit.server, instrumented
    command = [sys.executable, "-m", *module, *BACKEND_ARGS["faster-whisper"], "--port", str(port), *device_args()]
    with log.open("w", encoding="utf-8") as out:
        env = instrumented_env(events.path)
        process = subprocess.Popen(command, cwd=folder, env=env, stdout=out, stderr=subprocess.STDOUT)
    try:  # the server stops even when a check below fails
        wait_until_healthy(f"http://127.0.0.1:{port}/health", process, log)
        if not events.record():  # the model loads before the server answers; this checks its device
            pytest.fail(f"the instrumentation recorded no model load:\n{read_log(log)}")
        yield Server(f"http://127.0.0.1:{port}/v1", events)
    finally:
        process.terminate()  # TerminateProcess on Windows, SIGTERM elsewhere
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@pytest.fixture(scope="module")
def client(server: Server) -> Iterator[Any]:
    from openai import OpenAI

    with OpenAI(base_url=server.url, api_key="unused", max_retries=0, timeout=300) as client:
        yield client


@pytest.fixture(autouse=True)
def record_requests(server: Server) -> Iterator[None]:
    """After each test, its transcriptions go in the report, with the WER the test checked."""
    yield
    server.events.record()


def create_transcription(client: Any, path: pathlib.Path, **options: Any) -> Any:
    with path.open("rb") as audio:
        return client.audio.transcriptions.create(model="whisper-1", file=audio, **options)


def test_json(client, speech):
    assert_wer(create_transcription(client, speech.path).text, speech, "faster-whisper")


def test_srt(client, speech):
    cues = parse_cues(create_transcription(client, speech.path, response_format="srt"))

    assert_timeline(((cue.start, cue.end) for cue in cues), speech.duration, "cues")
    assert_wer(" ".join(cue.text for cue in cues), speech, "faster-whisper")


def test_verbose_json_with_word_timestamps(client, dialogue):
    options = {"response_format": "verbose_json", "timestamp_granularities": ["word"]}
    verbose = create_transcription(client, dialogue.path, **options)

    assert verbose.duration == pytest.approx(dialogue.duration, abs=0.1)
    assert_timeline(((seg.start, seg.end) for seg in verbose.segments), dialogue.duration, "segments")
    assert_timeline(((word.start, word.end) for word in verbose.words), dialogue.duration, "words")
    assert_wer(" ".join(word.word for word in verbose.words), dialogue, "faster-whisper")
