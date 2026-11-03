"""transcribe-server with faster-whisper, driven by the official OpenAI SDK."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from typing import Any, Dict, Iterator, Tuple

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
    write_wav,
)

STARTUP_TIMEOUT = 300  # seconds, model download included
MAX_AUDIO_SECONDS = 30  # the server's --max-audio-seconds, longer than the samples


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
    options = ["--port", str(port), "--max-audio-seconds", str(MAX_AUDIO_SECONDS), *device_args()]
    command = [sys.executable, "-m", *module, *BACKEND_ARGS["faster-whisper"], *options]
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


def post_form(url: str, fields: Dict[str, str], path: pathlib.Path) -> Tuple[str, bytes]:
    """POST *fields* and the file at *path* as a form, as the SDK can't send them: its Content-Type and body."""
    boundary = uuid.uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
        for name, value in fields.items()
    ]
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n\r\n')
    body = "".join(parts).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # localhost, never through a proxy
    with opener.open(request, timeout=300) as response:
        return response.headers["Content-Type"], response.read()


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


def test_audio_over_the_length_limit(client, tmp_path):
    """The server decodes uploads itself, at most a second past --max-audio-seconds, and refuses longer audio."""
    import numpy as np
    from openai import BadRequestError

    path = tmp_path / "silence.wav"
    write_wav(path, np.zeros(2 * MAX_AUDIO_SECONDS * 16000, np.float32), 16000)

    with pytest.raises(BadRequestError, match=f"longer than this server's {MAX_AUDIO_SECONDS} s limit"):
        create_transcription(client, path)


def test_streaming(client, speech):
    """The SDK streams the transcript: a delta per segment as faster-whisper decodes it, then the whole text."""
    with speech.path.open("rb") as audio:
        events = list(client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=audio, stream=True))

    *deltas, done = events
    assert deltas and {event.type for event in deltas} == {"transcript.text.delta"}
    assert done.type == "transcript.text.done"
    assert "".join(event.delta for event in deltas) == done.text
    assert_wer(done.text, speech, "faster-whisper")


def test_whisper_1_ignores_stream(server, speech):
    """As OpenAI does: a client that asks whisper-1 for a stream gets JSON, with usage in seconds of audio."""
    fields = {"model": "whisper-1", "stream": "true"}
    content_type, body = post_form(f"{server.url}/audio/transcriptions", fields, speech.path)

    assert content_type == "application/json"
    response = json.loads(body)
    assert response["usage"]["type"] == "duration"
    assert response["usage"]["seconds"] == pytest.approx(speech.duration, abs=1)  # whole seconds, rounded up
    assert_wer(response["text"], speech, "faster-whisper")


def test_translation_from_the_requested_language(server, speech):
    """/v1/audio/translations takes the language of the audio, an extension to OpenAI's API."""
    fields = {"model": "whisper-1", "language": "en"}
    _, body = post_form(f"{server.url}/audio/translations", fields, speech.path)

    assert_wer(json.loads(body)["text"], speech, "faster-whisper")
