"""transcribe-server's OpenAI features of #37: stream=true as server-sent events, diarized_json, usage, the language
of translations, the same speaker labels in every format, and an empty API key that means no key."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from speech_toolkit import AudioDecodeError, Transcriber
from speech_toolkit.backends import TranscriptionResult, report_progress
from speech_toolkit.server import (
    CLIENT_CLOSED,
    KEEPALIVE,
    SERVER_ERROR,
    _delta,
    _StreamEvents,
    _TranscriptStream,
    create_app,
    main,
    parse_args,
)
from tests.conftest import FakeBackend
from tests.server_helpers import decoded  # a fixture
from tests.test_nvidia_backend import FakeBatch, nvidia_modules  # a stand-in for Canary's batch, and a fixture
from tests.test_server_queue import GatedBackend, acquire, gated  # gated: a fixture

URL = "/v1/audio/transcriptions"
TRANSLATIONS = "/v1/audio/translations"
STREAM = {"model": "gpt-4o-mini-transcribe", "stream": "true"}  # a model that streams, as OpenAI's do
BOUNDARY = "toolkit-boundary"
WAIT = 10  # seconds a test waits for the server before it fails; never reached when the server works
TURNS = [(0.0, 0.9, "SPEAKER_00"), (0.9, 1.5, "SPEAKER_01")]  # what the stand-in diarization finds

pytestmark = pytest.mark.usefixtures("decoded")  # the server decodes uploads; here without ffmpeg


def upload(name: str = "talk.wav", data: bytes = b"RIFF fake audio") -> dict:
    return {"file": (name, io.BytesIO(data), "audio/wav")}


def sse(text: str) -> List[Dict[str, Any]]:
    """The events of a server-sent event stream, as OpenAI's SDKs read them: the data lines' JSON, comments skipped."""
    events = []
    for block in text.split("\n\n"):
        data = [line[len("data: ") :] for line in block.splitlines() if line.startswith("data: ")]
        if data:
            events.append(json.loads("\n".join(data)))
    return events


def deltas(events: List[Dict[str, Any]]) -> List[str]:
    return [event["delta"] for event in events if event["type"] == "transcript.text.delta"]


class SegmentsBackend(FakeBackend):
    """Reports its segments one at a time, as faster-whisper does, and calls ``after(index)`` after each one.

    ``reported`` has the text of each segment whose report returned; ``calls`` the language and task of each call.
    With ``wraps``, an error from a report comes out as a RuntimeError raised from it, as some libraries wrap
    their callbacks' errors.
    """

    name = "segments"
    capabilities = frozenset({"translate", "language_detection", "waveform"})
    texts = (" Hello", " there,", " world.")
    after: Callable[[int], None] = staticmethod(lambda index: None)
    duration: Optional[float] = None
    wraps = False
    reported: List[str] = []
    calls: List[tuple] = []

    def transcribe(self, audio_path, language=None, task="transcribe", verbose=True):
        cls = SegmentsBackend
        cls.calls.append((language, task))
        segments = []
        for index, text in enumerate(cls.texts):
            segment = {"id": index, "start": index * 0.5, "end": (index + 1) * 0.5, "text": text}
            segments.append(segment)
            try:
                report_progress(segment["end"], 1.5, segment)
            except Exception as e:
                if cls.wraps:
                    raise RuntimeError("the callback failed") from e
                raise
            cls.reported.append(text)
            cls.after(index)
        text = "".join(cls.texts)
        return TranscriptionResult(text, segments, language or "en", duration=cls.duration)


@pytest.fixture
def segments_backend(monkeypatch):
    from speech_toolkit.backends import _BACKENDS

    monkeypatch.setattr("speech_toolkit.backends._BACKENDS", {**_BACKENDS, "segments": SegmentsBackend})
    monkeypatch.setattr(SegmentsBackend, "after", staticmethod(lambda index: None))
    monkeypatch.setattr(SegmentsBackend, "duration", None)
    monkeypatch.setattr(SegmentsBackend, "texts", SegmentsBackend.texts)
    monkeypatch.setattr(SegmentsBackend, "wraps", False)
    SegmentsBackend.reported, SegmentsBackend.calls = [], []
    return SegmentsBackend


@pytest.fixture
def temp_dir(tmp_path, monkeypatch):
    """Send the server's temporary uploads (and Starlette's spooled ones) to a folder the test can inspect."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


@pytest.fixture
def client(segments_backend, temp_dir):
    return TestClient(create_app(Transcriber("segments")))


@pytest.fixture
def diarized(monkeypatch):
    """Diarization finds TURNS; returns a client of a diarizing server with *backend*."""
    monkeypatch.setattr("speech_toolkit.api.load_diarization_pipeline", lambda device: "pipeline")
    monkeypatch.setattr("speech_toolkit.api.diarize_audio", lambda source, pipeline, **hints: TURNS)

    def client(backend: str = "segments") -> TestClient:
        return TestClient(create_app(Transcriber(backend, diarize=True)))

    return client


@pytest.fixture
def stopped(monkeypatch):
    """An event set when a stream is stopped (its client left), after _TranscriptStream.stop() has run."""
    event = threading.Event()
    stop = _TranscriptStream.stop

    def stop_and_tell(self):
        stop(self)
        event.set()

    monkeypatch.setattr(_TranscriptStream, "stop", stop_and_tell)
    return event


def multipart(fields: Dict[str, str], data: bytes = b"RIFF fake audio") -> bytes:
    """A multipart/form-data body with *fields* and a file."""
    parts = [
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
        for name, value in fields.items()
    ]
    parts.append(f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="file"; filename="talk.wav"\r\n\r\n')
    return "".join(parts).encode() + data + f"\r\n--{BOUNDARY}--\r\n".encode()


def asgi_stream(
    app: Any,
    fields: Dict[str, str] = STREAM,
    *,
    leave: Optional[Callable[[], Any]] = None,
    on_body: Callable[[bytes], None] = lambda chunk: None,
    send_error: Optional[Exception] = None,
) -> SimpleNamespace:
    """POST to *app* through ASGI directly, as a client that sees each chunk of the response as it is sent.

    Once the body is sent, the client disconnects after *leave()* returns (it runs in a thread) if given, else once
    the response is complete. *on_body* gets each body chunk as it is sent; with *send_error*, sending a body chunk
    raises it, as ASGI servers that implement ASGI 2.4 do once the client has gone.
    """
    result = SimpleNamespace(status=None, headers={}, chunks=[])
    complete = asyncio.Event()
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": multipart(fields), "more_body": False}
        if leave is not None:
            await asyncio.to_thread(leave)
        else:
            await complete.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            result.status = message["status"]
            result.headers = {name.decode(): value.decode() for name, value in message["headers"]}
            return
        if send_error is not None:
            raise send_error
        result.chunks.append(message.get("body", b""))
        on_body(message.get("body", b""))
        if not message.get("more_body", False):
            complete.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": URL,
        "raw_path": URL.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [(b"content-type", f"multipart/form-data; boundary={BOUNDARY}".encode())],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8000),
    }
    asyncio.run(app(scope, receive, send))
    return result


def wait_for(event: threading.Event) -> None:
    assert event.wait(WAIT), "the server didn't get there in time"


class TestStreaming:
    def test_a_delta_per_segment_then_done(self, client, temp_dir):
        response = client.post(URL, files=upload(), data=STREAM)

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert response.headers["cache-control"] == "no-cache"
        assert "content-length" not in response.headers
        assert sse(response.text) == [
            {"type": "transcript.text.delta", "delta": "Hello"},
            {"type": "transcript.text.delta", "delta": " there,"},
            {"type": "transcript.text.delta", "delta": " world."},
            {"type": "transcript.text.done", "text": "Hello there, world."},
        ]
        assert response.text.startswith('data: {"type":"transcript.text.delta","delta":"Hello"}\n\n')
        assert list(temp_dir.iterdir()) == []

    @pytest.mark.parametrize("model", [None, "whisper-1"])
    def test_whisper_1_ignores_stream(self, client, model):
        """As OpenAI does: whisper-1 can't stream, and the default model is whisper-1."""
        data = {"stream": "true", **({"model": model} if model else {})}

        response = client.post(URL, files=upload(), data=data)

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert response.json() == {"text": "Hello there, world.", "usage": {"type": "duration", "seconds": 2}}

    def test_whisper_1_ignores_stream_in_every_format(self, client):
        response = client.post(
            URL, files=upload(), data={"model": "whisper-1", "stream": "true", "response_format": "srt"}
        )

        assert response.status_code == 200
        assert response.text.startswith("1\n00:00:00,000 --> 00:00:00,500\n")

    def test_a_segment_without_text_sends_no_delta(self, client, segments_backend):
        segments_backend.texts = (" Hello", "", " world.")

        response = client.post(URL, files=upload(), data=STREAM)

        assert deltas(sse(response.text)) == ["Hello", " world."]

    def test_text_streams_as_json_does(self, client):
        response = client.post(URL, files=upload(), data={**STREAM, "response_format": "text"})

        assert [event["type"] for event in sse(response.text)][-1] == "transcript.text.done"

    @pytest.mark.parametrize("response_format", ["srt", "vtt", "verbose_json"])
    def test_formats_that_dont_stream(self, client, response_format):
        response = client.post(URL, files=upload(), data={**STREAM, "response_format": response_format})

        assert response.status_code == 400
        assert response.json()["error"] == {
            "message": "A streamed transcription (stream=true) takes response_format json, text, diarized_json; "
            f"send stream=false for {response_format}.",
            "type": "invalid_request_error",
            "param": "response_format",
            "code": None,
        }

    def test_an_upload_that_isnt_audio_is_an_http_error(self, client, decoded, segments_backend, temp_dir):
        decoded.error = AudioDecodeError("Failed to load audio: Invalid data found when processing input")

        response = client.post(URL, files=upload(), data=STREAM)

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_value"
        assert segments_backend.calls == []
        assert list(temp_dir.iterdir()) == []

    @pytest.mark.parametrize(
        "error, status, message",
        [
            (ValueError("Unsupported language: xx"), 400, "Unsupported language: xx"),
            (RuntimeError("CUDA out of memory"), 500, SERVER_ERROR),
        ],
    )
    def test_a_failure_before_the_first_segment_is_an_http_error(self, client, temp_dir, error, status, message):
        """The response starts with its first event, so errors until then get an HTTP status, as without stream."""
        with patch.object(SegmentsBackend, "transcribe", side_effect=error):
            response = client.post(URL, files=upload(), data={**STREAM, "language": "xx"})

        assert response.status_code == status
        assert response.headers["content-type"] == "application/json"
        assert response.json()["error"]["message"] == message
        assert list(temp_dir.iterdir()) == []

    def test_a_model_failure_mid_stream_is_an_error_event(self, client, segments_backend, temp_dir, caplog):
        def fail(index):
            raise RuntimeError("CUDA out of memory")

        segments_backend.after = staticmethod(fail)

        response = client.post(URL, files=upload(), data=STREAM)

        assert response.status_code == 200  # the stream had started
        assert sse(response.text) == [
            {"type": "transcript.text.delta", "delta": "Hello"},
            {
                "type": "error",
                "error": {"message": SERVER_ERROR, "type": "server_error", "param": None, "code": None},
            },
        ]
        assert "CUDA out of memory" in caplog.text  # only in the server's log
        assert list(temp_dir.iterdir()) == []
        assert client.get("/health").json()["transcribing"] == 0

    def test_a_failure_building_the_events_is_an_error_event(self, client, caplog, monkeypatch):
        monkeypatch.setattr(_StreamEvents, "finish", MagicMock(side_effect=KeyError("text")))

        response = client.post(URL, files=upload(), data=STREAM)

        assert sse(response.text)[-1]["error"]["message"] == SERVER_ERROR
        assert "Streaming a transcription failed" in caplog.text

    def test_the_request_language_and_the_server_default(self, segments_backend, temp_dir):
        client = TestClient(create_app(Transcriber("segments"), language="fr"))

        client.post(URL, files=upload(), data=STREAM)
        client.post(URL, files=upload(), data={**STREAM, "language": "de"})

        assert segments_backend.calls == [("fr", "transcribe"), ("de", "transcribe")]


class TestDeltas:
    @pytest.mark.parametrize(
        "texts, sent",
        [
            # Whisper: a space before each segment, which the transcript strips at its start
            ([" Hello", " there,", " world."], ["Hello", " there,", " world."]),
            # chunks (Canary, Voxtral): stripped text, which the transcript joins with spaces, empty ones left out
            (["Hallo", "", "Welt", "  "], ["Hallo", " Welt"]),
            # CJK: Whisper writes no space between segments
            (["你好。", "我很好。", "Fine"], ["你好。", "我很好。", "Fine"]),
            (["", " Hi"], ["Hi"]),
        ],
    )
    def test_the_deltas_add_up_to_the_transcript(self, texts, sent):
        events = _StreamEvents()

        chunks = [events.delta({"text": text}) for text in texts]

        assert deltas(sse(b"".join(chunks).decode())) == sent
        assert events.text == "".join(sent)

    def test_a_segment_without_text_sends_nothing(self):
        assert _StreamEvents().delta({"start": 0.0, "end": 1.0}) == b""
        assert _delta("Hi", "") == ""


class TestIncrementally:
    """Through ASGI directly, where the test sees each chunk as the server sends it."""

    def test_each_delta_is_sent_as_its_segment_is_reported(self, segments_backend, temp_dir):
        first_sent = threading.Event()
        segments_backend.after = staticmethod(lambda index: index == 0 and wait_for(first_sent))

        response = asgi_stream(
            create_app(Transcriber("segments")), on_body=lambda chunk: b"Hello" in chunk and first_sent.set()
        )

        assert response.status == 200
        assert [event["type"] for event in sse(b"".join(response.chunks).decode())] == [
            "transcript.text.delta",
            "transcript.text.delta",
            "transcript.text.delta",
            "transcript.text.done",
        ]
        assert response.chunks[0] == b'data: {"type":"transcript.text.delta","delta":"Hello"}\n\n'
        assert response.chunks[-1] == b""  # the end of the response
        assert list(temp_dir.iterdir()) == []

    @pytest.mark.parametrize("wraps", [False, True], ids=["as-raised", "wrapped-by-the-backend"])
    def test_a_client_that_leaves_mid_stream_stops_the_model(self, segments_backend, stopped, temp_dir, caplog, wraps):
        caplog.set_level(logging.INFO, logger="speech_toolkit")
        segments_backend.wraps = wraps
        first_sent = threading.Event()
        segments_backend.after = staticmethod(lambda index: wait_for(stopped))
        app = create_app(Transcriber("segments"))

        response = asgi_stream(app, leave=lambda: wait_for(first_sent), on_body=lambda chunk: first_sent.set())

        assert response.status == 200
        assert response.chunks == [b'data: {"type":"transcript.text.delta","delta":"Hello"}\n\n']
        assert segments_backend.reported == [" Hello"]  # the next segment's report stopped the model
        assert "A client disconnected from its streamed transcription; stopping it." in caplog.messages
        assert "A client disconnected during its streamed transcription; stopped the transcription." in caplog.messages
        assert "Transcription failed" not in caplog.text  # not an error
        assert list(temp_dir.iterdir()) == []
        client = TestClient(app)
        assert client.get("/health").json()["waiting"] == 0
        assert sse(client.post(URL, files=upload(), data=STREAM).text)[-1]["type"] == "transcript.text.done"

    def test_a_stream_whose_client_left_keeps_its_place_until_the_model_is_free(
        self, segments_backend, stopped, temp_dir
    ):
        gate, first_sent = threading.Event(), threading.Event()
        segments_backend.after = staticmethod(lambda index: wait_for(gate))
        app = create_app(Transcriber("segments"), max_queue=0)
        client = TestClient(app)
        with ThreadPoolExecutor(1) as pool:
            try:
                left = pool.submit(
                    asgi_stream, app, leave=lambda: wait_for(first_sent), on_body=lambda chunk: first_sent.set()
                )
                wait_for(stopped)
                refused = client.post(URL, files=upload(), data=STREAM)  # the model still runs the stopped stream
                health = client.get("/health").json()
            finally:
                gate.set()
            assert left.result(timeout=WAIT).status == 200

        assert refused.status_code == 503
        assert (health["transcribing"], health["waiting"]) == (1, 0)
        assert segments_backend.reported == [" Hello"]
        assert client.post(URL, files=upload(), data=STREAM).status_code == 200  # free again

    def test_a_client_that_leaves_while_its_upload_is_decoded(self, segments_backend, stopped, temp_dir, monkeypatch):
        """The job has started, so it can't be dropped: it stops before the model runs."""
        decoding = threading.Event()

        def decode(source, **options):
            decoding.set()
            wait_for(stopped)
            import numpy as np

            return np.zeros(24000, np.float32)

        monkeypatch.setattr("speech_toolkit.server.load_audio", decode)

        response = asgi_stream(create_app(Transcriber("segments")), leave=lambda: wait_for(decoding))

        assert response.status == CLIENT_CLOSED
        assert segments_backend.calls == []
        assert list(temp_dir.iterdir()) == []

    def test_a_client_that_leaves_while_it_waits_is_dropped(self, gated, decoded, temp_dir):
        app = create_app(Transcriber("gated"), max_queue=1)
        with TestClient(app) as client, ThreadPoolExecutor(1) as pool:
            try:
                first = pool.submit(client.post, URL, files=upload())
                acquire(gated.started)
                left = asgi_stream(app, leave=lambda: None)  # waits behind the first, then leaves
            finally:
                gated.gate.set()
            assert first.result(timeout=WAIT).status_code == 200

        assert left.status == CLIENT_CLOSED
        assert gated.calls == 1 and len(decoded.calls) == 1  # the stream never reached the decoder or the model
        assert list(temp_dir.iterdir()) == []

    def test_a_comment_keeps_the_connection_open_while_no_event_is_due(self, segments_backend, monkeypatch):
        monkeypatch.setattr("speech_toolkit.server.KEEPALIVE_SECONDS", 0.01)
        kept_alive = threading.Event()
        segments_backend.after = staticmethod(lambda index: index == 0 and wait_for(kept_alive))

        response = asgi_stream(
            create_app(Transcriber("segments")), on_body=lambda chunk: chunk == KEEPALIVE and kept_alive.set()
        )

        assert response.chunks[0].startswith(b"data: ") and response.chunks[1] == KEEPALIVE
        assert sse(b"".join(response.chunks).decode())[-1] == {
            "type": "transcript.text.done",
            "text": "Hello there, world.",
        }

    def test_a_send_that_raises_stops_the_stream(self, segments_backend, stopped, temp_dir):
        """As ASGI 2.4 servers do once the client has gone."""
        segments_backend.after = staticmethod(lambda index: wait_for(stopped))

        response = asgi_stream(create_app(Transcriber("segments")), send_error=OSError("client disconnected"))

        assert response.status == 200 and response.chunks == []
        assert segments_backend.reported == [" Hello"]
        assert list(temp_dir.iterdir()) == []

    def test_events_after_the_event_loop_closed_stop_the_stream(self):
        """As when the server stops while the model runs: the worker can't reach the loop any more."""
        loop = asyncio.new_event_loop()
        loop.close()
        stream = _TranscriptStream(loop)

        stream.send(b"data: {}\n\n")

        assert stream.stopped


class TestDiarizedJson:
    # The shape of OpenAI's diarized_json (openai.types.audio.TranscriptionDiarized), with the values of its docs.
    OPENAI = {
        "task": "transcribe",
        "duration": 27.4,
        "text": "Thanks for calling. Hi, I need help.",
        "segments": [
            {
                "type": "transcript.text.segment",
                "id": "seg_001",
                "start": 0.0,
                "end": 4.7,
                "text": "Thanks for calling.",
                "speaker": "A",
            },
        ],
        "usage": {"type": "duration", "seconds": 28},
    }

    @staticmethod
    def shape(value: Any) -> Any:
        """The keys and value types of *value*, recursively; a list by its first item, numbers as numbers."""
        if isinstance(value, dict):
            return {key: TestDiarizedJson.shape(item) for key, item in value.items()}
        if isinstance(value, list):
            return [TestDiarizedJson.shape(value[0])] if value else []
        return "number" if isinstance(value, (int, float)) else type(value).__name__

    def test_matches_openais_shape(self, segments_backend, diarized):
        body = diarized().post(URL, files=upload(), data={"response_format": "diarized_json"}).json()

        assert self.shape(body) == self.shape(self.OPENAI)

    def test_a_segment_per_speaker_turn(self, segments_backend, diarized):
        body = diarized().post(URL, files=upload(), data={"response_format": "diarized_json"}).json()

        assert (body["task"], body["duration"], body["usage"]) == (
            "transcribe",
            1.5,
            {"type": "duration", "seconds": 2},
        )
        assert [(s["id"], s["speaker"], s["text"], s["start"]) for s in body["segments"]] == [
            ("seg_0", "SPEAKER_00", "Hello", 0.0),
            ("seg_1", "SPEAKER_00", "there,", 0.5),
            ("seg_2", "SPEAKER_01", "world.", 1.0),
        ]
        assert body["text"] == "[SPEAKER_00] Hello there,\n[SPEAKER_01] world."

    def test_streamed_it_sends_the_speaker_turns_before_done(self, segments_backend, diarized):
        response = diarized().post(URL, files=upload(), data={**STREAM, "response_format": "diarized_json"})

        events = sse(response.text)
        assert [event["type"] for event in events] == ["transcript.text.delta"] * 3 + [
            "transcript.text.segment"
        ] * 3 + ["transcript.text.done"]
        assert events[3] == {
            "type": "transcript.text.segment",
            "id": "seg_0",
            "start": 0.0,
            "end": 0.5,
            "text": "Hello",
            "speaker": "SPEAKER_00",
        }
        assert events[-1]["text"] == "[SPEAKER_00] Hello there,\n[SPEAKER_01] world."

    def test_needs_diarization(self, client):
        response = client.post(URL, files=upload(), data={"response_format": "diarized_json"})

        assert response.status_code == 400
        assert "start the server with --diarize" in response.json()["error"]["message"]

    def test_translations_dont_take_it(self, segments_backend, diarized):
        response = diarized().post(TRANSLATIONS, files=upload(), data={"response_format": "diarized_json"})

        assert response.status_code == 400
        assert response.json()["error"]["message"] == (
            "response_format must be one of: json, text, srt, vtt, verbose_json."
        )


class TestSpeakers:
    def test_every_format_labels_the_speakers_as_text_does(self, segments_backend, diarized):
        client = diarized()

        def post(response_format):
            return client.post(URL, files=upload(), data={"response_format": response_format})

        text = post("text").text
        assert text == "[SPEAKER_00] Hello there,\n[SPEAKER_01] world."
        assert post("json").json()["text"] == text
        assert post("verbose_json").json()["text"] == text
        assert post("diarized_json").json()["text"] == text
        assert "[SPEAKER_01] world." in post("srt").text
        assert client.post(TRANSLATIONS, files=upload()).json()["text"] == text

    def test_without_diarization_the_text_is_the_transcript(self, client):
        assert client.post(URL, files=upload(), data={"response_format": "verbose_json"}).json()["text"] == (
            "Hello there, world."
        )


class TestUsage:
    @pytest.mark.parametrize("seconds, billed", [(2.0, 2), (2.01, 3), (0.2, 1)])
    def test_whole_seconds_of_the_decoded_audio_rounded_up(self, client, decoded, seconds, billed):
        decoded.seconds = seconds

        assert client.post(URL, files=upload()).json()["usage"] == {"type": "duration", "seconds": billed}

    def test_the_decoded_duration_not_the_backends(self, client, segments_backend):
        """The server decodes the upload itself, so it knows the audio's length."""
        segments_backend.duration = 99.0

        body = client.post(URL, files=upload(), data={"response_format": "verbose_json"}).json()

        assert (body["duration"], body["usage"]) == (1.5, {"type": "duration", "seconds": 2})

    def test_translations_have_it_too(self, client):
        assert client.post(TRANSLATIONS, files=upload()).json() == {
            "text": "Hello there, world.",
            "usage": {"type": "duration", "seconds": 2},
        }

    @pytest.mark.parametrize("response_format", ["text", "srt", "vtt"])
    def test_not_in_text_formats(self, client, response_format):
        assert "usage" not in client.post(URL, files=upload(), data={"response_format": response_format}).text


class TestTranslationLanguage:
    def test_canary_translates_from_the_requested_language(self, nvidia_modules, temp_dir):
        processor = nvidia_modules.processor
        processor.apply_transcription_request.side_effect = lambda **kw: FakeBatch(
            decoder_input_ids=SimpleNamespace(shape=(len(kw["audio"]), 4))
        )
        processor.batch_decode.return_value = ["Good morning."]
        client = TestClient(create_app(Transcriber("canary")))

        response = client.post(TRANSLATIONS, files=upload(), data={"model": "whisper-1", "language": "de"})

        assert response.json()["text"] == "Good morning."
        kwargs = processor.apply_transcription_request.call_args[1]
        assert (kwargs["source_language"], kwargs["target_language"]) == ("de", "en")

    def test_the_server_language_is_the_default(self, segments_backend, temp_dir):
        client = TestClient(create_app(Transcriber("segments"), language="fr"))

        client.post(TRANSLATIONS, files=upload())
        client.post(TRANSLATIONS, files=upload(), data={"language": "de"})

        assert segments_backend.calls == [("fr", "translate"), ("de", "translate")]

    def test_the_language_option(self, segments_backend, temp_dir):
        uvicorn = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": uvicorn}):
            main(["-b", "segments", "-l", "it"])

        (app,), _ = uvicorn.run.call_args
        TestClient(app).post(TRANSLATIONS, files=upload())
        assert segments_backend.calls == [("it", "translate")]
        assert parse_args(["--language", "de"]).language == "de"
        assert parse_args([]).language is None


class TestEmptyApiKey:
    def test_is_no_key(self, client, segments_backend, temp_dir):
        app_client = TestClient(create_app(Transcriber("segments"), api_key=""))

        assert app_client.post(URL, files=upload()).status_code == 200
        assert app_client.get("/openapi.json").status_code == 200  # the docs are on, as without a key

    def test_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("TRANSCRIBE_API_KEY", "")
        assert parse_args([]).api_key is None

    @pytest.mark.parametrize("argv", [["--api-key", ""], []])
    def test_warns_when_open_to_the_network(self, fake_backend, monkeypatch, capsys, argv):
        monkeypatch.setenv("TRANSCRIBE_API_KEY", "")
        uvicorn = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": uvicorn}):
            main(["-b", "fake", "--host", "0.0.0.0", *argv])

        assert "without --api-key" in capsys.readouterr().err
        (app,), _ = uvicorn.run.call_args
        assert TestClient(app).get("/v1/models").status_code == 200


class TestOpenAISDK:
    """The official SDK, its requests sent to the app through a TestClient."""

    @pytest.fixture
    def openai(self):
        return pytest.importorskip("openai")

    @pytest.fixture
    def sdk(self, openai, segments_backend, temp_dir):
        def sdk(app: Any) -> Any:
            return openai.OpenAI(
                base_url="http://testserver/v1", api_key="unused", http_client=TestClient(app), max_retries=0
            )

        return sdk

    def test_streams_a_transcription(self, openai, sdk):
        from openai.types.audio import TranscriptionTextDeltaEvent, TranscriptionTextDoneEvent

        with sdk(create_app(Transcriber("segments"))) as client:
            stream = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe", file=("talk.wav", b"RIFF fake audio"), stream=True
            )
            events = list(stream)

        assert [type(event) for event in events] == [TranscriptionTextDeltaEvent] * 3 + [TranscriptionTextDoneEvent]
        assert "".join(event.delta for event in events[:-1]) == events[-1].text == "Hello there, world."

    def test_raises_an_error_event(self, openai, sdk, segments_backend):
        def fail(index):
            raise RuntimeError("CUDA out of memory")

        segments_backend.after = staticmethod(fail)

        with sdk(create_app(Transcriber("segments"))) as client:
            stream = client.audio.transcriptions.create(
                model="gpt-4o-transcribe", file=("talk.wav", b"RIFF"), stream=True
            )
            with pytest.raises(openai.APIError, match="The server had an error"):
                list(stream)

    def test_the_responses_validate_as_the_sdks_types(self, openai, sdk, diarized):
        """Strictly, with pydantic: the SDK itself builds its objects without checking them."""
        from openai.types.audio import (
            Transcription,
            TranscriptionDiarized,
            TranscriptionTextDeltaEvent,
            TranscriptionTextDoneEvent,
            TranscriptionTextSegmentEvent,
            TranscriptionVerbose,
        )

        client = diarized()
        types = {"json": Transcription, "verbose_json": TranscriptionVerbose, "diarized_json": TranscriptionDiarized}
        for response_format, model in types.items():
            body = client.post(URL, files=upload(), data={"response_format": response_format}).json()
            assert model.model_validate(body, strict=True).usage.seconds == 2

        events = {
            "transcript.text.delta": TranscriptionTextDeltaEvent,
            "transcript.text.segment": TranscriptionTextSegmentEvent,
            "transcript.text.done": TranscriptionTextDoneEvent,
        }
        streamed = client.post(URL, files=upload(), data={**STREAM, "response_format": "diarized_json"})
        for event in sse(streamed.text):
            events[event["type"]].model_validate(event, strict=True)
