"""transcribe-server under load (#31): the bounded queue in front of the model, a /health that never waits for it,
clients that leave, decoding once with a length limit, CORS and the docs switch."""

from __future__ import annotations

import asyncio
import io
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from speech_toolkit import AudioDecodeError, Transcriber
from speech_toolkit.backends import TranscriptionResult, get_backend_class, list_backends
from speech_toolkit.media import SAMPLE_RATE
from speech_toolkit.server import (
    AUDIO_LIMIT_MARGIN,
    CLIENT_CLOSED,
    DECODE_TIMEOUT,
    DEFAULT_MAX_AUDIO_SECONDS,
    DEFAULT_MAX_QUEUE,
    _Service,
    create_app,
    main,
    parse_args,
)
from tests.conftest import FakeBackend
from tests.server_helpers import decoded  # a fixture

URL = "/v1/audio/transcriptions"
KEY = "s3cret"
AUTH = {"Authorization": f"Bearer {KEY}"}
ORIGIN = "http://localhost:3000"
BOUNDARY = "toolkit-boundary"
MIB = 1024 * 1024
WAIT = 10  # seconds a test waits for the server before it fails; never reached when the server works


def upload(name: str = "talk.wav", data: bytes = b"RIFF fake audio") -> dict:
    return {"file": (name, io.BytesIO(data), "audio/wav")}


def multipart(data: bytes = b"RIFF fake audio", name: str = "talk.wav") -> bytes:
    """A multipart/form-data body with one file field."""
    head = f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n\r\n'
    return head.encode() + data + f"\r\n--{BOUNDARY}--\r\n".encode()


FORM = [("content-type", f"multipart/form-data; boundary={BOUNDARY}")]


def asgi_post(
    app: Any,
    chunks: Iterable[bytes] = (multipart(),),
    *,
    headers: Iterable[tuple] = FORM,
    leave: Optional[Callable[[], Any]] = None,
) -> SimpleNamespace:
    """POST to *app* through ASGI directly, as a client whose body arrives in *chunks*. Once the body is sent, the
    client disconnects, after *leave()* returns (it runs in a thread) if given, else at once.

    Returns what the app read (``read``, the number of chunks) and answered (``status``, ``headers``, ``json``).
    """
    chunks = iter(chunks)
    result = SimpleNamespace(read=0, status=None, headers={}, json=None)
    body = []
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            chunk = next(chunks, None)
            if chunk is not None:
                result.read += 1
                return {"type": "http.request", "body": chunk, "more_body": True}
            sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        if leave is not None:
            await asyncio.to_thread(leave)
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            result.status = message["status"]
            result.headers = {name.decode(): value.decode() for name, value in message["headers"]}
        else:
            body.append(message.get("body", b""))

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
        "headers": [(name.encode(), value.encode()) for name, value in headers],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8000),
    }
    asyncio.run(app(scope, receive, send))
    result.json = json.loads(b"".join(body)) if body else None
    return result


def acquire(semaphore: threading.Semaphore, times: int = 1) -> None:
    for _ in range(times):
        assert semaphore.acquire(timeout=WAIT), "the server didn't get there in time"


def timed(call: Callable[..., Any], *args: Any) -> tuple:
    start = time.perf_counter()
    result = call(*args)
    return result, time.perf_counter() - start


class GatedBackend(FakeBackend):
    """Takes the samples, and finishes a transcription only once the test opens its gate.

    ``started`` is released as each transcription starts; ``calls`` counts them, ``most_at_once`` is the most that
    ran at the same time.
    """

    name = "gated"
    capabilities = frozenset({"waveform"})
    gate = threading.Event()
    started = threading.Semaphore(0)
    lock = threading.Lock()
    calls = running = most_at_once = 0

    def transcribe(self, audio_path, language=None, task="transcribe", verbose=True):
        cls = GatedBackend
        with cls.lock:
            cls.calls += 1
            cls.running += 1
            cls.most_at_once = max(cls.most_at_once, cls.running)
        cls.started.release()
        try:
            if not cls.gate.wait(timeout=WAIT):
                raise RuntimeError("the test never opened the gate")
        finally:
            with cls.lock:
                cls.running -= 1
        return TranscriptionResult(text=" Done.", segments=[{"start": 0.0, "end": 1.5, "text": " Done."}])


@pytest.fixture
def gated(monkeypatch):
    """Register GatedBackend as 'gated'; its gate opens when the test ends, so no request is left hanging."""
    from speech_toolkit.backends import _BACKENDS

    monkeypatch.setattr("speech_toolkit.backends._BACKENDS", {**_BACKENDS, "gated": GatedBackend})
    GatedBackend.gate, GatedBackend.started = threading.Event(), threading.Semaphore(0)
    GatedBackend.calls = GatedBackend.running = GatedBackend.most_at_once = 0
    yield GatedBackend
    GatedBackend.gate.set()


@pytest.fixture
def queued(monkeypatch):
    """A semaphore released as each request joins the queue for the model, its body read."""
    joined = threading.Semaphore(0)
    run = _Service.run

    async def run_and_count(self, receive, job):
        joined.release()
        return await run(self, receive, job)

    monkeypatch.setattr(_Service, "run", run_and_count)
    return joined


@pytest.fixture
def temp_dir(tmp_path, monkeypatch):
    """Send the server's temporary files (and Starlette's spooled uploads) to a folder the test can inspect."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


@pytest.mark.usefixtures("decoded")
class TestQueue:
    def test_a_full_queue_is_a_503_at_once(self, gated, queued, temp_dir):
        app = create_app(Transcriber("gated"), max_queue=1)
        with TestClient(app) as client, ThreadPoolExecutor(2) as pool:
            try:
                first = pool.submit(client.post, URL, files=upload())
                acquire(gated.started)  # the model transcribes it
                second = pool.submit(client.post, URL, files=upload())
                acquire(queued, 2)  # it waits

                streamed = asgi_post(app, (bytes(MIB) for _ in range(50)))  # 50 MB, chunked
                refused = client.post(URL, files=upload())
                health = client.get("/health").json()
            finally:
                gated.gate.set()
            assert (first.result(timeout=WAIT).status_code, second.result(timeout=WAIT).status_code) == (200, 200)

        assert (streamed.status, streamed.read) == (503, 0)  # refused before any of the body was read
        assert streamed.headers["retry-after"] == "1"  # no request has finished yet to time
        assert refused.status_code == 503
        assert refused.headers["retry-after"] == "1"
        assert refused.json()["error"] == {
            "message": "The server is busy with other requests; retry in 1 s.",
            "type": "server_error",
            "param": None,
            "code": None,
        }
        assert (health["transcribing"], health["waiting"], health["max_queue"]) == (1, 1, 1)
        assert gated.calls == 2
        assert list(temp_dir.iterdir()) == []

    def test_health_answers_at_once_while_50_requests_wait(self, gated, queued, temp_dir):
        app = create_app(Transcriber("gated"), max_queue=50)
        with TestClient(app) as client, ThreadPoolExecutor(51) as pool:
            try:
                posts = [pool.submit(client.post, URL, files=upload()) for _ in range(51)]
                acquire(queued, 51)
                acquire(gated.started)
                health, health_seconds = timed(client.get, "/health")
                models, models_seconds = timed(client.get, "/v1/models")
            finally:
                gated.gate.set()
            statuses = [post.result(timeout=WAIT).status_code for post in posts]

        assert health_seconds < 0.1
        assert health.json() == {
            "status": "ok",
            "backend": "gated",
            "model": "small",
            "transcribing": 1,
            "waiting": 50,
            "max_queue": 50,
        }
        assert models_seconds < 0.1 and models.status_code == 200
        assert statuses == [200] * 51
        assert (gated.calls, gated.most_at_once) == (51, 1)  # one at a time
        assert list(temp_dir.iterdir()) == []

    def test_max_queue_0_refuses_every_request_while_the_model_is_busy(self, gated, temp_dir):
        app = create_app(Transcriber("gated"), max_queue=0)
        with TestClient(app) as client, ThreadPoolExecutor(1) as pool:
            try:
                first = pool.submit(client.post, URL, files=upload())
                acquire(gated.started)
                refused = client.post(URL, files=upload())
            finally:
                gated.gate.set()
            assert first.result(timeout=WAIT).status_code == 200
            assert client.post(URL, files=upload()).status_code == 200  # free again

        assert refused.status_code == 503

    def test_requests_refused_from_their_headers_take_no_place(self, gated, temp_dir):
        app = create_app(Transcriber("gated"), api_key=KEY, max_queue=0, max_upload_mb=1)
        with TestClient(app) as client, ThreadPoolExecutor(1) as pool:
            try:
                first = pool.submit(client.post, URL, files=upload(), headers=AUTH)
                acquire(gated.started)
                without_key = client.post(URL, files=upload())  # the queue is full, but the key is checked first
                too_large = asgi_post(
                    app, headers=[*FORM, ("content-length", str(3 * MIB)), ("authorization", f"Bearer {KEY}")]
                )
                health = client.get("/health").json()
            finally:
                gated.gate.set()
            assert first.result(timeout=WAIT).status_code == 200

        assert without_key.status_code == 401
        assert too_large.status == 413
        assert (health["transcribing"], health["waiting"]) == (1, 0)

    def test_every_request_gives_its_place_back(self, fake_backend, decoded, temp_dir):
        """With max_queue=0, a place that wasn't given back would turn every later request away with a 503."""
        app = create_app(Transcriber("fake-words"), max_queue=0, max_upload_mb=0.001)
        client = TestClient(app)

        statuses = [client.post(URL, files=upload()).status_code]
        statuses.append(client.post(URL, files=upload(), data={"response_format": "docx"}).status_code)
        statuses.append(client.post(URL, files=upload(data=b"#EXTM3U\n")).status_code)  # a playlist
        body = multipart(bytes(3 * MIB))
        chunks = (body[i : i + MIB] for i in range(0, len(body), MIB))
        statuses.append(asgi_post(app, chunks).status)  # sent chunked, and cut off at the limit
        decoded.error = AudioDecodeError("Failed to load audio: Invalid data found when processing input")
        statuses.append(client.post(URL, files=upload()).status_code)
        decoded.error = None
        with patch.object(Transcriber, "transcribe", side_effect=ValueError("Unsupported language: xx")):
            statuses.append(client.post(URL, files=upload(), data={"language": "xx"}).status_code)
        statuses.append(client.post(URL, files=upload()).status_code)

        assert statuses == [200, 400, 400, 413, 400, 400, 200]
        assert client.get("/health").json()["waiting"] == 0
        assert list(temp_dir.iterdir()) == []

    def test_retry_after_is_the_average_time_of_recent_requests(self, fake_backend, monkeypatch):
        service = _Service(Transcriber("fake"), None, 1)
        clock = []
        monkeypatch.setattr("speech_toolkit.server.time", SimpleNamespace(monotonic=lambda: clock.pop(0)))

        def retry_after_one_taking(seconds: float) -> str:
            clock.extend([100.0, 100.0 + seconds])
            assert service._work(lambda: "done") == "done"
            return service.busy().headers["Retry-After"]

        assert service.busy().headers["Retry-After"] == "1"
        assert retry_after_one_taking(9.2) == "10"  # rounded up
        assert retry_after_one_taking(109.2) == "40"  # 9.2 + 0.3 * (109.2 - 9.2): weighted to the latest
        assert retry_after_one_taking(1000) == "60"  # at most a minute
        service.request_seconds = 0.2
        assert service.busy().headers["Retry-After"] == "1"  # at least a second


@pytest.mark.usefixtures("decoded")
class TestClientsThatLeave:
    def test_while_waiting_frees_its_place_and_never_reaches_the_model(self, gated, decoded, temp_dir):
        app = create_app(Transcriber("gated"), max_queue=1)
        with TestClient(app) as client, ThreadPoolExecutor(1) as pool:
            try:
                first = pool.submit(client.post, URL, files=upload())
                acquire(gated.started)
                left = asgi_post(app)  # waits behind the first, then its client disconnects
                health = client.get("/health").json()
            finally:
                gated.gate.set()
            assert first.result(timeout=WAIT).status_code == 200
            assert client.post(URL, files=upload()).status_code == 200  # the worker skipped the dropped request

        assert left.status == CLIENT_CLOSED
        assert (health["transcribing"], health["waiting"]) == (1, 0)
        assert gated.calls == 2 and len(decoded.calls) == 2  # the first and the last: the dropped one never ran
        assert list(temp_dir.iterdir()) == []

    def test_during_its_transcription_keeps_its_place_until_the_model_is_done(self, gated, temp_dir):
        app = create_app(Transcriber("gated"), max_queue=0)
        client_left = threading.Event()
        with TestClient(app) as client, ThreadPoolExecutor(1) as pool:
            try:
                left = pool.submit(asgi_post, app, leave=lambda: client_left.wait(WAIT))
                acquire(gated.started)
                client_left.set()
                refused = client.post(URL, files=upload())  # the model is still busy with it
            finally:
                gated.gate.set()
            response = left.result(timeout=WAIT)
            health = client.get("/health").json()

        assert response.status == 200  # it waited for the model, which can't be stopped mid-way
        assert refused.status_code == 503
        assert (health["transcribing"], health["waiting"]) == (0, 0)
        assert list(temp_dir.iterdir()) == []  # the upload was deleted when the model was done

    def test_a_request_cancelled_while_it_waits_is_dropped(self, fake_backend):
        """As when the server stops: the request's task is cancelled, and the job it queued never runs."""
        service = _Service(Transcriber("fake"), None, 1)
        release, ran = threading.Event(), []

        async def scenario():
            blocker = service.worker.submit(release.wait, WAIT)  # the worker is busy
            never = asyncio.Event()

            async def receive():
                await never.wait()

            task = asyncio.ensure_future(service.run(receive, lambda: ran.append("ran")))
            await asyncio.sleep(0)  # it queues the job and waits
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()
            await asyncio.wrap_future(blocker)
            await asyncio.wrap_future(service.worker.submit(lambda: None))  # after the dropped job's turn

        asyncio.run(scenario())
        assert ran == []

    def test_run_waits_for_a_job_that_started_before_its_client_left(self, fake_backend):
        """The service's run() itself: the model can't be stopped mid-way, so the job runs to the end."""
        service = _Service(Transcriber("fake"), None, 1)
        started, finish = threading.Event(), threading.Event()

        def job():
            started.set()
            assert finish.wait(WAIT)
            return "done"

        async def scenario():
            disconnected = asyncio.Event()

            async def receive():
                await asyncio.to_thread(started.wait, WAIT)
                disconnected.set()
                return {"type": "http.disconnect"}

            task = asyncio.ensure_future(service.run(receive, job))
            await disconnected.wait()
            for _ in range(10):  # let run() see the disconnect, find the job running, and wait for it
                await asyncio.sleep(0)
            finish.set()
            return await task

        assert asyncio.run(scenario()) == "done"
        assert service.status() == {"transcribing": 0, "waiting": 0, "max_queue": DEFAULT_MAX_QUEUE}


class TestDecoding:
    def test_the_model_gets_the_samples_decoded_once_from_the_upload(self, fake_waveform_backend, decoded, temp_dir):
        response = TestClient(create_app(Transcriber("fake-waveform"))).post(URL, files=upload("talk.m4a"))

        assert response.status_code == 200
        (decoding,) = decoded.calls
        assert (decoding.path.parent, decoding.path.suffix, decoding.existed) == (temp_dir, ".m4a", True)
        assert decoding.options == {
            "strict": True,  # common, self-contained formats only
            "max_seconds": DEFAULT_MAX_AUDIO_SECONDS + AUDIO_LIMIT_MARGIN,
            "timeout": DECODE_TIMEOUT,
        }
        (samples,) = fake_waveform_backend.received
        assert isinstance(samples, np.ndarray) and len(samples) == 1.5 * SAMPLE_RATE
        assert response.json()["text"].startswith("24000 samples")
        assert list(temp_dir.iterdir()) == []

    def test_every_built_in_backend_takes_samples(self):
        """So none of them decodes an upload itself."""
        assert all("waveform" in get_backend_class(name).capabilities for name in list_backends())

    def test_a_backend_without_samples_gets_a_wav_file_of_them(self, fake_backend, decoded, temp_dir, monkeypatch):
        received = []

        def transcribe(self, audio_path, language=None, task="transcribe", verbose=True):
            with wave.open(str(audio_path)) as wav:
                received.append((audio_path.suffix, wav.getframerate(), wav.getnframes()))
            return TranscriptionResult(text=" Hi.", segments=[])

        monkeypatch.setattr(FakeBackend, "transcribe", transcribe)

        response = TestClient(create_app(Transcriber("fake"))).post(URL, files=upload("talk.mp3"))

        assert response.status_code == 200
        assert received == [(".wav", SAMPLE_RATE, 1.5 * SAMPLE_RATE)]  # not the upload: the decoded samples
        assert list(temp_dir.iterdir()) == []

    def test_audio_over_the_limit_is_a_400(self, fake_waveform_backend, decoded, temp_dir):
        client = TestClient(create_app(Transcriber("fake-waveform"), max_audio_seconds=2))
        decoded.seconds = 2 + AUDIO_LIMIT_MARGIN  # where ffmpeg stopped

        response = client.post(URL, files=upload())

        assert response.status_code == 400
        assert response.json()["error"] == {
            "message": "The audio is longer than this server's 2 s limit.",
            "type": "invalid_request_error",
            "param": "file",
            "code": None,
        }
        assert decoded.calls[0].options["max_seconds"] == 3
        assert fake_waveform_backend.received == []
        assert list(temp_dir.iterdir()) == []

    def test_audio_up_to_the_limit_is_transcribed(self, fake_waveform_backend, decoded, temp_dir):
        client = TestClient(create_app(Transcriber("fake-waveform"), max_audio_seconds=2))
        decoded.seconds = 2

        assert client.post(URL, files=upload()).status_code == 200

    def test_no_limit(self, fake_waveform_backend, decoded, temp_dir):
        client = TestClient(create_app(Transcriber("fake-waveform"), max_audio_seconds=None))
        decoded.seconds = 60

        assert client.post(URL, files=upload()).status_code == 200
        assert decoded.calls[0].options["max_seconds"] is None

    def test_ffmpeg_decodes_only_a_second_past_the_limit(self, fake_waveform_backend, temp_dir):
        """The real load_audio, with ffmpeg mocked: it is told to stop (-t), so a long file is never decoded whole."""
        commands = []

        def ffmpeg(cmd, **kwargs):
            commands.append(cmd)
            seconds = float(cmd[cmd.index("-t") + 1])
            return subprocess.CompletedProcess(cmd, 0, stdout=bytes(2 * round(seconds * SAMPLE_RATE)), stderr=b"")

        client = TestClient(create_app(Transcriber("fake-waveform"), max_audio_seconds=60))
        with patch("speech_toolkit.media.subprocess.run", side_effect=ffmpeg):
            response = client.post(URL, files=upload("x.m3u8"))

        assert response.status_code == 400
        (cmd,) = commands
        assert cmd[cmd.index("-t") + 1] == "61"
        assert "-format_whitelist" in cmd  # strict
        source = cmd[cmd.index("-i") + 1]
        assert source.startswith(f"file:{temp_dir}") and source.endswith(".bin")  # the saved upload, by its path
        assert fake_waveform_backend.received == []

    def test_create_app_checks_its_limits(self, fake_backend):
        transcriber = Transcriber("fake")
        for options in [{"max_queue": -1}, {"max_audio_seconds": 0}, {"max_audio_seconds": float("nan")}]:
            with pytest.raises(ValueError, match="must be"):
                create_app(transcriber, **options)


def wav_bytes(seconds: float, rate: int = 44100) -> bytes:
    """A WAV file of a 440 Hz tone."""
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        t = np.arange(round(seconds * rate)) / rate
        wav.writeframes((np.sin(2 * np.pi * 440 * t) * 8000).astype("<i2").tobytes())
    return out.getvalue()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg on PATH")
class TestWithFfmpeg:
    @pytest.mark.parametrize("seconds, status", [(2.0, 200), (2.01, 400), (30.0, 400)])
    def test_length_limit(self, fake_waveform_backend, temp_dir, seconds, status):
        client = TestClient(create_app(Transcriber("fake-waveform"), max_audio_seconds=2))

        response = client.post(URL, files=upload(data=wav_bytes(seconds)))

        assert response.status_code == status
        assert [len(samples) for samples in fake_waveform_backend.received] == ([32000] if status == 200 else [])
        assert list(temp_dir.iterdir()) == []

    def test_a_playlist_is_refused_by_the_decoder_too(self, fake_waveform_backend, temp_dir, monkeypatch):
        """Strict decoding refuses an HLS playlist that the upload sniffing let through."""
        monkeypatch.setattr("speech_toolkit.server._is_playlist", lambda head: False)
        playlist = b"#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10,\nother.wav\n#EXT-X-ENDLIST\n"

        response = TestClient(create_app(Transcriber("fake-waveform"))).post(URL, files=upload("x.m3u8", playlist))

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_value"
        assert fake_waveform_backend.received == []


@pytest.mark.usefixtures("decoded")
class TestCors:
    PREFLIGHT = {
        "Origin": ORIGIN,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization, content-type, x-stainless-os",
    }

    def test_a_preflight_needs_no_key(self, fake_backend):
        client = TestClient(create_app(Transcriber("fake-words"), api_key=KEY, cors_origins=[ORIGIN]))

        response = client.options(URL, headers=self.PREFLIGHT)

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == ORIGIN
        assert "POST" in response.headers["access-control-allow-methods"]
        assert "x-stainless-os" in response.headers["access-control-allow-headers"]  # OpenAI's JavaScript SDK

    def test_answers_and_errors_carry_the_headers(self, fake_backend, temp_dir):
        client = TestClient(create_app(Transcriber("fake-words"), api_key=KEY, cors_origins=[ORIGIN, "http://b"]))

        answered = client.post(URL, files=upload(), headers={"Origin": ORIGIN, **AUTH})
        refused = client.post(URL, files=upload(), headers={"Origin": "http://b"})  # by the guard

        assert answered.status_code == 200
        assert answered.headers["access-control-allow-origin"] == ORIGIN
        assert answered.headers["access-control-expose-headers"] == "Retry-After"  # for the 503s
        assert refused.status_code == 401
        assert refused.headers["access-control-allow-origin"] == "http://b"

    def test_other_origins_get_no_access(self, fake_backend):
        client = TestClient(create_app(Transcriber("fake-words"), cors_origins=[ORIGIN]))

        assert client.options(URL, headers={**self.PREFLIGHT, "Origin": "https://elsewhere.example"}).status_code == 400
        assert (
            "access-control-allow-origin"
            not in client.get("/health", headers={"Origin": "https://elsewhere.example"}).headers
        )

    def test_any_origin(self, fake_backend):
        client = TestClient(create_app(Transcriber("fake-words"), cors_origins=["*"]))

        assert (
            client.get("/health", headers={"Origin": "https://elsewhere.example"}).headers[
                "access-control-allow-origin"
            ]
            == "*"
        )

    def test_off_by_default(self, fake_backend):
        client = TestClient(create_app(Transcriber("fake-words")))

        assert client.options(URL, headers=self.PREFLIGHT).status_code == 405
        assert "access-control-allow-origin" not in client.get("/health", headers={"Origin": ORIGIN}).headers


class TestDocsSwitch:
    @pytest.mark.parametrize("page", ["/docs", "/redoc", "/openapi.json"])
    def test_off(self, fake_backend, page):
        assert TestClient(create_app(Transcriber("fake-words"), docs=False)).get(page).status_code == 404

    @pytest.mark.parametrize("page", ["/docs", "/redoc", "/openapi.json"])
    def test_on_with_a_key_needs_the_key(self, fake_backend, page):
        client = TestClient(create_app(Transcriber("fake-words"), api_key=KEY, docs=True))

        assert client.get(page).status_code == 401
        assert client.get(page, headers=AUTH).status_code == 200


class TestCommandLine:
    @pytest.fixture
    def uvicorn(self):
        module = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": module}):
            yield module

    def test_defaults(self):
        args = parse_args([])

        assert (args.max_queue, args.max_audio_seconds, args.cors_origin, args.docs) == (8, 10800, [], None)
        assert (args.max_upload_mb, args.port, args.host) == (100, 8000, "127.0.0.1")  # unchanged

    def test_options(self):
        argv = ["--max-queue", "0", "--max-audio-seconds", "90.5", "--cors-origin", ORIGIN, "--cors-origin", "*"]
        args = parse_args([*argv, "--no-docs"])

        assert (args.max_queue, args.max_audio_seconds, args.cors_origin, args.docs) == (0, 90.5, [ORIGIN, "*"], False)
        assert parse_args(["--docs"]).docs is True

    @pytest.mark.parametrize(
        "argv",
        [
            ["--max-queue", "-1"],
            ["--max-queue", "two"],
            ["--max-audio-seconds", "-1"],
            ["--max-audio-seconds", "nan"],
            ["--max-audio-seconds", "inf"],
        ],
    )
    def test_bad_values(self, argv, capsys):
        with pytest.raises(SystemExit):
            parse_args(argv)
        assert argv[0] in capsys.readouterr().err

    def test_main_passes_them_on(self, fake_backend, decoded, uvicorn, temp_dir):
        main(
            ["-b", "fake-words", "--api-key", KEY, "--max-queue", "3", "--max-audio-seconds", "0"]
            + ["--cors-origin", ORIGIN, "--docs"]
        )

        (app,), kwargs = uvicorn.run.call_args
        client = TestClient(app)
        assert kwargs == {"host": "127.0.0.1", "port": 8000}
        assert client.get("/health").json()["max_queue"] == 3
        assert client.get("/openapi.json", headers=AUTH).status_code == 200
        assert client.get("/health", headers={"Origin": ORIGIN}).headers["access-control-allow-origin"] == ORIGIN
        assert client.post(URL, files=upload(), headers=AUTH).status_code == 200
        assert decoded.calls[0].options["max_seconds"] is None  # 0: no limit
