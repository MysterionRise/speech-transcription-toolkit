"""Security of transcribe-server (#21): requests refused before their body is read, safe temporary files, playlists
refused, and error responses without temporary paths, decoder output or exception details."""

from __future__ import annotations

import asyncio
import codecs
import io
import json
import subprocess
import sys
import tempfile
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from speech_toolkit import AudioDecodeError, ModelLoadError, Transcriber, UnsupportedOptionError
from speech_toolkit.server import (
    FORM_OVERHEAD,
    NOT_AUDIO,
    SERVER_ERROR,
    APIError,
    _is_decode_error,
    _is_playlist,
    _RequestGuard,
    _Service,
    _upload_suffix,
    create_app,
)
from tests.conftest import FakeBackend, FakeWordsBackend

URL = "/v1/audio/transcriptions"
KEY = "s3cret"
AUTH = {"Authorization": f"Bearer {KEY}"}
MIB = 1024 * 1024
BOUNDARY = "toolkit-boundary"
# A streamed multipart upload: no Content-Length
CHUNKED_FORM = [("content-type", f"multipart/form-data; boundary={BOUNDARY}"), ("transfer-encoding", "chunked")]
NOT_AUDIO_ERROR = {"message": NOT_AUDIO, "type": "invalid_request_error", "param": "file", "code": "invalid_value"}
FFMPEG_BANNER = "ffmpeg version 6.1.1 Copyright (c) 2000-2023 the FFmpeg developers\n  built with gcc 13\n"


def upload(name: str = "talk.wav", data: bytes = b"RIFF fake audio") -> dict:
    return {"file": (name, io.BytesIO(data), "audio/wav")}


def multipart(data: bytes, name: str = "talk.wav") -> bytes:
    """A multipart/form-data body with one file field."""
    head = f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n\r\n'
    return head.encode() + data + f"\r\n--{BOUNDARY}--\r\n".encode()


def asgi_request(app, chunks=(), *, headers=(), method="POST", path=URL, root_path="") -> SimpleNamespace:
    """Send *app* one HTTP request whose body arrives in *chunks*, counting the chunks the app reads.

    Driving the app directly shows exactly how much of the body was read, which TestClient (it reads the whole body
    up front) can't.
    """
    chunks = iter(chunks)
    result = SimpleNamespace(read=0, status=None, json=None, done=False)
    body = []

    async def receive():
        chunk = None if result.done else next(chunks, None)
        if chunk is None:
            if result.done:
                return {"type": "http.disconnect"}
            result.done = True
            return {"type": "http.request", "body": b"", "more_body": False}
        result.read += 1
        return {"type": "http.request", "body": chunk, "more_body": True}

    async def send(message):
        if message["type"] == "http.response.start":
            result.status = message["status"]
        else:
            body.append(message.get("body", b""))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": root_path + path,
        "raw_path": (root_path + path).encode(),
        "root_path": root_path,
        "query_string": b"",
        "headers": [(name.encode(), value.encode()) for name, value in headers],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8000),
    }
    asyncio.run(app(scope, receive, send))
    result.json = json.loads(b"".join(body)) if body else None
    return result


@pytest.fixture
def temp_dir(tmp_path, monkeypatch):
    """Send the server's temporary files (and Starlette's spooled uploads) to a folder the test can inspect."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


class TestRefusedBeforeTheBodyIsRead:
    def test_unauthenticated_streamed_upload(self, fake_backend, temp_dir):
        app = create_app(Transcriber("fake-words"), api_key=KEY)
        fifty_mb = (bytes(MIB) for _ in range(50))

        response = asgi_request(app, fifty_mb, headers=CHUNKED_FORM)

        assert response.status == 401
        assert response.json["error"]["code"] == "invalid_api_key"
        assert response.read == 0  # #21 allows one chunk
        assert list(temp_dir.iterdir()) == []
        assert FakeWordsBackend.calls == []

    def test_wrong_key(self, fake_backend, temp_dir):
        app = create_app(Transcriber("fake-words"), api_key=KEY)
        headers = [*CHUNKED_FORM, ("authorization", "Bearer guess")]

        response = asgi_request(app, (bytes(MIB) for _ in range(5)), headers=headers)

        assert (response.status, response.read) == (401, 0)

    def test_declared_size_over_the_limit(self, fake_backend, temp_dir):
        app = create_app(Transcriber("fake-words"), api_key=KEY, max_upload_mb=1)
        headers = [
            ("content-type", f"multipart/form-data; boundary={BOUNDARY}"),
            ("content-length", str(MIB + FORM_OVERHEAD + 1)),
            ("authorization", f"Bearer {KEY}"),
        ]

        response = asgi_request(app, (bytes(MIB) for _ in range(3)), headers=headers)

        assert response.status == 413
        assert response.json["error"]["param"] == "file"
        assert response.read == 0
        assert list(temp_dir.iterdir()) == []

    def test_unauthenticated_beats_too_large(self, fake_backend):
        app = create_app(Transcriber("fake-words"), api_key=KEY, max_upload_mb=1)

        response = asgi_request(app, headers=[("content-length", str(50 * MIB))])

        assert response.status == 401  # the limit isn't revealed without the key

    def test_chunked_upload_is_cut_off_at_the_limit(self, fake_backend, temp_dir):
        app = create_app(Transcriber("fake-words"), max_upload_mb=0.0001)  # about 100 bytes, plus FORM_OVERHEAD
        limit = int(0.0001 * MIB) + FORM_OVERHEAD
        body = multipart(bytes(3 * MIB))
        chunk = 64 * 1024
        chunks = (body[i : i + chunk] for i in range(0, len(body), chunk))

        response = asgi_request(app, chunks, headers=CHUNKED_FORM)

        assert response.status == 413
        assert response.json["error"] == {
            "message": "The file is larger than this server's 0.0001 MB limit.",
            "type": "invalid_request_error",
            "param": "file",
            "code": None,
        }
        assert response.read == limit // chunk + 1  # just past the limit
        assert FakeWordsBackend.calls == []
        assert list(temp_dir.iterdir()) == []  # nor a partial upload

    def test_cut_off_is_answered_when_the_app_lets_the_error_through(self, fake_backend):
        async def app(scope, receive, send):  # reads the body without handling errors, unlike FastAPI
            while (await receive()).get("more_body"):
                pass

        guard = _RequestGuard(app, _Service(Transcriber("fake-words"), None, 0.0001))

        response = asgi_request(guard, (bytes(64 * 1024) for _ in range(40)), headers=CHUNKED_FORM)

        assert response.status == 413

    def test_the_limit_is_for_the_file(self, fake_backend, temp_dir):
        """The other form fields and the multipart framing get FORM_OVERHEAD on top of the limit."""
        client = TestClient(create_app(Transcriber("fake-words"), max_upload_mb=0.01))
        exactly = int(0.01 * MIB)

        assert client.post(URL, files=upload(data=b"R" * exactly), data={"prompt": "p" * 1000}).status_code == 200
        assert client.post(URL, files=upload(data=b"R" * (exactly + 1))).status_code == 413

    def test_health_needs_no_key_but_other_paths_do(self, fake_backend):
        client = TestClient(create_app(Transcriber("fake-words"), api_key=KEY))

        assert client.get("/health").status_code == 200
        assert client.get("/nope").status_code == 401
        assert client.get("/nope", headers=AUTH).status_code == 404

    def test_health_under_a_root_path(self, fake_backend):
        app = create_app(Transcriber("fake-words"), api_key=KEY)

        assert asgi_request(app, method="GET", path="/health", root_path="/stt").status == 200
        assert asgi_request(app, method="GET", path="/v1/models", root_path="/stt").status == 401

    def test_lifespan_passes_through(self, fake_backend):
        with TestClient(create_app(Transcriber("fake-words"), api_key=KEY)) as client:  # runs startup and shutdown
            assert client.get("/v1/models", headers=AUTH).status_code == 200


class TestDocsPages:
    @pytest.mark.parametrize("page", ["/docs", "/redoc", "/openapi.json"])
    def test_off_with_an_api_key(self, fake_backend, page):
        client = TestClient(create_app(Transcriber("fake-words"), api_key=KEY))

        assert client.get(page).status_code == 401
        assert client.get(page, headers=AUTH).status_code == 404

    @pytest.mark.parametrize("page", ["/docs", "/redoc", "/openapi.json"])
    def test_on_without_a_key(self, fake_backend, page):
        assert TestClient(create_app(Transcriber("fake-words"))).get(page).status_code == 200


class TestTempFiles:
    @pytest.mark.parametrize(
        "filename, suffix",
        [
            ("talk.wav", ".wav"),
            ("Talk.MP3", ".mp3"),
            ("clip.webm", ".webm"),
            ("x.m3u8", ".bin"),
            ("talk.mp3.m3u8", ".bin"),
            ("script.vpy", ".bin"),
            ("../../etc/passwd", ".bin"),
            ("noext", ".bin"),
            (".wav", ".bin"),
            ("", ".bin"),
            (None, ".bin"),
        ],
    )
    def test_suffix_comes_from_an_allowlist(self, filename, suffix):
        assert _upload_suffix(filename) == suffix

    def test_m3u8_upload_is_stored_as_bin(self, fake_backend, temp_dir, monkeypatch):
        stored = []
        transcribe = FakeWordsBackend.transcribe

        def record(self, audio_path, *args, **kwargs):
            stored.append((audio_path, audio_path.is_file()))
            return transcribe(self, audio_path, *args, **kwargs)

        monkeypatch.setattr(FakeWordsBackend, "transcribe", record)

        response = TestClient(create_app(Transcriber("fake-words"))).post(URL, files=upload("x.m3u8"))

        assert response.status_code == 200
        ((path, existed),) = stored
        assert (path.parent, path.suffix, existed) == (temp_dir, ".bin", True)
        assert "m3u8" not in path.name  # a random name, not the client's
        assert list(temp_dir.iterdir()) == []

    def test_no_temp_file_after_an_oserror_during_the_copy(self, fake_backend, temp_dir, caplog):
        class DiskFull:
            """The upload's file: one chunk, then the copy fails partway, as when the disk fills up."""

            chunks = [b"RIFF" + bytes(100)]

            def read(self, size=-1):
                if self.chunks:
                    return self.chunks.pop()
                raise OSError(28, f"No space left on device: {tempfile.gettempdir()}")

        service = _Service(Transcriber("fake-words"), None, 1)

        with pytest.raises(APIError) as raised:
            service.transcribe(SimpleNamespace(filename="talk.wav", file=DiskFull()), "transcribe", None, None, False)

        assert (raised.value.status, raised.value.body["error"]["message"]) == (500, SERVER_ERROR)
        assert list(temp_dir.iterdir()) == []
        assert "No space left on device" in caplog.text
        assert FakeWordsBackend.calls == []

    def test_oserror_during_the_copy_is_a_generic_500(self, fake_backend, temp_dir):
        def fill_disk(upload, out):
            out.write(b"RIFF partial")
            raise OSError(28, f"No space left on device: {out.name}")

        with patch.object(_Service, "save_upload", side_effect=fill_disk):
            response = TestClient(create_app(Transcriber("fake-words"))).post(URL, files=upload())

        assert response.status_code == 500
        assert response.json()["error"] == {
            "message": SERVER_ERROR,
            "type": "server_error",
            "param": None,
            "code": None,
        }
        assert list(temp_dir.iterdir()) == []


DASH = (
    b'<?xml version="1.0"?>\n<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" '
    b'profiles="urn:mpeg:dash:profile:isoff-on-demand:2011"><BaseURL>http://169.254.169.254/</BaseURL></MPD>\n'
)
PLAYLISTS = {
    "dash": DASH,
    "dash-with-bom-and-blank-lines": codecs.BOM_UTF8 + b"\r\n\r\n  " + DASH,
    "dash-past-whitespace-padding": b" " * 10_000 + DASH,
    "dash-utf-16-le": codecs.BOM_UTF16_LE + DASH.decode().encode("utf-16-le"),
    "dash-utf-16-be": codecs.BOM_UTF16_BE + DASH.decode().encode("utf-16-be"),
    "dash-utf-16-be-without-bom": DASH.decode().encode("utf-16-be"),
    "imf": b'<?xml version="1.0"?>\n<CompositionPlaylist><ContentTitle>x</ContentTitle></CompositionPlaylist>\n',
    "hls": b"#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10,\n/tmp/other-upload.wav\n#EXT-X-ENDLIST\n",
    "ffconcat": b"ffconcat version 1.0\nfile other-upload.wav\n",
    "empty": b"",
    "blank": b" \r\n\t",
}
MEDIA_HEADS = {
    "wav": b"RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00",
    "mp3-id3": b"ID3\x04\x00\x00\x00\x00\x00\x23TSSE\x00\x00\x00\x0f",
    "mp3-frame": b"\xff\xfb\x90\x64" + bytes(60),
    "flac": b"fLaC\x00\x00\x00\x22\x10\x00",
    "ogg": b"OggS\x00\x02" + bytes(20),
    "mp4-60-byte-ftyp": b"\x00\x00\x00\x3cftypisom\x00\x00\x02\x00isomiso2",
    "m2ts-timestamp-byte-<": b"<\x12\x34\x56G@\x00\x10\x00\x00\xb0\x0d\x00\x01",
    "mpeg-layer-1-ff-fe": b"\xff\xfe\x20\x00" + bytes(range(256)),
    "amr": b"#!AMR\n<\x10\x00",
    "xm-module": b"Extended Module: song",
}


class TestPlaylists:
    """DASH and IMF manifests, HLS playlists and ffconcat scripts make ffmpeg and PyAV open what they list."""

    @pytest.mark.parametrize("content", PLAYLISTS.values(), ids=PLAYLISTS.keys())
    def test_refused(self, fake_backend, temp_dir, content):
        response = TestClient(create_app(Transcriber("fake-words"))).post(URL, files=upload("talk.wav", content))

        assert response.status_code == 400
        assert response.json()["error"] == NOT_AUDIO_ERROR
        assert FakeWordsBackend.calls == []
        assert list(temp_dir.iterdir()) == []

    @pytest.mark.parametrize("head", MEDIA_HEADS.values(), ids=MEDIA_HEADS.keys())
    def test_media_is_not_a_playlist(self, head):
        assert not _is_playlist(head)


# Errors as each decoding path raises them, for a file at *path*: what the backends let through.
def audio_decode_error(path):  # speech_toolkit.media.load_audio: voxtral, parakeet, canary and diarization
    raise AudioDecodeError(f"Failed to load audio: {FFMPEG_BANNER}{path}: Invalid data found when processing input")


def ffmpeg_run_error(path):  # openai-whisper's load_audio (and media.load_audio before AudioDecodeError)
    try:
        raise subprocess.CalledProcessError(1, ["ffmpeg", "-i", str(path)], stderr=FFMPEG_BANNER.encode())
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}{path}: Invalid data found") from e


class FFmpegError(Exception):
    """Stand-in for PyAV's av.error.FFmpegError, the base of the errors it raises."""


class InvalidDataError(FFmpegError, ValueError):
    """PyAV's error for data it can't demux, also a ValueError."""


class FFmpegOSError(FFmpegError, OSError):
    """PyAV's errors with an errno, also OSErrors."""


AV_ERROR = types.ModuleType("av.error")
AV_ERROR.FFmpegError = FFmpegError


def pyav_invalid_data(path):  # faster-whisper, decoding with PyAV
    raise InvalidDataError(f"[Errno 1094995529] Invalid data found when processing input: '{path}'")


def pyav_os_error(path):  # faster-whisper: an ffconcat file listing itself, say
    raise FFmpegOSError(24, "Too many open files", str(path))


def _first_audio_stream(path):
    return ()[0]  # IndexError, as PyAV raises for a file without an audio stream


# Stand-in for faster_whisper.audio.decode_audio: it runs in that module, as the real one does.
faster_whisper_decode_audio = types.FunctionType(_first_audio_stream.__code__, {"__name__": "faster_whisper.audio"})


def video_without_sound(path):  # faster-whisper: a video file with no audio stream
    faster_whisper_decode_audio(path)


DECODE_ERRORS = {
    "AudioDecodeError": audio_decode_error,
    "ffmpeg-exit-status": ffmpeg_run_error,
    "pyav-InvalidDataError": pyav_invalid_data,
    "pyav-OSError": pyav_os_error,
    "faster-whisper-no-audio-stream": video_without_sound,
}


def assert_not_audio(response, path=None):
    """A 400 for undecodable audio, without the temporary file's path or the decoder's output."""
    assert response.status_code == 400
    assert response.json()["error"] == NOT_AUDIO_ERROR
    for leak in ["ffmpeg version", "Copyright", "Errno", "Error", "tmp", *([str(path), path.name] if path else [])]:
        assert leak not in response.text


@pytest.fixture
def failing_backend(fake_backend, temp_dir, monkeypatch):
    """Make the fake backend's transcribe() call ``fail(path)``; returns the paths it was given."""
    paths = []

    def use(fail):
        def transcribe(self, audio_path, *args, **kwargs):
            paths.append(audio_path)
            fail(audio_path)

        monkeypatch.setattr(FakeBackend, "transcribe", transcribe)
        return paths

    return use


class TestSanitizedErrors:
    @pytest.mark.parametrize("fail", DECODE_ERRORS.values(), ids=DECODE_ERRORS.keys())
    def test_undecodable_audio_is_a_400_on_every_path(self, failing_backend, temp_dir, caplog, fail):
        paths = failing_backend(fail)

        with patch.dict(sys.modules, {"av.error": AV_ERROR}):
            response = TestClient(create_app(Transcriber("fake"))).post(URL, files=upload())

        (path,) = paths
        assert_not_audio(response, path)
        (logged,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Couldn't decode an upload")]
        assert "\n" not in logged  # the details, on one line: the file's metadata can't forge log lines
        assert list(temp_dir.iterdir()) == []

    def test_pyav_errors_need_pyav(self, failing_backend):
        """Without PyAV loaded, a ValueError naming the file is just an error: a generic 500."""
        failing_backend(pyav_invalid_data)

        with patch.dict(sys.modules, {"av.error": None}):
            response = TestClient(create_app(Transcriber("fake"))).post(URL, files=upload())

        assert (response.status_code, response.json()["error"]["message"]) == (500, SERVER_ERROR)

    def test_diarization_decoding_error(self, fake_backend, temp_dir, monkeypatch):
        monkeypatch.setattr("speech_toolkit.api.load_diarization_pipeline", lambda device: "pipeline")
        monkeypatch.setattr(
            "speech_toolkit.api.diarize_audio", lambda path, pipeline, **hints: audio_decode_error(path)
        )

        response = TestClient(create_app(Transcriber("fake-words", diarize=True))).post(URL, files=upload())

        assert_not_audio(response)

    @pytest.mark.parametrize(
        "error, message",
        [
            (ValueError("Unsupported language: xx"), "Unsupported language: xx"),
            (
                UnsupportedOptionError("Parakeet only transcribes.\nTranslate with canary."),
                "Parakeet only transcribes.",
            ),
        ],
    )
    def test_bad_requests_keep_the_first_line_of_their_message(self, failing_backend, error, message):
        def fail(path):
            raise error

        failing_backend(fail)

        response = TestClient(create_app(Transcriber("fake"))).post(URL, files=upload())

        assert response.status_code == 400
        assert response.json()["error"]["message"] == message

    @pytest.mark.parametrize(
        "error",
        [
            lambda path: ValueError(f"can't read {path}"),
            lambda path: ValueError(f"{path.parent} is read-only"),
            lambda path: ValueError(""),
            lambda path: FileNotFoundError(f"Audio file not found: {path}"),
            lambda path: RuntimeError(f"CUDA out of memory while reading {path}"),
            lambda path: ModelLoadError("No model loaded. Call load_model() first."),
            lambda path: MemoryError(),
        ],
        ids=["value-error-naming-the-file", "value-error-naming-the-folder", "empty-value-error", "file-not-found"]
        + ["runtime-error", "model-load-error", "memory-error"],
    )
    def test_other_errors_are_generic_500s_with_details_in_the_log(self, failing_backend, temp_dir, caplog, error):
        def fail(path):
            raise error(path)

        paths = failing_backend(fail)

        response = TestClient(create_app(Transcriber("fake"))).post(URL, files=upload())

        assert response.status_code == 500
        assert response.json()["error"] == {
            "message": SERVER_ERROR,
            "type": "server_error",
            "param": None,
            "code": None,
        }
        assert str(paths[0].parent) not in response.text
        assert "Transcription failed" in caplog.text and "Traceback" in caplog.text
        assert list(temp_dir.iterdir()) == []

    def test_unexpected_errors_outside_transcription_are_generic_500s(self, fake_backend, temp_dir):
        client = TestClient(create_app(Transcriber("fake-words")), raise_server_exceptions=False)

        with patch("speech_toolkit.server._response_body", side_effect=RuntimeError("bug near /srv/secret")):
            response = client.post(URL, files=upload())

        assert response.status_code == 500
        assert response.json()["error"]["message"] == SERVER_ERROR
        assert "secret" not in response.text

    def test_error_chains_with_a_cycle_end(self):
        first, second = ValueError("first"), ValueError("second")
        first.__cause__, second.__cause__ = second, first

        assert not _is_decode_error(first)


class TestRealBackends:
    """The backends themselves, with their libraries mocked, let decoding errors through as they come."""

    def test_whisper(self, temp_dir):
        whisper = MagicMock()
        whisper.load_model.return_value.parameters.return_value = iter([MagicMock(device="cpu")])
        whisper.load_model.return_value.transcribe.side_effect = lambda path, **options: ffmpeg_run_error(path)

        with patch.dict(sys.modules, {"whisper": whisper}):
            response = TestClient(create_app(Transcriber("whisper", "tiny"))).post(URL, files=upload())

        assert_not_audio(response)

    @pytest.mark.parametrize("fail", [pyav_invalid_data, video_without_sound])
    def test_faster_whisper(self, temp_dir, fail):
        faster_whisper, ctranslate2 = MagicMock(), MagicMock()
        ctranslate2.get_cuda_device_count.return_value = 0
        faster_whisper.WhisperModel.return_value.transcribe.side_effect = lambda path, **options: fail(path)
        modules = {"faster_whisper": faster_whisper, "ctranslate2": ctranslate2, "av.error": AV_ERROR}

        with patch.dict(sys.modules, modules):
            response = TestClient(create_app(Transcriber("faster-whisper", "tiny"))).post(URL, files=upload())

        assert_not_audio(response)

    @pytest.mark.parametrize(
        "backend, module",
        [("voxtral", "voxtral_backend"), ("parakeet", "nvidia_backend"), ("canary", "nvidia_backend")],
    )
    def test_transformers_backends(self, temp_dir, backend, module):
        torch = MagicMock()
        torch.cuda.is_available.return_value = False

        with (
            patch.dict(sys.modules, {"torch": torch, "transformers": MagicMock()}),
            patch(f"speech_toolkit.backends.{module}.load_audio", side_effect=audio_decode_error),
        ):
            client = TestClient(create_app(Transcriber(backend)))
            response = client.post(URL, files=upload(), data={"language": "en"})

        assert_not_audio(response)
