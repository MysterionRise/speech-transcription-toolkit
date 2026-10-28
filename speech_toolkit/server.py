"""The ``transcribe-server`` command: OpenAI's audio API (``/v1/audio/transcriptions``), served locally.

Apps and SDKs written for OpenAI's speech-to-text endpoints work offline against any backend:

    transcribe-server -b faster-whisper -m small

    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
    print(client.audio.transcriptions.create(model="whisper-1", file=open("talk.mp3", "rb")).text)

The server runs one model, so the request's ``model`` field is accepted and ignored. One worker thread runs it,
taking the requests in turn from a bounded queue: when the queue is full, more requests get a 503 at once, and
``/health`` answers straight away however busy the model is. Needs the ``server`` extra.
"""

# No "from __future__ import annotations": FastAPI reads the endpoints' annotations at runtime, and the
# endpoints are defined inside create_app() where its imports live.

import argparse
import asyncio
import codecs
import contextlib
import functools
import ipaddress
import json
import logging
import math
import os
import pathlib
import secrets
import subprocess  # nosec B404 - only for CalledProcessError, the error of a failed ffmpeg run
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    BinaryIO,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    TypeVar,
)

from . import AudioDecodeError, __version__
from .api import Transcriber
from .backends import TranscriptionResult
from .cli import FILE_ERRORS, add_model_options, load_transcriber, positive_int
from .media import SAMPLE_RATE, load_audio

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

INSTALL_HINT = 'transcribe-server needs the server extra: pip install "speech-transcription-toolkit[server]"'
RESPONSE_FORMATS = ("json", "text", "srt", "vtt", "verbose_json")
CHUNK_BYTES = 1024 * 1024
# A request's body may be this much larger than --max-upload-mb, for the other form fields and the multipart framing.
FORM_OVERHEAD = 1024 * 1024
# Extensions an upload's temporary file may keep: OpenAI's formats and the CLI's. Anything else is stored as .bin.
UPLOAD_SUFFIXES = frozenset(".aac .flac .m4a .mkv .mov .mp3 .mp4 .mpeg .mpga .oga .ogg .opus .wav .webm .wma".split())
# What OpenAI answers for a file it can't decode.
NOT_AUDIO = "Audio file might be corrupted or unsupported"
SERVER_ERROR = "The server had an error while processing the request; its log has the details."
# The endpoints whose requests take a place in the queue for the model.
TRANSCRIPTION_PATHS = frozenset({"/v1/audio/transcriptions", "/v1/audio/translations"})
DEFAULT_MAX_QUEUE = 8
# Three hours: decoding them takes some 1 GB of memory (see media.load_audio).
DEFAULT_MAX_AUDIO_SECONDS = 3 * 60 * 60
# Uploads are decoded this many seconds past the limit, so that audio longer than the limit can be told from audio
# that ends at it. ffmpeg stops there: the rest of the file is never decoded.
AUDIO_LIMIT_MARGIN = 1.0
# Seconds that decoding one upload may take. The longest audio allowed by default decodes in a minute or two.
DECODE_TIMEOUT = 600.0
# A 503's Retry-After is the average time of recent requests, rounded up and kept within these bounds (seconds).
MIN_RETRY_AFTER, MAX_RETRY_AFTER = 1, 60
# The weight of the latest request in that average, an exponential moving average.
RECENT_WEIGHT = 0.3
# The status logged for a request whose client left while it waited: nginx's "client closed request".
CLIENT_CLOSED = 499
# How main() shows the package's log on stderr, next to uvicorn's own lines.
LOG_FORMAT = "%(levelname)s: %(name)s: %(message)s"

# By name, not __name__: under "python -m speech_toolkit.server" this module is __main__, outside the package's logger.
logger = logging.getLogger("speech_toolkit.server")

# ASGI types, as Starlette defines them (Starlette is imported only when the app is built).
Message = MutableMapping[str, Any]
Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

T = TypeVar("T")


class APIError(Exception):
    """An error sent to the client in OpenAI's format."""

    def __init__(
        self,
        status: int,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        param: Optional[str] = None,
        code: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = {"error": {"message": message, "type": error_type, "param": param, "code": code}}
        self.headers = dict(headers or {})


class _Service:
    """Request handling behind the routes: auth, upload limits, validation, and the queue in front of the model.

    One worker thread runs the model, which isn't thread-safe, taking the requests in turn. Besides the request it
    transcribes, at most *max_queue* requests are let in to wait (from the start of their upload); the guard refuses
    more with a 503 before reading them. The counts are kept under a lock of their own, never held for longer than
    it takes to change or read them, so ``/health`` never waits for the model.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        api_key: Optional[str],
        max_upload_mb: float,
        *,
        max_queue: int = DEFAULT_MAX_QUEUE,
        max_audio_seconds: Optional[float] = DEFAULT_MAX_AUDIO_SECONDS,
    ) -> None:
        self.transcriber = transcriber
        self.api_key = api_key
        self.max_upload_mb = max_upload_mb
        self.max_bytes = int(max_upload_mb * 1024 * 1024)
        self.max_request_bytes = self.max_bytes + FORM_OVERHEAD
        self.max_queue = max_queue
        self.max_audio_seconds = max_audio_seconds
        # The worker thread starts with the first request; it ends when the executor is garbage-collected or Python
        # exits, after finishing its request.
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="transcribe-server")
        self._counts = threading.Lock()
        self.admitted = 0  # requests let in and not finished: being received, waiting, or being transcribed
        self.transcribing = 0  # 1 while the worker runs a request
        self.request_seconds: Optional[float] = None  # the average time of recent requests in the worker

    def admit(self) -> bool:
        """Let a transcription request in, if the model or a place in the queue is free; release it with leave()."""
        with self._counts:
            if self.admitted > self.max_queue:
                return False
            self.admitted += 1
            return True

    def leave(self) -> None:
        with self._counts:
            self.admitted -= 1

    def status(self) -> Dict[str, int]:
        """The queue, for /health."""
        with self._counts:
            waiting = max(0, self.admitted - self.transcribing)
            return {"transcribing": self.transcribing, "waiting": waiting, "max_queue": self.max_queue}

    def busy(self) -> APIError:
        """The 503 for a request that finds the queue full."""
        with self._counts:
            average = self.request_seconds or MIN_RETRY_AFTER
        seconds = min(MAX_RETRY_AFTER, max(MIN_RETRY_AFTER, math.ceil(average)))
        message = f"The server is busy with other requests; retry in {seconds} s."
        return APIError(503, message, error_type="server_error", headers={"Retry-After": str(seconds)})

    async def run(self, receive: Receive, job: Callable[[], T]) -> T:
        """Run *job* in the worker thread, after the requests before it, and return what it returns.

        A job whose client disconnects while it waits is dropped: the request ends with a 499 and frees its place.
        One whose client disconnects while it runs goes on to the end, since the model can't be stopped mid-way: the
        request keeps its place until then, and the upload that the job reads stays open.

        Args:
            receive: The request's ASGI receive(). The body has been read, so it returns only on a disconnect.
            job: What the worker runs.
        """
        future = self.worker.submit(self._work, job)
        waiter = asyncio.wrap_future(future)
        left = asyncio.ensure_future(_disconnected(receive))
        try:
            await asyncio.wait((waiter, left), return_when=asyncio.FIRST_COMPLETED)
            if not waiter.done():  # the client left
                if future.cancel():  # the job hadn't started, and now never will
                    logger.info("A client disconnected while its request waited for the model; dropped the request.")
                    raise APIError(CLIENT_CLOSED, "The client closed the request.")
                await asyncio.wait((waiter,))
            return waiter.result()
        finally:
            left.cancel()
            future.cancel()  # in case this request was cancelled, as when the server stops: the job then never runs

    def _work(self, job: Callable[[], T]) -> T:
        """Run *job*, in the worker thread, counting it for /health and timing it for Retry-After."""
        with self._counts:
            self.transcribing += 1
        start = time.monotonic()
        try:
            return job()
        finally:
            seconds = time.monotonic() - start
            with self._counts:
                self.transcribing -= 1
                average = self.request_seconds
                self.request_seconds = seconds if average is None else average + RECENT_WEIGHT * (seconds - average)

    def check_key(self, authorization: str) -> None:
        if self.api_key is None:
            return
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(token.strip().encode(), self.api_key.encode()):
            raise APIError(401, "Missing or wrong API key: send 'Authorization: Bearer <key>'.", code="invalid_api_key")

    def check_request(self, task: str, response_format: str, granularities: List[str], stream: bool) -> bool:
        """Reject what the server can't do; returns whether word timestamps were asked for."""
        if task == "translate" and not self.transcriber.supports("translate"):
            raise APIError(400, f"The {self.transcriber.backend_name} backend can't translate.")
        words = "word" in granularities
        if stream:
            raise APIError(400, "Streaming isn't supported; leave stream unset.", param="stream")
        if response_format not in RESPONSE_FORMATS:
            raise APIError(
                400, f"response_format must be one of: {', '.join(RESPONSE_FORMATS)}.", param="response_format"
            )
        if set(granularities) - {"word", "segment"}:
            raise APIError(
                400, "timestamp_granularities[] takes 'word' and 'segment'.", param="timestamp_granularities[]"
            )
        if words and response_format != "verbose_json":
            raise APIError(400, "Word timestamps need response_format=verbose_json.", param="timestamp_granularities[]")
        if words and not self.transcriber.supports("word_timestamps"):
            message = f"The {self.transcriber.backend_name} backend doesn't give word timestamps."
            raise APIError(400, message, param="timestamp_granularities[]")
        return words

    def check_headers(self, path: str, headers: Mapping[bytes, bytes]) -> None:
        """The checks a request's headers allow, made before its body is read: the API key and Content-Length.

        Args:
            path: The request's path in the app; ``/health`` needs no key.
            headers: The request's headers, with lower-case names, as ASGI gives them.
        """
        if path != "/health":
            self.check_key(headers.get(b"authorization", b"").decode("latin-1"))
        length = headers.get(b"content-length", b"")
        if length.isdigit() and int(length) > self.max_request_bytes:
            raise self.too_large()

    def too_large(self) -> APIError:
        return APIError(413, f"The file is larger than this server's {self.max_upload_mb:g} MB limit.", param="file")

    def save_upload(self, upload: Any, out: BinaryIO) -> None:
        """Copy the upload to *out*, refusing a file over the size limit and one that isn't media (_is_playlist)."""
        chunk = upload.file.read(CHUNK_BYTES)
        if _is_playlist(chunk[:4096]):
            logger.warning("Refused an upload that is a playlist or manifest, or blank, rather than audio.")
            raise _not_audio()
        size = 0
        while chunk:
            size += len(chunk)
            if size > self.max_bytes:
                raise self.too_large()
            out.write(chunk)
            chunk = upload.file.read(CHUNK_BYTES)

    async def handle(
        self,
        receive: Receive,
        task: str,
        upload: Any,
        language: Optional[str],
        prompt: Optional[str],
        response_format: str,
        granularities: List[str],
        stream: bool,
    ) -> Any:
        """A transcription request, from the route: validate it, then wait for the worker to transcribe it and render
        the response. All of that stays off the event loop, which keeps answering other requests."""
        words = self.check_request(task, response_format, granularities, stream)
        job = functools.partial(self.respond, upload, task, language, prompt, words, response_format)
        return await self.run(receive, job)

    def respond(
        self, upload: Any, task: str, language: Optional[str], prompt: Optional[str], words: bool, response_format: str
    ) -> Any:
        """The response to a request, in the worker thread: transcribe the upload, and render the result.

        It is a Response, serialized here, because FastAPI would encode a dict that a route returns on the event loop.
        """
        from starlette.responses import JSONResponse, PlainTextResponse  # imported already, with the app

        result = self.transcribe(upload, task, language, prompt, words)
        body = _response_body(result, response_format, task, words)
        return JSONResponse(body) if isinstance(body, dict) else PlainTextResponse(body)

    def transcribe(
        self, upload: Any, task: str, language: Optional[str], prompt: Optional[str], words: bool
    ) -> TranscriptionResult:
        """Save the upload to a temporary file, decode it and transcribe the samples; the file is deleted whatever
        happens. The worker thread runs this, one request at a time."""
        fd, name = tempfile.mkstemp(suffix=_upload_suffix(upload.filename))
        path = pathlib.Path(name)
        try:
            with os.fdopen(fd, "wb") as out:
                self.save_upload(upload, out)
            samples = self.decode(path)
            return self.transcriber.transcribe(
                samples, language or None, task, prompt=prompt or None, word_timestamps=True if words else None
            )
        except APIError:
            raise
        except Exception as e:  # decoding, model, disk and out-of-memory errors come in many types
            raise _api_error(e, path) from e
        finally:
            path.unlink(missing_ok=True)

    def decode(self, path: pathlib.Path) -> "np.ndarray":
        """The samples of the upload saved at *path*, decoded once, for the model and for diarization alike.

        ffmpeg reads only common, self-contained formats (``strict``), from the file, which it can seek in (an MP4
        whose index comes last decodes only so), and stops a little past the length limit.
        """
        limit = self.max_audio_seconds
        max_seconds = None if limit is None else limit + AUDIO_LIMIT_MARGIN
        samples = load_audio(path, strict=True, max_seconds=max_seconds, timeout=DECODE_TIMEOUT)
        if limit is not None and len(samples) > limit * SAMPLE_RATE:
            raise APIError(400, f"The audio is longer than this server's {limit:g} s limit.", param="file")
        return samples


class _BodyTooLarge(Exception):
    """Raised to the app, from receive(), when the request body passes the size limit."""


class _LimitedBody:
    """The receive() and send() the app gets, which cut the request body off at *limit* bytes.

    Past the limit, receive() raises and send() drops the app's own answer to the cut-off body: the guard sends a 413.
    """

    def __init__(self, receive: Receive, send: Send, limit: int) -> None:
        self._receive = receive
        self._send = send
        self.limit = limit
        self.received = 0
        self.too_large = False

    async def receive(self) -> Message:
        message = await self._receive()
        self.received += len(message.get("body", b""))
        if self.received > self.limit:
            self.too_large = True
            raise _BodyTooLarge()
        return message

    async def send(self, message: Message) -> None:
        if not self.too_large:
            await self._send(message)


class _RequestGuard:
    """ASGI middleware that refuses requests from their headers, before anything reads the body.

    FastAPI parses the form, and Starlette spools the upload to disk, before the routes run. So the API key and
    ``Content-Length`` are checked here, answering 401 or 413 straight away, and a body sent without
    ``Content-Length`` (chunked) is cut off with a 413 as soon as it passes the limit. A transcription request that
    passes takes a place in the queue for the model, held until its response, or gets a 503 if the queue is full.
    """

    def __init__(self, app: ASGIApp, service: _Service) -> None:
        self.app = app
        self.service = service

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = _route_path(scope)
        try:
            self.service.check_headers(path, dict(scope["headers"]))
        except APIError as e:
            await _send_error(send, e)
            return
        queued = scope["method"] == "POST" and path in TRANSCRIPTION_PATHS
        if queued and not self.service.admit():
            await _send_error(send, self.service.busy())
            return
        body = _LimitedBody(receive, send, self.service.max_request_bytes)
        try:
            await self.app(scope, body.receive, body.send)
        except _BodyTooLarge:
            pass  # answered below
        finally:
            if queued:
                self.service.leave()
        if body.too_large:  # the routes read the whole body before they respond, so no response has started
            await _send_error(send, self.service.too_large())


async def _send_error(send: Send, error: APIError) -> None:
    """Send *error* as the response, in OpenAI's format, from outside the FastAPI app."""
    body = json.dumps(error.body, separators=(",", ":")).encode()  # compact, as FastAPI's own responses
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    headers += [(name.lower().encode("latin-1"), value.encode("latin-1")) for name, value in error.headers.items()]
    await send({"type": "http.response.start", "status": error.status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


async def _disconnected(receive: Receive) -> None:
    """Return when the client disconnects. Once a request's body has been read, receive() returns only then."""
    message: Message = {}
    while message.get("type") != "http.disconnect":
        message = await receive()


def _route_path(scope: Scope) -> str:
    """The request's path in the app, without the root path the app may be served under."""
    path: str = scope["path"]
    root: str = scope.get("root_path", "")
    return path[len(root) :] if root and path.startswith(root) else path


def _upload_suffix(filename: Optional[str]) -> str:
    """The extension for an upload's temporary file: the upload's own, lower-cased, if in UPLOAD_SUFFIXES, else .bin.

    The client's file name is never used: a name ending in ``.m3u8`` would have ffmpeg read the upload as an HLS
    playlist, whose entries can name other files on the server. ffmpeg and PyAV (faster-whisper) recognise audio and
    video by their content, so ``.bin`` decodes like the right extension: checked with WAV, MP3 (with and without an
    ID3 tag), FLAC, Ogg, Opus, AAC, M4A, MP4, MOV, MKV, WebM, WMA and MPEG, on ffmpeg 6.1 and PyAV 18.1 (FFmpeg 8.1).
    """
    suffix = pathlib.PurePath(filename or "").suffix.lower()
    return suffix if suffix in UPLOAD_SUFFIXES else ".bin"


def _is_playlist(head: bytes) -> bool:
    """Whether an upload starting with *head* is a playlist or manifest, or blank, rather than audio or video.

    ffmpeg and PyAV recognise DASH and IMF manifests (XML), HLS playlists and ffconcat scripts by their content,
    whatever the file's name, and open the files and URLs they list: a DASH manifest makes them fetch URLs on the
    server's network. No audio or video format starts with XML, ``#EXTM3U``, ``ffconcat`` or whitespace; XML is
    recognised as libxml2 reads it, in UTF-8 or UTF-16, and has no control characters, unlike media headers.
    """
    if head.startswith((b"#EXTM3U", b"ffconcat")):
        return True
    encoding = "latin-1"  # one character per byte, enough to recognise ASCII markup (and UTF-8)
    if head.startswith((codecs.BOM_UTF16_LE, b"<\x00?\x00")):
        encoding = "utf-16-le"
    elif head.startswith((codecs.BOM_UTF16_BE, b"\x00<\x00?")):
        encoding = "utf-16-be"
    text = head.removeprefix(codecs.BOM_UTF8).decode(encoding, "replace").lstrip("\ufeff \t\r\n")
    return not text or (text.startswith("<") and all(c >= " " or c in "\t\r\n" for c in text[:256]))


def _not_audio() -> APIError:
    return APIError(400, NOT_AUDIO, param="file", code="invalid_value")


def _api_error(error: Exception, path: pathlib.Path) -> APIError:
    """The response to a failed transcription of *path*; the error's details go only to the server's log."""
    if _is_decode_error(error):
        # repr: ffmpeg's output can quote the upload's metadata, which mustn't be able to write log lines of its own
        logger.warning("Couldn't decode an upload: %r", error)
        return _not_audio()
    message = next(iter(str(error).strip().splitlines()), "")
    if isinstance(error, ValueError) and message and path.name not in message and str(path.parent) not in message:
        return APIError(400, message)  # a bad request, such as an unknown language
    logger.error("Transcription failed", exc_info=error)
    return APIError(500, SERVER_ERROR, error_type="server_error")


def _is_decode_error(error: BaseException) -> bool:
    """Whether *error*, or an error behind it, means the upload couldn't be decoded: the client's mistake.

    Backends report it in their own ways: ``AudioDecodeError`` (speech_toolkit's own decoding), ffmpeg exiting with
    an error (a ``CalledProcessError``, behind openai-whisper's ``RuntimeError``), PyAV's ``av.error.FFmpegError``
    subclasses (faster-whisper; also ``ValueError`` or ``OSError``), or any error raised in faster-whisper's audio
    decoding, such as the ``IndexError`` for a video without sound. One behind a ``FileNotFoundError``, such as a
    missing ffmpeg, is the server's problem instead.
    """
    av_errors = sys.modules.get("av.error")  # PyAV, if faster-whisper imported it
    causes = list(_error_chain(error))
    if any(isinstance(cause, FileNotFoundError) for cause in causes):
        return False
    for cause in causes:
        if isinstance(cause, (AudioDecodeError, subprocess.CalledProcessError)):
            return True
        if av_errors is not None and isinstance(cause, av_errors.FFmpegError):
            return True
        if any(
            frame.f_globals.get("__name__") == "faster_whisper.audio"
            for frame, _ in traceback.walk_tb(cause.__traceback__)
        ):
            return True
    return False


def _error_chain(error: BaseException) -> Iterator[BaseException]:
    """*error*, then the errors it was raised from or while handling."""
    seen = set()
    current: Optional[BaseException] = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def create_app(
    transcriber: Transcriber,
    *,
    api_key: Optional[str] = None,
    max_upload_mb: float = 100.0,
    max_queue: int = DEFAULT_MAX_QUEUE,
    max_audio_seconds: Optional[float] = DEFAULT_MAX_AUDIO_SECONDS,
    cors_origins: Sequence[str] = (),
    docs: Optional[bool] = None,
) -> Any:
    """Build the FastAPI app around a loaded :class:`Transcriber`; requests take turns on the model.

    Args:
        transcriber: The loaded model (and diarization pipeline, if any) to serve.
        api_key: If set, every request except ``/health`` needs ``Authorization: Bearer <api_key>``.
        max_upload_mb: Larger uploads are refused with HTTP 413, before they are read when they declare their size.
        max_queue: How many transcription requests may wait while the model transcribes another; more are refused
            with HTTP 503 and ``Retry-After``, before they are read. 0 refuses every request while the model is busy.
        max_audio_seconds: Longer audio is refused with HTTP 400, and decoded only a second past this. None for no
            limit.
        cors_origins: Web pages from these origins (such as ``http://localhost:3000``, or ``*`` for any) may call the
            API from the browser.
        docs: Serve the API docs (``/docs``, ``/redoc`` and ``/openapi.json``). The default (None) serves them
            only without an API key; with one, they too need the key.

    Raises:
        ValueError: If *max_queue* is negative, or *max_audio_seconds* isn't a positive number.
    """
    _check_limits(max_queue, max_audio_seconds)
    from fastapi import FastAPI, File, Form, Request, UploadFile
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from starlette.exceptions import HTTPException

    # By default no docs with a key: the docs pages can't send it, and would show the API to anyone.
    docs = api_key is None if docs is None else docs
    app = FastAPI(
        title="speech-transcription-toolkit",
        version=__version__,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
    )
    service = _Service(transcriber, api_key, max_upload_mb, max_queue=max_queue, max_audio_seconds=max_audio_seconds)
    app.add_middleware(_RequestGuard, service=service)
    _allow_origins(app, cors_origins)  # after the guard, so it runs first

    def api_error(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content=exc.body, headers=exc.headers or None)

    def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """FastAPI answers bad fields with 422; OpenAI clients expect a 400 in OpenAI's format."""
        return api_error(request, _validation_error(exc.errors()))

    def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return api_error(request, APIError(exc.status_code, str(exc.detail)))

    def server_error(request: Request, exc: Exception) -> JSONResponse:
        """Any other error: a generic 500 (the server logs the traceback)."""
        return api_error(request, APIError(500, SERVER_ERROR, error_type="server_error"))

    app.add_exception_handler(APIError, api_error)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, http_error)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, server_error)

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        request: Request,
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),  # accepted for compatibility: the server runs the model it loaded
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),  # accepted for compatibility, not used
        timestamp_granularities: List[str] = Form([], alias="timestamp_granularities[]"),
        stream: bool = Form(False),
    ) -> Any:
        return await service.handle(
            request.receive, "transcribe", file, language, prompt, response_format, timestamp_granularities, stream
        )

    @app.post("/v1/audio/translations")
    async def translations(
        request: Request,
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),
        prompt: Optional[str] = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
    ) -> Any:
        return await service.handle(request.receive, "translate", file, None, prompt, response_format, [], False)

    @app.get("/v1/models")
    async def models() -> Dict[str, Any]:
        owner = f"speech-transcription-toolkit ({transcriber.backend_name})"
        return {
            "object": "list",
            "data": [{"id": transcriber.model_name, "object": "model", "created": 0, "owned_by": owner}],
        }

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """The loaded model and the queue, answered at once however busy the model is."""
        return {
            "status": "ok",
            "backend": transcriber.backend_name,
            "model": transcriber.model_name,
            **service.status(),
        }

    return app


def _check_limits(max_queue: int, max_audio_seconds: Optional[float]) -> None:
    """Raise ValueError for create_app() limits that make no sense."""
    if max_queue < 0:
        raise ValueError(f"max_queue must be 0 or more, got {max_queue!r}")
    if max_audio_seconds is not None and not 0 < max_audio_seconds < math.inf:
        raise ValueError(f"max_audio_seconds must be a positive number or None, got {max_audio_seconds!r}")


def _allow_origins(app: Any, origins: Sequence[str]) -> None:
    """Let web pages from *origins* call the API (CORS), if there are any.

    Added after the guard, so it runs before it: a browser's preflight request carries no API key, and the guard's
    errors need the CORS headers for the page to read them.
    """
    if not origins:
        return
    from starlette.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],  # OpenAI's JavaScript SDK sends headers of its own (x-stainless-*)
        expose_headers=["Retry-After"],  # for the 503s
    )


def _validation_error(errors: Sequence[Any]) -> APIError:
    """The first of FastAPI's validation errors, e.g. "Field required: file"."""
    error = errors[0] if errors else {}
    param = str(error["loc"][-1]) if error.get("loc") else None
    message = f"{error.get('msg', 'Invalid request')}" + (f": {param}" if param else "")
    return APIError(400, message, param=param)


def _response_body(result: TranscriptionResult, response_format: str, task: str, words: bool) -> Any:
    """A dict for the JSON formats, or the text of txt/srt/vtt."""
    if response_format == "json":
        return {"text": result.text.strip()}
    if response_format == "verbose_json":
        return verbose_json(result, task, words)
    return result.render("txt" if response_format == "text" else response_format)


def verbose_json(result: TranscriptionResult, task: str, words: bool = False) -> Dict[str, Any]:
    """The result as OpenAI's ``verbose_json``: its segment fields, and top-level ``words`` when asked for."""
    segments = result.speaker_segments or result.segments
    duration = result.duration
    if duration is None:  # not reported: a duration in raw (as some backends give), else the last segment's end
        duration = result.raw.get("duration") or (segments[-1].get("end", 0.0) if segments else 0.0)
    body: Dict[str, Any] = {
        "task": task,
        "language": result.language,
        "duration": duration,
        "text": result.text.strip(),
        "segments": [_openai_segment(i, seg) for i, seg in enumerate(segments)],
    }
    if words:
        body["words"] = [
            {"word": w["word"].strip(), "start": w["start"], "end": w["end"]}
            for seg in result.segments
            for w in seg.get("words") or []
        ]
    return body


def _openai_segment(index: int, seg: Dict[str, Any]) -> Dict[str, Any]:
    """A segment with every field OpenAI's ``verbose_json`` has (defaults for what the backend lacks)."""
    out = {
        "id": index,
        "seek": seg.get("seek") or 0,
        "start": seg.get("start", 0.0),
        "end": seg.get("end", 0.0),
        "text": seg.get("text", ""),
        "tokens": seg.get("tokens") or [],
        "temperature": seg.get("temperature") or 0.0,
        "avg_logprob": seg.get("avg_logprob") or 0.0,
        "compression_ratio": seg.get("compression_ratio") or 0.0,
        "no_speech_prob": seg.get("no_speech_prob") or 0.0,
    }
    if "speaker" in seg:
        out["speaker"] = seg["speaker"]
    return out


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="transcribe-server",
        description="Serve OpenAI's audio API (/v1/audio/transcriptions, /v1/audio/translations) locally.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--host", default="127.0.0.1", help="Address to listen on (default: 127.0.0.1, this machine).")
    parser.add_argument("--port", type=positive_int, default=8000, help="Port to listen on (default: 8000).")
    add_model_options(parser)  # --backend, --model and --device, as for transcribe
    parser.add_argument("--diarize", action="store_true", help="Label speakers (needs the diarize extra and HF_TOKEN).")
    parser.add_argument("--hf-token", metavar="TOKEN", help="Hugging Face token (default: HF_TOKEN env var).")
    parser.add_argument(
        "--api-key",
        default=os.getenv("TRANSCRIBE_API_KEY"),
        help="Require 'Authorization: Bearer KEY' (default: the TRANSCRIBE_API_KEY env var; unset means no key).",
    )
    parser.add_argument(
        "--max-upload-mb", type=positive_int, default=100, help="Refuse larger uploads (default: 100 MB)."
    )
    parser.add_argument(
        "--max-queue",
        type=_non_negative_int,
        default=DEFAULT_MAX_QUEUE,
        metavar="N",
        help="How many requests may wait while the model transcribes another; more get HTTP 503 with Retry-After "
        f"(default: {DEFAULT_MAX_QUEUE}).",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=_seconds,
        default=DEFAULT_MAX_AUDIO_SECONDS,
        metavar="SECONDS",
        help="Refuse longer audio with HTTP 400, without decoding the rest "
        f"(default: {DEFAULT_MAX_AUDIO_SECONDS}, 3 hours; 0 for no limit).",
    )
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help="Let web pages from ORIGIN (e.g. http://localhost:3000, or * for any) call the API; repeat for more.",
    )
    parser.add_argument(
        "--docs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Serve /docs, /redoc and /openapi.json (default: only without --api-key; with one, they need the key).",
    )
    parser.set_defaults(quiet=False)  # for cli.load_transcriber
    return parser.parse_args(argv)


def _non_negative_int(value: str) -> int:
    """argparse type for whole numbers from 0."""
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {value}")
    return number


def _seconds(value: str) -> float:
    """argparse type for a number of seconds: 0 or more, and finite."""
    number = float(value)
    if not 0 <= number < math.inf:
        raise argparse.ArgumentTypeError(f"must be a number of seconds, 0 or more, got {value}")
    return number


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        import fastapi  # noqa: F401
        import python_multipart  # noqa: F401
        import uvicorn
    except ImportError as e:
        sys.exit(f"Error: {INSTALL_HINT} ({e})")

    if args.api_key is None and not _is_loopback(args.host):
        print(
            f"Warning: listening on {args.host} without --api-key; anyone who can reach this port can use it.",
            file=sys.stderr,
        )
    with _log_to_stderr():
        try:
            transcriber = load_transcriber(args)
        except (ImportError, *FILE_ERRORS) as e:
            sys.exit(f"Error: {e}")

        app = create_app(
            transcriber,
            api_key=args.api_key,
            max_upload_mb=args.max_upload_mb,
            max_queue=args.max_queue,
            max_audio_seconds=args.max_audio_seconds or None,  # 0: no limit
            cors_origins=args.cors_origin,
            docs=args.docs,
        )
        uvicorn.run(app, host=args.host, port=args.port)


@contextlib.contextmanager
def _log_to_stderr() -> Iterator[None]:
    """While the server runs, show the speech_toolkit log on stderr, from INFO: undecodable uploads, dropped requests,
    and errors with their tracebacks.

    Uvicorn sets up only its own loggers, and the package's logger may have just a NullHandler, which keeps Python
    from printing anything. A logger that already has another handler is left as it is, and so is the root logger.
    """
    package = logging.getLogger("speech_toolkit")
    if any(not isinstance(handler, logging.NullHandler) for handler in package.handlers):
        yield
        return
    handler = logging.StreamHandler()  # sys.stderr
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    level = package.level
    package.addHandler(handler)
    if level == logging.NOTSET:  # else the root logger's level applies: WARNING, unless set otherwise
        package.setLevel(logging.INFO)
    try:
        yield
    finally:
        package.removeHandler(handler)
        package.setLevel(level)


if __name__ == "__main__":  # pragma: no cover
    main()
