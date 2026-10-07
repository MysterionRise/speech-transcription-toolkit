"""The ``transcribe-server`` command: OpenAI's audio API (``/v1/audio/transcriptions``), served locally.

Apps and SDKs written for OpenAI's speech-to-text endpoints work offline against any backend:

    transcribe-server -b faster-whisper -m small

    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
    print(client.audio.transcriptions.create(model="whisper-1", file=open("talk.mp3", "rb")).text)

The server runs one model, so the request's ``model`` field is accepted and ignored. Needs the
``server`` extra.
"""

# No "from __future__ import annotations": FastAPI reads the endpoints' annotations at runtime, and the
# endpoints are defined inside create_app() where its imports live.

import argparse
import codecs
import ipaddress
import json
import logging
import os
import pathlib
import secrets
import subprocess  # nosec B404 - only for CalledProcessError, the error of a failed ffmpeg run
import sys
import tempfile
import threading
import traceback
from typing import Any, Awaitable, BinaryIO, Callable, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence

from . import AudioDecodeError, __version__
from .api import Transcriber
from .backends import DEFAULT_BACKEND, TranscriptionResult, list_backends
from .cli import FILE_ERRORS, load_transcriber, positive_int

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

logger = logging.getLogger(__name__)

# ASGI types, as Starlette defines them (Starlette is imported only when the app is built).
Message = MutableMapping[str, Any]
Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


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
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = {"error": {"message": message, "type": error_type, "param": param, "code": code}}


class _Service:
    """Request handling behind the routes: auth, upload limits, validation and one-at-a-time transcription."""

    def __init__(self, transcriber: Transcriber, api_key: Optional[str], max_upload_mb: float) -> None:
        self.transcriber = transcriber
        self.api_key = api_key
        self.max_upload_mb = max_upload_mb
        self.max_bytes = int(max_upload_mb * 1024 * 1024)
        self.max_request_bytes = self.max_bytes + FORM_OVERHEAD
        self.lock = threading.Lock()  # one model: requests are transcribed one at a time

    def check_key(self, authorization: str) -> None:
        if self.api_key is None:
            return
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(token.strip().encode(), self.api_key.encode()):
            raise APIError(401, "Missing or wrong API key: send 'Authorization: Bearer <key>'.", code="invalid_api_key")

    def check_request(self, response_format: str, granularities: List[str], stream: bool) -> bool:
        """Reject what the server can't do; returns whether word timestamps were asked for."""
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

    def transcribe(
        self, upload: Any, task: str, language: Optional[str], prompt: Optional[str], words: bool
    ) -> TranscriptionResult:
        """Save the upload to a temporary file and transcribe it; the file is deleted whatever happens."""
        fd, name = tempfile.mkstemp(suffix=_upload_suffix(upload.filename))
        path = pathlib.Path(name)
        try:
            with os.fdopen(fd, "wb") as out:
                self.save_upload(upload, out)
            with self.lock:
                return self.transcriber.transcribe(
                    path, language or None, task, prompt=prompt or None, word_timestamps=True if words else None
                )
        except APIError:
            raise
        except Exception as e:  # decoding, model, disk and out-of-memory errors come in many types
            raise _api_error(e, path) from e
        finally:
            path.unlink(missing_ok=True)


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
    ``Content-Length`` (chunked) is cut off with a 413 as soon as it passes the limit.
    """

    def __init__(self, app: ASGIApp, service: _Service) -> None:
        self.app = app
        self.service = service

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        try:
            self.service.check_headers(_route_path(scope), dict(scope["headers"]))
        except APIError as e:
            await _send_error(send, e)
            return
        body = _LimitedBody(receive, send, self.service.max_request_bytes)
        try:
            await self.app(scope, body.receive, body.send)
        except _BodyTooLarge:
            pass  # answered below
        if body.too_large:  # the routes read the whole body before they respond, so no response has started
            await _send_error(send, self.service.too_large())


async def _send_error(send: Send, error: APIError) -> None:
    """Send *error* as the response, in OpenAI's format, from outside the FastAPI app."""
    body = json.dumps(error.body, separators=(",", ":")).encode()  # compact, as FastAPI's own responses
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    await send({"type": "http.response.start", "status": error.status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


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


def create_app(transcriber: Transcriber, *, api_key: Optional[str] = None, max_upload_mb: float = 100.0) -> Any:
    """Build the FastAPI app around a loaded :class:`Transcriber`; requests take turns on the model.

    Args:
        transcriber: The loaded model (and diarization pipeline, if any) to serve.
        api_key: If set, every request except ``/health`` needs ``Authorization: Bearer <api_key>``, and the API
            docs (``/docs``, ``/redoc`` and ``/openapi.json``) are turned off.
        max_upload_mb: Larger uploads are refused with HTTP 413, before they are read when they declare their size.
    """
    from fastapi import FastAPI, File, Form, Request, UploadFile
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse, PlainTextResponse
    from starlette.exceptions import HTTPException

    docs = api_key is None  # the docs pages can't send the key, and would show the API to anyone
    app = FastAPI(
        title="speech-transcription-toolkit",
        version=__version__,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
    )
    service = _Service(transcriber, api_key, max_upload_mb)
    app.add_middleware(_RequestGuard, service=service)

    def api_error(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content=exc.body)

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

    def handle(
        task: str,
        upload: UploadFile,
        language: Optional[str],
        prompt: Optional[str],
        response_format: str,
        granularities: List[str],
        stream: bool,
    ) -> Any:
        words = service.check_request(response_format, granularities, stream)
        result = service.transcribe(upload, task, language, prompt, words)
        body = _response_body(result, response_format, task, words)
        return body if isinstance(body, dict) else PlainTextResponse(body)

    @app.post("/v1/audio/transcriptions")
    def transcriptions(
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),  # accepted for compatibility: the server runs the model it loaded
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),  # accepted for compatibility, not used
        timestamp_granularities: List[str] = Form([], alias="timestamp_granularities[]"),
        stream: bool = Form(False),
    ) -> Any:
        return handle("transcribe", file, language, prompt, response_format, timestamp_granularities, stream)

    @app.post("/v1/audio/translations")
    def translations(
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),
        prompt: Optional[str] = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
    ) -> Any:
        return handle("translate", file, None, prompt, response_format, [], False)

    @app.get("/v1/models")
    def models() -> Dict[str, Any]:
        owner = f"speech-transcription-toolkit ({transcriber.backend_name})"
        return {
            "object": "list",
            "data": [{"id": transcriber.model_name, "object": "model", "created": 0, "owned_by": owner}],
        }

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "backend": transcriber.backend_name, "model": transcriber.model_name}

    return app


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
    body: Dict[str, Any] = {
        "task": task,
        "language": result.language,
        "duration": result.raw.get("duration") or (segments[-1].get("end", 0.0) if segments else 0.0),
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
    parser.add_argument(
        "-b",
        "--backend",
        default=DEFAULT_BACKEND,
        help=f"Transcription backend (default: {DEFAULT_BACKEND}). Available: {', '.join(list_backends())}",
    )
    parser.add_argument("-m", "--model", default=None, help="Model name/size (default: the backend's default).")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None, help="Force device (default: auto).")
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
    parser.set_defaults(quiet=False)  # for cli.load_transcriber
    return parser.parse_args(argv)


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
    try:
        transcriber = load_transcriber(args)
    except (ImportError, *FILE_ERRORS) as e:
        sys.exit(f"Error: {e}")

    uvicorn.run(
        create_app(transcriber, api_key=args.api_key, max_upload_mb=args.max_upload_mb), host=args.host, port=args.port
    )


if __name__ == "__main__":  # pragma: no cover
    main()
