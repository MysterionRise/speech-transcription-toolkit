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
import ipaddress
import os
import pathlib
import secrets
import sys
import tempfile
import threading
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .api import Transcriber
from .backends import DEFAULT_BACKEND, TranscriptionResult, list_backends
from .cli import FILE_ERRORS, load_transcriber, positive_int

INSTALL_HINT = 'transcribe-server needs the server extra: pip install "speech-transcription-toolkit[server]"'
RESPONSE_FORMATS = ("json", "text", "srt", "vtt", "verbose_json")
CHUNK_BYTES = 1024 * 1024


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

    def save_upload(self, upload: Any) -> pathlib.Path:
        """Copy the upload to a temporary file (keeping its extension as a hint for ffmpeg)."""
        size = 0
        with tempfile.NamedTemporaryFile(suffix=pathlib.Path(upload.filename or "").suffix, delete=False) as out:
            while size <= self.max_bytes and (chunk := upload.file.read(CHUNK_BYTES)):
                size += len(chunk)
                out.write(chunk)
        path = pathlib.Path(out.name)
        if size > self.max_bytes:
            path.unlink(missing_ok=True)
            raise APIError(413, f"The file is larger than this server's {self.max_upload_mb:g} MB limit.", param="file")
        return path

    def transcribe(
        self, upload: Any, task: str, language: Optional[str], prompt: Optional[str], words: bool
    ) -> TranscriptionResult:
        path = self.save_upload(upload)
        try:
            with self.lock:
                return self.transcriber.transcribe(
                    path, language or None, task, prompt=prompt or None, word_timestamps=True if words else None
                )
        except ValueError as e:
            raise APIError(400, str(e)) from e
        except Exception as e:  # audio decoding, model and out-of-memory errors come in many types
            raise APIError(500, str(e), error_type="server_error") from e
        finally:
            path.unlink(missing_ok=True)


def create_app(transcriber: Transcriber, *, api_key: Optional[str] = None, max_upload_mb: float = 100.0) -> Any:
    """Build the FastAPI app around a loaded :class:`Transcriber`; requests take turns on the model.

    Args:
        transcriber: The loaded model (and diarization pipeline, if any) to serve.
        api_key: If set, every ``/v1`` request needs ``Authorization: Bearer <api_key>``.
        max_upload_mb: Larger uploads are refused with HTTP 413.
    """
    from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse, PlainTextResponse
    from starlette.exceptions import HTTPException

    app = FastAPI(title="speech-transcription-toolkit", version=__version__)
    service = _Service(transcriber, api_key, max_upload_mb)

    def api_error(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content=exc.body)

    def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """FastAPI answers bad fields with 422; OpenAI clients expect a 400 in OpenAI's format."""
        return api_error(request, _validation_error(exc.errors()))

    def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return api_error(request, APIError(exc.status_code, str(exc.detail)))

    app.add_exception_handler(APIError, api_error)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, http_error)  # type: ignore[arg-type]

    def check_key(request: Request) -> None:
        service.check_key(request.headers.get("authorization", ""))

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

    @app.post("/v1/audio/transcriptions", dependencies=[Depends(check_key)])
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

    @app.post("/v1/audio/translations", dependencies=[Depends(check_key)])
    def translations(
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),
        prompt: Optional[str] = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
    ) -> Any:
        return handle("translate", file, None, prompt, response_format, [], False)

    @app.get("/v1/models", dependencies=[Depends(check_key)])
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
