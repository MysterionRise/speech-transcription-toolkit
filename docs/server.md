# OpenAI-compatible server

`transcribe-server` serves OpenAI's speech-to-text API locally, so apps and SDKs written for it work offline with any
backend.

```bash
pip install "speech-transcription-toolkit[server]"
transcribe-server -m small      # faster-whisper's small model on http://127.0.0.1:8000/v1
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
print(client.audio.transcriptions.create(model="whisper-1", file=open("talk.mp3", "rb")).text)
```

**How it behaves:**
- The server runs one model, so the request's `model` field is accepted and ignored. Clients that hard-code `whisper-1`
  keep working.
- Requests are transcribed one at a time, in the order they arrive; see [Load](#load).
- The server decodes each upload once, with ffmpeg, and gives the model the samples, so every backend reads the same
  formats.
- `python -m speech_toolkit.server` starts it too.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/audio/transcriptions` | Transcribe an uploaded file. |
| `POST /v1/audio/translations` | Translate an uploaded file to English (backends that translate: whisper, faster-whisper, canary). |
| `GET /v1/models` | The loaded model. |
| `GET /health` | Status, backend, model and queue, answered at once even while the model is busy. Needs no API key. |
| `GET /docs`, `/redoc`, `/openapi.json` | Interactive API docs and the OpenAPI schema. Off when an API key is set, unless `--docs`. |

`/health` answers like this:

```json
{"status": "ok", "backend": "faster-whisper", "model": "small", "transcribing": 1, "waiting": 3, "max_queue": 8}
```

`transcribing` is 1 while the model works on a request. `waiting` counts the requests waiting for their turn, including
those still uploading.

**Transcription form fields:**
- `file`
- `model` (ignored)
- `language`
- `prompt`
- `response_format`: `json` (default), `text`, `srt`, `vtt` or `verbose_json`.
- `temperature`: accepted, not used.
- `timestamp_granularities[]`: `segment`, or `word`. `word` needs `verbose_json` and a backend with word timestamps,
  and adds a top-level `words` list.
- `stream`: must be false or absent.

The translations endpoint takes `file`, `model`, `prompt`, `response_format` and `temperature`.

**Diarization:** with `--diarize`, `verbose_json` segments carry a `speaker` field, and `text`, `srt` and `vtt`
responses include the labels.

**Errors** use OpenAI's JSON shape: `{"error": {"message", "type", "param", "code"}}`.
- A file that isn't audio or video ffmpeg can decode gets a 400 with `"code": "invalid_value"` and OpenAI's message,
  `Audio file might be corrupted or unsupported`. So does a format outside the common, self-contained ones, such as a
  playlist (see [Security](#security)).
- Audio longer than `--max-audio-seconds` gets a 400 with `"param": "file"`.
- When the queue is full, a request gets a 503 with a `Retry-After` header; see [Load](#load).
- Any other failure gets a 500 with a generic message. The details, such as ffmpeg's output or the traceback, go to the
  server's log on stderr.

## Load

One worker runs the model, a request at a time; models can't transcribe two at once. The other requests wait in a queue.
- **Bounded queue.** Up to `--max-queue` requests (default: 8) wait while the model works on another, counted from the
  start of their upload. Further requests get a 503 at once, before their upload is read, with a `Retry-After` header:
  the average time of recent requests, from 1 to 60 seconds. OpenAI's SDKs retry 503s on their own (twice, by default),
  waiting as `Retry-After` says.
- **Responsive.** `/health` and `/v1/models` answer at once, however many requests wait.
- **Clients that disconnect.** A request whose client disconnects while it waits is dropped, and its place is freed.
  One being transcribed runs to the end, since the model can't be stopped mid-way.
- **Long audio.** Audio longer than `--max-audio-seconds` (default: 3 hours) gets a 400. ffmpeg stops a second past the
  limit, so the rest of the file is never decoded. Decoding three hours takes about 1 GB of memory, and the server
  decodes one upload at a time. `--max-audio-seconds 0` removes the limit.

## Options

| Option | Description |
|---|---|
| `--host` | Address to listen on (default: `127.0.0.1`, this machine only). |
| `--port` | Port (default: 8000). |
| `-b`, `--backend`, `-m`, `--model`, `--device` | As for `transcribe`; see [Command line](cli.md). |
| `--diarize` | Label speakers; needs the `diarize` extra and `HF_TOKEN`. |
| `--hf-token TOKEN` | Hugging Face token (default: the `HF_TOKEN` environment variable). |
| `--api-key KEY` | Require `Authorization: Bearer KEY` on every request except `/health`, and turn off `/docs`, `/redoc` and `/openapi.json` unless `--docs` is given (default: the `TRANSCRIBE_API_KEY` environment variable; unset means no key). |
| `--max-upload-mb N` | Refuse larger files with HTTP 413 (default: 100). |
| `--max-queue N` | How many requests may wait while the model transcribes another; more get HTTP 503 with `Retry-After` (default: 8). `0` refuses every request while the model is busy. |
| `--max-audio-seconds SECONDS` | Refuse longer audio with HTTP 400, without decoding the rest (default: 10800, 3 hours; `0` for no limit). |
| `--cors-origin ORIGIN` | Let web pages from `ORIGIN`, such as `http://localhost:3000`, call the API from the browser; repeat it for more origins, or give `*` for any. Off by default. |
| `--docs`, `--no-docs` | Serve `/docs`, `/redoc` and `/openapi.json`, or don't (default: only without `--api-key`). With an API key, `--docs` serves them to requests that send the key. |

## Browser clients (CORS)

Browsers let a web page call the server only if the server allows the page's origin: pass it with `--cors-origin`.
- Every endpoint gets the CORS headers, error responses included. `Retry-After` is exposed to the page.
- Any request header is allowed, such as the `x-stainless-*` headers of OpenAI's JavaScript SDK.
- Browsers send a preflight request without the API key first; the server answers it without asking for the key.
- An API key in a web page is visible to anyone who loads the page. Use one only for pages that aren't public.

## Security

- **Bind address.** The server listens on localhost unless you pass `--host`, and warns when it's exposed without an API
  key.
- **Checks before the upload is read.** The API key and the upload's size are checked from the request's headers,
  before the server reads the body:
  - A request without the right key gets a 401.
  - A request whose `Content-Length` is over the limit gets a 413.
  - A transcription request that finds the queue full gets a 503. The key is checked first, so only clients with the
    key learn that the server is busy.
  - An upload sent without a `Content-Length` (chunked) is cut off with a 413 when it reaches the limit.

  The limit is for the file: the whole request may be up to 1 MB larger, for the other form fields.
- **Temp files.** Uploads go to temporary files with random names, which are always deleted after the request: when
  it succeeds or fails, and when its client disconnects. A request refused for a full queue never gets one.
  - A file keeps the upload's extension only if it's a common audio or video one (such as `.mp3`, `.wav` or `.webm`);
    others are stored as `.bin`. ffmpeg recognises formats by their content, so this changes nothing for real audio.
  - Uploads that are playlists or manifests (HLS, DASH, ffconcat) get a 400, because ffmpeg would open the files and
    URLs they list.
- **Decoding.** ffmpeg decodes only the common, self-contained audio and video formats (the strict mode of
  `speech_toolkit.media.load_audio`), opens only the uploaded file, and is stopped after 10 minutes. So a playlist
  that the check above missed is refused as well.
- **Errors.** Error responses never include temporary paths, ffmpeg's output or exception details.
- **API docs.** With an API key, `/docs`, `/redoc` and `/openapi.json` are off: they would show the API to anyone who
  can reach the server. With `--docs` they are on, and need the key like the other endpoints.

**Breaking change:** `/docs`, `/redoc` and `/openapi.json` used to be served with an API key set too. To browse them,
run a server without `--api-key` on localhost.
