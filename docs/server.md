# OpenAI-compatible server

`transcribe-server` serves OpenAI's speech-to-text API locally, so apps and SDKs written for it work offline with any
backend.

```bash
pip install "speech-transcription-toolkit[server,faster-whisper]"
transcribe-server -b faster-whisper -m small      # http://127.0.0.1:8000/v1
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
print(client.audio.transcriptions.create(model="whisper-1", file=open("talk.mp3", "rb")).text)
```

**How it behaves:**
- The server runs one model, so the request's `model` field is accepted and ignored. Clients that hard-code `whisper-1`
  keep working.
- Requests are transcribed one at a time.
- `python -m speech_toolkit.server` starts it too.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/audio/transcriptions` | Transcribe an uploaded file. |
| `POST /v1/audio/translations` | Translate an uploaded file to English (backends that translate: whisper, faster-whisper, canary). |
| `GET /v1/models` | The loaded model. |
| `GET /health` | Status, backend and model. Needs no API key. |
| `GET /docs`, `/redoc`, `/openapi.json` | Interactive API docs and the OpenAPI schema. Off when an API key is set. |

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
- A file that isn't audio or video the backend can decode gets a 400 with `"code": "invalid_value"` and OpenAI's
  message, `Audio file might be corrupted or unsupported`, on every backend.
- Any other failure gets a 500 with a generic message. The details, such as ffmpeg's output or the traceback, go to the
  server's log on stderr.

## Options

| Option | Description |
|---|---|
| `--host` | Address to listen on (default: `127.0.0.1`, this machine only). |
| `--port` | Port (default: 8000). |
| `-b`, `--backend`, `-m`, `--model`, `--device` | As for `transcribe`; see [Command line](cli.md). |
| `--diarize` | Label speakers; needs the `diarize` extra and `HF_TOKEN`. |
| `--hf-token TOKEN` | Hugging Face token (default: the `HF_TOKEN` environment variable). |
| `--api-key KEY` | Require `Authorization: Bearer KEY` on every request except `/health`, and turn off `/docs`, `/redoc` and `/openapi.json` (default: the `TRANSCRIBE_API_KEY` environment variable; unset means no key). |
| `--max-upload-mb N` | Refuse larger files with HTTP 413 (default: 100). |

## Security

- **Bind address.** The server listens on localhost unless you pass `--host`, and warns when it's exposed without an API
  key.
- **Checks before the upload is read.** The API key and the upload's size are checked from the request's headers,
  before the server reads the body:
  - A request without the right key gets a 401.
  - A request whose `Content-Length` is over the limit gets a 413.
  - An upload sent without a `Content-Length` (chunked) is cut off with a 413 when it reaches the limit.

  The limit is for the file: the whole request may be up to 1 MB larger, for the other form fields.
- **Temp files.** Uploads go to temporary files with random names, which are always deleted after the request.
  - A file keeps the upload's extension only if it's a common audio or video one (such as `.mp3`, `.wav` or `.webm`);
    others are stored as `.bin`. ffmpeg recognises formats by their content, so this changes nothing for real audio.
  - Uploads that are playlists or manifests (HLS, DASH, ffconcat) get a 400, because ffmpeg would open the files and
    URLs they list.
- **Errors.** Error responses never include temporary paths, ffmpeg's output or exception details.
- **API docs.** With an API key, `/docs`, `/redoc` and `/openapi.json` are off: they would show the API to anyone who
  can reach the server.

**Breaking change:** `/docs`, `/redoc` and `/openapi.json` used to be served with an API key set too. To browse them,
run a server without `--api-key` on localhost.
