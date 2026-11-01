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
  keep working. The one exception is `stream=true`, which streams the transcript for any model but `whisper-1`, as
  OpenAI does; see [Streaming](#streaming).
- Requests are transcribed one at a time, in the order they arrive; see [Load](#load).
- The server decodes each upload once, with ffmpeg, and gives the model the samples, so every backend reads the same
  formats: the common, self-contained audio and video formats, such as WAV, FLAC, MP3, AAC, M4A, MP4, MOV, Ogg, Opus,
  WebM, MKV, WMA, AIFF, AMR and AVI. Convert other files first, for example with
  `ffmpeg -i input -ac 1 -ar 16000 output.wav`.
- The log goes to stderr: uvicorn's lines, and the server's own from `INFO` up, such as undecodable uploads, dropped
  requests, and errors with their tracebacks. If your program has set up the `speech_toolkit` logger already, the
  server uses that instead.
- `python -m speech_toolkit.server` starts it too.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/audio/transcriptions` | Transcribe an uploaded file. |
| `POST /v1/audio/translations` | Translate an uploaded file to English (backends that translate: whisper, faster-whisper, canary), from the language given. |
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
- `model`: ignored, except that `whisper-1` (the default) ignores `stream`.
- `language`: the language of the audio, such as `de`. Without it, the server's `--language`, else the backend
  detects it.
- `prompt`
- `response_format`: `json` (default), `text`, `srt`, `vtt`, `verbose_json`, or `diarized_json` with `--diarize`.
- `temperature`: accepted, not used.
- `timestamp_granularities[]`: `segment`, or `word`. `word` needs `verbose_json` and a backend with word timestamps,
  and adds a top-level `words` list.
- `stream`: `true` streams the transcript as server-sent events, for any `model` but `whisper-1`; see
  [Streaming](#streaming).

The translations endpoint takes `file`, `model`, `language`, `prompt`, `response_format` (not `diarized_json`) and
`temperature`. `language` is the language translated from, an extension to OpenAI's API, which has no such field;
Canary needs it, as it can't detect the language and otherwise assumes English. With OpenAI's Python SDK, send it as
`extra_body={"language": "de"}`.

**Usage:** `json`, `verbose_json` and `diarized_json` responses have OpenAI's `usage` for models billed by the audio's
length: `{"type": "duration", "seconds": 9}`, the whole seconds of the decoded audio, rounded up as OpenAI's are.

**Diarization:** with `--diarize`, every response labels the speakers the same way:
- `text`, and the `text` field of `json`, `verbose_json` and `diarized_json`: a line per speaker turn,
  `[SPEAKER_00] Hello there.`;
- `srt` and `vtt`: the label in each cue;
- `verbose_json` segments: a `speaker` field;
- `diarized_json`: OpenAI's format for it, a segment per speaker turn.

```json
{
  "task": "transcribe",
  "duration": 12.0,
  "text": "[SPEAKER_00] Good morning, did you finish the report?\n[SPEAKER_01] Yes, I sent it.",
  "segments": [
    {"type": "transcript.text.segment", "id": "seg_0", "start": 0.0, "end": 2.9,
     "text": "Good morning, did you finish the report?", "speaker": "SPEAKER_00"},
    {"type": "transcript.text.segment", "id": "seg_1", "start": 3.1, "end": 4.6,
     "text": "Yes, I sent it.", "speaker": "SPEAKER_01"}
  ],
  "usage": {"type": "duration", "seconds": 12}
}
```

The speakers keep the labels of the other formats (`SPEAKER_00`, `SPEAKER_01`, ...) rather than OpenAI's `A`, `B`.
Without `--diarize`, `diarized_json` gets a 400.

## Streaming

With `stream=true` and any `model` but `whisper-1`, the response is a stream of
[server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events),
the events of OpenAI's API, which OpenAI's SDKs read:

```python
stream = client.audio.transcriptions.create(model="gpt-4o-transcribe", file=open("talk.mp3", "rb"), stream=True)
for event in stream:
    if event.type == "transcript.text.delta":
        print(event.delta, end="", flush=True)
```

```text
data: {"type":"transcript.text.delta","delta":"Good morning, did you finish the report?"}

data: {"type":"transcript.text.delta","delta":" Yes, I sent it."}

data: {"type":"transcript.text.done","text":"Good morning, did you finish the report? Yes, I sent it."}
```

- **Events:** a `transcript.text.delta` per segment as the backend transcribes it, then `transcript.text.done` with the
  whole transcript, which the deltas add up to. faster-whisper reports each segment as it is decoded (a 30-second
  window at a time), Canary and Voxtral after each batch of chunks, the other backends all at the end.
- **Formats:** `json` and `text` give these events. `diarized_json` (with `--diarize`) also gives a
  `transcript.text.segment` per speaker turn once diarization is done, before `transcript.text.done`. The deltas come
  before the speakers are known, so they have no labels; the done event's `text` has them, as `json`'s has. `srt`,
  `vtt` and `verbose_json` get a 400: send `stream=false` for them.
- **`whisper-1`:** OpenAI ignores `stream` for `whisper-1`, which can't stream, and so does the server: the response is
  the usual one. `model` defaults to `whisper-1`, so send another, such as `gpt-4o-transcribe`, to stream.
- **No `usage`** in the done event: OpenAI's is in tokens, which the server doesn't count.
- **Errors:** the response starts with its first event, so errors until then, such as an upload that isn't audio, an
  unknown language or a full queue, get the usual HTTP status. A failure after that ends the stream with an error
  event,
  `data: {"type":"error","error":{"message":...,"type":"server_error","param":null,"code":null}}`, which OpenAI's SDKs
  raise as an `APIError`.
- **Keep-alive:** while no event is due for 15 seconds, as while diarizing, the server sends an SSE comment
  (`: keep-alive`) that clients ignore, so that proxies don't close the connection.
- **Proxies:** the response has `Cache-Control: no-cache` and `X-Accel-Buffering: no`, so nginx passes each event on
  at once.

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
  One being transcribed runs to the end, since the model can't be stopped mid-way. A stream stops at the next segment
  the backend reports (for faster-whisper, once it has decoded the 30-second window it is on). Either way, the request
  keeps its place until the model is free.
- **Long audio.** Audio longer than `--max-audio-seconds` (default: 3 hours) gets a 400. ffmpeg stops a second past the
  limit, so the rest of the file is never decoded. Decoding three hours takes about 1 GB of memory, and the server
  decodes one upload at a time. `--max-audio-seconds 0` removes the limit.

## Options

| Option | Description |
|---|---|
| `--host` | Address to listen on (default: `127.0.0.1`, this machine only). |
| `--port` | Port (default: 8000). |
| `-b`, `--backend`, `-m`, `--model`, `--device`, `--compute-type`, `-O key=value` | As for `transcribe`; see [Command line](cli.md). |
| `-l`, `--language` | The language of the audio, such as `de`, for requests that don't send one: transcriptions and translations (the language translated from). Default: the backend detects it; Canary, which can't, assumes English. |
| `--diarize` | Label speakers; needs the `diarize` extra and `HF_TOKEN`. |
| `--hf-token TOKEN` | Hugging Face token (default: the `HF_TOKEN` environment variable). |
| `--api-key KEY` | Require `Authorization: Bearer KEY` on every request except `/health`, and turn off `/docs`, `/redoc` and `/openapi.json` unless `--docs` is given (default: the `TRANSCRIBE_API_KEY` environment variable; unset or empty means no key). |
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
