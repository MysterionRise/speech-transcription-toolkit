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

## Options

| Option | Description |
|---|---|
| `--host` | Address to listen on (default: `127.0.0.1`, this machine only). |
| `--port` | Port (default: 8000). |
| `-b`, `--backend`, `-m`, `--model`, `--device` | As for `transcribe`; see [Command line](cli.md). |
| `--diarize` | Label speakers; needs the `diarize` extra and `HF_TOKEN`. |
| `--hf-token TOKEN` | Hugging Face token (default: the `HF_TOKEN` environment variable). |
| `--api-key KEY` | Require `Authorization: Bearer KEY` on `/v1` requests (default: the `TRANSCRIBE_API_KEY` environment variable; unset means no key). |
| `--max-upload-mb N` | Refuse larger uploads with HTTP 413 (default: 100). |

## Security

- **Bind address.** The server listens on localhost unless you pass `--host`, and warns when it's exposed without an API
  key.
- **Upload limits.** Today the upload size limit and the API key are checked only after the whole upload has been
  received. Until [#21](https://github.com/MysterionRise/speech-transcription-toolkit/issues/21) lands, don't expose the
  server to untrusted networks.
- **Temp files.** Uploads go to temporary files that are always deleted after the request.
