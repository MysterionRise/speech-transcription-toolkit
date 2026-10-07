# Installation

You need Python 3.10 or newer, and [ffmpeg](https://ffmpeg.org/download.html) on your `PATH`
(`apt install ffmpeg` / `brew install ffmpeg`). Every backend, speaker diarization and `ogg2wav` use ffmpeg to decode audio.

```bash
pip install speech-transcription-toolkit                    # Whisper (default backend)
pip install "speech-transcription-toolkit[faster-whisper]"  # add the faster-whisper backend
```

> **Not on PyPI yet.** The first release is tracked in
> [#28](https://github.com/MysterionRise/speech-transcription-toolkit/issues/28). Until then, install from GitHub:
>
> ```bash
> pip install "speech-transcription-toolkit @ git+https://github.com/MysterionRise/speech-transcription-toolkit"
> pip install "speech-transcription-toolkit[faster-whisper] @ git+https://github.com/MysterionRise/speech-transcription-toolkit"
> ```

The default backend, openai-whisper, runs on PyTorch, so a plain install also installs torch.

## Extras

| Extra | Adds |
|---|---|
| `faster-whisper` | The faster-whisper backend (CTranslate2) |
| `voxtral` | Mistral Voxtral (transformers, mistral-common) |
| `nvidia` | NVIDIA Parakeet and Canary (transformers 5.18+, librosa) |
| `diarize` | Speaker labels with pyannote.audio (see [Speaker labels](diarization.md)) |
| `server` | `transcribe-server`, the OpenAI-compatible API (FastAPI, uvicorn) |
| `all` | All of the above |

## Models and offline use

Models download the first time you use them:
- Whisper models from OpenAI's servers, cached in `~/.cache/whisper`.
- All other models from the Hugging Face Hub, cached in `~/.cache/huggingface`.

After that, transcription runs offline: your audio never leaves your machine.

The GPU is used when available; force a device with `--device cpu` or `--device cuda`.
