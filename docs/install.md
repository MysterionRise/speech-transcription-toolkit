# Installation

You need Python 3.10 or newer, and [ffmpeg](https://ffmpeg.org/download.html) on your `PATH`
(`apt install ffmpeg` / `brew install ffmpeg`). Every backend, speaker diarization and `ogg2wav` use ffmpeg to decode audio.

```bash
pip install speech-transcription-toolkit             # faster-whisper, the default backend; no PyTorch
pip install "speech-transcription-toolkit[whisper]"  # add OpenAI's Whisper (installs PyTorch)
```

> **Not on PyPI yet.** The first release is tracked in
> [#28](https://github.com/MysterionRise/speech-transcription-toolkit/issues/28). Until then, install from GitHub:
>
> ```bash
> pip install "speech-transcription-toolkit @ git+https://github.com/MysterionRise/speech-transcription-toolkit"
> pip install "speech-transcription-toolkit[whisper] @ git+https://github.com/MysterionRise/speech-transcription-toolkit"
> ```

The default backend, faster-whisper, runs on [CTranslate2](https://github.com/OpenNMT/CTranslate2) instead of PyTorch,
so a plain install is light: no torch, which takes several GB on Linux. The `whisper`, `voxtral`, `nvidia` and `diarize`
extras install PyTorch.

## Extras

| Extra | Adds |
|---|---|
| `whisper` | OpenAI Whisper, `-b whisper` (openai-whisper, PyTorch) |
| `voxtral` | Mistral Voxtral (PyTorch, transformers, mistral-common) |
| `nvidia` | NVIDIA Parakeet and Canary (PyTorch, transformers 5.18+, librosa) |
| `diarize` | Speaker labels with pyannote.audio (see [Speaker labels](diarization.md)) |
| `server` | `transcribe-server`, the OpenAI-compatible API (FastAPI, uvicorn) |
| `all` | All of the above |

`faster-whisper` is also accepted as an extra, so older instructions keep working; it adds nothing, since every install
includes faster-whisper.

## Models and offline use

Models download the first time you use them:
- faster-whisper's and the other backends' models from the Hugging Face Hub, cached in `~/.cache/huggingface`.
- Whisper's (`-b whisper`) from OpenAI's servers, cached in `~/.cache/whisper`.

After that, transcription runs offline: your audio never leaves your machine.

The GPU is used when available; force a device with `--device cpu` or `--device cuda`.
