# Installation

You need Python 3.10 or newer, and [ffmpeg](https://ffmpeg.org/download.html) on your `PATH`
(`apt install ffmpeg` / `brew install ffmpeg`). Every backend, speaker diarization and `ogg2wav` use ffmpeg to decode audio.

```bash
pip install speech-transcription-toolkit             # faster-whisper, the default backend; no PyTorch
pip install "speech-transcription-toolkit[whisper]"  # add OpenAI's Whisper (installs PyTorch)
```

To try the latest changes before they're released, install from GitHub:

```bash
pip install "speech-transcription-toolkit @ git+https://github.com/MysterionRise/speech-transcription-toolkit"
```

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

The GPU is used when available (see [GPU](#gpu) for the libraries it needs); force a device with `--device cpu` or
`--device cuda`.

## GPU

Running on an NVIDIA GPU takes NVIDIA's CUDA libraries, which pip doesn't always install. Without them, transcription
runs on the CPU, only slower.

### faster-whisper (the default backend)

faster-whisper runs on [CTranslate2](https://github.com/OpenNMT/CTranslate2), which loads NVIDIA's cuBLAS for CUDA 12
when it uses the GPU. Older CTranslate2 releases also load cuDNN:

| CTranslate2 (`pip show ctranslate2`) | Needs |
|---|---|
| 4.6.3 or newer (what a new install gets) | cuBLAS 12 |
| 4.5.0 to 4.6.2 | cuBLAS 12 and cuDNN 9 |
| 4.0 to 4.4 | cuBLAS 12 and cuDNN 8 (`"nvidia-cudnn-cu12==8.*"`) |

Install them with pip, then tell CTranslate2 where pip put them. It doesn't look in Python's `site-packages` itself.

**Windows** (PowerShell). CTranslate2 finds the libraries through `PATH`:

```powershell
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"   # cuDNN matters only before CTranslate2 4.6.3
$sp = python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
$env:PATH = "$sp\nvidia\cublas\bin;$sp\nvidia\cudnn\bin;$env:PATH"
transcribe talk.mp3
```

`$env:PATH` lasts for this terminal. To keep it, add both folders to your user `PATH` ("Edit environment variables
for your account" in Settings).

**Linux.** CTranslate2 finds the libraries through `LD_LIBRARY_PATH`, which is read when a program starts, so set it in
the shell (or in `~/.bashrc`), not from Python:

```bash
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"   # cuDNN matters only before CTranslate2 4.6.3
sp=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export LD_LIBRARY_PATH="$sp/nvidia/cublas/lib:$sp/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
transcribe talk.mp3
```

**Other ways:**
- Install the [CUDA 12 Toolkit](https://developer.nvidia.com/cuda-downloads) and
  [cuDNN](https://developer.nvidia.com/cudnn-downloads) from NVIDIA. On Windows, CTranslate2 also looks in
  `%CUDA_PATH%\bin`, which the Toolkit's installer sets.
- Use a CUDA 12 `cudnn-runtime` image of [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda) on Docker, which has both.
- See also [faster-whisper's GPU notes](https://github.com/SYSTRAN/faster-whisper#gpu).

**Checking:**
- `python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"` prints how many GPUs CTranslate2 sees.
- Before loading a model on the GPU, faster-whisper checks that these libraries load:
  - If one is missing, it prints a warning naming it and runs on the CPU.
  - With `--device cuda` (`device="cuda"`), it stops with that message instead.
  - `--device cpu` skips the check.

### PyTorch backends (`whisper`, `voxtral`, `nvidia`, `diarize`)

PyTorch ships its own CUDA libraries, but only in its CUDA builds. On Linux, PyPI's `torch` is one. On Windows, PyPI's
`torch` runs on the CPU only, so install torch from PyTorch's CUDA wheel index first, then the extra:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "speech-transcription-toolkit[diarize]"
python -c "import torch; print(torch.cuda.is_available())"   # True: torch sees the GPU
```

[pytorch.org](https://pytorch.org/get-started/locally/) gives the index URL for your system and driver (`cu126`,
`cu128`, …). Without a CUDA build of torch, these backends run on the CPU.
