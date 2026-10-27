# Docker

Two images run `transcribe-server` and `transcribe` without a Python setup. Both have the faster-whisper backend, the
server and ffmpeg, but no PyTorch:

| Image | Runs on | Size, unpacked |
|---|---|---|
| `ghcr.io/mysterionrise/speech-transcription-toolkit:latest` | The CPU | about 1 GB |
| `ghcr.io/mysterionrise/speech-transcription-toolkit:cuda` | NVIDIA GPUs, or the CPU: the CPU image plus cuBLAS (see [GPU](#gpu)) | about 2 GB |

[Image tags](#image-tags) lists every tag. The images are published from v0.5.0.

## Run the server

```bash
docker run -d --name transcribe -p 127.0.0.1:8000:8000 -v speech-models:/models \
  ghcr.io/mysterionrise/speech-transcription-toolkit transcribe-server --host 0.0.0.0 -m small
docker logs -f transcribe            # the first start downloads the model
curl http://127.0.0.1:8000/health    # {"status":"ok","backend":"faster-whisper","model":"small"}
```

Clients then use `http://127.0.0.1:8000/v1`, as in [OpenAI-compatible server](server.md):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
print(client.audio.transcriptions.create(model="whisper-1", file=open("talk.mp3", "rb")).text)
```

**How it behaves:**
- **Command:** without one, the image runs `transcribe-server --host 0.0.0.0` with the default model, turbo (a 1.6 GB
  download, and slow on a CPU). A command after the image name replaces it, so repeat `transcribe-server --host
  0.0.0.0` before options such as `-m small`. [Options](server.md#options) lists them.
- **Health:** `docker ps` shows the container as `healthy` once the model has loaded and the server answers `/health`.
  On the first start the model downloads first: the container has 30 minutes for that before it counts as unhealthy.
- **Port:** the health check calls port 8000 in the container. To use another port, change the host side only:
  `-p 127.0.0.1:9000:8000`.

## API key

Inside the container the server listens on every address (`--host 0.0.0.0`), since that is how the published port
reaches it. So who can use it depends on `-p`:
- `-p 127.0.0.1:8000:8000` publishes the port on this machine only. The server still warns at start that it runs
  without an API key, because it can't see how the port is published.
- `-p 8000:8000` publishes it on every network the machine is on. Set an API key first. On Linux, Docker's published
  ports bypass firewalls such as ufw.

Set the key with `TRANSCRIBE_API_KEY`, from a file that keeps it out of your shell history:

```bash
echo "TRANSCRIBE_API_KEY=$(openssl rand -hex 32)" > transcribe.env
docker run -d --name transcribe -p 8000:8000 --env-file transcribe.env -v speech-models:/models \
  ghcr.io/mysterionrise/speech-transcription-toolkit transcribe-server --host 0.0.0.0 -m small
```

- Clients send it as `Authorization: Bearer <key>`; with the OpenAI SDK, `api_key="<key>"`.
- `/health` needs no key, so the health check works with one set.
- For no key, leave the variable out: `TRANSCRIBE_API_KEY=` with no value sets an empty key.
- [Security](server.md#security) has the rest: upload limits, and the API docs, which a key turns off.

## Transcribe files

`/data` is the working folder, so mount the folder with your audio there:

```bash
docker run --rm -v "$PWD:/data" -v speech-models:/models \
  ghcr.io/mysterionrise/speech-transcription-toolkit transcribe -m small talk.mp3 > talk.txt
docker run --rm -v "$PWD:/data" -v speech-models:/models \
  ghcr.io/mysterionrise/speech-transcription-toolkit transcribe -m small recordings/ -f srt --outdir subtitles
cat talk.mp3 | docker run --rm -i -v speech-models:/models \
  ghcr.io/mysterionrise/speech-transcription-toolkit transcribe -m small - > talk.txt
```

- [Command line](cli.md) has every option. Progress goes to stderr, so `> talk.txt` gets the transcript only.
- The health check is the server's: a `transcribe` run longer than 30 minutes shows as unhealthy. `docker run
  --no-healthcheck` turns it off.

**File ownership.** The images run as the user `app`, uid 1000, not root. Writing files into a mounted folder, such
as the subtitles above, needs that folder to be writable by uid 1000 (on Linux; Docker Desktop on macOS and Windows
maps the owner). Otherwise run as yourself, and give the models a folder you own too, such as your Hugging Face
cache:

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/data" -v "$HOME/.cache/huggingface:/models" \
  ghcr.io/mysterionrise/speech-transcription-toolkit transcribe -m small recordings/ -f srt
```

## Models and offline use

- **Where:** models download from the Hugging Face Hub into `/models` (the images set `HF_HOME=/models`).
- **Keep them:** mount a named volume, `-v speech-models:/models`, and each model downloads once for every container
  that mounts it. Without it, each container downloads its own copy, which `docker run --rm` deletes with the
  container. `docker volume rm speech-models` frees the space.
- **Offline:** once a model is in the volume, nothing needs the network. Set `HF_HUB_OFFLINE=1` so that nothing tries
  to reach the Hub either; the CLI then runs without a network:

  ```bash
  docker run --rm --network none -e HF_HUB_OFFLINE=1 -v "$PWD:/data" -v speech-models:/models \
    ghcr.io/mysterionrise/speech-transcription-toolkit transcribe -m small talk.mp3
  ```

- **Private or gated models:** pass a Hugging Face token with `-e HF_TOKEN`.

## GPU

The `cuda` image runs faster-whisper on an NVIDIA GPU. It needs:
- an NVIDIA driver for CUDA 12 or newer (`nvidia-smi` shows "CUDA Version: 12.x" or higher);
- the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  or Docker Desktop with WSL 2 on Windows.

```bash
docker run -d --name transcribe --gpus all -p 127.0.0.1:8000:8000 -v speech-models:/models \
  ghcr.io/mysterionrise/speech-transcription-toolkit:cuda
docker run --rm --gpus all ghcr.io/mysterionrise/speech-transcription-toolkit:cuda \
  python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"   # GPUs CTranslate2 sees: 1 or more
```

- The image is the CPU image plus NVIDIA's cuBLAS for CUDA 12 (and NVRTC, which cuBLAS uses). From version 4.6.3,
  cuBLAS is the only CUDA library CTranslate2 loads: it has the CUDA runtime built in, and the toolkit adds the driver
  (`--gpus all`). There is no cuDNN.
- The server and `transcribe` use the GPU when they see one. Without `--gpus`, they run on the CPU; with
  `--device cuda`, they stop with an error instead.
- On the GPU the default model, turbo, is fast, so the command above keeps it.
- [GPU](install.md#gpu) explains the libraries, and [Troubleshooting](troubleshooting.md) the messages about them.

## Docker Compose

The repository's [docker-compose.yml](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/docker-compose.yml)
runs the CPU server:
- with `-m small`, on port 8000 of this machine only (`127.0.0.1`);
- with the models in a volume, which `docker compose down` keeps;
- with `TRANSCRIBE_API_KEY` from a `.env` file next to it, if there is one;
- restarting it unless you stop it.

```bash
echo "TRANSCRIBE_API_KEY=$(openssl rand -hex 32)" > .env   # optional; .gitignore keeps it out of git
docker compose up -d
docker compose logs -f transcribe
```

In a checkout, `docker compose up` builds the image when it can't pull it, and `docker compose up --build` rebuilds
it from your changes. For a GPU, its comments show what to change: the `cuda` image, and a `deploy` section that
reserves the GPU.

## Image tags

| Tag | Image |
|---|---|
| `latest`, `<version>` (such as `0.5.0`) | CPU |
| `cuda`, `<version>-cuda` (such as `0.5.0-cuda`) | CUDA 12 |

- Each release tag publishes them, from `.github/workflows/docker.yml`. A pre-release, such as `0.5.0rc1`, gets only
  the version tags.
- Pin a version in production: `latest` and `cuda` move with each release.
- The images are for `linux/amd64`.
- Inside: Python 3.12 on Debian 13 (trixie) slim, ffmpeg from Debian, and the package with its `server` extra, in
  `/opt/venv`. The other backends and diarization need PyTorch and aren't included; see [Installation](install.md).

## Build the images yourself

From a checkout:

```bash
docker build --target cpu -t speech-transcription-toolkit .
docker build --target cuda -t speech-transcription-toolkit:cuda .
```

Only `pyproject.toml`, `README.md`, `LICENSE`, `MANIFEST.in` and `speech_toolkit/` go into the build
(`.dockerignore`).
