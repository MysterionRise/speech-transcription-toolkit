# syntax=docker/dockerfile:1
#
# Images for transcribe-server and the transcribe command; docs/docker.md explains how to run them.
#
#   docker build --target cpu -t speech-transcription-toolkit .        # faster-whisper on the CPU, no PyTorch
#   docker build --target cuda -t speech-transcription-toolkit:cuda .  # the same, plus cuBLAS for NVIDIA GPUs
#
# Stages: "runtime" (Python, ffmpeg and the user), "build" (the virtualenv), then the two targets. "cuda" is "cpu" plus
# one layer, so the two images share every other layer.

ARG PYTHON_VERSION=3.12
ARG DEBIAN_RELEASE=trixie

FROM python:${PYTHON_VERSION}-slim-${DEBIAN_RELEASE} AS runtime

# ffmpeg decodes audio for every backend, diarization and ogg2wav.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Runs as "app" (uid 1000), not root. /models holds the downloaded models; /data is the working folder, where the
# docs mount the audio.
RUN useradd --uid 1000 --user-group --create-home --shell /usr/sbin/nologin app \
    && mkdir /models /data \
    && chown app:app /models /data


FROM python:${PYTHON_VERSION}-slim-${DEBIAN_RELEASE} AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /src

# The dependencies (core and the server extra) first, in a layer that only pyproject.toml changes rebuild.
COPY pyproject.toml ./
RUN python -c 'import tomllib; p = tomllib.load(open("pyproject.toml", "rb"))["project"]; \
print("\n".join(p["dependencies"] + p["optional-dependencies"]["server"]))' > requirements.txt \
    && pip install -r requirements.txt

COPY README.md LICENSE MANIFEST.in ./
COPY speech_toolkit ./speech_toolkit
RUN pip install --no-deps . \
    && pip check


# faster-whisper (CTranslate2) on the CPU, and transcribe-server: no PyTorch.
FROM runtime AS cpu

LABEL org.opencontainers.image.title="speech-transcription-toolkit" \
      org.opencontainers.image.description="Offline speech-to-text: transcribe and an OpenAI-compatible transcribe-server" \
      org.opencontainers.image.source="https://github.com/MysterionRise/speech-transcription-toolkit" \
      org.opencontainers.image.documentation="https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/docs/docker.md" \
      org.opencontainers.image.licenses="MIT"

# Read-only for the app user: it owns only /models and /data.
COPY --from=build /opt/venv /opt/venv

# HF_HOME: models download to /models, a volume, so they survive the container.
ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/models \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER 1000:1000
WORKDIR /data
VOLUME /models
EXPOSE 8000

# The server answers /health once its model has loaded, which on the first start includes the download (1.6 GB for
# the default model, turbo), hence the long start period. /health needs no API key. No proxy: a container started
# with HTTP_PROXY set would send the check there.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30m --start-interval=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.build_opener(u.ProxyHandler({})).open('http://127.0.0.1:8000/health', timeout=8)"]

CMD ["transcribe-server", "--host", "0.0.0.0"]


# The cpu image plus cuBLAS for CUDA 12, for NVIDIA GPUs. From 4.6.3, CTranslate2 needs no other CUDA library: it
# links the CUDA runtime statically, and the NVIDIA Container Toolkit adds the driver (docker run --gpus all).
FROM cpu AS cuda
ARG PYTHON_VERSION

USER root
RUN pip install --no-cache-dir --disable-pip-version-check "nvidia-cublas-cu12==12.*" "ctranslate2>=4.6.3,<5" \
    && pip check
USER 1000:1000

# CTranslate2 loads libcublas.so.12 through the library search path, and cuBLAS loads libnvrtc.so.12 (its pip
# dependency) the same way; pip puts both in site-packages.
ENV LD_LIBRARY_PATH=/opt/venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/cublas/lib:/opt/venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/cuda_nvrtc/lib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
