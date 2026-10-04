# speech-transcription-toolkit

[![CI](https://github.com/MysterionRise/speech-transcription-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/MysterionRise/speech-transcription-toolkit/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Offline speech-to-text from the command line or Python. Pick a backend (Whisper, faster-whisper or Voxtral), optionally label speakers, and get plain text, subtitles or JSON. Your audio never leaves your machine; models download once on first use.

## Install

Needs Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html) (`apt install ffmpeg` / `brew install ffmpeg`).

```bash
pip install speech-transcription-toolkit                    # Whisper (default backend)
pip install "speech-transcription-toolkit[faster-whisper]"  # add the faster-whisper backend
```

Other extras: `[voxtral]`, `[diarize]` (speaker labels) and `[all]`.

## Usage

```bash
transcribe audio.mp3                           # transcript to stdout
transcribe audio.mp3 -o audio.srt              # subtitles; format from extension (txt, srt, vtt, json)
transcribe recordings/ --outdir out -f vtt     # every audio file in a folder
transcribe audio.mp3 -m large-v3 -l de         # another model, known language
transcribe audio.mp3 -t translate -m large-v3  # translate to English (turbo can't translate)
transcribe audio.mp3 -b faster-whisper         # faster, especially on CPU
transcribe --list-backends                     # backends and their models
```

Progress goes to stderr, so `transcribe a.mp3 > a.txt` gives a clean file (`-q` hides progress). See `transcribe --help` for all options.

## Python API

```python
from speech_toolkit import Transcriber, transcribe

result = transcribe("talk.mp3", backend="faster-whisper", model="small")
print(result.text)        # result.segments has start, end and text for each segment
result.save("talk.srt")   # txt, srt, vtt or json, from the extension

transcriber = Transcriber(backend="faster-whisper", model="small")  # load once, reuse
for path in ["a.mp3", "b.mp3"]:
    transcriber.transcribe(path, language="en").save(f"{path}.vtt")
```

## Backends

| Backend | Models (default in bold) | Notes |
|---|---|---|
| `whisper` | tiny … large-v3, **turbo** | OpenAI Whisper |
| `faster-whisper` | tiny … large-v3, distil-\*, **turbo** | Same models, several times faster, no torch needed |
| `voxtral` | **voxtral-mini**, voxtral-small | Mistral Voxtral; transcription only, timestamps per ~30 s |

The GPU is used when available; force it with `--device cpu` or `--device cuda`.

## Speaker labels

1. `pip install "speech-transcription-toolkit[diarize]"`
2. Accept the terms of [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1) and create a [Hugging Face token](https://huggingface.co/settings/tokens).
3. Run:

```bash
export HF_TOKEN=hf_...
transcribe meeting.wav --diarize --num-speakers 3 -o meeting.txt
```

Lines look like `[SPEAKER_00] Hello there.`, and subtitles get the same labels. In Python: `Transcriber(diarize=True).transcribe("meeting.wav", num_speakers=3)`. pyannote's usage telemetry stays off unless you set `PYANNOTE_METRICS_ENABLED=true`.

## Convert OGG/Opus to WAV

```bash
ogg2wav recordings/ --outdir wav --rate 16000 --channels 1
```

## Troubleshooting

- **Out of memory:** use a smaller model (`-m small`) or `-b faster-whisper`.
- **Diarization "could not download":** accept the model terms (step 2) and set `HF_TOKEN`.
- **`ffmpeg` not found:** install it and make sure it is on your `PATH`.

## Development

```bash
pip install -r requirements-dev.txt   # enough for the unit tests: models are mocked
pip install -e ".[all]"               # optional: run real models from the checkout
pre-commit install
pytest                                # 80% coverage required
```

See [CONTRIBUTING.md](CONTRIBUTING.md). MIT licensed.
