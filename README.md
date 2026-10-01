# speech-transcription-toolkit

[![CI](https://github.com/MysterionRise/speech-transcription-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/MysterionRise/speech-transcription-toolkit/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Offline speech-to-text from the command line. Pick a backend (Whisper, faster-whisper or Voxtral), optionally label speakers, and get plain text, subtitles or JSON. Your audio never leaves your machine; models download once on first use.

## Install

Needs Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html) (`apt install ffmpeg` / `brew install ffmpeg`).

```bash
pip install -r requirements.txt                  # Whisper (default backend)
pip install -r requirements-faster-whisper.txt   # optional: faster-whisper backend
pip install -r requirements-voxtral.txt          # optional: Voxtral backend
pip install -r requirements-diarize.txt          # optional: speaker labels
```

## Usage

```bash
python main.py audio.mp3                           # transcript to stdout
python main.py audio.mp3 -o audio.srt              # subtitles; format from extension (txt, srt, vtt, json)
python main.py recordings/ --outdir out -f vtt     # every audio file in a folder
python main.py audio.mp3 -m large-v3 -l de         # another model, known language
python main.py audio.mp3 -t translate -m large-v3  # translate to English (turbo can't translate)
python main.py audio.mp3 -b faster-whisper         # faster, especially on CPU
python main.py --list-backends                     # backends and their models
```

Progress goes to stderr, so `python main.py a.mp3 > a.txt` gives a clean file (`-q` hides progress). See `python main.py --help` for all options.

## Backends

| Backend | Models (default in bold) | Notes |
|---|---|---|
| `whisper` | tiny … large-v3, **turbo** | OpenAI Whisper |
| `faster-whisper` | tiny … large-v3, distil-\*, **turbo** | Same models, several times faster, no torch needed |
| `voxtral` | **voxtral-mini**, voxtral-small | Mistral Voxtral; transcription only, timestamps per ~30 s |

The GPU is used when available; force it with `--device cpu` or `--device cuda`.

## Speaker labels

1. `pip install -r requirements-diarize.txt`
2. Accept the terms of [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1) and create a [Hugging Face token](https://huggingface.co/settings/tokens).
3. Run:

```bash
export HF_TOKEN=hf_...
python main.py meeting.wav --diarize --num-speakers 3 -o meeting.txt
```

Lines look like `[SPEAKER_00] Hello there.`, and subtitles get the same labels. pyannote's usage telemetry stays off unless you set `PYANNOTE_METRICS_ENABLED=true`.

## Convert OGG/Opus to WAV

```bash
python convert.py recordings/ --outdir wav --rate 16000 --channels 1
```

## Troubleshooting

- **Out of memory:** use a smaller model (`-m small`) or `-b faster-whisper`.
- **Diarization "could not download":** accept the model terms (step 2) and set `HF_TOKEN`.
- **`ffmpeg` not found:** install it and make sure it is on your `PATH`.

## Development

```bash
pip install -r requirements-dev.txt   # enough for the unit tests: models are mocked
pre-commit install
pytest                                # 80% coverage required
```

See [CONTRIBUTING.md](CONTRIBUTING.md). MIT licensed.
