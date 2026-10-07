# speech-transcription-toolkit

[![CI](https://github.com/MysterionRise/speech-transcription-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/MysterionRise/speech-transcription-toolkit/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/LICENSE)

Offline speech-to-text from the command line or Python.
- Pick a backend: Whisper, faster-whisper, Voxtral, or NVIDIA's Parakeet and Canary.
- Optionally label speakers.
- Get plain text, subtitles or JSON.

Your audio never leaves your machine; models download once, on first use.

## Install

Needs Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html) (`apt install ffmpeg` / `brew install ffmpeg`).

```bash
pip install speech-transcription-toolkit                    # Whisper (default backend)
pip install "speech-transcription-toolkit[faster-whisper]"  # add the faster-whisper backend
```

The first PyPI release is on its way. Until then, install from GitHub:
`pip install "speech-transcription-toolkit @ git+https://github.com/MysterionRise/speech-transcription-toolkit"`.

Other extras: `[nvidia]` (Parakeet, Canary), `[voxtral]`, `[diarize]` (speaker labels), `[server]` (OpenAI-compatible API) and
`[all]`. See [Installation](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/docs/install.md).

## Quick start

```bash
transcribe audio.mp3                           # transcript to stdout
transcribe audio.mp3 -o audio.srt              # subtitles; format from extension (txt, srt, vtt, json)
transcribe recordings/ --outdir out -f vtt     # every audio file in a folder
transcribe audio.mp3 -b faster-whisper         # faster, especially on CPU
transcribe meeting.wav --diarize -o notes.txt  # label speakers (needs the diarize extra and HF_TOKEN)
transcribe-server -b faster-whisper            # OpenAI-compatible API on http://127.0.0.1:8000/v1
```

```python
from speech_toolkit import Transcriber, transcribe

result = transcribe("talk.mp3", backend="faster-whisper", model="small")
print(result.text)        # result.segments has start, end and text for each segment
result.save("talk.srt")   # txt, srt, vtt or json, from the extension

transcriber = Transcriber(backend="faster-whisper", model="small")  # load once, reuse
for path in ["a.mp3", "b.mp3"]:
    transcriber.transcribe(path, language="en").save(f"{path}.vtt")
```

## Documentation

Every page lives in [docs/](https://github.com/MysterionRise/speech-transcription-toolkit/tree/main/docs):
- [Command line](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/docs/cli.md): every `transcribe` option, batch mode and `ogg2wav`.
- [Python API](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/docs/python-api.md): `Transcriber`, results and custom backends.
- [Backends](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/docs/backends.md): models, what each backend supports, devices.
- [Speaker labels](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/docs/diarization.md): diarization with pyannote.audio.
- [Output formats](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/docs/formats.md): txt, srt, vtt and json.
- [OpenAI-compatible server](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/docs/server.md): `transcribe-server`.
- [Troubleshooting](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/docs/troubleshooting.md)

## Contributing

See [CONTRIBUTING.md](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/CONTRIBUTING.md) and
[Development](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/docs/development.md). The roadmap is
tracked in [#12](https://github.com/MysterionRise/speech-transcription-toolkit/issues/12).

MIT licensed.
