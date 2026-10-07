# Development

```bash
pip install -r requirements-dev.txt   # enough for the unit tests: models are mocked
pip install -e ".[all]"               # optional: run real models from the checkout
pre-commit install
pytest                                # 80% coverage required
```

See [CONTRIBUTING.md](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/CONTRIBUTING.md) for:
- the checks to run before opening a pull request;
- how roadmap work is split into parallel issues: branches, changelog fragments, files that take one open PR at a time.

## Project layout

```
speech_toolkit/
├── api.py          Transcriber (loads diarization, then the model, once) and the one-shot transcribe()
├── cli.py          the transcribe command
├── diarization.py  pyannote pipeline loading, diarization and merging speakers into segments
├── formats.py      txt/srt/vtt/json rendering and subtitle line splitting
├── media.py        folder scanning, ffmpeg decoding, splitting long audio in pauses
├── server.py       transcribe-server, the OpenAI-compatible API
├── convert.py      the ogg2wav command
└── backends/       the backend registry, the TranscriptionBackend base class and one module per backend
docs/               user documentation (one page per topic)
changelog.d/        changelog fragments, assembled into CHANGELOG.md at release time
tests/              unit tests; heavy model libraries are mocked
```

`main.py` and `convert.py` at the top level are shims for running the two commands from a checkout; they aren't packaged.

## Releases

Pushing a `v*` tag runs `.github/workflows/release.yml`, which publishes to PyPI with trusted publishing and creates a
GitHub release. Bump `speech_toolkit.__version__` first.
