# Development

```bash
pip install -r requirements-dev.txt   # enough for the unit tests: models are mocked
pip install -e ".[all]"               # optional: run real models from the checkout
pre-commit install
pytest --cov-fail-under=95            # what CI runs: fails below 95% coverage
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

## Integration tests and release QA

`pytest -m integration` runs real models on the speech samples in `tests/data`; plain `pytest` skips these tests.
- `INTEGRATION_DEVICE` picks the device: `auto`, `cpu` or `cuda`.
- `INTEGRATION_TIER=full` adds the larger models.
- Each run ends with a report, `integration-report.md`.

[Manual testing](manual-testing.md) explains these, and is the QA pass the maintainer runs before each release, on
Windows with an NVIDIA GPU and on a Mac, between the TestPyPI dry run and the tag.

## Releases

Pushing a `v*` tag runs `.github/workflows/release.yml`, which publishes to PyPI with trusted publishing and creates a
GitHub release. Bump `speech_toolkit.__version__` first.
