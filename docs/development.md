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
├── errors.py       the exceptions (SpeechToolkitError and its subclasses) and SpeechToolkitWarning
├── formats.py      txt/srt/vtt/json rendering and subtitle line splitting
├── media.py        folder scanning, ffmpeg decoding, splitting long audio in pauses
├── server.py       transcribe-server, the OpenAI-compatible API
├── types.py        the Segment and Word TypedDicts, and the JSON schema version
├── convert.py      the ogg2wav command
└── backends/       the backend registry, the TranscriptionBackend base class, option and device parsing
                    (config.py), the shared transformers loader (_transformers.py) and one module per backend
docs/               user documentation (one page per topic), built into a site by MkDocs (mkdocs.yml)
examples/           runnable scripts; tests/test_examples.py runs them against the fake backend
benchmarks/         the WER and speed harness, python -m benchmarks.run (see benchmarks/README.md)
changelog.d/        changelog fragments, assembled into CHANGELOG.md at release time
tests/              unit tests, with heavy model libraries mocked; tests/integration runs real models
Dockerfile          the cpu and cuda images (see docs/docker.md)
```

The docs site builds with `pip install -r requirements-docs.txt` and `mkdocs build --strict`; a new page in `docs/` must
also be added to `nav` in `mkdocs.yml`.

`main.py` and `convert.py` at the top level are shims for running the two commands from a checkout; they aren't packaged.

## Integration tests and release QA

`pytest -m integration` runs real models on the speech samples in `tests/data`; plain `pytest` skips these tests.
- `INTEGRATION_DEVICE` picks the device: `auto`, `cpu` or `cuda`.
- `INTEGRATION_TIER=full` adds the larger models.
- Each run ends with a report, `integration-report.md`.

[Manual testing](manual-testing.md) explains these, and is the QA pass the maintainer runs before each release, on
Windows with an NVIDIA GPU and on a Mac, between the TestPyPI dry run and the tag.

## Releases

[RELEASING.md](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/RELEASING.md) has the steps:
1. A release pull request bumps `speech_toolkit.__version__` and runs `towncrier build`.
2. After the merge, the maintainer runs the TestPyPI dry run and the [QA pass](manual-testing.md), pushes a `v*` tag
   and approves the `pypi` environment.

`.github/workflows/release.yml` runs CI on the tagged commit and builds the package once. It then publishes with trusted
publishing, and creates the GitHub release with the version's `CHANGELOG.md` section as its notes.
