# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Offline speech-to-text library and CLI (`pip install speech-transcription-toolkit`, import name `speech_toolkit`) with pluggable backends (Whisper, faster-whisper, Voxtral, Parakeet, Canary), optional speaker diarization (pyannote.audio) and txt/srt/vtt/json output. No cloud APIs.

## Common Commands

```bash
pip install -r requirements-dev.txt   # enough for unit tests (model libraries are mocked)
pip install -e ".[all]"               # real model runs; extras: faster-whisper, voxtral, nvidia, diarize
pre-commit install

pytest                                           # all tests, 80% coverage gate
pytest tests/test_backends.py::TestWhisperBackend -v

black . && isort --profile black . && flake8 .   # format and lint (120-char lines)
mypy speech_toolkit
bandit -c pyproject.toml -r .
python -m build && twine check --strict dist/*   # package check (CI does this too)
```

## Architecture

```
speech_toolkit/
├── __init__.py     public API: Transcriber, transcribe, TranscriptionResult, list/register_backend, __version__
├── api.py          Transcriber (loads diarization, then the model, once) and the one-shot transcribe()
├── cli.py          `transcribe` command: parse_args → plan_outputs → run_jobs (load_transcriber, write_output)
├── diarization.py  load_diarization_pipeline, diarize_audio, merge_diarization (max-overlap)
├── formats.py      render(result, fmt, max_line_width) for txt/srt/vtt/json; split_cues, format_for_path, write_text
├── media.py        collect_files() for folders; load_audio() (ffmpeg → 16 kHz float32), split_audio() (cuts in pauses)
├── convert.py      `ogg2wav` command: OGG/Opus → 16-bit WAV via pydub/ffmpeg
└── backends/
    ├── __init__.py                 registry: list_backends(), get_backend(), register_backend()
    ├── base.py                     TranscriptionBackend ABC, TranscriptionResult (render, save)
    ├── whisper_backend.py          openai-whisper (default)
    ├── faster_whisper_backend.py   faster-whisper / CTranslate2
    ├── voxtral_backend.py          Mistral Voxtral via transformers (30 s chunks → segments)
    └── nvidia_backend.py           Parakeet TDT (word timings from generate durations) and Canary, via transformers
main.py, convert.py   checkout shims for the two commands (not packaged)
```

Packaging lives in `pyproject.toml` (setuptools; version from `speech_toolkit.__version__`). Pushing a `v*` tag runs `.github/workflows/release.yml` (PyPI trusted publishing + GitHub release); bump `__version__` first.

**Adding a backend:** subclass `TranscriptionBackend`, implement `available_models()`, `load_model()`, `transcribe()`, register it in `speech_toolkit/backends/__init__.py`, and add its packages as an extra in `pyproject.toml`. Optional features (`prompt`, `vad`, `word_timestamps`) go in the class's `capabilities` and are keyword-only `transcribe()` arguments; `Transcriber` passes only declared ones and warns about the rest. Word timings use Whisper's format: `segment["words"] = [{"word": " Hi", "start": 0.0, "end": 0.4}]`.

## Key Design Rules

- **Lazy heavy imports:** torch, whisper, transformers, faster_whisper and pyannote are imported inside `load_model()` / diarization functions, never at module level, so `--help`/`--list-*` stay instant and unit tests run without them.
- **stdout is for the transcript only:** `cli.main()` wraps all work in `redirect_stdout(sys.stderr)`; progress and messages go to stderr. Library warnings (`warnings.warn`) print as one `Warning: …` line.
- **Errors:** the library raises (`ValueError`, `RuntimeError`, `ImportError`, `FileNotFoundError`) and never prints or exits; only `cli.main()`/`run_jobs()` turn errors into `sys.exit`. In batch mode a failing file doesn't stop the others.
- **Hugging Face token:** `--hf-token` / `Transcriber(hf_token=...)` / `HUGGINGFACE_TOKEN` are exported as `HF_TOKEN`.
- **Offline:** `PYANNOTE_METRICS_ENABLED` defaults to `false` (pyannote 4 telemetry).

## Testing Notes

- Mock heavy libraries with `patch.dict(sys.modules, {"whisper": mock, ...})`, not by patching module attributes.
- `tests/conftest.py` has a `FakeBackend` and the `fake_backend` fixture; `tests/test_cli.py` drives `cli.main([...])` end to end with it, `tests/test_api.py` the `Transcriber`.
- When testing `sys.exit()`, expect `SystemExit` (e.g. `pytest.raises(SystemExit, match=...)`).
- CI: unit tests on Python 3.10–3.13 without torch; the integration job runs real `tiny` models on synthesized speech.

## External Requirements

- ffmpeg on PATH (audio decoding for every backend, diarization and `ogg2wav`).
- Diarization: accept the terms of `pyannote/speaker-diarization-community-1` on Hugging Face and set `HF_TOKEN`.
