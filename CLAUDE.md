# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Offline speech-to-text CLI with pluggable backends (Whisper, faster-whisper, Voxtral), optional speaker diarization (pyannote.audio) and txt/srt/vtt/json output. No cloud APIs.

## Common Commands

```bash
pip install -r requirements-dev.txt   # enough for unit tests (model libraries are mocked)
pip install -r requirements.txt       # real Whisper runs; extras: requirements-{faster-whisper,voxtral,diarize}.txt
pre-commit install

pytest                                           # all tests, 80% coverage gate
pytest tests/test_backends.py::TestWhisperBackend -v

black . && isort --profile black . && flake8 .   # format and lint (120-char lines)
mypy main.py convert.py formats.py media.py backends/
bandit -c pyproject.toml -r .
```

## Architecture

```
main.py        CLI: parse_args → plan_outputs → run_jobs (load_backend, transcribe_file, write_output)
               diarization: load_diarization_pipeline, diarize_audio, merge_diarization (max-overlap)
formats.py     render(result, fmt) for txt/srt/vtt/json; format_for_path picks the format from -o
media.py       collect_files(): expand folders into (file, relative path) pairs
convert.py     OGG/Opus → 16-bit WAV via pydub/ffmpeg (mirrors folder trees under --outdir)
backends/
├── __init__.py                 registry: list_backends(), get_backend(), register_backend()
├── base.py                     TranscriptionBackend ABC, TranscriptionResult
├── whisper_backend.py          openai-whisper (default)
├── faster_whisper_backend.py   faster-whisper / CTranslate2
└── voxtral_backend.py          Mistral Voxtral via transformers (30 s chunks → segments)
```

**Adding a backend:** subclass `TranscriptionBackend`, implement `available_models()`, `load_model()`, `transcribe()`, and register it in `backends/__init__.py`.

## Key Design Rules

- **Lazy heavy imports:** torch, whisper, transformers, faster_whisper and pyannote are imported inside `load_model()` / diarization functions, never at module level, so `--help`/`--list-*` stay instant and unit tests run without them.
- **stdout is for the transcript only:** `main()` wraps all work in `redirect_stdout(sys.stderr)`; progress and messages go to stderr.
- **Errors:** backends raise (`ValueError`, `RuntimeError`, `ImportError`, `FileNotFoundError`); only `main()`/`run_jobs()` turn them into `sys.exit`. In batch mode a failing file doesn't stop the others.
- **Hugging Face token:** `--hf-token` / `HUGGINGFACE_TOKEN` are exported as `HF_TOKEN` once at startup.
- **Offline:** `PYANNOTE_METRICS_ENABLED` defaults to `false` (pyannote 4 telemetry).

## Testing Notes

- Mock heavy libraries with `patch.dict(sys.modules, {"whisper": mock, ...})`, not by patching module attributes.
- `tests/test_main.py` drives `main.main([...])` end to end with a `FakeBackend` registered via `monkeypatch`.
- When testing `sys.exit()`, expect `SystemExit` (e.g. `pytest.raises(SystemExit, match=...)`).
- CI: unit tests on Python 3.10–3.13 without torch; the integration job runs real `tiny` models on synthesized speech.

## External Requirements

- ffmpeg on PATH (audio decoding for every backend and convert.py).
- Diarization: accept the terms of `pyannote/speaker-diarization-community-1` on Hugging Face and set `HF_TOKEN`.
