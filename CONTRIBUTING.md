# Contributing

Bug reports and pull requests are welcome.

## Setup

```bash
git clone https://github.com/MysterionRise/speech-transcription-toolkit.git
cd speech-transcription-toolkit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # unit tests mock the models, no torch needed
pip install -e ".[all]"               # optional: run real models from the checkout
pre-commit install
```

## Before opening a PR

```bash
pytest                        # all tests pass, coverage at least 80%
pre-commit run --all-files    # black, isort, flake8, mypy, bandit
```

- Add tests for new behaviour and keep PRs focused.
- Use [Conventional Commits](https://www.conventionalcommits.org/) (`fix: …`, `feat: …`).
- New backend: subclass `TranscriptionBackend` in `speech_toolkit/backends/`, import heavy libraries inside `load_model()`, register it in `speech_toolkit/backends/__init__.py`, and add its packages as an extra in `pyproject.toml`.
