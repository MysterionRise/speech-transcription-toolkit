# Contributing

Bug reports and pull requests are welcome.

## Setup

```bash
git clone https://github.com/MysterionRise/speech-transcription-toolkit.git
cd speech-transcription-toolkit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # unit tests mock the models, no torch needed
pre-commit install
```

## Before opening a PR

```bash
pytest                        # all tests pass, coverage at least 80%
pre-commit run --all-files    # black, isort, flake8, mypy, bandit
```

- Add tests for new behaviour and keep PRs focused.
- Use [Conventional Commits](https://www.conventionalcommits.org/) (`fix: …`, `feat: …`).
- New backend: subclass `TranscriptionBackend` in `backends/`, import heavy libraries inside `load_model()`, and register it in `backends/__init__.py`.
