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
- New backend: see [Adding a backend](docs/backends.md#adding-a-backend).

## Changelog and docs

- **Changelog:** add a fragment, `changelog.d/<issue>.<type>.md`, for every change users will notice; see
  [changelog.d/README.md](changelog.d/README.md).
  - Label the PR `skip-changelog` when users won't notice the change (tests, CI, refactoring).
  - Don't edit `CHANGELOG.md`; it is assembled at release time.
- **Docs:** user documentation lives in [docs/](docs/), one page per topic; update the page your change affects. The
  README has only the pitch, install and a quick start.

## Parallel work on roadmap issues

The roadmap ([#12](https://github.com/MysterionRise/speech-transcription-toolkit/issues/12)) is split into issues, each
sized for one person or agent and one pull request. Issues are grouped into waves that can run in parallel.

- **Branch:** start from the latest `main`. Agents name branches `claude/issue-<n>-<slug>`.
- **Draft PR early:** use a Conventional Commits title and `Fixes #<n>`, and fill in the template's Coordination
  section.
- **Scope:** touch only the files the issue lists. Put follow-up ideas in a comment on the issue.
- **Shared files:**
  - `speech_toolkit/server.py`, `speech_toolkit/formats.py` and `.github/workflows/ci.yml` take one open pull request
    at a time; #12 gives the order.
  - Put new tests in new test files where possible.
- **Catching up with `main`:** merge it into your branch. Don't rebase or force-push once the pull request is open.
  Pull requests are squash-merged.
- **Definition of done:**
  - The issue's acceptance criteria are met.
  - `pytest`, `pre-commit run --all-files` and `mypy speech_toolkit` pass.
  - Docs and a changelog fragment are added.
  - CI is green, including Integration Tests.

## Releases

See [docs/development.md](docs/development.md#releases).
