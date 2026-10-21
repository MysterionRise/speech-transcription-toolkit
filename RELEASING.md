# Releasing

Releases go to PyPI through [`.github/workflows/release.yml`](.github/workflows/release.yml), with
[trusted publishing](https://docs.pypi.org/trusted-publishers/): PyPI trusts that workflow, so no API token is stored
anywhere.

## What the release workflow does

| Started by | Publishes to | GitHub release |
|---|---|---|
| **Run workflow** (the dry run) | TestPyPI | None |
| A pre-release tag, such as `v0.4.0rc1` | TestPyPI | A pre-release |
| A final release tag, such as `v0.4.0` | PyPI, once you approve the `pypi` environment | The release |

Each run has these jobs:
1. **CI:** all of `ci.yml`, Integration Tests included, on the commit being released.
2. **Build:**
   - Builds the wheel and sdist once, with the tools pinned in [`requirements-release.txt`](requirements-release.txt),
     and runs `twine check --strict`.
   - Checks that a tag is `v` plus the package version: `speech_toolkit.__version__`, written as PyPI writes it
     (`v0.4.0rc1`, not `v0.4.0-rc1`).
   - Takes the version's `CHANGELOG.md` section as the release notes and shows them in the run's summary. A final
     release tag without a section stops here.
3. **Publish:** uploads exactly the files the build job made. The publish jobs get no permission except the OIDC token
   that trusted publishing needs.
4. **GitHub release** (tags only): created after the upload, with the release notes and the wheel and sdist attached.

So nothing is published if CI fails, if the tag doesn't match the version, or if a final release has no changelog
section.

## One-time setup

The maintainer does this once; it is part of the checklist in
[#13](https://github.com/MysterionRise/speech-transcription-toolkit/issues/13).
- **PyPI:** add a pending trusted publisher:
  - project `speech-transcription-toolkit`
  - owner `MysterionRise`
  - repository `speech-transcription-toolkit`
  - workflow `release.yml`
  - environment `pypi`
- **TestPyPI:** the same on [test.pypi.org](https://test.pypi.org/), with environment `testpypi`.
- **GitHub, Settings → Environments:**
  - Create `pypi`, with yourself as a required reviewer, and `testpypi`.
  - Optionally, under "Deployment branches and tags", limit `pypi` to tags matching `v*`.

## Making a release

Steps 1 and 2 are an ordinary pull request (the release issue, such as
[#28](https://github.com/MysterionRise/speech-transcription-toolkit/issues/28)). The maintainer does the rest.

1. **Bump the version** in `speech_toolkit/__init__.py`, such as `__version__ = "0.4.0"`.
   - Write it the way PyPI shows versions ([PEP 440](https://peps.python.org/pep-0440/)): `0.4.0`, `0.4.0rc1` or
     `0.4.0.post1`.
   - The tag must match it.
2. **Collect the changelog** and open the release pull request:
   ```bash
   pip install towncrier
   towncrier build --version 0.4.0   # --yes skips the question about deleting the fragments
   ```
   - This writes the `## [0.4.0](...) - <date>` section into `CHANGELOG.md` and deletes the fragments in `changelog.d/`.
   - Read the section over: it becomes the release notes.
   - The Changelog check passes on its own because the pull request changes `CHANGELOG.md`.
3. **Dry run on TestPyPI**, after the merge.
   - Start it with **Actions → Release → Run workflow** on `main`, or with `gh workflow run release.yml --ref main`.
     It runs CI, builds, and uploads to TestPyPI.
   - Check that the run's summary shows the release notes you expect.
   - Check that the package installs from TestPyPI, with its dependencies from PyPI:
     ```bash
     python -m venv /tmp/dry-run && . /tmp/dry-run/bin/activate
     pip download --no-deps --only-binary :all: --index-url https://test.pypi.org/simple/ --dest /tmp/dry-run-dist \
         speech-transcription-toolkit==0.4.0
     pip install /tmp/dry-run-dist/*.whl   # its dependencies come from PyPI
     transcribe --version
     ```
   - TestPyPI, like PyPI, takes each version only once, so a second dry run of the same version fails at the upload.
     To rehearse again after a fix, release a candidate: set `__version__ = "0.4.0rc1"`, merge, and push the tag
     `v0.4.0rc1`. It goes to TestPyPI and becomes a GitHub pre-release.
4. **Tag the release** on the merged commit, and push the tag:
   ```bash
   git switch main && git pull
   git tag -a v0.4.0 -m "v0.4.0"
   git push origin v0.4.0
   ```
   - The workflow runs CI and builds again, then waits for you.
   - Open the run and approve the `pypi` deployment (**Review deployments**).
   - After the upload, it creates the GitHub release.
5. **Verify** in a clean virtualenv:
   ```bash
   python -m venv /tmp/release-check && . /tmp/release-check/bin/activate
   pip install speech-transcription-toolkit==0.4.0
   transcribe --version
   ```
   Then run the checks the release issue lists, such as transcribing a sample.

## If something goes wrong

- **The run stopped before publishing** (red CI, a tag that doesn't match, no changelog section): nothing was uploaded.
  Fix the problem on `main`, then move the tag:
  ```bash
  git push --delete origin v0.4.0 && git tag -d v0.4.0
  ```
  Then tag again, as in step 4.
- **The upload failed** (for example, the trusted publisher isn't set up yet): fix the cause, then use **Re-run failed
  jobs** on the run. The build job's files are reused.
- **The package is published, but the GitHub release failed:** re-run the failed job.
- **A broken release reached PyPI:** PyPI never takes the same version twice, even after you delete it. Yank the
  release on PyPI and publish a fixed version, such as `0.4.1`.
