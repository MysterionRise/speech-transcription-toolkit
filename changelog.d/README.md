# Changelog fragments

Every pull request with a user-visible change adds one small file here. At release time,
[towncrier](https://towncrier.readthedocs.io/) collects the files into `CHANGELOG.md` and deletes them. Never edit
`CHANGELOG.md` directly: with several pull requests in flight, everyone would edit the same lines.

**Name:** `<issue>.<type>.md`, for example `19.fixed.md`. Use the issue the PR fixes, or the PR number if there's no
issue.

| Type | For |
|---|---|
| `added` | New features |
| `changed` | Changes to existing behaviour |
| `deprecated` | Features that will be removed |
| `removed` | Removed features |
| `fixed` | Bug fixes |
| `security` | Security fixes |

Two changes of the same type for the same issue go in `19.fixed.1.md` and `19.fixed.2.md`.

**Content:**
- One or two sentences for users: what changed and what they need to do.
- Start breaking changes with `**Breaking:**`.

```markdown
WebVTT subtitles now escape `&`, `<` and `>`, so cues with these characters are valid.
```

**Skipping a fragment:** changes users don't see, such as tests, CI or refactoring, can skip the fragment by labelling
the pull request `skip-changelog`. The changelog check workflow also skips Dependabot.

**Preview** what the next release's notes will look like:

```bash
pip install towncrier
towncrier build --draft --version 0.4.0
```
