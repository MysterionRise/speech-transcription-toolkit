# QA Report: speech-transcription-toolkit

**Date:** 2026-02-18
**Scope:** Full codebase audit (source, tests, CI, config)
**Tools used:** pytest, flake8, mypy, bandit, black, isort, manual code review

---

## Executive Summary

- **83 tests pass** (100% pass rate), **98.48% code coverage**
- **1 confirmed functional bug** (data overwrite in `to_dict()`)
- **4 formatting violations** (black)
- **2 security findings** (bandit, medium severity)
- **Multiple configuration and test quality issues**

---

## BUG-01: `TranscriptionResult.to_dict()` silently overwrites standardized fields [CRITICAL]

**File:** `backends/base.py:36-43`
**Type:** Logic bug

The `to_dict()` method unpacks `**self.raw` after the standard keys, allowing raw backend data to overwrite the standardized `text`, `segments`, and `language` fields:

```python
def to_dict(self) -> Dict[str, Any]:
    return {
        "text": self.text,
        "segments": self.segments,
        "language": self.language,
        **self.raw,  # <-- overwrites all above keys if present in raw
    }
```

**Impact:** The Whisper backend passes the entire raw result dict (which contains `text`, `segments`, and `language`) as the `raw` parameter (`whisper_backend.py:115`). This means:
- The carefully standardized segments (lines 96-109) are thrown away
- The segment normalization step is **dead code**
- `to_dict()` returns raw Whisper segments (with extra fields like `seek`) instead of the cleaned ones

**Reproduction:**
```python
result = TranscriptionResult(
    text="standardized", segments=[{"clean": True}],
    language="en", raw={"text": "RAW", "segments": [{"raw": True}], "language": "xx"}
)
d = result.to_dict()
assert d["text"] == "RAW"       # Overwrites "standardized"
assert d["segments"] == [{"raw": True}]  # Overwrites clean segments
```

**Fix:** Either:
- (A) Reverse the dict order: `{**self.raw, "text": self.text, ...}` so standard fields take priority
- (B) Filter conflicting keys from `raw` before merging
- (C) Nest raw data under a `"raw"` key instead of flat-merging

---

## BUG-02: Diarized output has double-space after speaker labels [LOW]

**File:** `main.py:226-235`
**Type:** Formatting bug

Speaker labels are appended with a trailing space (`f"\n[{spk}] "`) and then `" ".join(lines)` adds another space, producing double-spaces:

```
[SPEAKER_00]  Hello.    <-- double space after label
[SPEAKER_01]  World.
```

**Reproduction:**
```python
result = {'speaker_segments': [
    {'speaker': 'SPEAKER_00', 'text': 'Hello.'},
    {'speaker': 'SPEAKER_01', 'text': 'World.'},
]}
# Output: '[SPEAKER_00]  Hello. \n[SPEAKER_01]  World.'
```

**Fix:** Remove trailing space from the f-string or change join logic.

---

## BUG-03: `merge_diarization()` crashes on malformed segments [MEDIUM]

**File:** `main.py:208`
**Type:** Missing input validation

If a transcription segment is missing `start` or `end` keys, the function raises an unhandled `KeyError`:

```python
mid = (seg["start"] + seg["end"]) / 2.0  # KeyError if keys missing
```

While built-in backends always provide these keys, custom backends registered via `register_backend()` might not. There is no defensive check.

**Reproduction:**
```python
merge_diarization(
    {"segments": [{"text": "no timestamps"}]},
    [(0.0, 5.0, "SPEAKER_00")]
)
# Raises: KeyError: 'start'
```

---

## BUG-04: `register_backend()` allows silent overwrite of built-in backends [MEDIUM]

**File:** `backends/__init__.py:82-95`
**Type:** Missing validation

No guard prevents overwriting built-in backends (`whisper`, `voxtral`). No type validation ensures the class actually subclasses `TranscriptionBackend`.

```python
def register_backend(name: str, backend_class: Type[TranscriptionBackend]) -> None:
    _BACKENDS[name] = backend_class  # No checks at all
```

**Reproduction:**
```python
register_backend("whisper", SomeArbitraryClass)  # Silently overwrites
```

---

## SEC-01: Unpinned HuggingFace model revisions [MEDIUM]

**File:** `backends/voxtral_backend.py:136, 145`
**Tool:** bandit B615

Both `from_pretrained()` calls lack a `revision` parameter, making the application vulnerable to supply-chain attacks if a malicious model is pushed to the HuggingFace Hub under the same name.

```python
self._model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, ...)  # No revision=
self._processor = AutoProcessor.from_pretrained(model_id, ...)          # No revision=
```

**Fix:** Pin to specific commit hashes or release tags.

---

## CFG-01: Black formatting violations (4 files) [LOW]

**Files:** `main.py`, `convert.py`, `tests/test_main.py`, `tests/test_convert.py`
**Issue:** Missing blank line between module docstring and `from __future__` import.

This means `pre-commit` hooks are either not installed or not running.

---

## CFG-02: `.flake8` contains wrong application name [LOW]

**File:** `.flake8:40`

```ini
application-import-names = whisper_danger_zone
```

Should be the current project name. This is a leftover from a repository rename and means flake8 import ordering doesn't correctly classify first-party imports.

---

## CFG-03: Duplicate pytest/coverage configuration [MEDIUM]

**Files:** `pytest.ini` vs `pyproject.toml`

Both files define pytest options and coverage configuration, but they **diverge**:

| Setting | `pytest.ini` | `pyproject.toml` |
|---------|-------------|-----------------|
| Coverage omit `voxtral_backend.py` | No | Yes |
| `fail_under` | Not set | 80% |

This creates ambiguity about which config is authoritative. The `pyproject.toml` omits `backends/voxtral_backend.py` from coverage, but `pytest.ini` does not. Which one takes effect depends on tool resolution order.

**Fix:** Remove the duplicate config from `pytest.ini` and use `pyproject.toml` as the single source of truth, or keep `pytest.ini` minimal with just a pointer.

---

## CFG-04: CI mypy step omits `backends/` directory [LOW]

**File:** `.github/workflows/ci.yml:174`

```yaml
run: mypy main.py convert.py --ignore-missing-imports --no-strict-optional
```

The `backends/` directory is not type-checked in CI, despite being documented in `CLAUDE.md`:
```bash
mypy main.py convert.py backends/ --ignore-missing-imports
```

---

## CFG-05: CI does not test Python 3.12 despite targeting it [LOW]

**File:** `.github/workflows/ci.yml:91` and `pyproject.toml:3`

`pyproject.toml` lists `target-version = ['py310', 'py311', 'py312']` but CI only tests `['3.10', '3.11']`. Python 3.12 is a declared target with no CI coverage.

---

## TEST-01: `test_register_custom_backend` pollutes global state [MEDIUM]

**File:** `tests/test_backends.py:129`

```python
register_backend("custom_test", CustomBackend)
```

This modifies the global `_BACKENDS` dict and never cleans up. The `custom_test` entry persists for all subsequent tests, which can cause:
- Order-dependent test failures
- Incorrect backend count assertions in future tests

**Fix:** Use a pytest fixture with teardown to restore `_BACKENDS`, or use `monkeypatch`.

---

## TEST-02: `parse_args()` functions marked `# pragma: no cover` but ARE tested [LOW]

**Files:** `main.py:51`, `convert.py:33`

Both `parse_args()` functions are marked `# pragma: no cover`, yet they are directly called in test methods (`test_parse_args_minimal`, etc.). The pragma:
- Excludes tested lines from coverage metrics, inflating the percentage artificially
- Hides any future untested code added to these functions

---

## TEST-03: Missing test for `parse_args` error when no audio argument provided [LOW]

**File:** `main.py:116-117`

The validation branch `parser.error("the following arguments are required: audio")` is never tested. This is a user-facing error path that should be covered.

---

## TEST-04: No tests for Voxtral backend [LOW]

**File:** `tests/test_backends.py`

The Voxtral backend (`voxtral_backend.py`, 235 lines) has zero dedicated test methods. Only `test_get_backend_voxtral` confirms the registry returns an instance. No tests cover:
- `load_model()` behavior
- `transcribe()` behavior
- `_check_dependencies()` error paths
- Edge cases (missing HF token, invalid model name)

The file is omitted from coverage in `pyproject.toml`, masking this gap.

---

## TEST-05: Unused import in test file [LOW]

**File:** `tests/test_convert.py:6`

```python
import tempfile  # Never used; tmp_path fixture is used instead
```

---

## CODE-01: Redundant `_check_dependencies()` call in Voxtral transcribe [LOW]

**File:** `backends/voxtral_backend.py:188`

`transcribe()` calls `self._check_dependencies()` even though `load_model()` already verified dependencies. If `transcribe()` is called after `load_model()`, the check runs twice unnecessarily.

---

## CODE-02: Unused variable `sr` in Voxtral transcribe [LOW]

**File:** `backends/voxtral_backend.py:197`

```python
audio, sr = librosa.load(str(audio_path), sr=16000)
# sr is never used
```

---

## CODE-03: No validation for `rate` parameter in `convert_file()` [LOW]

**File:** `convert.py:91`

The function accepts any integer for `rate`. Negative or zero values would produce corrupted WAV files. While `argparse` restricts CLI input, the function is importable as a library without those guards.

---

## Summary Table

| ID | Severity | Category | Status |
|----|----------|----------|--------|
| BUG-01 | **Critical** | Logic bug | Confirmed & reproduced |
| BUG-02 | Low | Formatting | Confirmed & reproduced |
| BUG-03 | Medium | Robustness | Confirmed & reproduced |
| BUG-04 | Medium | Validation | Confirmed & reproduced |
| SEC-01 | Medium | Security | Bandit finding |
| CFG-01 | Low | Formatting | 4 files fail `black --check` |
| CFG-02 | Low | Config | Stale project name |
| CFG-03 | Medium | Config | Divergent coverage configs |
| CFG-04 | Low | CI | Missing type-check scope |
| CFG-05 | Low | CI | Missing Python version |
| TEST-01 | Medium | Test quality | Global state pollution |
| TEST-02 | Low | Test quality | Misleading coverage pragma |
| TEST-03 | Low | Test coverage | Missing edge case |
| TEST-04 | Low | Test coverage | Entire backend untested |
| TEST-05 | Low | Code quality | Dead import |
| CODE-01 | Low | Performance | Redundant check |
| CODE-02 | Low | Code quality | Unused variable |
| CODE-03 | Low | Robustness | Missing validation |

---

## Test Results Summary

```
Tests:     83 passed, 0 failed
Coverage:  98.48% (with voxtral_backend.py excluded)
Flake8:    0 issues
Mypy:      0 errors (2 notes about unchecked function bodies)
Bandit:    2 medium findings (B615)
Black:     4 files need reformatting
Isort:     Clean
```
