"""The pre-commit hooks run the same tool versions as requirements-dev.txt, which CI installs."""

from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PRE_COMMIT_CONFIG = ROOT / ".pre-commit-config.yaml"
REQUIREMENTS_DEV = ROOT / "requirements-dev.txt"
# Tools that are both a pre-commit hook and pinned in requirements-dev.txt (the hook id is the package name).
SHARED_TOOLS = {"bandit", "black", "flake8", "isort", "mypy"}

pytestmark = pytest.mark.skipif(
    not (PRE_COMMIT_CONFIG.is_file() and REQUIREMENTS_DEV.is_file()),
    reason="needs a source checkout: the sdist has no .pre-commit-config.yaml or requirements-dev.txt",
)


def pinned_versions() -> dict[str, str]:
    """Package name -> version for every ``name==version`` line in requirements-dev.txt (extras dropped)."""
    pins = {}
    for line in REQUIREMENTS_DEV.read_text(encoding="utf-8").splitlines():
        match = re.match(r"([A-Za-z0-9._-]+)(\[[^\]]*\])?==([^\s;#]+)", line.strip())
        if match:
            pins[match.group(1).lower()] = match.group(3)
    return pins


def hook_versions() -> dict[str, str]:
    """Hook id -> the rev of its repository in .pre-commit-config.yaml, without a leading "v"."""
    yaml = pytest.importorskip("yaml")  # PyYAML comes with pre-commit
    config = yaml.safe_load(PRE_COMMIT_CONFIG.read_text(encoding="utf-8"))
    return {
        hook["id"]: str(repo["rev"]).removeprefix("v")
        for repo in config["repos"]
        if "rev" in repo
        for hook in repo["hooks"]
    }


def test_hook_versions_match_requirements_dev():
    hooks, pins = hook_versions(), pinned_versions()
    shared = hooks.keys() & pins.keys()
    assert SHARED_TOOLS <= shared, f"not both a hook and a requirements-dev.txt pin: {sorted(SHARED_TOOLS - shared)}"

    mismatched = [
        f"{tool} (hook {hooks[tool]}, pinned {pins[tool]})" for tool in sorted(shared) if hooks[tool] != pins[tool]
    ]
    assert not mismatched, f"update .pre-commit-config.yaml to match requirements-dev.txt: {', '.join(mismatched)}"
