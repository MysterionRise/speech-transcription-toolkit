"""The release workflow: nothing is published before CI and the build job pass, and the build job's scripts check
the tag against the version and take the release notes from CHANGELOG.md.

release.yml runs only at release time, so these tests run its build-job scripts the way the runner does
(``shell: python``) to catch mistakes before then.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
CHECK_VERSION = "Check the tag matches the version"
EXTRACT_NOTES = "Extract the release notes from CHANGELOG.md"

pytestmark = pytest.mark.skipif(
    not RELEASE_WORKFLOW.is_file(), reason="needs a source checkout: the sdist has no .github/ directory"
)


@pytest.fixture(scope="module")
def workflow() -> dict:
    yaml = pytest.importorskip("yaml")  # PyYAML comes with pre-commit
    return yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))


def action(step: dict) -> str:
    return step.get("uses", "").split("@")[0]


def test_publishing_waits_for_ci_and_uploads_only_what_the_build_job_made(workflow):
    jobs = workflow["jobs"]
    assert workflow["permissions"] == {}
    assert jobs["ci"]["uses"] == "./.github/workflows/ci.yml"
    assert jobs["build"]["needs"] == ["ci"]
    assert jobs["build"]["permissions"] == {"contents": "read"}
    uploads = {
        step["with"]["name"]: step["with"]["path"]
        for step in jobs["build"]["steps"]
        if action(step) == "actions/upload-artifact"
    }
    assert uploads["release-dist"] == "dist/"

    publishers = sorted(
        name
        for name, job in jobs.items()
        if any(action(step) == "pypa/gh-action-pypi-publish" for step in job.get("steps", []))
    )
    assert publishers == ["publish-pypi", "publish-testpypi"]
    for name in publishers:
        job = jobs[name]
        assert {"ci", "build"} <= set(job["needs"]), name
        assert job["permissions"] == {"id-token": "write"}, name
        # No checkout and no scripts: download the build job's files, then upload them.
        assert [action(step) for step in job["steps"]] == [
            "actions/download-artifact",
            "pypa/gh-action-pypi-publish",
        ], name
        assert job["steps"][0]["with"]["name"] == "release-dist", name

    assert jobs["github-release"]["permissions"] == {"contents": "write"}


def run_build_step(workflow: dict, name: str, workspace: pathlib.Path, **env: str):
    """Runs a ``shell: python`` step of the build job in ``workspace``; returns the result and the step's outputs."""
    step = next(step for step in workflow["jobs"]["build"]["steps"] if step.get("name") == name)
    assert step["shell"] == "python"
    assert "${{" not in step["run"], "pass expressions through env, not into the script"
    script, outputs, summary = (workspace.parent / file for file in ("step.py", "outputs", "summary"))
    script.write_text(step["run"], encoding="utf-8")
    outputs.write_text("", encoding="utf-8")
    env = {**os.environ, "GITHUB_OUTPUT": str(outputs), "GITHUB_STEP_SUMMARY": str(summary), **env}
    result = subprocess.run(
        [sys.executable, str(script)], cwd=workspace, env=env, capture_output=True, text=True, check=False
    )
    return result, dict(line.split("=", 1) for line in outputs.read_text(encoding="utf-8").splitlines())


def built(tmp_path: pathlib.Path, version: str) -> pathlib.Path:
    """A workspace whose dist/ has what ``python -m build`` makes for ``version``."""
    dist = tmp_path / "workspace" / "dist"
    dist.mkdir(parents=True)
    (dist / f"speech_transcription_toolkit-{version}-py3-none-any.whl").touch()
    (dist / f"speech_transcription_toolkit-{version}.tar.gz").touch()
    return dist.parent


@pytest.mark.parametrize(
    "ref_type, ref_name, version, prerelease",
    [
        ("tag", "v0.4.0", "0.4.0", "false"),
        ("tag", "v0.4.0rc1", "0.4.0rc1", "true"),
        ("tag", "v0.4.0.post1", "0.4.0.post1", "false"),
        ("branch", "main", "0.4.0", "false"),  # the dry run: there is no tag to check
    ],
)
def test_version_check_passes_a_matching_tag(workflow, tmp_path, ref_type, ref_name, version, prerelease):
    workspace = built(tmp_path, version)
    result, outputs = run_build_step(
        workflow, CHECK_VERSION, workspace, GITHUB_REF_TYPE=ref_type, GITHUB_REF_NAME=ref_name
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert outputs == {"version": version, "prerelease": prerelease}


@pytest.mark.parametrize(
    "tag, version",
    [
        ("v0.4.1", "0.4.0"),
        ("0.4.0", "0.4.0"),
        ("v0.4.0-rc1", "0.4.0rc1"),  # PyPI writes 0.4.0-rc1 as 0.4.0rc1, so the tag must too
    ],
)
def test_version_check_stops_a_tag_that_does_not_match(workflow, tmp_path, tag, version):
    workspace = built(tmp_path, version)
    result, outputs = run_build_step(workflow, CHECK_VERSION, workspace, GITHUB_REF_TYPE="tag", GITHUB_REF_NAME=tag)
    assert result.returncode == 1
    assert f"::error::Tag {tag} doesn't match speech_toolkit.__version__ ({version}" in result.stdout
    assert outputs == {}


def changelog(*versions: str) -> str:
    """CHANGELOG.md after ``towncrier build`` for each version, newest first, with towncrier.toml's title format."""
    config = (ROOT / "towncrier.toml").read_text(encoding="utf-8")
    title_format = re.search(r'^title_format = "(.+)"$', config, re.MULTILINE).group(1)
    sections = "".join(
        f"{title_format.format(version=version, project_date='2026-10-20')}\n\n### Fixed\n\n- Fixed in {version}.\n\n\n"
        for version in versions
    )
    return f"# Changelog\n\n<!-- towncrier release notes start -->\n\n{sections}## Before 0.4.0\n\n- Older work.\n"


def extract_notes(workflow, tmp_path, changelog_text: str, version: str, ref_type: str = "tag"):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "CHANGELOG.md").write_text(changelog_text, encoding="utf-8")
    prerelease = "true" if "rc" in version else "false"
    result, _ = run_build_step(
        workflow, EXTRACT_NOTES, workspace, GITHUB_REF_TYPE=ref_type, VERSION=version, PRERELEASE=prerelease
    )
    notes = workspace / "release-notes.md"
    return result, notes.read_text(encoding="utf-8") if notes.exists() else None


@pytest.mark.parametrize("version", ["0.5.0", "0.4.1rc1", "0.4.1", "0.4.0"])
def test_release_notes_are_the_versions_changelog_section(workflow, tmp_path, version):
    # 0.4.1 comes after the 0.4.1rc1 section, whose heading must not count as its own.
    text = changelog("0.5.0", "0.4.1rc1", "0.4.1", "0.4.0")
    result, notes = extract_notes(workflow, tmp_path, text, version)
    assert result.returncode == 0, result.stdout + result.stderr
    assert notes == f"### Fixed\n\n- Fixed in {version}.\n"


def test_a_final_release_without_a_changelog_section_stops(workflow, tmp_path):
    result, notes = extract_notes(workflow, tmp_path, changelog("0.4.1rc1", "0.4.0"), "0.4.1")
    assert result.returncode == 1
    assert "::error::CHANGELOG.md has no section for 0.4.1: run towncrier build --version 0.4.1" in result.stdout
    assert notes is None


@pytest.mark.parametrize("ref_type, version", [("tag", "0.4.1rc2"), ("branch", "0.4.1")])
def test_a_pre_release_or_the_dry_run_without_a_section_only_warns(workflow, tmp_path, ref_type, version):
    result, notes = extract_notes(workflow, tmp_path, changelog("0.4.1rc1", "0.4.0"), version, ref_type)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"::warning::CHANGELOG.md has no section for {version}" in result.stdout
    assert notes == ""  # a GitHub pre-release then gets GitHub's generated notes; the dry run makes no release


def test_the_release_build_uses_the_same_build_and_twine_as_ci():
    """CI's package check installs requirements-dev.txt; the release build installs requirements-release.txt."""

    def pins(name: str) -> dict[str, str]:
        text = (ROOT / f"requirements-{name}.txt").read_text(encoding="utf-8")
        return dict(re.findall(r"^([\w.-]+)==(\S+)", text, re.MULTILINE))

    dev, release = pins("dev"), pins("release")
    assert {tool: release.get(tool) for tool in ("build", "twine")} == {tool: dev[tool] for tool in ("build", "twine")}
