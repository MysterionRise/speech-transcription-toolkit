"""The benchmark harness (benchmarks/): the comparison that gates the benchmark workflow, the evaluation set's manifest
and download, the measurements and the results, all without a model library.

The word error rate needs jiwer and whisper-normalizer (requirements-bench.txt), and reading Parquet files pyarrow:
their tests skip without them, as in the unit test jobs, and run in the benchmark workflow, which installs them.
Elsewhere a simple scorer stands in for them.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import pathlib
import shlex
import sys
import tarfile
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from speech_toolkit.backends import TranscriptionBackend, TranscriptionResult

ROOT = pathlib.Path(__file__).resolve().parent.parent
if not (ROOT / "benchmarks").is_dir():
    pytest.skip("needs a source checkout: the sdist has no benchmarks/ directory", allow_module_level=True)

from benchmarks import SCHEMA, compare, data, metrics, run  # noqa: E402  (after the skip: the sdist has none)
from benchmarks.data import Clip, DataError  # noqa: E402
from benchmarks.metrics import Errors  # noqa: E402

WORKFLOW = ROOT / ".github" / "workflows" / "benchmark.yml"
BASELINE = ROOT / "benchmarks" / "baseline.json"


# --- Results documents, as python -m benchmarks.run writes them -------------------------------------------------


def model_entry(backend="faster-whisper", model="tiny", device="cpu", wer=0.08, fleurs_wer=0.3, rtfx=20.0, **extra):
    """A model's results on two sets, with two clips in the first."""
    return {
        "backend": backend,
        "model": model,
        "device": device,
        "precision": "int8",
        "load_seconds": 1.0,
        "rtfx": rtfx,
        "sets": {
            "librispeech": {"fingerprint": "aaaa", "wer": wer, "rtfx": rtfx, "clips": 2},
            "fleurs": {"fingerprint": "bbbb", "wer": fleurs_wer, "rtfx": rtfx, "clips": 1},
        },
        "clips": {
            "librispeech/1": {"set": "librispeech", "words": 10, "errors": 1, "hypothesis": "one"},
            "librispeech/2": {"set": "librispeech", "words": 10, "errors": 0, "hypothesis": "two"},
            "fleurs/fr/3": {"set": "fleurs", "words": 10, "errors": 3, "hypothesis": "trois"},
        },
        **extra,
    }


def document(*entries, scoring="whisper-normalizer 0.1.15, jiwer 4.0.0", cpu="Xeon"):
    return {
        "schema": SCHEMA,
        "environment": {"machine": "Linux", "cpu": cpu, "cpus": 4},
        "settings": {"scoring": scoring},
        "results": list(entries) if entries else [model_entry()],
    }


# --- compare: the gate --------------------------------------------------------------------------------------------


class TestCompare:
    def test_identical_results_pass(self):
        comparison = compare.compare(document(), document())

        assert not comparison.failed
        assert [line.verdict for line in comparison.lines] == ["ok", "ok"]
        assert "**Result: passed.**" in compare.markdown(comparison)

    def test_a_rise_up_to_the_tolerance_passes(self):
        comparison = compare.compare(document(model_entry(wer=0.08)), document(model_entry(wer=0.09)))

        assert not comparison.failed, comparison.regressions

    def test_a_rise_beyond_the_tolerance_fails_and_names_the_clips_that_got_worse(self):
        worse = model_entry(wer=0.0801 + compare.DEFAULT_MAX_WER_INCREASE)
        worse["clips"]["librispeech/2"].update(errors=4, hypothesis="too")

        comparison = compare.compare(document(), document(worse))

        assert comparison.failed
        assert comparison.lines[0].verdict == "regressed"
        assert "the WER on librispeech rose from 0.0800 to 0.0901" in comparison.regressions[0]
        assert "`librispeech/2`: 0 → 4 errors of 10 words; was 'two', now 'too'" in comparison.regressions[1]
        text = compare.markdown(comparison)
        assert "### Regressions" in text and "**Result: FAILED.**" in text

    def test_the_tolerance_is_absolute_and_can_be_tightened(self):
        comparison = compare.compare(document(), document(model_entry(wer=0.085)), max_wer_increase=0.001)

        assert comparison.failed

    def test_an_improvement_passes_with_a_note_to_update_the_baseline(self):
        comparison = compare.compare(document(), document(model_entry(wer=0.05)))

        assert not comparison.failed
        assert comparison.lines[0].verdict == "improved"
        assert "update the baseline" in comparison.notes[0]

    def test_a_model_of_the_baseline_that_wasnt_measured_fails(self):
        baseline = document(model_entry(), model_entry(backend="whisper"))

        comparison = compare.compare(baseline, document())

        assert comparison.failed
        assert comparison.problems == ["whisper tiny on cpu is in the baseline but wasn't measured"]

    def test_results_without_any_model_fail(self):
        empty = document()
        empty["results"] = []

        comparison = compare.compare(empty, empty)

        assert comparison.failed
        assert "the results have no models" in comparison.problems

    def test_a_model_without_a_baseline_fails(self):
        comparison = compare.compare(document(), document(model_entry(), model_entry(model="base")))

        assert comparison.failed
        assert comparison.problems == ["faster-whisper base on cpu has no baseline; add it (see benchmarks/README.md)"]

    def test_the_device_is_part_of_what_is_compared(self):
        comparison = compare.compare(document(), document(model_entry(device="cuda")))

        assert len(comparison.problems) == 2  # cpu not measured, cuda without a baseline

    def test_results_on_other_clips_dont_compare(self):
        other = model_entry()
        other["sets"]["fleurs"]["fingerprint"] = "cccc"

        comparison = compare.compare(document(), document(other))

        assert comparison.failed
        assert "the fleurs set was measured on other clips or references" in comparison.problems[0]
        assert [line.set for line in comparison.lines] == ["librispeech"]

    def test_a_set_measured_on_one_side_only_fails(self):
        fewer = model_entry()
        del fewer["sets"]["fleurs"]

        comparison = compare.compare(document(), document(fewer))

        assert comparison.problems == ["faster-whisper tiny on cpu: only the baseline measured the fleurs set"]

    def test_another_scoring_version_is_noted_and_the_wer_still_gated(self):
        newer = "whisper-normalizer 0.2.0, jiwer 4.0.0"

        comparison = compare.compare(document(), document(scoring=newer))

        assert not comparison.failed
        assert f"but {newer} now, which alone may change it" in comparison.notes[0]
        assert compare.compare(document(), document(model_entry(wer=0.2), scoring=newer)).failed

    def test_another_results_format_doesnt_compare(self):
        old = document()
        old["schema"] = SCHEMA + 1

        comparison = compare.compare(old, document())

        assert comparison.failed and not comparison.lines
        assert f"the baseline file is in results format {SCHEMA + 1}" in comparison.problems[0]

    def test_speed_is_only_reported_by_default(self):
        comparison = compare.compare(document(), document(model_entry(rtfx=5.0)))

        assert not comparison.failed
        assert "| 20.0 | 5.0 | -75% | ok |" in compare.markdown(comparison)
        assert "RTFx is only reported" in compare.markdown(comparison)

    def test_a_speed_drop_beyond_the_given_fraction_fails(self):
        assert not compare.compare(document(), document(model_entry(rtfx=18.0)), max_rtfx_drop=0.15).failed

        comparison = compare.compare(document(), document(model_entry(rtfx=16.0)), max_rtfx_drop=0.15)

        assert comparison.regressions == ["faster-whisper tiny on cpu: RTFx fell from 20.0 to 16.0, by more than 15%"]

    def test_notes_another_machine_and_precision(self):
        other = model_entry(precision="float32")

        comparison = compare.compare(document(), document(other, cpu="EPYC"))

        assert not comparison.failed
        assert any("ran in float32, the baseline in int8" in note for note in comparison.notes)
        assert any("different machines" in note for note in comparison.notes)

    def test_main_passes_and_fails_with_the_exit_status(self, tmp_path, capsys):
        before, after = tmp_path / "before.json", tmp_path / "after.json"
        before.write_text(json.dumps(document()), encoding="utf-8")
        after.write_text(json.dumps(document(model_entry(wer=0.2))), encoding="utf-8")

        compare.main([str(before), str(before)])
        assert "**Result: passed.**" in capsys.readouterr().out

        with pytest.raises(SystemExit) as exit_info:
            compare.main([str(before), str(after)])
        assert exit_info.value.code == 1
        assert "**Result: FAILED.**" in capsys.readouterr().out

    def test_main_rejects_a_file_that_isnt_results(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("[]", encoding="utf-8")

        with pytest.raises(SystemExit, match="isn't a results file"):
            compare.main([str(bad), str(bad)])
        with pytest.raises(SystemExit, match="can't read"):
            compare.main([str(tmp_path / "missing.json"), str(bad)])

    @pytest.mark.parametrize("option", [["--max-wer-increase", "-0.1"], ["--max-rtfx-drop", "1.5"]])
    def test_main_rejects_tolerances_out_of_range(self, option, tmp_path):
        with pytest.raises(SystemExit):
            compare.main(["a.json", "b.json", *option])


# --- The committed manifest, baseline and workflow ---------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest() -> data.Manifest:
    return data.load_manifest()


class TestCommittedFiles:
    def test_the_manifest_lists_librispeech_and_fleurs_clips(self, manifest):
        librispeech = [clip for clip in manifest.clips if clip.set == "librispeech"]
        fleurs = [clip for clip in manifest.clips if clip.set == "fleurs"]

        assert len(librispeech) == 100 and {clip.language for clip in librispeech} == {"en"}
        assert len({clip.id.split("/")[1].split("-")[0] for clip in librispeech}) == 40  # every test-clean speaker
        assert len(fleurs) == 30 and {clip.language for clip in fleurs} == {"de", "es", "fr"}

    def test_the_datasets_are_pinned_and_their_licenses_stated(self, manifest):
        for source in manifest.sets.values():
            assert len(source["revision"]) == 40 and int(source["revision"], 16) >= 0  # a commit, not a branch
            assert source["license"] == "CC BY 4.0"
            assert source["citation"]

    def test_the_baseline_was_measured_on_the_committed_manifest(self, manifest):
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

        for entry in baseline["results"]:
            for name, results in entry["sets"].items():
                clips = [clip for clip in manifest.clips if clip.set == name]
                assert results["fingerprint"] == data.fingerprint(clips), f"{entry['model']} on {name}"
        assert not compare.compare(baseline, baseline).failed

    def test_the_workflow_measures_the_models_of_the_baseline(self):
        yaml = pytest.importorskip("yaml")  # PyYAML comes with pre-commit
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        steps = workflow["jobs"]["benchmark"]["steps"]
        command = next(step["run"] for step in steps if "python -m benchmarks.run" in step.get("run", ""))
        options = command.split("python -m benchmarks.run", 1)[1].split("|")[0]
        args = run.parse_args(shlex.split(options.replace("\\\n", " ")))
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

        measured = {(backend, model, args.device) for backend, model in run.model_list(args.specs)}
        assert measured == {(entry["backend"], entry["model"], entry["device"]) for entry in baseline["results"]}
        assert any("benchmarks.compare benchmarks/baseline.json" in step.get("run", "") for step in steps)
        assert "benchmarks/**" in workflow[True]["pull_request"]["paths"]  # PyYAML reads the key "on" as True


# --- data: the manifest and the download ----------------------------------------------------------------------------


def clip(clip_id="ls/a", set_name="ls", language="en", audio=b"audio a", text="hello world", **where):
    if not where:
        where = {"member": f"test/{clip_id.split('/')[-1]}.wav"}
    sha256 = hashlib.sha256(audio).hexdigest()
    return Clip(clip_id, set_name, language, "audio.tar.gz", f"{clip_id}.wav", sha256, 1.0, text, **where)


def write_manifest(path: pathlib.Path, clips: List[Clip], sets=("ls", "fl")) -> pathlib.Path:
    manifest = {
        "sets": {name: {"dataset": f"org/{name}", "revision": "0" * 40} for name in sets},
        "clips": [{key: value for key, value in vars(c).items() if value is not None} for c in clips],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def tar_archive(files: Dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


class TestManifest:
    def test_load_and_select(self, tmp_path):
        clips = [clip("ls/a"), clip("ls/b", audio=b"b"), clip("fl/c", "fl", "fr", audio=b"c")]
        manifest = data.load_manifest(write_manifest(tmp_path / "manifest.json", clips))

        assert manifest.clips == tuple(clips)
        assert manifest.select() == clips
        assert manifest.select(["fl"]) == clips[2:]
        assert manifest.select(limit=1) == [clips[0], clips[2]]
        with pytest.raises(DataError, match="no set named nope"):
            manifest.select(["nope"])

    @pytest.mark.parametrize(
        "bad, problem",
        [
            ({"set": "other"}, "isn't in the manifest's sets"),
            ({"audio": "/etc/passwd"}, "isn't a relative path"),
            ({"audio": "../outside.wav"}, "isn't a relative path"),
            ({"audio": "C:\\clip.wav"}, "isn't a relative path"),
            ({"member": None}, "needs either a row"),
            ({"row": 3}, "needs either a row"),
            ({"sha256": "ABC"}, "sha256 isn't"),
            ({"text": " "}, "no reference text"),
            ({"seconds": 0}, "no reference text or no length"),
        ],
    )
    def test_a_bad_clip_is_refused(self, tmp_path, bad, problem):
        fields = {**vars(clip()), **bad}
        path = tmp_path / "manifest.json"
        clips = [{key: value for key, value in fields.items() if value is not None}]
        path.write_text(json.dumps({"sets": {"ls": {}}, "clips": clips}), encoding="utf-8")

        with pytest.raises(DataError, match=problem):
            data.load_manifest(path)

    def test_repeated_ids_are_refused(self, tmp_path):
        path = write_manifest(tmp_path / "manifest.json", [clip(), clip()])

        with pytest.raises(DataError, match="repeats the id of ls/a"):
            data.load_manifest(path)

    def test_an_unreadable_manifest_is_a_data_error(self, tmp_path):
        with pytest.raises(DataError, match="can't read the manifest"):
            data.load_manifest(tmp_path / "missing.json")

    def test_the_cache_folder(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BENCHMARK_DATA_DIR", str(tmp_path / "data"))
        assert data.default_data_dir() == tmp_path / "data"

        monkeypatch.delenv("BENCHMARK_DATA_DIR")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        assert data.default_data_dir() == tmp_path / "cache" / "speech-toolkit-benchmarks"

    def test_fingerprints_change_with_the_clips_and_references(self):
        base = data.fingerprint([clip()])

        assert data.fingerprint([clip()]) == base
        assert data.fingerprint([clip(text="hello word")]) != base
        assert data.fingerprint([clip(audio=b"other")]) != base
        assert data.fingerprint([clip(), clip("ls/b")]) != base


class TestDownload:
    @pytest.fixture
    def opened(self, monkeypatch):
        """Replace the download with an archive in memory; records each file opened."""
        calls: List[tuple] = []
        archive = tar_archive({"test/a.wav": b"audio a", "test/b.wav": b"audio b", "test/c.wav": b"audio c"})

        def open_dataset_file(dataset, revision, file, streaming):
            calls.append((dataset, revision, file, streaming))
            return io.BytesIO(archive)

        monkeypatch.setattr(data, "open_dataset_file", open_dataset_file)
        return calls

    def test_downloads_the_missing_clips_once_and_checks_them(self, tmp_path, opened):
        clips = [clip("ls/a"), clip("ls/b", audio=b"audio b")]
        manifest = data.load_manifest(write_manifest(tmp_path / "manifest.json", clips))
        messages: List[str] = []

        data.prepare(manifest, tmp_path / "data", log=messages.append)
        data.prepare(manifest, tmp_path / "data", log=messages.append)

        assert opened == [("org/ls", "0" * 40, "audio.tar.gz", True)]
        assert (tmp_path / "data" / "ls" / "a.wav").read_bytes() == b"audio a"
        assert (tmp_path / "data" / "ls" / "b.wav").read_bytes() == b"audio b"
        assert messages[-1].endswith("2 clips, all intact.")

    def test_a_damaged_clip_is_downloaded_again(self, tmp_path, opened):
        manifest = data.load_manifest(write_manifest(tmp_path / "manifest.json", [clip("ls/a")]))
        (tmp_path / "data" / "ls").mkdir(parents=True)
        (tmp_path / "data" / "ls" / "a.wav").write_bytes(b"damaged")

        data.prepare(manifest, tmp_path / "data", log=lambda message: None)

        assert len(opened) == 1
        assert (tmp_path / "data" / "ls" / "a.wav").read_bytes() == b"audio a"

    def test_audio_that_doesnt_match_its_sha256_isnt_stored(self, tmp_path, opened):
        changed = clip("ls/a", audio=b"what the manifest says")
        manifest = data.load_manifest(write_manifest(tmp_path / "manifest.json", [changed]))

        with pytest.raises(DataError, match="ls/a: the downloaded audio has SHA-256"):
            data.prepare(manifest, tmp_path / "data", log=lambda message: None)
        assert not (tmp_path / "data" / "ls" / "a.wav").exists()
        assert len(opened) == 1  # not tried again: the dataset changed

    def test_a_member_missing_from_the_archive_is_an_error(self, tmp_path, opened):
        manifest = data.load_manifest(write_manifest(tmp_path / "manifest.json", [clip("ls/z")]))

        with pytest.raises(DataError, match="no file named test/z.wav"):
            data.prepare(manifest, tmp_path / "data", log=lambda message: None)

    def test_network_errors_are_retried(self, tmp_path, monkeypatch):
        archive = tar_archive({"test/a.wav": b"audio a"})
        attempts = iter([ConnectionError("reset"), io.BytesIO(archive)])

        def open_dataset_file(*args, **kwargs):
            attempt = next(attempts)
            if isinstance(attempt, Exception):
                raise attempt
            return attempt

        monkeypatch.setattr(data, "open_dataset_file", open_dataset_file)
        monkeypatch.setattr(data.time, "sleep", lambda seconds: None)
        manifest = data.load_manifest(write_manifest(tmp_path / "manifest.json", [clip("ls/a")]))
        messages: List[str] = []

        data.prepare(manifest, tmp_path / "data", log=messages.append)

        assert "Download failed (ConnectionError: reset); trying again" in messages
        assert (tmp_path / "data" / "ls" / "a.wav").read_bytes() == b"audio a"

    def test_gives_up_after_a_few_attempts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data, "open_dataset_file", MagicMock(side_effect=TimeoutError("timed out")))
        monkeypatch.setattr(data.time, "sleep", lambda seconds: None)
        manifest = data.load_manifest(write_manifest(tmp_path / "manifest.json", [clip("ls/a")]))

        with pytest.raises(DataError, match="can't download org/ls@000000000000: audio.tar.gz: TimeoutError"):
            data.prepare(manifest, tmp_path / "data", log=lambda message: None)
        assert data.open_dataset_file.call_count == data.ATTEMPTS

    def test_opens_dataset_files_with_huggingface_hub(self):
        hub = MagicMock()

        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            data.open_dataset_file("google/fleurs", "abc", "data/x.tar.gz", streaming=True)
            data.open_dataset_file("google/fleurs", "abc", "data/x.parquet", streaming=False)

        assert hub.HfFileSystem.return_value.open.call_args_list == [
            (("datasets/google/fleurs@abc/data/x.tar.gz", "rb"), {"block_size": 0}),
            (("datasets/google/fleurs@abc/data/x.parquet", "rb"), {"block_size": None}),
        ]

    def test_says_what_to_install(self):
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            with pytest.raises(DataError, match="needs huggingface_hub"):
                data.open_dataset_file("a/b", "c", "d", streaming=True)
        with patch.dict(sys.modules, {"pyarrow": None, "pyarrow.parquet": None}):
            with pytest.raises(DataError, match="needs pyarrow: pip install -r requirements-bench.txt"):
                list(data.read_rows(io.BytesIO(), [clip(row=0)]))

    def test_reads_only_the_rows_asked_for_from_parquet(self):
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        audio = [{"bytes": f"audio {row}".encode(), "path": f"{row}.flac"} for row in range(5)]
        buffer = io.BytesIO()
        pq.write_table(pa.table({"id": list(range(5)), "audio": audio}), buffer, row_group_size=2)
        clips = [clip("ls/a", row=3), clip("ls/b", row=0)]

        found = list(data.read_rows(io.BytesIO(buffer.getvalue()), clips))

        assert found == [(clips[1], b"audio 0"), (clips[0], b"audio 3")]
        with pytest.raises(DataError, match="the Parquet file has 5 rows; no row 9"):
            list(data.read_rows(io.BytesIO(buffer.getvalue()), [clip(row=9)]))

    def test_main(self, tmp_path, monkeypatch):
        path = write_manifest(tmp_path / "manifest.json", [clip("ls/a")])
        prepare = MagicMock()
        monkeypatch.setattr(data, "prepare", prepare)

        data.main(["--manifest", str(path), "--data-dir", str(tmp_path)])
        assert prepare.call_args.args[1] == tmp_path

        prepare.side_effect = DataError("offline")
        with pytest.raises(SystemExit, match="Error: offline"):
            data.main(["--manifest", str(path)])


# --- metrics --------------------------------------------------------------------------------------------------------


class TestMetrics:
    def test_errors_add_up(self):
        total = Errors(10, 1, 2, 3) + Errors(5, 1, 0, 0)

        assert (total.words, total.errors, total.wer) == (15, 7, 7 / 15)
        assert Errors().wer is None

    def test_scores_with_whispers_english_normalizer(self):
        pytest.importorskip("jiwer")
        pytest.importorskip("whisper_normalizer")

        assert metrics.score("TEN COLOURFUL BIRDS", "10 colorful birds.", "en") == Errors(3, 0, 0, 0)
        assert metrics.score("HELLO WORLD", "", "en") == Errors(2, 0, 2, 0)
        assert metrics.score("hello world", "hello there world", "en") == Errors(2, 0, 0, 1)

    def test_scores_other_languages_with_the_basic_normalizer(self):
        pytest.importorskip("jiwer")
        pytest.importorskip("whisper_normalizer")

        assert metrics.score("über die straße", "Über die Straße!", "de") == Errors(3, 0, 0, 0)
        assert metrics.score("l'accident", "L’accident.", "fr").errors == 0
        with pytest.raises(ValueError, match="no words"):
            metrics.score("...", "text", "fr")

    def test_says_what_to_install(self):
        metrics.normalizer.cache_clear()
        try:
            with patch.dict(sys.modules, {"whisper_normalizer": None, "whisper_normalizer.basic": None}):
                with pytest.raises(ImportError, match="whisper-normalizer: pip install -r requirements-bench.txt"):
                    metrics.normalizer("en")
            with patch.dict(sys.modules, {"jiwer": None}):
                with pytest.raises(ImportError, match="jiwer: pip install -r requirements-bench.txt"):
                    metrics.score("a", "a", "en")
        finally:
            metrics.normalizer.cache_clear()

    def test_scoring_names_the_versions(self):
        assert metrics.scoring().startswith("whisper-normalizer ")

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
    def test_peak_rss_on_linux(self):
        assert metrics.peak_rss() > 1024 * 1024

    def test_peak_rss_on_macos_and_windows(self, monkeypatch):
        resource = SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda who: SimpleNamespace(ru_maxrss=123456))
        monkeypatch.setattr(sys, "platform", "darwin")
        with patch.dict(sys.modules, {"resource": resource}):
            assert metrics.peak_rss() == 123456

        monkeypatch.setattr(sys, "platform", "win32")
        psutil = MagicMock()
        psutil.Process.return_value.memory_info.return_value.peak_wset = 654321
        with patch.dict(sys.modules, {"resource": None, "psutil": psutil}):
            assert metrics.peak_rss() == 654321
        with patch.dict(sys.modules, {"resource": None, "psutil": None}):
            assert metrics.peak_rss() is None

    def test_gpu_name(self, monkeypatch):
        torch = MagicMock()
        torch.cuda.is_initialized.return_value = True
        torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        assert metrics.gpu_name("cpu") is None
        with patch.dict(sys.modules, {"torch": torch}):
            assert metrics.gpu_name("cuda") == "NVIDIA RTX 4090"

        with patch.dict(sys.modules, {"torch": None}):
            monkeypatch.setattr(metrics.shutil, "which", lambda name: None)
            assert metrics.gpu_name("cuda") is None
            monkeypatch.setattr(metrics.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
            done = SimpleNamespace(stdout="NVIDIA L4\n")
            with patch.object(metrics.subprocess, "run", return_value=done) as nvidia_smi:
                assert metrics.gpu_name("cuda") == "NVIDIA L4"
            assert nvidia_smi.call_args.args[0][:2] == ["/usr/bin/nvidia-smi", "--query-gpu=name"]

    def test_environment(self):
        env = metrics.environment()

        assert env["cpu"] and env["cpus"] and env["python"]
        assert env["speech_toolkit"]
        assert "numpy" in env["packages"]


# --- run: measuring and the results ---------------------------------------------------------------------------------


class EchoBackend(TranscriptionBackend):
    """Says the reference text of the clip it hears, which it recognises by its number of samples."""

    name = "echo"
    description = "Echo backend for tests"
    capabilities = frozenset({"language_detection", "waveform"})
    texts: Dict[int, str] = {}
    heard: List[Any] = []

    @classmethod
    def available_models(cls):
        return ["small", "large"]

    def load_model(self, model_name, device=None):
        if model_name not in self.available_models():
            raise ValueError(f"Unknown echo model: {model_name}")
        self._model = object()
        self._device = device or "cpu"

    def transcribe(self, audio, language=None, task="transcribe", verbose=True):
        EchoBackend.heard.append((len(audio), language))
        return TranscriptionResult(text=EchoBackend.texts.get(len(audio), "something else"), segments=[])


@pytest.fixture
def echo(monkeypatch, tmp_path):
    """The echo backend, the clips of a small manifest stored in tmp_path/data, and decoding that needs no ffmpeg.

    Each clip lasts as many hundredths of a second as its position (1, 2, 3...) times 100 samples.
    """
    import numpy as np

    from speech_toolkit.backends import _BACKENDS

    monkeypatch.setattr("speech_toolkit.backends._BACKENDS", {**_BACKENDS, "echo": EchoBackend})
    contents = [b"a", b"b", b"c", b"d"]
    clips = [
        clip("ls/a", audio=contents[0], text="hello world"),
        clip("ls/b", audio=contents[1], text="good morning"),
        clip("fl/c", "fl", "fr", audio=contents[2], text="bonjour tout le monde"),
        clip("fl/d", "fl", "de", audio=contents[3], text="guten tag"),
    ]
    lengths = {c.audio: (index + 1) * 100 for index, c in enumerate(clips)}
    EchoBackend.texts = {lengths[c.audio]: c.text for c in clips if c.id != "ls/b"}  # one mistake: ls/b
    EchoBackend.heard = []
    for c, content in zip(clips, contents):
        (tmp_path / "data" / c.audio).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / c.audio).write_bytes(content)

    def load_audio(path):
        relative = pathlib.Path(path).relative_to(tmp_path / "data").as_posix()
        return np.zeros(lengths[relative], dtype=np.float32)

    monkeypatch.setattr(run, "load_audio", load_audio)
    manifest = write_manifest(tmp_path / "manifest.json", clips)
    return SimpleNamespace(clips=clips, manifest=manifest, data=tmp_path / "data")


def simple_score(reference: str, hypothesis: str, language: str) -> Errors:
    """Stands in for metrics.score() where jiwer isn't installed: substitutions only, word by word."""
    from tests.integration.wer import edit_distance, normalize

    words = normalize(reference)
    return Errors(len(words), edit_distance(words, normalize(hypothesis)))


@pytest.fixture
def scorer(monkeypatch):
    """metrics.score(), or a simple scorer if jiwer or whisper-normalizer isn't installed."""
    try:
        import jiwer  # noqa: F401
        import whisper_normalizer  # noqa: F401
    except ImportError:
        monkeypatch.setattr(run, "score", simple_score)
        monkeypatch.setattr(run, "normalizer", lambda language: str)


class TestArguments:
    def test_each_model_belongs_to_the_backend_before_it(self):
        args = run.parse_args(["-b", "faster-whisper", "-m", "tiny", "-m", "base", "-b", "whisper", "-m", "tiny"])

        assert args.specs == [("faster-whisper", ["tiny", "base"]), ("whisper", ["tiny"])]
        assert run.model_list(args.specs) == [
            ("faster-whisper", "tiny"),
            ("faster-whisper", "base"),
            ("whisper", "tiny"),
        ]

    def test_a_backend_without_models_gets_its_default_one_once(self):
        args = run.parse_args(["-b", "parakeet", "-b", "parakeet"])

        assert run.model_list(args.specs) == [("parakeet", "parakeet-tdt-0.6b-v3")]

    @pytest.mark.parametrize("argv", [[], ["-m", "tiny"], ["-b", "whisper", "--limit", "0"]])
    def test_refuses(self, argv):
        with pytest.raises(SystemExit):
            run.parse_args(argv)

    def test_an_unknown_backend_is_an_error(self):
        with pytest.raises(SystemExit, match="Error: .*nope"):
            run.main(["-b", "nope", "--sets", "librispeech", "--limit", "1", "--data-dir", "/nonexistent"])


class TestMeasure:
    def test_decodes_each_clip_once_warms_up_and_times_each_transcription(self, echo):
        messages: List[str] = []

        measured = run.measure("echo", "small", None, echo.clips, echo.data, log=messages.append)

        # The first clip twice: the warm-up, then the timed run. Each with its language.
        assert EchoBackend.heard == [(100, "en"), (100, "en"), (200, "en"), (300, "fr"), (400, "de")]
        assert measured["clips"]["ls/b"]["text"] == "something else"
        assert measured["clips"]["fl/c"] == {
            "audio_seconds": 300 / 16000,
            "seconds": pytest.approx(0, abs=1),
            "text": "bonjour tout le monde",
        }
        assert (measured["device"], measured["peak_gpu_bytes"], measured["gpu"]) == ("cpu", None, None)
        assert measured["load_seconds"] >= 0
        assert messages[0].startswith("echo small: loaded in") and messages[-1] == "echo small: 4/4 clips"

    def test_resets_the_random_generators_before_each_transcription(self):
        torch, ctranslate2 = MagicMock(), MagicMock()

        with patch.dict(sys.modules, {"torch": torch, "ctranslate2": ctranslate2}):
            run.reset_seeds()

        torch.manual_seed.assert_called_once_with(run.SEED)
        ctranslate2.set_random_seed.assert_called_once_with(run.SEED)

    def test_samples_the_gpu_unless_on_the_cpu(self, monkeypatch):
        monkeypatch.setattr(run.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
        sampler = MagicMock()
        monkeypatch.setattr(run, "GpuSampler", sampler)

        assert run._gpu_sampler("cpu") is None
        assert run._gpu_sampler("cuda") is sampler.return_value
        sampler.assert_called_once_with("/usr/bin/nvidia-smi")

    def test_in_a_process_of_its_own(self, echo, monkeypatch):
        def worker_in_this_process(command, cwd, env, stdout, check):
            assert command[1:4] == ["-m", "benchmarks.run", "--worker"] and cwd == run.ROOT
            assert env["PYTHONPATH"].split(run.os.pathsep)[0] == str(run.ROOT)
            run.main(command[3:])
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(run.subprocess, "run", worker_in_this_process)

        measured = run.measure_isolated("echo", "small", None, echo.clips, echo.data, echo.manifest)

        assert set(measured["clips"]) == {"ls/a", "ls/b", "fl/c", "fl/d"}

    def test_a_failed_process_is_an_error(self, echo, monkeypatch):
        monkeypatch.setattr(run.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))

        with pytest.raises(run.BenchmarkError, match="measuring echo small failed"):
            run.measure_isolated("echo", "small", None, echo.clips, echo.data, echo.manifest)


class TestRun:
    def test_one_command_measures_every_model_and_writes_json_and_markdown(self, echo, scorer, tmp_path, capsys):
        out = tmp_path / "results.json"
        argv = ["-b", "echo", "-m", "small", "-m", "large", "--in-process", "--json", str(out)]

        run.main([*argv, "--manifest", str(echo.manifest), "--data-dir", str(echo.data)])

        results = json.loads(out.read_text(encoding="utf-8"))
        assert results["schema"] == SCHEMA and results["settings"]["isolated"] is False
        assert [(entry["backend"], entry["model"]) for entry in results["results"]] == [
            ("echo", "small"),
            ("echo", "large"),
        ]
        entry = results["results"][0]
        assert entry["audio_seconds"] == pytest.approx(1000 / 16000, abs=1e-3)
        assert entry["rtfx"] > 0
        assert entry["peak_rss_bytes"] is None or entry["peak_rss_bytes"] > 0
        ls, fl = entry["sets"]["ls"], entry["sets"]["fl"]
        assert (ls["clips"], ls["words"], ls["errors"], ls["wer"]) == (2, 4, 2, 0.5)
        assert (fl["clips"], fl["wer"], sorted(fl["languages"])) == (2, 0.0, ["de", "fr"])
        assert ls["fingerprint"] == data.fingerprint(echo.clips[:2])
        assert entry["clips"]["ls/b"] == {
            **entry["clips"]["ls/b"],
            "set": "ls",
            "errors": 2,
            "hypothesis": "something else",
        }

        table = capsys.readouterr().out
        assert "| Backend | Model | Device | Precision | Load (s) | Audio (s) | RTFx | Peak RSS (MiB)" in table
        assert "| echo | small | cpu | - |" in table and "| 0.500 | 0.000 |" in table
        assert "| Backend | Model | ls en | fl de | fl fr |" in table  # a set with several languages
        assert table.startswith("Machine: ")

        compare.main([str(out), str(out)])  # results compare with themselves

    def test_limit_and_sets_pick_the_clips(self, echo, scorer, tmp_path):
        out = tmp_path / "results.json"
        argv = ["-b", "echo", "--in-process", "--sets", "ls", "--limit", "1", "--json", str(out)]

        run.main([*argv, "--manifest", str(echo.manifest), "--data-dir", str(echo.data)])

        results = json.loads(out.read_text(encoding="utf-8"))
        assert list(results["results"][0]["clips"]) == ["ls/a"]
        assert results["settings"]["limit"] == 1 and results["settings"]["sets"] == ["ls"]

    def test_a_model_that_fails_fails_the_run_but_not_the_others(self, echo, scorer, tmp_path, monkeypatch, capsys):
        out = tmp_path / "results.json"
        argv = ["-b", "echo", "-m", "small", "-m", "huge", "--in-process", "--json", str(out)]

        with pytest.raises(SystemExit, match="1 model failed: echo huge"):
            run.main([*argv, "--manifest", str(echo.manifest), "--data-dir", str(echo.data)])

        assert [entry["model"] for entry in json.loads(out.read_text(encoding="utf-8"))["results"]] == ["small"]

    def test_nothing_measured_is_a_failure_and_writes_nothing(self, echo, scorer, tmp_path):
        out = tmp_path / "results.json"
        argv = ["-b", "echo", "-m", "huge", "--in-process", "--json", str(out)]

        with pytest.raises(SystemExit, match="1 model failed"):
            run.main([*argv, "--manifest", str(echo.manifest), "--data-dir", str(echo.data)])
        assert not out.exists()

    def test_the_default_json_path(self):
        path = run.default_json()

        assert path.parent == ROOT / "benchmarks" / "results" and path.suffix == ".json"

    def test_markdown_shows_the_gpu(self):
        entry = {
            "backend": "faster-whisper",
            "model": "turbo",
            "device": "cuda",
            "gpu": "NVIDIA L4",
            "precision": "float16",
            "load_seconds": 2.0,
            "audio_seconds": 1000.0,
            "rtfx": 150.0,
            "peak_rss_bytes": 900 * run.MIB,
            "peak_gpu_bytes": 1500 * run.MIB,
            "peak_gpu_sampled": True,
            "sets": {"librispeech": {"wer": 0.04, "languages": {"en": {"wer": 0.04}}}},
        }
        env = {
            "machine": "Linux",
            "cpu": "EPYC",
            "cpus": 8,
            "ram_bytes": None,
            "python": "3.12",
            "speech_toolkit": "0.5",
        }

        table = run.markdown({"environment": {**env, "packages": {}}, "results": [entry]})

        assert (
            "| faster-whisper | turbo | cuda (NVIDIA L4) | float16 | 2.0 | 1000 | 150.0 | 900 | ~1500 | 0.040 |"
            in table
        )
        assert "WER by language" not in table
