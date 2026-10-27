"""Measure backends and models on the evaluation set: word error rate, speed (RTFx) and peak memory.

    python -m benchmarks.run -b faster-whisper -m tiny
    python -m benchmarks.run -b faster-whisper -m tiny -m base -b whisper -m tiny --device cpu
    python -m benchmarks.run -b parakeet                             # the backend's default model
    python -m benchmarks.run -b faster-whisper -m tiny --sets librispeech --limit 10    # a quick look

Each ``-m`` names a model of the ``-b`` before it, so adding a model or a backend is one more option. The table goes
to stdout as Markdown, progress to stderr, and the results to a JSON file (``--json``; by default
``benchmarks/results/<time>.json``) that ``python -m benchmarks.compare`` compares with another.

Each model runs in a process of its own, so that its peak memory is its own:

1. The model loads, timed on its own (``load_seconds``; a first run's includes the download).
2. The first clip is transcribed once to warm up, untimed and unscored.
3. Each clip is decoded (ffmpeg, 16 kHz mono) outside the timed region, then transcribed from the array, with its
   language, after resetting the random generators: a temperature fallback then samples the same way every run.

RTFx is seconds of audio per second of transcription, so higher is faster. The word error rate is scored afterwards,
in this process (see metrics.py).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import json
import os
import pathlib
import random
import shutil
import subprocess  # nosec B404 (only this module again, with this interpreter, to measure a model)
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from benchmarks import SCHEMA
from benchmarks.data import (
    MANIFEST,
    Clip,
    DataError,
    Log,
    default_data_dir,
    fingerprint,
    load_manifest,
    log_to_stderr,
    prepare,
)
from benchmarks.metrics import Errors, environment, gpu_name, normalizer, peak_rss, score, scoring
from speech_toolkit.media import SAMPLE_RATE, load_audio
from tests.integration.instrument import GpuSampler, torch_peak, transcriber_devices
from tests.integration.report import Run

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = pathlib.Path(__file__).resolve().parent / "results"
MIB = 1024 * 1024
SEED = 0
PROGRESS_EVERY = 10  # clips


class BenchmarkError(RuntimeError):
    """A model couldn't be measured."""


class _Backend(argparse.Action):
    """-b BACKEND: starts a new entry, which the -m options after it add models to."""

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, value: Any, option: Any = None):
        namespace.specs = [*(namespace.specs or []), (value, [])]


class _Model(argparse.Action):
    """-m MODEL: a model of the -b before it."""

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, value: Any, option: Any = None):
        if not namespace.specs:
            parser.error("-m/--model names a model of the -b/--backend before it: give -b first")
        namespace.specs[-1][1].append(value)


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.run",
        description="Measure the word error rate, speed (RTFx) and peak memory of backends and models on the "
        "benchmark's evaluation set. Prints a Markdown table and writes the results as JSON.",
        epilog="Example: python -m benchmarks.run -b faster-whisper -m tiny -m base -b whisper -m tiny",
    )
    parser.add_argument("-b", "--backend", dest="specs", action=_Backend, help="a backend to measure (repeatable)")
    parser.add_argument(
        "-m",
        "--model",
        dest="specs",
        action=_Model,
        help="a model of the -b before it (repeatable; default: the backend's default model)",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), help="where the models run (default: as the backend picks)"
    )
    parser.add_argument("--sets", nargs="+", metavar="SET", help="only these sets of the manifest (default: all)")
    parser.add_argument("--limit", type=int, metavar="N", help="only the first N clips of each set, for a quick look")
    parser.add_argument("--json", type=pathlib.Path, help="where to write the results (default: benchmarks/results/)")
    parser.add_argument(
        "--data-dir", type=pathlib.Path, help=f"where the clips are kept (default: {default_data_dir()})"
    )
    parser.add_argument("--manifest", type=pathlib.Path, default=MANIFEST, help=argparse.SUPPRESS)
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="measure every model in this process: quicker to debug, but each "
        "model's peak memory then includes the ones before it",
    )
    parser.add_argument("--worker", nargs=2, metavar=("JOB", "OUT"), type=pathlib.Path, help=argparse.SUPPRESS)
    parser.set_defaults(specs=None)
    args = parser.parse_args(argv)
    if args.worker is None and not args.specs:
        parser.error("name a backend with -b, and its models with -m (e.g. -b faster-whisper -m tiny)")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def model_list(specs: Sequence[Tuple[str, List[str]]]) -> List[Tuple[str, str]]:
    """The (backend, model) pairs to measure: the -m options of each -b, or the backend's default model."""
    from speech_toolkit.backends import get_backend_class

    pairs: List[Tuple[str, str]] = []
    for backend, models in specs:
        for model in models or [get_backend_class(backend).default_model()]:
            if (backend, model) not in pairs:
                pairs.append((backend, model))
    return pairs


def reset_seeds(seed: int = SEED) -> None:
    """Reset the random generators a backend may sample with (Python's, NumPy's, torch's, CTranslate2's)."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
    ctranslate2 = sys.modules.get("ctranslate2")
    if ctranslate2 is not None and hasattr(ctranslate2, "set_random_seed"):
        ctranslate2.set_random_seed(seed)


def _transcribe(transcriber: Any, audio: Any, language: str) -> Tuple[float, str]:
    """Transcribe *audio*; returns the seconds it took and the transcript."""
    reset_seeds()
    started = time.perf_counter()
    result = transcriber.transcribe(audio, language=language)
    return time.perf_counter() - started, result.text


def _gpu_sampler(device: Optional[str]) -> Optional[GpuSampler]:
    """A sampler of the GPU's memory use (with nvidia-smi), unless the model runs on the CPU or there's no nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi") if device != "cpu" else None
    return GpuSampler(nvidia_smi) if nvidia_smi else None


def measure(
    backend: str,
    model: str,
    device: Optional[str],
    clips: Sequence[Clip],
    data_dir: pathlib.Path,
    log: Log = log_to_stderr,
) -> Dict[str, Any]:
    """Load *model* of *backend* and transcribe *clips*, in this process.

    Returns the measurements: where the model ran, its load time, peak memory, and per clip the audio's length, the
    transcription's seconds and the transcript (scored later, by summarize()).
    """
    from speech_toolkit import Transcriber

    sampler = _gpu_sampler(device)
    try:
        started = time.perf_counter()
        transcriber = Transcriber(backend, model, device=device)
        load_seconds = time.perf_counter() - started
        devices, precision = transcriber_devices(transcriber)
        log(
            f"{backend} {model}: loaded in {load_seconds:.1f} s on {devices['model']} ({precision or 'precision unknown'})"
        )
        measured: Dict[str, Dict[str, Any]] = {}
        for index, clip in enumerate(clips):
            audio = load_audio(data_dir / clip.audio)  # decoded once per clip, outside the timed region
            if index == 0:
                _transcribe(transcriber, audio, clip.language)  # warm-up
            seconds, text = _transcribe(transcriber, audio, clip.language)
            measured[clip.id] = {"audio_seconds": len(audio) / SAMPLE_RATE, "seconds": seconds, "text": text}
            if (index + 1) % PROGRESS_EVERY == 0 or index + 1 == len(clips):
                log(f"{backend} {model}: {index + 1}/{len(clips)} clips")
        torch_bytes, sampled = torch_peak(), sampler.peak_above_baseline() if sampler else None
    finally:
        if sampler is not None:
            sampler.stop()
    # The integration report's choice of GPU number: torch's count, else nvidia-smi's samples (CTranslate2).
    peak = Run("", backend, model, devices, precision, load_seconds, None, 0.0, torch_bytes, sampled).peak
    return {
        "backend": backend,
        "model": model,
        "device": devices["model"],
        "precision": precision,
        "gpu": gpu_name(devices["model"]),
        "load_seconds": load_seconds,
        "peak_rss_bytes": peak_rss(),
        "peak_gpu_bytes": peak[0] if peak else None,
        "peak_gpu_sampled": bool(peak and peak[1]),
        "clips": measured,
    }


def _worker_env() -> Dict[str, str]:
    """The environment of a measuring process: this checkout first on the import path."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))
    return env


def _stderr() -> Any:
    """Where a measuring process's stdout goes: this process's stderr, so that stdout carries only the table."""
    try:
        sys.stderr.fileno()
        return sys.stderr
    except (AttributeError, OSError, ValueError):  # replaced by something without a file descriptor
        return None


def measure_isolated(
    backend: str,
    model: str,
    device: Optional[str],
    clips: Sequence[Clip],
    data_dir: pathlib.Path,
    manifest: pathlib.Path,
) -> Dict[str, Any]:
    """measure(), in a new process (``python -m benchmarks.run --worker JOB OUT``) so that its peak memory is its own.

    Raises:
        BenchmarkError: if the process fails.
    """
    with tempfile.TemporaryDirectory(prefix="speech-toolkit-benchmark-") as folder:
        job, out = pathlib.Path(folder) / "job.json", pathlib.Path(folder) / "measured.json"
        settings = {"backend": backend, "model": model, "device": device, "data_dir": str(data_dir)}
        job.write_text(
            json.dumps({**settings, "manifest": str(manifest), "clips": [clip.id for clip in clips]}), encoding="utf-8"
        )
        command = [sys.executable, "-m", "benchmarks.run", "--worker", str(job), str(out)]
        done = subprocess.run(command, cwd=ROOT, env=_worker_env(), stdout=_stderr(), check=False)  # nosec B603
        if done.returncode != 0 or not out.is_file():
            raise BenchmarkError(f"measuring {backend} {model} failed (exit status {done.returncode}); see above")
        measured: Dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
        return measured


def worker(job_file: pathlib.Path, out: pathlib.Path) -> None:
    """The measuring process: measure() the job's model on its clips and write the measurements to *out*."""
    job = json.loads(job_file.read_text(encoding="utf-8"))
    by_id = {clip.id: clip for clip in load_manifest(pathlib.Path(job["manifest"])).clips}
    clips = [by_id[clip_id] for clip_id in job["clips"]]
    with contextlib.redirect_stdout(sys.stderr):  # libraries that print: stdout is the parent's table
        measured = measure(job["backend"], job["model"], job["device"], clips, pathlib.Path(job["data_dir"]))
    out.write_text(json.dumps(measured), encoding="utf-8")


@dataclasses.dataclass
class Totals:
    """A set's clips, added up."""

    clips: List[Clip] = dataclasses.field(default_factory=list)
    audio_seconds: float = 0.0
    seconds: float = 0.0
    errors: Errors = dataclasses.field(default_factory=Errors)
    languages: Dict[str, Errors] = dataclasses.field(default_factory=dict)

    def add(self, clip: Clip, audio_seconds: float, seconds: float, errors: Errors) -> None:
        self.clips.append(clip)
        self.audio_seconds += audio_seconds
        self.seconds += seconds
        self.errors += errors
        self.languages[clip.language] = self.languages.get(clip.language, Errors()) + errors

    def as_dict(self) -> Dict[str, Any]:
        return {
            "clips": len(self.clips),
            "fingerprint": fingerprint(self.clips),  # the same clips and references: only then do results compare
            "audio_seconds": round(self.audio_seconds, 3),
            "seconds": round(self.seconds, 3),
            "rtfx": rtfx(self.audio_seconds, self.seconds),
            **errors_dict(self.errors),
            "languages": {language: errors_dict(errors) for language, errors in sorted(self.languages.items())},
        }


def rtfx(audio_seconds: float, seconds: float) -> Optional[float]:
    return round(audio_seconds / seconds, 2) if seconds > 0 else None


def errors_dict(errors: Errors) -> Dict[str, Any]:
    return {
        "words": errors.words,
        "errors": errors.errors,
        "substitutions": errors.substitutions,
        "deletions": errors.deletions,
        "insertions": errors.insertions,
        "wer": None if errors.wer is None else round(errors.wer, 6),
    }


def summarize(measured: Dict[str, Any], clips: Sequence[Clip]) -> Dict[str, Any]:
    """A model's results: *measured* (from measure()) with each transcript scored, added up per set."""
    sets: Dict[str, Totals] = {}
    per_clip: Dict[str, Dict[str, Any]] = {}
    for clip in clips:
        found = measured["clips"][clip.id]
        errors = score(clip.text, found["text"], clip.language)
        sets.setdefault(clip.set, Totals()).add(clip, found["audio_seconds"], found["seconds"], errors)
        per_clip[clip.id] = {
            "set": clip.set,
            "audio_seconds": round(found["audio_seconds"], 3),
            "seconds": round(found["seconds"], 4),
            "words": errors.words,
            "errors": errors.errors,
            "hypothesis": found["text"],
        }
    audio_seconds = sum(totals.audio_seconds for totals in sets.values())
    seconds = sum(totals.seconds for totals in sets.values())
    summary = {key: value for key, value in measured.items() if key != "clips"}
    summary["load_seconds"] = round(summary["load_seconds"], 3)
    summary.update(
        audio_seconds=round(audio_seconds, 3),
        seconds=round(seconds, 3),
        rtfx=rtfx(audio_seconds, seconds),
        sets={name: totals.as_dict() for name, totals in sets.items()},
        clips=per_clip,
    )
    return summary


def _cell(value: Optional[float], digits: int, scale: float = 1.0, prefix: str = "") -> str:
    return "-" if value is None else f"{prefix}{value / scale:.{digits}f}"


def markdown(results: Dict[str, Any]) -> str:
    """The results as Markdown: the machine, one table row per model, and the WER per language when a set has several."""
    env = results["environment"]
    ram = f", {env['ram_bytes'] / 1024**3:.1f} GB of RAM" if env.get("ram_bytes") else ""
    packages = ", ".join(f"{name} {version}" for name, version in env["packages"].items())
    lines = [
        f"Machine: {env['machine']}, {env['cpu']} ({env['cpus']} CPUs){ram}. Python {env['python']}, speech_toolkit "
        f"{env['speech_toolkit']}, {packages}.",
        "",
    ]
    entries = results["results"]
    sets = list(dict.fromkeys(name for entry in entries for name in entry["sets"]))
    header = ["Backend", "Model", "Device", "Precision", "Load (s)", "Audio (s)", "RTFx", "Peak RSS (MiB)"]
    header += ["Peak GPU (MiB)", *(f"WER {name}" for name in sets)]
    lines += ["| " + " | ".join(header) + " |", "|---|---|---|---|" + "--:|" * (len(header) - 4)]
    for entry in entries:
        device = entry["device"] + (f" ({entry['gpu']})" if entry.get("gpu") else "")
        gpu = _cell(entry["peak_gpu_bytes"], 0, MIB, "~" if entry["peak_gpu_sampled"] else "")
        row = [entry["backend"], entry["model"], device, entry["precision"] or "-", _cell(entry["load_seconds"], 1)]
        row += [_cell(entry["audio_seconds"], 0), _cell(entry["rtfx"], 1), _cell(entry["peak_rss_bytes"], 0, MIB), gpu]
        row += [_cell(entry["sets"][name]["wer"], 3) if name in entry["sets"] else "-" for name in sets]
        lines.append("| " + " | ".join(row) + " |")
    lines += _language_table(entries, sets)
    lines += ["", NOTES]
    return "\n".join(lines) + "\n"


def _language_table(entries: List[Dict[str, Any]], sets: List[str]) -> List[str]:
    columns = [(name, language) for name in sets for language in _languages(entries, name)]
    if len(columns) == len(sets):  # every set has one language: the main table says it all
        return []
    header = ["Backend", "Model", *(f"{name} {language}" for name, language in columns)]
    lines = ["", "WER by language:", "", "| " + " | ".join(header) + " |", "|---|---|" + "--:|" * len(columns)]
    for entry in entries:
        cells = [
            _cell(entry["sets"].get(name, {}).get("languages", {}).get(language, {}).get("wer"), 3)
            for name, language in columns
        ]
        lines.append("| " + " | ".join([entry["backend"], entry["model"], *cells]) + " |")
    return lines


def _languages(entries: List[Dict[str, Any]], name: str) -> List[str]:
    return sorted({language for entry in entries for language in entry["sets"].get(name, {}).get("languages", {})})


NOTES = (
    "Load: the model load (a first run's includes the download). RTFx: seconds of audio per second of transcription "
    "(higher is faster; decoding the audio and loading the model excluded). Peak RSS: the measuring process's peak "
    "resident memory. Peak GPU: torch's count, or with ~ the GPU's use sampled with nvidia-smi (CTranslate2). WER: "
    "word error rate over all of a set's words, after Whisper's text normalizers."
)


def default_json() -> pathlib.Path:
    return RESULTS / f"{datetime.datetime.now(datetime.timezone.utc):%Y%m%d-%H%M%S}.json"


def measure_all(
    args: argparse.Namespace, models: List[Tuple[str, str]], clips: List[Clip], data_dir: pathlib.Path
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Measure and score *models*; returns the results, and a message per model that failed."""
    results: List[Dict[str, Any]] = []
    failed: List[str] = []
    for backend, model in models:
        log_to_stderr(f"Measuring {backend} {model} on {len(clips)} clips")
        try:
            if args.in_process:
                measured = measure(backend, model, args.device, clips, data_dir)
            else:
                measured = measure_isolated(backend, model, args.device, clips, data_dir, args.manifest)
        except (BenchmarkError, ImportError, RuntimeError, ValueError, OSError) as e:
            failed.append(f"{backend} {model}: {e}")
            log_to_stderr(f"Error: {backend} {model}: {e}")
            continue
        results.append(summarize(measured, clips))
    return results, failed


def run(args: argparse.Namespace) -> List[str]:
    """Measure the models of *args*, print the table, write the JSON; returns the failures."""
    # An unknown backend, or missing scoring packages, fail before anything is downloaded or measured.
    models = model_list(args.specs)
    normalizer("en")
    score("a", "a", "en")
    manifest = load_manifest(args.manifest)
    clips = manifest.select(args.sets, args.limit)
    data_dir = args.data_dir or default_data_dir()
    prepare(manifest, data_dir, clips)
    results, failed = measure_all(args, models, clips, data_dir)
    if not results:
        return failed or ["nothing was measured"]
    document = {
        "schema": SCHEMA,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "environment": environment(),
        "settings": {
            "device": args.device,
            "isolated": not args.in_process,
            "sets": sorted({clip.set for clip in clips}),
            "limit": args.limit,
            "scoring": scoring(),
        },
        "results": results,
    }
    path = args.json or default_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sys.stdout.write(markdown(document))
    log_to_stderr(f"Results written to {path}")
    return failed


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.worker is not None:
        worker(*args.worker)
        return
    try:
        failed = run(args)
    except (DataError, ImportError, ValueError) as e:
        sys.exit(f"Error: {e}")
    if failed:
        sys.exit(f"Error: {len(failed)} model{'s' * (len(failed) != 1)} failed: " + "; ".join(failed))


if __name__ == "__main__":
    main()
