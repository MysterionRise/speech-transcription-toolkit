"""The device the integration tests run on, whether the GPU is usable, and whether a model fits in memory.

- ``INTEGRATION_DEVICE``: ``auto`` (the default: each backend picks, as with no ``--device``), ``cpu`` or ``cuda``.
  The commands get ``--device``, the Python API ``device=``, and each test checks the device its models ran on.
  With ``cuda``, a test fails, saying why, when there is no GPU its backend can use.
- ``INTEGRATION_TIER``: ``default``, or ``full`` for the larger models (test_full_tier.py), which are skipped, with
  the reason, when they don't fit in the GPU's memory (or in RAM on the CPU).

What torch and CTranslate2 see of the GPU is probed once, in a subprocess, so the test process holds no GPU memory.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import pytest

from tests.integration.instrument import MIB, PROBE_PREFIX

ROOT = pathlib.Path(__file__).resolve().parents[2]
GB = 1024**3

DEVICES = ("auto", "cpu", "cuda")
TIERS = ("default", "full")
DEVICE = os.getenv("INTEGRATION_DEVICE", "").strip().lower() or "auto"
TIER = os.getenv("INTEGRATION_TIER", "").strip().lower() or "default"

# The library that runs each backend (and "diarize", pyannote) on the GPU.
RUNTIMES: Dict[str, str] = {
    "whisper": "torch",
    "faster-whisper": "ctranslate2",
    "parakeet": "torch",
    "canary": "torch",
    "voxtral": "torch",
    "diarize": "torch",
}

# Approximate peak memory, in GB, each model needs to transcribe the long clip, as (GPU, CPU). On the GPU the backends
# load float16 or bfloat16 weights (openai-whisper keeps float32 weights and decodes in float16); on the CPU float32
# (int8 for faster-whisper). From the projects' published figures, rounded up; the report's peak GPU column shows
# what the models really used, to correct these by.
MEMORY_GB: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("faster-whisper", "large-v3"): (5, 4),
    ("faster-whisper", "turbo"): (3, 3),
    ("whisper", "turbo"): (6, 6),
    ("whisper", "large-v3"): (10, 10),
    ("parakeet", "parakeet-tdt-0.6b-v2"): (3, 4),
    ("parakeet", "parakeet-tdt-0.6b-v3"): (3, 4),
    ("canary", "canary-1b-v2"): (4, 6),
    ("voxtral", "voxtral-mini"): (11, 20),  # 4.7B parameters: 9.4 GB in bfloat16, 19 GB in float32
    ("diarize", "pyannote"): (1, 2),
}
# The share of a device's memory a model may count on; the rest is for the system, the desktop and the CUDA context.
USABLE_SHARE = 0.85

PROBE_TIMEOUT = 300  # seconds: importing torch can be slow the first time


def check_settings() -> None:
    """Fail the test if INTEGRATION_DEVICE or INTEGRATION_TIER has a value the suite doesn't know."""
    if DEVICE not in DEVICES:
        pytest.fail(f"INTEGRATION_DEVICE must be auto, cpu or cuda, not {DEVICE!r}", pytrace=False)
    if TIER not in TIERS:
        pytest.fail(f"INTEGRATION_TIER must be default or full, not {TIER!r}", pytrace=False)


def device_args() -> Tuple[str, ...]:
    """The --device option for the commands; none with auto."""
    return () if DEVICE == "auto" else ("--device", DEVICE)


def api_device() -> Optional[str]:
    """The device= argument for Transcriber; None (pick automatically) with auto."""
    return None if DEVICE == "auto" else DEVICE


def check_devices(backend: str, devices: Dict[str, str]) -> None:
    """Fail unless every model of a run, ``{"model": "cuda", "diarization": "cuda"}``, is on INTEGRATION_DEVICE."""
    if DEVICE == "auto":
        return
    names = {"model": f"the {backend} model", "diarization": "the pyannote pipeline"}
    wrong = [f"{names.get(part, part)} ran on {device}" for part, device in devices.items() if device != DEVICE]
    if wrong:
        pytest.fail(f"INTEGRATION_DEVICE={DEVICE}, but {' and '.join(wrong)}", pytrace=False)


@dataclasses.dataclass(frozen=True)
class Card:
    """A GPU as nvidia-smi lists it."""

    name: str
    memory: int  # bytes
    driver: str


def parse_nvidia_smi(output: str) -> List[Card]:
    """The GPUs in the output of ``nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits``."""
    cards = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.rsplit(",", 2)]  # the name is the only field that could hold a comma
        try:
            cards.append(Card(parts[0], int(float(parts[1]) * MIB), parts[2]))
        except (IndexError, ValueError):
            continue
    return cards


def parse_probe(output: str) -> Optional[Dict[str, Any]]:
    """What ``python -m tests.integration.instrument --probe`` found, from its stdout; None if it printed nothing."""
    for line in output.splitlines():
        if line.startswith(PROBE_PREFIX):
            found = json.loads(line[len(PROBE_PREFIX) :])
            return found if isinstance(found, dict) else None
    return None


@dataclasses.dataclass(frozen=True)
class Gpu:
    """What torch and CTranslate2 see of the GPU (None: not installed), and the GPUs nvidia-smi lists."""

    torch: Optional[Dict[str, Any]] = None
    ctranslate2: Optional[Dict[str, Any]] = None
    cards: Optional[Tuple[Card, ...]] = None  # None: nvidia-smi isn't installed
    problem: str = ""  # why the probe itself failed, if it did

    def usable(self, runtime: str) -> bool:
        """Whether *runtime*, "torch" or "ctranslate2", can run models on a CUDA GPU."""
        if runtime == "torch":
            return bool(self.torch and self.torch.get("available"))
        if runtime == "ctranslate2":
            return bool(self.ctranslate2 and self.ctranslate2.get("cuda_devices", 0) > 0)
        return False

    @property
    def found(self) -> bool:
        """Whether any library, or nvidia-smi, finds an NVIDIA GPU."""
        return bool(self.cards) or self.usable("torch") or self.usable("ctranslate2")

    @property
    def name(self) -> str:
        if self.usable("torch"):
            return str(self.torch["name"])  # type: ignore[index]
        return self.cards[0].name if self.cards else ""

    @property
    def memory(self) -> Optional[int]:
        """The first GPU's memory in bytes (the one "cuda" means), or None if unknown."""
        if self.usable("torch"):
            return int(self.torch["memory"])  # type: ignore[index]
        return self.cards[0].memory if self.cards else None

    def why_not(self, runtime: str) -> str:
        """Why *runtime* can't use the GPU, and what to do about it."""
        if runtime == "ctranslate2":
            info = self.ctranslate2
            if info is None:
                return "CTranslate2 (faster-whisper) isn't installed"
            if "version" not in info:
                return f"CTranslate2 fails to import: {info.get('error')}"
            return f"CTranslate2 {info['version']} finds no CUDA GPU{_error(info)}; check the driver with nvidia-smi"
        info = self.torch
        if info is None:
            return "torch isn't installed"
        if "version" not in info:
            return f"torch fails to import: {info.get('error')}"
        if not info.get("cuda"):
            return (
                f"torch {info['version']} is a CPU-only build; install a CUDA build of torch "
                "(see docs/manual-testing.md)"
            )
        return (
            f"torch {info['version']} (CUDA {info['cuda']}) finds no usable GPU{_error(info)}; check the driver with "
            "nvidia-smi, and CUDA_VISIBLE_DEVICES"
        )

    def summary(self) -> str:
        """One line on what was found, for messages and the report."""
        if self.cards:
            card = self.cards[0]
            parts = [f"{card.name} ({card.memory / GB:.1f} GB, driver {card.driver})"]
        elif self.name:
            parts = [f"{self.name} ({(self.memory or 0) / GB:.1f} GB)"]
        else:
            parts = ["nvidia-smi: not found" if self.cards is None else "nvidia-smi: no GPU listed"]
        if self.torch is None:
            parts.append("torch: not installed")
        elif "version" in self.torch:
            cuda = f"CUDA {self.torch['cuda']}" if self.torch.get("cuda") else "CPU-only build"
            usable = "uses the GPU" if self.usable("torch") else "no GPU"
            bf16 = ", bfloat16" if self.torch.get("bf16") else ""
            parts.append(f"torch {self.torch['version']} ({cuda}, {usable}{bf16})")
        else:
            parts.append(f"torch: {self.torch.get('error')}")
        if self.ctranslate2 is None:
            parts.append("CTranslate2: not installed")
        elif "version" in self.ctranslate2:
            count = self.ctranslate2.get("cuda_devices", 0)
            parts.append(f"CTranslate2 {self.ctranslate2['version']} ({count} CUDA device{'s' * (count != 1)})")
        else:
            parts.append(f"CTranslate2: {self.ctranslate2.get('error')}")
        if self.problem:
            parts.append(f"probe: {self.problem}")
        return "; ".join(parts)


def _error(info: Dict[str, Any]) -> str:
    return f" ({info['error']})" if info.get("error") else ""


def query_cards() -> Optional[Tuple[Card, ...]]:
    """The GPUs nvidia-smi lists (none if it fails), or None if it isn't installed."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    command = [nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]
    try:
        done = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return ()
    return tuple(parse_nvidia_smi(done.stdout))


def probe_gpu() -> Gpu:
    """Ask torch and CTranslate2, in a subprocess, what GPU they can use, and nvidia-smi which GPUs there are."""
    command = [sys.executable, "-m", "tests.integration.instrument", "--probe"]
    env = dict(os.environ, PYTHONUTF8="1")
    problem, found = "", None
    try:
        done = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=PROBE_TIMEOUT
        )
        found = parse_probe(done.stdout)
        if found is None:
            problem = f"exited with {done.returncode}: {done.stderr.strip()[-500:]}"
    except (OSError, subprocess.SubprocessError) as e:
        problem = f"{type(e).__name__}: {e}"
    found = found or {}
    return Gpu(found.get("torch"), found.get("ctranslate2"), query_cards(), problem)


_probed: Optional[Gpu] = None


def gpu() -> Gpu:
    """The GPU as the probe found it; probed once per session."""
    global _probed
    if _probed is None:
        _probed = probe_gpu()
    return _probed


def probed_gpu() -> Optional[Gpu]:
    """The GPU, if this session has probed it already."""
    return _probed


def require_gpu() -> None:
    """With INTEGRATION_DEVICE=cuda, fail the test unless some library, or nvidia-smi, finds an NVIDIA GPU."""
    if DEVICE == "cuda" and not gpu().found:
        pytest.fail(
            f"INTEGRATION_DEVICE=cuda, but there is no usable CUDA GPU ({gpu().summary()}). Install the NVIDIA "
            "driver and the CUDA libraries (see docs/manual-testing.md), or run with INTEGRATION_DEVICE=cpu.",
            pytrace=False,
        )


def require_device(*features: str) -> None:
    """With INTEGRATION_DEVICE=cuda, fail the test unless the library each feature runs on can use the GPU."""
    if DEVICE != "cuda":
        return
    for feature in features:
        runtime = RUNTIMES.get(feature)
        if runtime and not gpu().usable(runtime):
            pytest.fail(
                f"INTEGRATION_DEVICE=cuda, but {feature} can't use the GPU: {gpu().why_not(runtime)}", pytrace=False
            )


def memory_device(feature: str) -> str:
    """The device *feature* runs on: INTEGRATION_DEVICE, or with auto the GPU if its library can use one."""
    if DEVICE != "auto":
        return DEVICE
    runtime = RUNTIMES.get(feature)
    return "cuda" if runtime and gpu().usable(runtime) else "cpu"


def total_ram() -> Optional[int]:
    """This machine's physical memory in bytes, or None if it can't be told."""
    if sys.platform == "win32":
        return _windows_ram()
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return None


def _windows_ram() -> Optional[int]:
    import ctypes

    class MemoryStatus(ctypes.Structure):  # MEMORYSTATUSEX
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
        return None
    return int(status.ullTotalPhys)


def device_memory(device: str) -> Tuple[str, Optional[int]]:
    """What *device*'s memory is called in messages, and its size in bytes (None if unknown)."""
    if device == "cuda":
        found = gpu()
        return found.name or "the GPU", found.memory
    return "this machine", total_ram()


def require_memory(*models: Tuple[str, str]) -> None:
    """Skip the test unless the models, (backend or "diarize", model) pairs, fit in memory together (MEMORY_GB)."""
    needs: Dict[str, float] = {}
    for feature, model in models:
        device = memory_device(feature)
        on_gpu, on_cpu = MEMORY_GB[(feature, model)]
        needs[device] = needs.get(device, 0.0) + (on_gpu if device == "cuda" else on_cpu)
    for device, need in needs.items():
        where, total = device_memory(device)
        if total is None:
            continue  # unknown: let the model try
        usable = total * USABLE_SHARE / GB
        if need > usable:
            kind = "GPU memory" if device == "cuda" else "RAM"
            names = " with ".join(f"{feature} {model}" for feature, model in models)
            pytest.skip(
                f"{names} needs about {need:g} GB of {kind}; {where} has {total / GB:.1f} GB, of which "
                f"{USABLE_SHARE:.0%} ({usable:.1f} GB) is counted as usable"
            )


def sampling_nvidia_smi() -> Optional[str]:
    """nvidia-smi, if the commands should sample the GPU's memory with it: on a run that may use the GPU."""
    return None if DEVICE == "cpu" else shutil.which("nvidia-smi")


# Distributions whose versions the report lists, when installed.
REPORTED_PACKAGES = (
    "faster-whisper",
    "ctranslate2",
    "openai-whisper",
    "torch",
    "transformers",
    "mistral-common",
    "pyannote.audio",
    "fastapi",
)


def environment_lines() -> List[str]:
    """The report's description of this machine and run, as Markdown list items."""
    from importlib import metadata

    import speech_toolkit

    versions = []
    for distribution in REPORTED_PACKAGES:
        try:
            versions.append(f"{distribution} {metadata.version(distribution)}")
        except metadata.PackageNotFoundError:
            continue
    machine = f"- Machine: {platform.platform()} ({platform.machine()}), Python {platform.python_version()}"
    ram = total_ram()
    if ram:
        machine += f", {ram / GB:.1f} GB of RAM"
    found = probed_gpu()
    if found is None and DEVICE != "cpu" and shutil.which("nvidia-smi"):
        found = gpu()  # a GPU machine: worth the probe, for the report
    return [
        machine,
        f"- speech_toolkit {speech_toolkit.__version__}; INTEGRATION_DEVICE={DEVICE}, INTEGRATION_TIER={TIER}",
        f"- GPU: {found.summary()}" if found else "- GPU: not checked",
        f"- Packages: {', '.join(versions) or 'no model libraries'}",
    ]
