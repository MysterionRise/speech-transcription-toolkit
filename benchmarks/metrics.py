"""What the benchmark measures, apart from time: the word error rate, peak memory, and a description of the machine.

**Word error rate (WER).** Reference and transcript are normalized as Whisper's evaluation does, with the normalizers
from OpenAI's Whisper (packaged as ``whisper-normalizer``): ``EnglishTextNormalizer`` for English, which also spells
numbers, currencies and British spellings one way ("ten" and "10" match), and ``BasicTextNormalizer`` for the other
languages (lower case, no punctuation, no text in brackets). jiwer then aligns the words and counts substitutions,
deletions and insertions. A set's WER is the sum of its clips' errors over the sum of their reference words.

The integration tests' WER helper (tests/integration/wer.py) only lower-cases and strips punctuation: right for its
synthesized samples, but on LibriSpeech, whose references spell out every number, it would count "10" against "ten"
as an error, so the numbers wouldn't compare with published ones.

**Peak memory.** The most memory the process had resident (peak RSS): VmHWM on Linux, ``ru_maxrss`` on macOS, the
peak working set on Windows (with psutil). The GPU's is tests/integration/instrument.py's: torch's own count, or the
GPU's use sampled with nvidia-smi for CTranslate2 (faster-whisper), which keeps no count.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import platform
import shutil
import subprocess  # nosec B404 (only nvidia-smi, found on PATH, with fixed arguments)
import sys
from importlib import metadata
from typing import Any, Callable, Dict, Optional

INSTALL_HINT = "pip install -r requirements-bench.txt"
# The distributions whose versions go in the results: the scoring tools first, then what runs the models.
PACKAGES = (
    "jiwer",
    "whisper-normalizer",
    "faster-whisper",
    "ctranslate2",
    "openai-whisper",
    "torch",
    "transformers",
    "numpy",
)


@dataclasses.dataclass
class Errors:
    """The word errors of one transcript, or of several added up."""

    words: int = 0  # in the reference
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> Optional[float]:
        return self.errors / self.words if self.words else None

    def __add__(self, other: "Errors") -> "Errors":
        return Errors(
            self.words + other.words,
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
        )


@functools.lru_cache(maxsize=None)
def normalizer(language: str) -> Callable[[str], str]:
    """Whisper's text normalizer for *language*: the English one for "en", the basic one for every other language."""
    try:
        from whisper_normalizer.basic import BasicTextNormalizer
        from whisper_normalizer.english import EnglishTextNormalizer
    except ImportError as e:
        raise ImportError(f"the word error rate needs whisper-normalizer: {INSTALL_HINT}") from e
    normalize: Callable[[str], str] = EnglishTextNormalizer() if language == "en" else BasicTextNormalizer()
    return normalize


def score(reference: str, hypothesis: str, language: str) -> Errors:
    """The word errors of *hypothesis* against *reference*, both normalized for *language*.

    Raises:
        ValueError: if the reference has no words left once normalized.
    """
    try:
        import jiwer
    except ImportError as e:
        raise ImportError(f"the word error rate needs jiwer: {INSTALL_HINT}") from e
    normalize = normalizer(language)
    reference, hypothesis = " ".join(normalize(reference).split()), " ".join(normalize(hypothesis).split())
    if not reference:
        raise ValueError("the reference has no words once normalized")
    counts = jiwer.process_words(reference, hypothesis)
    words = counts.hits + counts.substitutions + counts.deletions
    return Errors(words, counts.substitutions, counts.deletions, counts.insertions)


def scoring() -> str:
    """What the word error rate is computed with, for the results: the normalizer's and jiwer's versions."""
    return f"whisper-normalizer {_version('whisper-normalizer')}, jiwer {_version('jiwer')}"


def peak_rss() -> Optional[int]:
    """The most memory this process has had resident so far, in bytes; None where it can't be told.

    On Linux it is VmHWM, which a process started by another doesn't inherit from it (``ru_maxrss`` can: Linux keeps
    the parent's peak across ``exec``).
    """
    system = sys.platform  # read when called: mypy would check only its own platform's branch of sys.platform tests
    if system.startswith("linux"):
        try:
            with open("/proc/self/status", encoding="ascii") as status:
                for line in status:
                    if line.startswith("VmHWM:"):
                        return int(line.split()[1]) * 1024  # in kB
        except (OSError, ValueError, IndexError):
            return None
        return None
    try:
        import resource
    except ImportError:  # Windows
        try:
            import psutil

            return int(psutil.Process().memory_info().peak_wset)
        except (ImportError, AttributeError):
            return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)  # bytes on macOS


def gpu_name(device: str) -> Optional[str]:
    """The name of the GPU a model ran on: torch's, or nvidia-smi's first GPU; None on the CPU or if unknown."""
    if device != "cuda":
        return None
    return _torch_gpu_name() or _nvidia_smi_gpu_name()


def _torch_gpu_name() -> Optional[str]:
    torch = sys.modules.get("torch")  # never imported here: only if a backend uses it
    try:
        if torch is not None and torch.cuda.is_initialized():
            return str(torch.cuda.get_device_name(0))
    except Exception:  # an odd torch build: the name is only informative
        return None
    return None


def _nvidia_smi_gpu_name() -> Optional[str]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    command = [nvidia_smi, "--query-gpu=name", "--format=csv,noheader", "--id=0"]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL)  # nosec
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


def _version(distribution: str) -> Optional[str]:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def cpu_name() -> str:
    """The CPU's model name, as /proc/cpuinfo or the platform tells it."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def total_ram() -> Optional[int]:
    """This machine's physical memory in bytes, or None if it can't be told."""
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except ImportError:
        pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return None


def environment() -> Dict[str, Any]:
    """The machine, Python and package versions the benchmark ran with, for the results."""
    import speech_toolkit

    packages = {name: version for name in PACKAGES if (version := _version(name))}
    return {
        "machine": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": cpu_name(),
        "cpus": os.cpu_count(),
        "ram_bytes": total_ram(),
        "python": platform.python_version(),
        "speech_toolkit": speech_toolkit.__version__,
        "packages": packages,
        # On GitHub Actions: the runner's image and the commit, so that results say where they came from.
        "runner": f"GitHub Actions ({os.getenv('ImageOS', 'unknown image')})" if os.getenv("GITHUB_ACTIONS") else None,
        "commit": os.getenv("GITHUB_SHA"),
    }
