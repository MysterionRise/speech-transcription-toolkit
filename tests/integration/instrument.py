"""What runs inside the commands the integration tests start: the instrumentation, and the GPU probe.

The tests run ``transcribe`` and ``transcribe-server`` through this module::

    python -m tests.integration.instrument speech_toolkit <args>          # as python -m speech_toolkit <args>
    python -m tests.integration.instrument speech_toolkit.server <args>   # as python -m speech_toolkit.server <args>

The command runs unchanged. With INTEGRATION_EVENTS naming a file, one JSON line is appended to it per event:

- ``{"event": "load", ...}`` once a ``Transcriber`` has loaded its models: the backend and model, the device each
  model is on (``{"model": "cuda", "diarization": "cuda"}``) and the model's precision, read from the model objects
  themselves, so a model that stayed on the CPU shows up whatever ``--device`` asked for; and the seconds it took.
- ``{"event": "transcribe", ...}`` after each file: the audio's length, the seconds it took, and the peak GPU memory
  so far: torch's own count, and with INTEGRATION_NVIDIA_SMI naming nvidia-smi, the GPU's memory use sampled with it,
  above the level before the model loaded.

``python -m tests.integration.instrument --probe`` prints what torch and CTranslate2 see of the GPU, as JSON.

Nothing here writes to stdout, which carries only the transcript, and only ``--probe`` imports torch.
"""

from __future__ import annotations

import functools
import json
import os
import pathlib
import runpy
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional, Sequence, Tuple

EVENTS_VARIABLE = "INTEGRATION_EVENTS"
NVIDIA_SMI_VARIABLE = "INTEGRATION_NVIDIA_SMI"
PROBE_PREFIX = "integration-probe: "  # marks the probe's line: an imported library could print to stdout too
MIB = 1024 * 1024


def device_type(device: Any) -> str:
    """The kind of *device*, "cuda" or "cpu": from a torch.device, a string such as "cuda:0", or None (unknown)."""
    kind = getattr(device, "type", device)
    return str(kind).partition(":")[0] if kind else "unknown"


def model_device(backend: Any) -> Tuple[str, str]:
    """The device type and precision of a loaded backend's model, read from the model object."""
    model = getattr(backend, "_model", None)
    inner = getattr(model, "model", None)  # faster-whisper's WhisperModel holds a ctranslate2.models.Whisper
    if isinstance(getattr(inner, "device", None), str):
        return device_type(inner.device), str(getattr(inner, "compute_type", ""))
    parameters = getattr(model, "parameters", None)
    first = next(iter(parameters()), None) if callable(parameters) else None
    if first is not None:  # a torch module: openai-whisper, or a transformers model
        device, precision = device_type(first.device), str(first.dtype).replace("torch.", "")
        if getattr(backend, "name", "") == "whisper" and device != "cpu":
            precision = "float16"  # openai-whisper keeps float32 weights and decodes in float16 on a GPU
        return device, precision
    return device_type(getattr(backend, "device", None)), ""


def pipeline_device(pipeline: Any) -> str:
    """The device of a pyannote pipeline: its ``device``, else its segmentation model's."""
    for part in (pipeline, getattr(pipeline, "_segmentation", None)):
        device = getattr(part, "device", None)
        if device is not None:
            return device_type(device)
    return "unknown"


def transcriber_devices(transcriber: Any) -> Tuple[Dict[str, str], str]:
    """Where a Transcriber's models are, ``{"model": "cuda", "diarization": "cuda"}``, and the model's precision."""
    device, precision = model_device(transcriber.backend)
    devices = {"model": device}
    pipeline = getattr(transcriber, "_pipeline", None)
    if pipeline is not None:
        devices["diarization"] = pipeline_device(pipeline)
    return devices, precision


def torch_peak() -> Optional[int]:
    """The most GPU memory torch's allocator has held in this process, in bytes; None if torch hasn't used CUDA."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_initialized():
            return int(torch.cuda.max_memory_reserved())
    except Exception:  # an old or partly installed torch: no number rather than a failed run
        return None
    return None


def audio_seconds(audio: Any) -> Optional[float]:
    """The length of the audio file *audio* in seconds, or None if it can't be decoded."""
    try:
        from speech_toolkit.media import SAMPLE_RATE, load_audio

        return len(load_audio(audio)) / SAMPLE_RATE
    except Exception:  # the measurement must never fail a transcription that worked
        return None


class GpuSampler:
    """Samples the GPU's memory use with nvidia-smi in a background thread, from before the model loads.

    It sees every process on the GPU, so the peak above the starting level is approximate; but it is the only number
    for CTranslate2 (faster-whisper), which keeps no count torch can read, and works on Windows, where nvidia-smi
    can't tell processes' memory apart.
    """

    def __init__(self, nvidia_smi: str, interval: float = 0.2) -> None:
        self.command = [nvidia_smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"]
        self.interval = interval
        self.baseline = self.used()
        self.peak = self.baseline
        self._stop = threading.Event()
        if self.baseline is not None:
            threading.Thread(target=self._sample, name="gpu-memory-sampler", daemon=True).start()

    def used(self) -> Optional[int]:
        """The GPU's memory in use now, in bytes, or None if nvidia-smi can't tell."""
        try:
            done = subprocess.run(self.command, capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL)
            return int(float(done.stdout.split()[0]) * MIB)
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None

    def _sample(self) -> None:
        while not self._stop.wait(self.interval):
            used = self.used()
            if used is not None and (self.peak is None or used > self.peak):
                self.peak = used

    def stop(self) -> None:
        self._stop.set()

    def peak_above_baseline(self) -> Optional[int]:
        """The highest use seen above the starting level, in bytes."""
        if self.baseline is None or self.peak is None:
            return None
        return max(0, self.peak - self.baseline)


def write_event(events: pathlib.Path, event: Dict[str, Any]) -> None:
    """Append *event* to *events* as one JSON line, written at once so that a reader never sees half of it."""
    with events.open("a", encoding="utf-8") as out:
        out.write(json.dumps(event) + "\n")


def install(events: pathlib.Path, sampler: Optional[GpuSampler] = None, target: Any = None) -> None:
    """Wrap ``Transcriber`` (or *target*, a subclass) so that loading and each transcription append an event."""
    if target is None:
        from speech_toolkit import api

        target = api.Transcriber
    load, transcribe = target.__init__, target.transcribe

    @functools.wraps(load)
    def instrumented_init(self: Any, *args: Any, **kwargs: Any) -> None:
        started = time.perf_counter()
        load(self, *args, **kwargs)
        seconds = time.perf_counter() - started
        devices, precision = transcriber_devices(self)
        event = {"backend": self.backend_name, "model": self.model_name, "devices": devices, "precision": precision}
        write_event(events, {"event": "load", **event, "seconds": seconds})

    @functools.wraps(transcribe)
    def instrumented_transcribe(self: Any, audio: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = transcribe(self, audio, *args, **kwargs)
        seconds = time.perf_counter() - started
        event = {
            "event": "transcribe",
            "audio": str(audio),
            "audio_seconds": audio_seconds(audio),
            "seconds": seconds,
            "torch_peak_bytes": torch_peak(),
            "sampled_peak_bytes": sampler.peak_above_baseline() if sampler else None,
        }
        write_event(events, event)
        return result

    target.__init__ = instrumented_init
    target.transcribe = instrumented_transcribe


def _import_error(error: BaseException) -> Dict[str, Any]:
    return {"error": f"{type(error).__name__}: {error}"}


def probe_torch() -> Optional[Dict[str, Any]]:
    """torch's version, the CUDA version it was built for (None: a CPU-only build), and the GPU it can use."""
    try:
        import torch
    except ModuleNotFoundError as e:
        return None if e.name == "torch" else _import_error(e)
    except Exception as e:  # e.g. a DLL that fails to load on Windows
        return _import_error(e)
    info: Dict[str, Any] = {"version": str(torch.__version__), "cuda": torch.version.cuda, "available": False}
    try:
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            info.update(available=True, name=properties.name, memory=int(properties.total_memory))
            info["bf16"] = bool(torch.cuda.is_bf16_supported())
    except Exception as e:
        info.update(_import_error(e))
    return info


def probe_ctranslate2() -> Optional[Dict[str, Any]]:
    """CTranslate2's version and the number of CUDA GPUs it can use."""
    try:
        import ctranslate2
    except ModuleNotFoundError as e:
        return None if e.name == "ctranslate2" else _import_error(e)
    except Exception as e:
        return _import_error(e)
    info: Dict[str, Any] = {"version": str(ctranslate2.__version__), "cuda_devices": 0}
    try:
        info["cuda_devices"] = int(ctranslate2.get_cuda_device_count())
    except Exception as e:
        info.update(_import_error(e))
    return info


def main(argv: Sequence[str]) -> None:
    if list(argv[:1]) == ["--probe"]:
        print(PROBE_PREFIX + json.dumps({"torch": probe_torch(), "ctranslate2": probe_ctranslate2()}))
        return
    if not argv:
        sys.exit("usage: python -m tests.integration.instrument (--probe | MODULE [ARG ...])")
    module, args = argv[0], list(argv[1:])
    events = os.getenv(EVENTS_VARIABLE)
    if events:
        nvidia_smi = os.getenv(NVIDIA_SMI_VARIABLE)
        install(pathlib.Path(events), GpuSampler(nvidia_smi) if nvidia_smi else None)
    sys.argv = [module, *args]
    runpy.run_module(module, run_name="__main__", alter_sys=True)  # exits as the command does


if __name__ == "__main__":
    main(sys.argv[1:])
