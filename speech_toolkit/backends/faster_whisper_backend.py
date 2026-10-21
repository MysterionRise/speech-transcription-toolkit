"""faster-whisper transcription backend, the default.

faster-whisper re-implements Whisper on CTranslate2: it is several times faster than openai-whisper, uses less memory
and does not need torch. It is a dependency of the package, so every install has it; models download from the Hugging
Face Hub on first use.

On an NVIDIA GPU, CTranslate2 also needs NVIDIA's CUDA 12 libraries (cuBLAS, plus cuDNN before CTranslate2 4.6.3),
which pip doesn't install. ``load_model()`` checks that they load before it uses the GPU.
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from ..errors import BackendUnavailableError, ModelLoadError, ModelNotFoundError, SpeechToolkitWarning
from ..types import Segment
from .base import TranscriptionBackend, TranscriptionResult


class FasterWhisperBackend(TranscriptionBackend):
    """faster-whisper (CTranslate2) transcription backend."""

    name = "faster-whisper"
    description = "faster-whisper - CTranslate2 Whisper: faster, lighter, no torch needed"
    capabilities = frozenset({"translate", "language_detection", "prompt", "vad", "word_timestamps"})
    requires = ("faster_whisper",)

    # Mirrors faster_whisper.available_models(); kept static so listing models needs no import.
    MODELS = [
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "distil-small.en",
        "medium",
        "medium.en",
        "distil-medium.en",
        "large-v1",
        "large-v2",
        "large-v3",
        "large",
        "distil-large-v2",
        "distil-large-v3",
        "distil-large-v3.5",
        "large-v3-turbo",
        "turbo",
    ]

    @classmethod
    def available_models(cls) -> List[str]:
        """Return list of available faster-whisper model names."""
        return cls.MODELS.copy()

    @classmethod
    def default_model(cls) -> str:
        """Return 'turbo' as the default - best balance of speed and accuracy."""
        return "turbo"

    def load_model(self, model_name: str, device: Optional[str] = None) -> None:
        """Load a faster-whisper model (int8 on CPU, float16 on CUDA).

        Args:
            model_name: One of the available model names.
            device: 'cpu', 'cuda', or None for the GPU when CTranslate2 sees one. If the CUDA libraries it needs
                don't load, None warns and uses the CPU.

        Raises:
            BackendUnavailableError: If faster-whisper is not installed.
            ModelNotFoundError: If model_name is not recognized.
            ModelLoadError: If the model cannot be downloaded or loaded, or device is 'cuda' and the CUDA libraries
                are missing.
        """
        if model_name not in self.MODELS:
            raise ModelNotFoundError(f"Unknown faster-whisper model: {model_name}. Available: {', '.join(self.MODELS)}")

        try:
            import ctranslate2
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise BackendUnavailableError(
                f"faster-whisper backend needs faster-whisper ({e}), which speech-transcription-toolkit depends on. "
                "Install it with: pip install speech-transcription-toolkit"
            ) from e

        auto = device is None
        if auto:
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        # A missing library would fail the first transcription, or on Windows end the process, so check first. Machines
        # without a GPU skip the check.
        if device == "cuda" and (auto or ctranslate2.get_cuda_device_count() > 0):
            problem = _cuda_library_problem(getattr(ctranslate2, "__version__", ""))
            if problem and not auto:
                raise ModelLoadError(f"faster-whisper can't run on CUDA. {problem}")
            if problem:
                warnings.warn(
                    f"faster-whisper runs on the CPU instead of the GPU. {problem} Pass --device cpu to skip this check.",
                    SpeechToolkitWarning,
                    stacklevel=2,
                )
                device = "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        try:
            self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception as e:  # download and CTranslate2 errors come in many types
            self._model = None
            raise ModelLoadError(f"Failed to load faster-whisper model '{model_name}': {e}") from e

        self._model_name = model_name
        self._device = device

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = True,
        *,
        prompt: Optional[str] = None,
        vad: bool = False,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio using faster-whisper.

        Args:
            audio_path: Path to the audio file.
            language: Language code or None for auto-detection.
            task: 'transcribe' or 'translate'.
            verbose: Show a progress bar during transcription.
            prompt: Names, terms or a sample sentence that guide spelling (initial_prompt).
            vad: Skip silence with the built-in Silero VAD filter.
            word_timestamps: Add per-word timings ("words") to each segment.

        Returns:
            TranscriptionResult with transcript text and segments.

        Raises:
            ModelLoadError: If no model is loaded.
            FileNotFoundError: If audio file doesn't exist.
        """
        if self._model is None:
            raise ModelLoadError("No model loaded. Call load_model() first.")

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        segment_iter, info = self._model.transcribe(
            str(audio_path),
            language=language,
            task=task,
            log_progress=verbose,
            initial_prompt=prompt,
            vad_filter=vad,
            word_timestamps=word_timestamps,
        )
        # segment_iter is a generator: decoding happens while it is iterated.
        segments = [_segment(index, seg) for index, seg in enumerate(segment_iter)]

        return TranscriptionResult(
            text="".join(seg["text"] for seg in segments).strip(),
            segments=segments,
            language=info.language,
            # Also in raw, where earlier versions put them.
            raw={"duration": info.duration, "language_probability": info.language_probability},
            duration=info.duration,
            language_probability=info.language_probability,
        )


def _segment(index: int, seg: Any) -> Segment:
    """A faster-whisper Segment in the standard format (with "words" when word timestamps were asked for).

    faster-whisper numbers segments from 1; *index*, the segment's position, numbers them from 0 like other backends.
    """
    segment: Segment = {
        "id": index,
        "start": seg.start,
        "end": seg.end,
        "text": seg.text,
        "tokens": seg.tokens,
        "temperature": seg.temperature,
        "avg_logprob": seg.avg_logprob,
        "compression_ratio": seg.compression_ratio,
        "no_speech_prob": seg.no_speech_prob,
    }
    if getattr(seg, "words", None):
        segment["words"] = [
            {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in seg.words
        ]
    return segment


_GPU_DOCS = "https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/docs/install.md#gpu"


@dataclass
class _CudaLibrary:
    """A CUDA library that CTranslate2 loads on the GPU and pip doesn't install with it."""

    requirement: str  # what to pip install
    folder: str  # where that puts it: site-packages/nvidia/<folder>/bin on Windows, .../lib on Linux
    windows: List[str]  # the files CTranslate2 loads; each pulls in the rest of the library
    linux: List[str]


_CUBLAS_12 = _CudaLibrary("nvidia-cublas-cu12", "cublas", ["cublas64_12.dll"], ["libcublas.so.12"])
# Up to CTranslate2 4.6.2, cuDNN runs the convolutions of Whisper's encoder: cuDNN 8 before 4.5.0, cuDNN 9 from it.
_CUDNN_8 = _CudaLibrary(
    '"nvidia-cudnn-cu12==8.*"',
    "cudnn",
    ["cudnn_ops_infer64_8.dll", "cudnn_cnn_infer64_8.dll"],
    ["libcudnn_ops_infer.so.8", "libcudnn_cnn_infer.so.8"],
)
_CUDNN_9 = _CudaLibrary(
    '"nvidia-cudnn-cu12==9.*"',
    "cudnn",
    ["cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"],
    ["libcudnn_ops.so.9", "libcudnn_cnn.so.9"],
)


def _cuda_libraries(ct2_version: str) -> List[_CudaLibrary]:
    """The CUDA libraries that CTranslate2 *ct2_version* loads on the GPU, as its 4.x wheels do.

    Other versions return none, so the GPU is used without a check, as before.
    """
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", ct2_version)
    version = tuple(int(part) for part in match.groups()) if match else ()
    if not (4, 0, 0) <= version < (5, 0, 0):
        return []
    if version < (4, 5, 0):
        return [_CUBLAS_12, _CUDNN_8]
    if version < (4, 6, 3):
        return [_CUBLAS_12, _CUDNN_9]
    return [_CUBLAS_12]


def _cuda_library_problem(ct2_version: str) -> Optional[str]:
    """The CUDA libraries CTranslate2 *ct2_version* needs that don't load, and how to install them.

    None when they all load, and on systems other than Windows and Linux, where CTranslate2 has no CUDA build.
    """
    platform = sys.platform
    if platform not in ("win32", "linux"):
        return None
    windows = platform == "win32"
    missing: List[str] = []
    to_install: List[_CudaLibrary] = []
    for library in _cuda_libraries(ct2_version):
        absent = [name for name in (library.windows if windows else library.linux) if not _can_load(name, windows)]
        if absent:
            missing += absent
            to_install.append(library)
    if not missing:
        return None

    separator, subfolder, variable = ("\\", "bin", "PATH") if windows else ("/", "lib", "LD_LIBRARY_PATH")
    folders = " and ".join(separator.join(("site-packages", "nvidia", lib.folder, subfolder)) for lib in to_install)
    fix = f"pip install {' '.join(lib.requirement for lib in to_install)}, then add {folders} to {variable}"
    if not windows:
        fix += " before starting Python"
    if any(lib.folder == "cudnn" for lib in to_install):
        fix += " (or upgrade CTranslate2, which needs no cuDNN since 4.6.3: pip install -U ctranslate2)"
    return f"CUDA libraries not found: {', '.join(missing)}. To use the GPU, {fix}; see {_GPU_DOCS}."


def _can_load(name: str, windows: bool) -> bool:
    """Whether the library *name* loads from where CTranslate2 would find it.

    A library that loads stays loaded, and CTranslate2's own lookup by name then finds it.
    """
    if not windows:
        return _loads(name)  # dlopen, like CTranslate2: LD_LIBRARY_PATH, the ldconfig cache, the system folders
    # ctypes' default search covers libraries already loaded, Python's folder, System32 and the os.add_dll_directory()
    # folders, such as CTranslate2's own. CTranslate2 also searches %CUDA_PATH%\bin and PATH (SetDllDirectoryA, then
    # LoadLibraryA): those are tried by full path, which keeps a DLL in the current folder from being loaded.
    if _loads(name):
        return True
    folders = [os.path.join(os.environ["CUDA_PATH"], "bin")] if os.environ.get("CUDA_PATH") else []
    folders += [folder.strip('"') for folder in os.environ.get("PATH", "").split(os.pathsep)]
    return any(os.path.isfile(path) and _loads(path) for path in (os.path.join(f, name) for f in folders if f))


def _loads(path: str) -> bool:
    """Whether ctypes loads the library *path*; a missing one raises OSError rather than ending the process."""
    import ctypes

    try:
        ctypes.CDLL(path)
    except OSError:
        return False
    return True
