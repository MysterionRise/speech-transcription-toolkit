"""faster-whisper transcription backend, the default.

faster-whisper re-implements Whisper on CTranslate2: it is several times faster than openai-whisper, uses less memory
and does not need torch. It is a dependency of the package, so every install has it; models download from the Hugging
Face Hub on first use.

On an NVIDIA GPU, CTranslate2 also needs NVIDIA's CUDA 12 libraries (cuBLAS, plus cuDNN before CTranslate2 4.6.3),
which pip doesn't install. ``load_model()`` checks that they load before it uses the GPU.

Besides the names in ``MODELS``, it loads CTranslate2 conversions of Whisper models from the Hugging Face Hub (such as
``Systran/faster-whisper-small``) or from a local folder. It runs on the CPU or an NVIDIA GPU (``cuda:N`` picks one),
not on Apple GPUs.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union

from ..errors import BackendUnavailableError, ModelLoadError, SpeechToolkitWarning
from ..media import SAMPLE_RATE, load_audio
from ..types import Segment
from ..vad import VAD_OPTIONS, faster_whisper_parameters, has_speech
from ..vad import settings as vad_settings
from .base import TranscriptionBackend, TranscriptionResult, report_progress
from .config import BackendOption
from .whisper_backend import HALLUCINATION_OPTIONS, decoding_arguments

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = logging.getLogger(__name__)

# What --compute-type takes: "auto", or one of CTranslate2's types ("default" keeps the type the model was saved in).
COMPUTE_TYPES = (
    "auto",
    "default",
    "int8",
    "int8_float32",
    "int8_float16",
    "int8_bfloat16",
    "int16",
    "float16",
    "bfloat16",
    "float32",
)
# What "auto" picks: the first of these the device supports, so a GPU without fast float16 falls back to int8.
AUTO_COMPUTE_TYPES: Dict[str, Tuple[str, ...]] = {
    "cuda": ("float16", "int8_float16", "int8", "float32"),
    "cpu": ("int8", "float32"),
}


class FasterWhisperBackend(TranscriptionBackend):
    """faster-whisper (CTranslate2) transcription backend."""

    name = "faster-whisper"
    description = "faster-whisper - CTranslate2 Whisper: faster, lighter, no torch needed"
    capabilities = frozenset({"translate", "language_detection", "prompt", "vad", "word_timestamps", "waveform"})
    requires = ("faster_whisper",)
    MODEL_SOURCES = {
        "hub": "a Hugging Face repo id (org/name) of a CTranslate2 model",
        "directory": "a local CTranslate2 model folder",
    }
    DEVICES = frozenset({"cpu", "cuda"})  # CTranslate2 doesn't run on Apple GPUs (mps)
    OPTIONS = {
        "compute_type": BackendOption(
            str,
            "auto",
            "Precision; auto: float16 on a GPU and int8 on the CPU, or the next best the device supports",
            choices=COMPUTE_TYPES,
        ),
        "cpu_threads": BackendOption(int, 0, "Threads on the CPU; 0 uses OMP_NUM_THREADS if set, else 4", minimum=0),
        "num_workers": BackendOption(
            int,
            1,
            "Transcriptions the model runs at once when several threads share it; each worker takes memory",
            minimum=1,
        ),
        "batch_size": BackendOption(
            int,
            1,
            "Pieces of audio decoded at a time; above 1, the batched pipeline: several times faster on a GPU, it cuts "
            "the audio at pauses with voice activity detection",
            minimum=1,
        ),
        "beam_size": BackendOption(int, 5, "Beams in beam search; 1 decodes greedily, the fastest", minimum=1),
        **HALLUCINATION_OPTIONS,
        **VAD_OPTIONS,
    }
    # The options that are keyword arguments of the model's transcribe(), with faster-whisper's defaults.
    DECODING_OPTIONS = ("beam_size", *HALLUCINATION_OPTIONS)
    # What the batched pipeline ignores: it decodes the pieces of audio independently, and skips silence with voice
    # activity detection instead.
    SEQUENTIAL_OPTIONS = tuple(HALLUCINATION_OPTIONS)

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

    def load_model(
        self, model_name: str, device: Optional[str] = None, *, options: Optional[Mapping[str, Any]] = None
    ) -> None:
        """Load a faster-whisper model (float16 on a GPU, int8 on the CPU, unless ``compute_type`` says otherwise).

        Args:
            model_name: One of the available model names, a Hugging Face repo id of a CTranslate2 model, or a local
                CTranslate2 model folder.
            device: 'cpu', 'cuda', 'cuda:N' (the N-th GPU), or None or 'auto' for the GPU when CTranslate2 sees one.
                If the CUDA libraries it needs don't load, the automatic choice warns and uses the CPU.
            options: Values for ``OPTIONS``. ``compute_type``, ``cpu_threads`` and ``num_workers`` load the model;
                afterwards ``self.options`` shows the compute type that "auto" picked. The others are for
                transcribe(): ``batch_size`` (above 1, faster-whisper's batched pipeline), ``beam_size``,
                ``condition_on_previous_text``, ``no_speech_threshold`` and ``hallucination_silence_threshold``, and
                the ``vad_`` settings, which tune the voice activity detection. The batched pipeline ignores
                ``condition_on_previous_text``, ``no_speech_threshold`` and ``hallucination_silence_threshold``, so
                setting them with a ``batch_size`` above 1 warns.

        Raises:
            ModelNotFoundError: If model_name is none of these.
            UnsupportedOptionError: If the device is 'mps' or isn't a device, or an option is unknown or invalid.
            BackendUnavailableError: If faster-whisper is not installed.
            ModelLoadError: If the model cannot be downloaded or loaded, or a CUDA device was asked for and the CUDA
                libraries are missing.
        """
        model_id = self.resolve_model(model_name)
        target = self._parse_device(device)
        settings = self._resolve_options(options)
        ignored = [name for name in self.SEQUENTIAL_OPTIONS if settings[name] != self.OPTIONS[name].default]
        if settings["batch_size"] > 1 and ignored:
            warnings.warn(
                f"faster-whisper's batched pipeline (batch_size above 1) ignores {' and '.join(ignored)}; it skips "
                "silence with voice activity detection instead.",
                SpeechToolkitWarning,
                stacklevel=2,
            )

        try:
            import ctranslate2
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise BackendUnavailableError(
                f"faster-whisper backend needs faster-whisper ({e}), which speech-transcription-toolkit depends on. "
                "Install it with: pip install speech-transcription-toolkit"
            ) from e

        auto = target.auto
        kind = ("cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu") if auto else target.kind
        # A missing library would fail the first transcription, or on Windows end the process, so check first. Machines
        # without a GPU skip the check.
        if kind == "cuda" and (auto or ctranslate2.get_cuda_device_count() > 0):
            problem = _cuda_library_problem(getattr(ctranslate2, "__version__", ""))
            if problem and not auto:
                raise ModelLoadError(f"faster-whisper can't run on CUDA. {problem}")
            if problem:
                warnings.warn(
                    f"faster-whisper runs on the CPU instead of the GPU. {problem} Pass --device cpu to skip this check.",
                    SpeechToolkitWarning,
                    stacklevel=2,
                )
                kind = "cpu"
        arguments = _model_arguments(ctranslate2, kind, target.index, settings)
        logger.debug("Loading faster-whisper model %r with %s", model_id, arguments)

        try:
            self._model = WhisperModel(model_id, **arguments)
        except Exception as e:  # download and CTranslate2 errors come in many types
            self._model = None
            raise ModelLoadError(f"Failed to load faster-whisper model '{model_name}': {e}") from e

        self._model_name = model_name
        self._device = kind if target.index is None else f"{kind}:{target.index}"
        self._options = {**settings, "compute_type": arguments["compute_type"]}

    def detect_language(self, audio: Union[Path, "np.ndarray"]) -> List[Tuple[str, float]]:
        """The languages spoken in the first 30 seconds of *audio*, with their probabilities, most likely first.

        It uses faster-whisper's ``detect_language()``, which looks at the same 30 seconds as when it detects the
        language to transcribe.

        Args:
            audio: A path, of which only the first 30 seconds are decoded, or the samples (mono float32 at 16 kHz).

        Returns:
            Every language the model knows, with its probability: ``[("de", 0.97), ("en", 0.01), ...]``. An
            English-only model gives ``[("en", 1.0)]``.

        Raises:
            ModelLoadError: If no model is loaded.
            FileNotFoundError: If the audio file doesn't exist.
        """
        if self._model is None:
            raise ModelLoadError("No model loaded. Call load_model() first.")
        if not self._model.model.is_multilingual:  # as faster-whisper's transcribe() treats it: English, for sure
            return [("en", 1.0)]
        if isinstance(audio, Path):
            from ..media import load_audio

            if not audio.exists():
                raise FileNotFoundError(f"Audio file not found: {audio}")
            audio = load_audio(audio, max_seconds=self._model.feature_extractor.chunk_length)
        _, _, probabilities = self._model.detect_language(audio)  # it takes the first 30 s of the samples
        return [(code, float(chance)) for code, chance in probabilities]

    def transcribe(
        self,
        audio_path: Union[Path, "np.ndarray"],
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
            audio_path: Path to the audio file, or its samples (mono float32 at 16 kHz), which it takes as is.
            language: Language code or None for auto-detection.
            task: 'transcribe' or 'translate'.
            verbose: Show faster-whisper's progress bar on stderr. Each segment is also passed to report_progress() as
                soon as it is decoded, whatever *verbose* is.
            prompt: Names, terms or a sample sentence that guide spelling (initial_prompt).
            vad: Skip silence with the built-in Silero VAD filter, tuned by the ``vad_`` options. A path is then
                decoded with load_audio(), and audio without speech returns an empty result without reaching the
                model. The batched pipeline (the ``batch_size`` option above 1) always filters, with the same
                settings: it decodes the pieces of speech it finds.
            word_timestamps: Add per-word timings ("words") to each segment.

        The options the model loaded with (``beam_size`` and the others in ``DECODING_OPTIONS``) go to faster-whisper's
        transcribe(), or with a ``batch_size`` above 1 to its ``BatchedInferencePipeline``. That makes a segment of
        each piece of speech of up to 30 s, and gives them in order, as they are decoded, with word timings if asked.

        Returns:
            TranscriptionResult with transcript text and segments.

        Raises:
            ModelLoadError: If no model is loaded.
            FileNotFoundError: If audio file doesn't exist.
        """
        if self._model is None:
            raise ModelLoadError("No model loaded. Call load_model() first.")

        if isinstance(audio_path, Path) and not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio: Any = str(audio_path) if isinstance(audio_path, Path) else audio_path
        settings = vad_settings(self._options)
        if vad:
            audio = load_audio(audio_path) if isinstance(audio_path, Path) else audio_path
            if not has_speech(audio, threshold=settings["threshold"]):  # its filter would leave nothing to transcribe
                duration = len(audio) / SAMPLE_RATE
                raw = {"duration": duration, "language_probability": None}
                return TranscriptionResult("", [], language, raw=raw, duration=duration)

        # The options set, as keyword arguments; the ones left at faster-whisper's defaults aren't passed.
        arguments = decoding_arguments(self._options, {name: self.OPTIONS[name] for name in self.DECODING_OPTIONS})
        model = self._model
        if self._options["batch_size"] > 1:
            model = self._batched_pipeline()
            for name in self.SEQUENTIAL_OPTIONS:  # it ignores them (load_model() warned)
                arguments.pop(name, None)
            # The pipeline decodes the pieces of speech that voice activity detection finds, of up to 30 s each, and
            # needs them for audio longer than that. Each piece becomes a segment: it decodes each piece once, so
            # asking it for timestamps (without_timestamps=False) would split the pieces, but lose the text after a
            # piece's first few sentences now and then.
            arguments["batch_size"] = self._options["batch_size"]
            vad = True
        # hallucination_silence_threshold works on word timings: they are computed for it, and kept if asked for.
        words = word_timestamps or bool(arguments.get("hallucination_silence_threshold"))
        segment_iter, info = model.transcribe(
            audio,
            language=language,
            task=task,
            log_progress=verbose,
            initial_prompt=prompt,
            vad_filter=vad,
            vad_parameters=faster_whisper_parameters(**settings) if vad else None,
            word_timestamps=words,
            **arguments,
        )
        # segment_iter is a generator: decoding happens while it is iterated, so each segment is reported as it comes.
        segments: List[Segment] = []
        for seg in segment_iter:
            segments.append(_segment(len(segments), seg, words=word_timestamps))
            report_progress(seg.end, info.duration, segments[-1])

        return TranscriptionResult(
            text="".join(seg["text"] for seg in segments).strip(),
            segments=segments,
            language=info.language,
            # Also in raw, where earlier versions put them.
            raw={"duration": info.duration, "language_probability": info.language_probability},
            duration=info.duration,
            language_probability=info.language_probability,
        )

    def _batched_pipeline(self) -> Any:
        """faster-whisper's batched pipeline on the loaded model: a new one for each transcription, as it keeps the
        last word's time between batches, which a transcription stopped halfway would leave behind."""
        from faster_whisper import BatchedInferencePipeline  # imported by load_model() already

        return BatchedInferencePipeline(self._model)


def _model_arguments(
    ctranslate2: Any, device: str, index: Optional[int], settings: Mapping[str, Any]
) -> Dict[str, Any]:
    """WhisperModel's keyword arguments for *device* ("cpu" or "cuda"), the GPU *index* and the backend's options.

    Only what differs from WhisperModel's defaults, device_index=0, cpu_threads=0 and num_workers=1, is included.
    """
    compute_type = settings["compute_type"]
    if compute_type == "auto":
        compute_type = _auto_compute_type(ctranslate2, device, index or 0)
    arguments: Dict[str, Any] = {"device": device, "compute_type": compute_type}
    if index is not None:
        arguments["device_index"] = index
    if settings["cpu_threads"]:
        arguments["cpu_threads"] = settings["cpu_threads"]
    if settings["num_workers"] != 1:
        arguments["num_workers"] = settings["num_workers"]
    return arguments


def _auto_compute_type(ctranslate2: Any, device: str, index: int) -> str:
    """The first of AUTO_COMPUTE_TYPES[device] that CTranslate2 supports on that device (the GPU *index*, for CUDA).

    The first one, as before "auto" existed, if CTranslate2 can't tell.
    """
    preferred = AUTO_COMPUTE_TYPES[device]
    try:
        supported = set(ctranslate2.get_supported_compute_types(device, index))
    except Exception as e:  # a CUDA error, say: use the usual type, as before
        logger.debug("CTranslate2 can't list the compute types of %s:%s: %s", device, index, e)
        return preferred[0]
    return next((compute_type for compute_type in preferred if compute_type in supported), preferred[0])


def _segment(index: int, seg: Any, words: bool = True) -> Segment:
    """A faster-whisper Segment in the standard format, with "words" when it has them and *words* is true.

    faster-whisper numbers segments from 1 (the batched pipeline, across the whole recording); *index*, the segment's
    position, numbers them from 0 like other backends.
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
    if words and getattr(seg, "words", None):
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
