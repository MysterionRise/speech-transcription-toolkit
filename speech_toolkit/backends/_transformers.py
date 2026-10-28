"""Loading shared by the backends that run a Hugging Face transformers model: Voxtral, Parakeet and Canary.

:class:`_TransformersBackend` loads a model class with its ``AutoProcessor``, from a name in ``MODELS`` (a repo id),
another repo id or a local folder. torch and transformers are imported when a model loads, never before, so listing
backends stays instant. :func:`torch_device` and :func:`torch_dtype` pick the torch device and precision from what the
backend was given, for other backends on torch to reuse.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union

from ..errors import BackendUnavailableError, ModelLoadError
from .base import TranscriptionBackend
from .config import BackendOption, Device

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = logging.getLogger(__name__)

# What --compute-type takes on these backends.
COMPUTE_TYPES = ("auto", "float32", "float16", "bfloat16")
COMPUTE_TYPE_OPTION = BackendOption(
    str,
    "auto",
    "Precision; auto: bfloat16 on a CUDA GPU that supports it, else float16 on a GPU, float32 on the CPU",
    choices=COMPUTE_TYPES,
)
# Canary and Voxtral write at most this many tokens per ~30 s chunk, plenty for normal speech (the max_new_tokens
# option); a chunk that reaches it may be cut off, so they warn.
MAX_NEW_TOKENS = 500
MAX_NEW_TOKENS_OPTION = BackendOption(
    int,
    MAX_NEW_TOKENS,
    "Most tokens of text per ~30 s chunk; a chunk that reaches it may be cut off, with a warning",
    minimum=1,
)


def torch_device(torch: Any, device: Device) -> str:
    """The torch device for *device*: itself as text, or for "auto" "cuda" when torch sees a GPU, else "cpu".

    "auto" never picks "mps": Apple GPUs are used only when asked for.
    """
    if device.auto:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def torch_dtype(torch: Any, device: str, compute_type: str = "auto") -> Any:
    """The torch dtype for *compute_type* (``"auto"``, ``"float32"``, ``"float16"`` or ``"bfloat16"``) on *device*.

    ``"auto"`` is bfloat16 on a CUDA GPU that supports it (the one *device* names, such as "cuda:1"), else float16;
    float16 on an Apple GPU ("mps"); float32 on the CPU.
    """
    if compute_type != "auto":
        return getattr(torch, compute_type)
    kind, _, index = device.partition(":")
    if kind == "cuda":
        # is_bf16_supported() asks about the current GPU: make it the one the model runs on.
        with torch.cuda.device(int(index)) if index else contextlib.nullcontext():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float16 if kind == "mps" else torch.float32


class _TransformersBackend(TranscriptionBackend):
    """A backend that runs the transformers class ``MODEL_CLASS``, with its ``AutoProcessor`` (both imported lazily).

    A subclass sets ``MODELS`` (its names for models, mapped to their Hugging Face repo ids; the first is the default),
    ``MODEL_CLASS``, and for the error when an import fails, ``EXTRA`` and ``NEEDS``.
    """

    requires: Tuple[str, ...] = ("torch", "transformers")
    MODELS: Dict[str, str] = {}
    MODEL_CLASS = ""  # transformers class name
    EXTRA = ""  # the extra that installs what the backend needs
    NEEDS = "torch and transformers"  # what it needs, in words
    MODEL_SOURCES = {"hub": "", "directory": ""}
    DEVICES = frozenset({"cpu", "cuda", "mps"})
    OPTIONS: Mapping[str, BackendOption] = {"compute_type": COMPUTE_TYPE_OPTION}

    def __init__(self) -> None:
        super().__init__()
        self._processor: Any = None
        self._repo_id: Optional[str] = None  # what the model loaded from: a repo id, or a local folder

    @classmethod
    def available_models(cls) -> List[str]:
        """Return the model names; the first is the default."""
        return list(cls.MODELS)

    @classmethod
    def _model_id(cls, model_name: str) -> str:
        return cls.MODELS[model_name]

    def load_model(
        self, model_name: str, device: Optional[str] = None, *, options: Optional[Mapping[str, Any]] = None
    ) -> None:
        """Load a model and its processor (bfloat16/float16 on a GPU, float32 on the CPU, unless ``compute_type``
        says otherwise).

        Args:
            model_name: One of ``MODELS``, a Hugging Face repo id (org/name) or a local model folder.
            device: 'cpu', 'cuda', 'cuda:N' (the N-th GPU), 'mps' (an Apple GPU), or None or 'auto' for CUDA when
                torch sees a GPU, else the CPU.
            options: Values for ``OPTIONS``, such as ``{"compute_type": "float16"}``.

        Raises:
            ModelNotFoundError: If model_name is none of these.
            UnsupportedOptionError: If the device isn't a device, or an option is unknown or invalid.
            BackendUnavailableError: If torch or a recent enough transformers is not installed.
            ModelLoadError: If the model cannot be downloaded or loaded.
        """
        repo_id = self.resolve_model(model_name)
        target = self._parse_device(device)
        settings = self._resolve_options(options)

        try:
            import torch
            import transformers

            model_class = getattr(transformers, self.MODEL_CLASS)
        except (ImportError, AttributeError) as e:
            raise BackendUnavailableError(
                f"The {self.name} backend needs {self.NEEDS} ({e}). "
                f'Install with: pip install "speech-transcription-toolkit[{self.EXTRA}]"'
            ) from e

        try:
            on = torch_device(torch, target)
            dtype = torch_dtype(torch, on, settings["compute_type"])
            logger.debug("Loading %s model %r on %s in %s", self.name, repo_id, on, dtype)
            self._processor = transformers.AutoProcessor.from_pretrained(repo_id)
            self._model = model_class.from_pretrained(repo_id, dtype=dtype).to(on)
        except Exception as e:  # download, auth, CUDA and out-of-memory errors come in many types
            self._model = None
            self._processor = None
            raise ModelLoadError(f"Failed to load {self.name} model '{model_name}': {e}") from e

        self._repo_id = repo_id
        self._model_name = model_name
        self._device = on
        self._options = settings

    def _check_ready(self, audio_path: Union[Path, "np.ndarray"]) -> None:
        if self._model is None or self._processor is None:
            raise ModelLoadError("No model loaded. Call load_model() first.")
        if isinstance(audio_path, Path) and not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
