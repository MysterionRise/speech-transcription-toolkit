"""Python API: load a model once, then transcribe as many files as you like.

    from speech_toolkit import Transcriber

    transcriber = Transcriber(backend="faster-whisper", model="small", diarize=True)
    for path in ["a.mp3", "b.mp3"]:
        transcriber.transcribe(path, num_speakers=2).save(f"{path}.srt")

Errors are raised, never turned into ``sys.exit``: ``ValueError`` for bad arguments,
``ImportError`` for a missing optional package, ``FileNotFoundError``/``RuntimeError`` for
audio or model problems.
"""

from __future__ import annotations

import os
import pathlib
import warnings
from typing import Optional, Union

from .backends import DEFAULT_BACKEND, TranscriptionBackend, TranscriptionResult, get_backend_class
from .diarization import diarize_audio, load_diarization_pipeline, merge_diarization

# Whisper's turbo weights weren't trained for translation: they return the original language.
TURBO_MODELS = ("turbo", "large-v3-turbo")

AudioPath = Union[str, "os.PathLike[str]"]


def configure_hf_token(token: Optional[str] = None) -> None:
    """Expose the Hugging Face token as HF_TOKEN, which huggingface_hub uses for every model download.

    Falls back to an existing HF_TOKEN or the older HUGGINGFACE_TOKEN variable.
    """
    token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token


class Transcriber:
    """A loaded model, plus a speaker-diarization pipeline with ``diarize=True``, reusable across files.

    Args:
        backend: Backend name, see :func:`speech_toolkit.list_backends`.
        model: Model name (default: the backend's default, e.g. Whisper ``turbo``).
        device: ``"cpu"``, ``"cuda"`` or None to pick automatically.
        diarize: Also label speakers with pyannote.audio (needs the ``diarize`` extra and a Hugging Face token).
        hf_token: Hugging Face token for model downloads (default: ``HF_TOKEN`` or ``HUGGINGFACE_TOKEN``).
    """

    def __init__(
        self,
        backend: str = DEFAULT_BACKEND,
        model: Optional[str] = None,
        *,
        device: Optional[str] = None,
        diarize: bool = False,
        hf_token: Optional[str] = None,
    ) -> None:
        configure_hf_token(hf_token)
        backend_class = get_backend_class(backend)
        self.backend_name = backend
        self.model_name = model or backend_class.default_model()

        # Diarization first: a missing package or HF token should fail before a big model download.
        self._pipeline = load_diarization_pipeline(device) if diarize else None
        self.backend: TranscriptionBackend = backend_class()
        self.backend.load_model(self.model_name, device=device)

    @property
    def diarize(self) -> bool:
        """Whether results get speaker labels."""
        return self._pipeline is not None

    def transcribe(
        self,
        audio: AudioPath,
        language: Optional[str] = None,
        task: str = "transcribe",
        *,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        verbose: bool = False,
    ) -> TranscriptionResult:
        """Transcribe one audio or video file.

        Args:
            audio: Path to anything ffmpeg can decode.
            language: Language code such as ``"en"`` (default: auto-detect).
            task: ``"transcribe"``, or ``"translate"`` to English (Whisper models only).
            num_speakers, min_speakers, max_speakers: Speaker-count hints for diarization.
            verbose: Show the backend's progress output on stderr.

        Returns:
            The transcript; ``result.speaker_segments`` holds speaker-labelled segments when diarizing.
        """
        if (num_speakers or min_speakers or max_speakers) and not self.diarize:
            raise ValueError("num_speakers, min_speakers and max_speakers need diarize=True")
        if task == "translate" and self.model_name in TURBO_MODELS:
            warnings.warn(
                f"'{self.model_name}' isn't trained for translation and keeps the original language; "
                "use model 'medium' or 'large-v3'.",
                stacklevel=2,
            )

        path = pathlib.Path(audio)
        result = self.backend.transcribe(audio_path=path, language=language, task=task, verbose=verbose)
        if self._pipeline is not None:
            turns = diarize_audio(
                path,
                self._pipeline,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            result.speaker_segments = merge_diarization({"segments": result.segments}, turns)
        return result


def transcribe(
    audio: AudioPath,
    backend: str = DEFAULT_BACKEND,
    model: Optional[str] = None,
    *,
    language: Optional[str] = None,
    task: str = "transcribe",
    device: Optional[str] = None,
    diarize: bool = False,
    hf_token: Optional[str] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    verbose: bool = False,
) -> TranscriptionResult:
    """Load a model and transcribe one file. To transcribe several files, create a :class:`Transcriber` once."""
    transcriber = Transcriber(backend, model, device=device, diarize=diarize, hf_token=hf_token)
    return transcriber.transcribe(
        audio,
        language,
        task,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        verbose=verbose,
    )
