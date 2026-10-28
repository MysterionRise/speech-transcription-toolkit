"""Mistral Voxtral transcription backend.

Runs Voxtral Mini (3B) or Voxtral Small (24B) locally through Hugging Face transformers.
Needs the ``voxtral`` extra; models download from the Hugging Face
Hub on first use (set HF_TOKEN if the download asks for authentication). Loading is shared with
Parakeet and Canary, in ``_transformers.py``.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..errors import ModelLoadError, SpeechToolkitWarning, UnsupportedOptionError
from ..media import SAMPLE_RATE, load_audio, split_audio
from ._transformers import MAX_NEW_TOKENS_OPTION, _TransformersBackend
from .base import TranscriptionResult
from .config import BackendOption
from .nvidia_backend import _cut_off_warning, _rows_at_limit

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

# Voxtral's transcription mode returns text without timestamps, so audio is split into
# chunks of up to 30 s (cut in pauses) and each chunk becomes one coarsely timestamped segment.
CHUNK_SECONDS = 30
BATCH_SIZE = 8


class VoxtralBackend(_TransformersBackend):
    """Mistral Voxtral transcription backend.

    Supports Voxtral Mini (3B parameters) and Voxtral Small (24B parameters)
    for multilingual speech-to-text transcription.

    Requirements:
        - torch
        - transformers>=5.17
        - mistral-common[audio]>=1.12
    """

    name = "voxtral"
    description = "Mistral Voxtral - open-weight multilingual speech-to-text"
    capabilities = frozenset({"waveform"})  # transcription only, and it doesn't report the language
    requires = ("torch", "transformers", "mistral_common")
    EXTRA = "voxtral"
    NEEDS = "torch, transformers>=5.17 and mistral-common[audio]"

    # Available Voxtral models
    MODELS = {
        "voxtral-mini": "mistralai/Voxtral-Mini-3B-2507",
        "voxtral-small": "mistralai/Voxtral-Small-24B-2507",
    }
    MODEL_CLASS = "VoxtralForConditionalGeneration"
    OPTIONS: Mapping[str, BackendOption] = {
        **_TransformersBackend.OPTIONS,
        "max_new_tokens": MAX_NEW_TOKENS_OPTION,
    }

    @classmethod
    def default_model(cls) -> str:
        """Return 'voxtral-mini' as the default - good balance of speed and quality."""
        return "voxtral-mini"

    def transcribe(
        self,
        audio_path: Union[Path, "np.ndarray"],
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = True,
    ) -> TranscriptionResult:
        """Transcribe audio using Voxtral.

        Args:
            audio_path: Path to the audio file, or its samples (mono float32 at 16 kHz).
            language: Language code or None for auto-detection.
            task: Only 'transcribe' is supported.
            verbose: Print progress to stderr.

        Returns:
            TranscriptionResult with one segment per ~30 s chunk of audio. A chunk whose text reaches the
            max_new_tokens option's limit may be cut off: one warning per file gives the number of such chunks and the
            first one's times.

        Raises:
            ModelLoadError: If no model is loaded.
            UnsupportedOptionError: If task is not 'transcribe'.
            FileNotFoundError: If audio file doesn't exist.
        """
        if self._model is None or self._processor is None:
            raise ModelLoadError("No model loaded. Call load_model() first.")

        if task != "transcribe":
            raise UnsupportedOptionError("The Voxtral backend only supports --task transcribe.")

        if isinstance(audio_path, Path) and not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio = load_audio(audio_path) if isinstance(audio_path, Path) else audio_path
        chunks = split_audio(audio, CHUNK_SECONDS)

        segments: List[Dict[str, Any]] = []
        cut_off: List[Dict[str, Any]] = []  # the segments of chunks that reached max_new_tokens
        for first in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[first : first + BATCH_SIZE]
            texts, at_limit = self._transcribe_batch([samples for _, samples in batch], language)
            for index, ((start, samples), text) in enumerate(zip(batch, texts), start=first):
                segments.append({"id": index, "start": start, "end": start + len(samples) / SAMPLE_RATE, "text": text})
            cut_off += [segments[first + row] for row in at_limit]
            if verbose:
                print(f"Voxtral: transcribed {len(segments)}/{len(chunks)} chunks", file=sys.stderr)
        if cut_off:
            limit = self._options["max_new_tokens"]
            warnings.warn(_cut_off_warning(audio_path, "Voxtral", limit, cut_off), SpeechToolkitWarning)

        return TranscriptionResult(
            text=" ".join(seg["text"] for seg in segments if seg["text"]),
            segments=segments,
            language=language,  # Voxtral doesn't report the detected language
            duration=len(audio) / SAMPLE_RATE,
        )

    def _transcribe_batch(self, chunks: Sequence[Any], language: Optional[str]) -> Tuple[List[str], List[int]]:
        """Transcribe a batch of audio chunks.

        Returns:
            One string per chunk, and the positions in the batch of the chunks whose text reached max_new_tokens.
        """
        inputs = self._processor.apply_transcription_request(
            audio=list(chunks),
            model_id=self._repo_id,
            language=language,
            sampling_rate=SAMPLE_RATE,
            format="wav",
        )
        inputs = inputs.to(self._device, dtype=self._model.dtype)
        outputs = self._model.generate(**inputs, max_new_tokens=self._options["max_new_tokens"])
        generated = outputs[:, inputs.input_ids.shape[1] :]  # without the prompt
        texts = self._processor.batch_decode(generated, skip_special_tokens=True)
        at_limit = _rows_at_limit(generated, self._model.generation_config, self._options["max_new_tokens"])
        return [text.strip() for text in texts], at_limit
