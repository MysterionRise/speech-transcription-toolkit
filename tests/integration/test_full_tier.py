"""The full tier, run with INTEGRATION_TIER=full before a release: the larger models, best on a GPU.

Each model transcribes the long clip, with the checks of test_long_audio.py. A model that doesn't fit in the memory
of the GPU (or in RAM, on the CPU) is skipped with the reason; devices.MEMORY_GB has the approximate needs. With
HF_TOKEN set, faster-whisper turbo and pyannote label the speakers of the long clip, sharing one process (and the GPU,
with INTEGRATION_DEVICE=cuda). See docs/manual-testing.md.
"""

from __future__ import annotations

import pytest

from tests.integration.devices import TIER, require_memory
from tests.integration.helpers import FULL_MODELS, check_long_audio, check_speaker_labels, require, require_hf_token

pytestmark = pytest.mark.skipif(TIER != "full", reason="full tier: set INTEGRATION_TIER=full")

LANGUAGE_ARGS = {"canary": ("-l", "en")}  # Canary can't detect the language


@pytest.mark.parametrize("backend, model", FULL_MODELS, ids=[f"{backend}-{model}" for backend, model in FULL_MODELS])
def test_large_model_on_long_audio(backend, model, long_clip, transcribe, tmp_path):
    require_memory((backend, model))
    require(backend)

    args = ("-b", backend, "-m", model, *LANGUAGE_ARGS.get(backend, ()))
    check_long_audio(backend, args, long_clip, transcribe, tmp_path)


def test_speaker_labels_with_a_large_model(long_clip, transcribe, tmp_path):
    require_hf_token()
    require_memory(("faster-whisper", "turbo"), ("diarize", "pyannote"))
    require("diarize", "faster-whisper")

    check_speaker_labels(long_clip, ("-b", "faster-whisper", "-m", "turbo"), transcribe, tmp_path)
