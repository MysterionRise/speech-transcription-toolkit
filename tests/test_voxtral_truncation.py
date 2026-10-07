"""Voxtral warns when a chunk's text reaches MAX_NEW_TOKENS and may be cut off (torch and transformers are mocked)."""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from speech_toolkit.backends.voxtral_backend import VoxtralBackend

RATE = 16000
PROMPT = [1, 1, 1]  # the instruction and audio tokens before the generated text
EOS, PAD = 2, 11  # end-of-text and padding token ids


class FakeInputs(dict):
    """Stand-in for the processor's BatchFeature."""

    def __init__(self, batch_size: int):
        super().__init__(input_ids="ids")
        self.input_ids = SimpleNamespace(shape=(batch_size, len(PROMPT)))

    def to(self, *args, **kwargs):
        return self


@pytest.fixture
def audio_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "talk.wav"
    path.touch()
    return path


@pytest.fixture
def voxtral():
    """A loaded VoxtralBackend whose (mocked) audio is 70 s long: three chunks, cut at 25.05 s and 50.1 s."""
    torch = MagicMock()
    torch.cuda.is_available.return_value = False
    transformers = MagicMock()
    with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
        with patch("speech_toolkit.backends.voxtral_backend.load_audio") as load_audio:
            load_audio.return_value = np.zeros(70 * RATE, dtype=np.float32)
            backend = VoxtralBackend()
            backend.load_model("voxtral-mini")
            processor = transformers.AutoProcessor.from_pretrained.return_value
            processor.apply_transcription_request.side_effect = lambda **kwargs: FakeInputs(len(kwargs["audio"]))
            model = transformers.VoxtralForConditionalGeneration.from_pretrained.return_value.to.return_value
            model.generation_config = SimpleNamespace(eos_token_id=EOS, pad_token_id=PAD)
            yield SimpleNamespace(backend=backend, processor=processor, model=model)


def generates(voxtral, *batches):
    """Make generate() return these batches of new tokens after the prompt."""
    voxtral.model.generate.side_effect = [np.array([PROMPT + row for row in batch]) for batch in batches]


def test_warns_once_naming_the_cut_off_chunk(voxtral, audio_file):
    generates(voxtral, [[5, EOS, PAD], [5, 6, 7], [5, 6, EOS]])  # ended early, cut off, ended at the limit
    voxtral.processor.batch_decode.side_effect = [["one", "two", "three"]]

    with patch("speech_toolkit.backends.voxtral_backend.MAX_NEW_TOKENS", 3), pytest.warns(UserWarning) as warned:
        result = voxtral.backend.transcribe(audio_file, verbose=False)

    assert voxtral.model.generate.call_args[1]["max_new_tokens"] == 3
    assert [str(warning.message) for warning in warned] == [
        f"{audio_file}: Voxtral reached its limit of 3 tokens in the chunk from 0:25 to 0:50, so the text may be cut off."
    ]
    assert [s["text"] for s in result.segments] == ["one", "two", "three"]  # the text is kept


def test_one_warning_for_several_cut_off_chunks(voxtral, audio_file):
    generates(voxtral, [[5, 6], [5, EOS]], [[7, 8]])
    voxtral.processor.batch_decode.side_effect = [["one", "two"], ["three"]]

    with patch.multiple("speech_toolkit.backends.voxtral_backend", BATCH_SIZE=2, MAX_NEW_TOKENS=2):
        with pytest.warns(UserWarning) as warned:
            voxtral.backend.transcribe(audio_file, verbose=False)

    assert [str(warning.message) for warning in warned] == [
        f"{audio_file}: Voxtral reached its limit of 2 tokens in 2 chunks, the first from 0:00 to 0:25, "
        "so the text may be cut off."
    ]


def test_no_warning_when_every_chunk_ends(voxtral, audio_file):
    generates(voxtral, [[5, EOS, PAD], [5, 6, EOS], [EOS, PAD, PAD]])
    voxtral.processor.batch_decode.side_effect = [["one", "two", ""]]

    with patch("speech_toolkit.backends.voxtral_backend.MAX_NEW_TOKENS", 3):
        result = voxtral.backend.transcribe(audio_file, verbose=False)  # a warning would fail the test

    assert result.text == "one two"
