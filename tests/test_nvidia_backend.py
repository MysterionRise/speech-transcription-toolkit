"""Tests for the NVIDIA Parakeet and Canary backends (torch and transformers are mocked)."""

from __future__ import annotations

import pathlib
import sys
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from speech_toolkit.backends import get_backend
from speech_toolkit.backends.nvidia_backend import (
    CanaryBackend,
    ParakeetBackend,
    _clock,
    _rows_at_limit,
    _segments_from_words,
    _words_from_tokens,
)
from speech_toolkit.errors import SpeechToolkitWarning

RATE = 16000
CANARY_EOS, CANARY_PAD = 3, 2  # canary-1b-v2's end-of-text and padding token ids


class FakeBatch(dict):
    """Minimal stand-in for a processor's BatchFeature."""

    def to(self, *args, **kwargs):
        return self


@pytest.fixture
def audio_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "talk.wav"
    path.touch()
    return path


@pytest.fixture
def nvidia_modules():
    """Stand-ins for torch, transformers and the ffmpeg audio loader (no GPU)."""
    torch = MagicMock()
    torch.cuda.is_available.return_value = False
    transformers = MagicMock()
    with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
        with patch("speech_toolkit.backends.nvidia_backend.load_audio") as load_audio:
            load_audio.return_value = np.zeros(5 * RATE, dtype=np.float32)
            yield SimpleNamespace(
                torch=torch,
                transformers=transformers,
                processor=transformers.AutoProcessor.from_pretrained.return_value,
                load_audio=load_audio,
            )


class TestLoading:
    @pytest.mark.parametrize(
        "backend_class, model, repo, model_class",
        [
            (ParakeetBackend, "parakeet-tdt-0.6b-v3", "nvidia/parakeet-tdt-0.6b-v3", "ParakeetForTDT"),
            (CanaryBackend, "canary-1b-v2", "nvidia/canary-1b-v2", "CanaryForConditionalGeneration"),
        ],
    )
    def test_load_model_cpu(self, nvidia_modules, backend_class, model, repo, model_class):
        backend = backend_class()
        backend.load_model(model)

        model_cls = getattr(nvidia_modules.transformers, model_class)
        nvidia_modules.transformers.AutoProcessor.from_pretrained.assert_called_once_with(repo)
        model_cls.from_pretrained.assert_called_once_with(repo, dtype=nvidia_modules.torch.float32)
        model_cls.from_pretrained.return_value.to.assert_called_once_with("cpu")
        assert (backend.is_loaded, backend.model_name, backend.device) == (True, model, "cpu")

    def test_defaults_and_registry(self):
        assert ParakeetBackend.default_model() == "parakeet-tdt-0.6b-v3"
        assert CanaryBackend.default_model() == "canary-1b-v2"
        assert isinstance(get_backend("parakeet"), ParakeetBackend)
        assert isinstance(get_backend("canary"), CanaryBackend)
        assert ParakeetBackend.capabilities == {"word_timestamps"}
        assert CanaryBackend.capabilities == frozenset()

    def test_load_model_cuda_uses_half_precision(self, nvidia_modules):
        nvidia_modules.torch.cuda.is_available.return_value = True
        nvidia_modules.torch.cuda.is_bf16_supported.return_value = False

        backend = ParakeetBackend()
        backend.load_model("parakeet-tdt-0.6b-v2")

        _, kwargs = nvidia_modules.transformers.ParakeetForTDT.from_pretrained.call_args
        assert kwargs["dtype"] is nvidia_modules.torch.float16
        assert backend.device == "cuda"

    def test_load_model_invalid(self):
        with pytest.raises(ValueError, match="Unknown parakeet model"):
            ParakeetBackend().load_model("parakeet-huge")

    def test_load_model_missing_dependency(self):
        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            with pytest.raises(ImportError, match=r"speech-transcription-toolkit\[nvidia\]"):
                CanaryBackend().load_model("canary-1b-v2")

    def test_transformers_too_old(self, nvidia_modules):
        del nvidia_modules.transformers.ParakeetForTDT  # older releases don't have the class

        with pytest.raises(ImportError, match=r"transformers>=5\.18"):
            ParakeetBackend().load_model("parakeet-tdt-0.6b-v3")

    def test_load_model_failure(self, nvidia_modules):
        nvidia_modules.transformers.AutoProcessor.from_pretrained.side_effect = OSError("repo not found")

        backend = CanaryBackend()
        with pytest.raises(RuntimeError, match="repo not found"):
            backend.load_model("canary-1b-v2")
        assert backend.is_loaded is False

    @pytest.mark.parametrize("backend_class", [ParakeetBackend, CanaryBackend])
    def test_transcribe_without_model(self, backend_class):
        with pytest.raises(RuntimeError, match="No model loaded"):
            backend_class().transcribe(pathlib.Path("talk.wav"))

    @pytest.mark.parametrize("backend_class", [ParakeetBackend, CanaryBackend])
    def test_transcribe_missing_file(self, nvidia_modules, backend_class, tmp_path):
        backend = backend_class()
        backend.load_model(backend_class.default_model())
        with pytest.raises(FileNotFoundError):
            backend.transcribe(tmp_path / "missing.wav", language="en")


class TestParakeet:
    def _backend(self):
        backend = ParakeetBackend()
        backend.load_model("parakeet-tdt-0.6b-v3")
        return backend

    def test_tokens_become_words_and_sentences(self, nvidia_modules, audio_file, capsys):
        processor = nvidia_modules.processor
        processor.return_value = FakeBatch(input_features="features")
        tokens = [
            {"token": "Hello", "start": 0.0, "end": 0.4},
            {"token": " world", "start": 0.5, "end": 0.9},
            {"token": ".", "start": 0.9, "end": 0.9},
            {"token": " How", "start": 2.5, "end": 2.7},
            {"token": "dy", "start": 2.7, "end": 2.8},
        ]
        processor.decode.return_value = ("Hello world. Howdy", [tokens])

        result = self._backend().transcribe(audio_file, word_timestamps=True)

        model = nvidia_modules.transformers.ParakeetForTDT.from_pretrained.return_value.to.return_value
        model.generate.assert_called_once_with(input_features="features")
        outputs = model.generate.return_value
        processor.decode.assert_called_once_with(outputs.sequences, durations=outputs.durations)
        assert result.text == "Hello world. Howdy"
        assert [(s["start"], s["end"], s["text"]) for s in result.segments] == [
            (0.0, 0.9, " Hello world."),
            (2.5, 2.8, " Howdy"),
        ]
        assert result.segments[0]["words"] == [
            {"word": " Hello", "start": 0.0, "end": 0.4},
            {"word": " world.", "start": 0.5, "end": 0.9},
        ]
        assert "1/1 chunks" in capsys.readouterr().err

    def test_long_audio_is_chunked_and_offset(self, nvidia_modules, audio_file):
        nvidia_modules.load_audio.return_value = np.zeros(130 * RATE, dtype=np.float32)
        nvidia_modules.processor.return_value = FakeBatch()
        nvidia_modules.processor.decode.side_effect = [
            ("", [[{"token": " One.", "start": 1.0, "end": 1.5}]]),
            ("", [[{"token": " Two.", "start": 2.0, "end": 2.5}]]),
        ]

        result = self._backend().transcribe(audio_file, verbose=False)

        # Silent audio is cut 5 s before the 120 s limit (plus half a 100 ms window): the second chunk starts at 115.05 s.
        assert [(s["start"], s["text"]) for s in result.segments] == [(1.0, " One."), (117.05, " Two.")]
        assert "words" not in result.segments[0]  # only kept when asked for

    def test_generate_max_length_warning_is_silenced(self, nvidia_modules, audio_file, recwarn):
        """transformers warns about the default max_length on every Parakeet call, though it's sized correctly."""
        nvidia_modules.processor.return_value = FakeBatch()
        nvidia_modules.processor.decode.return_value = ("", [[]])
        backend = self._backend()
        model = nvidia_modules.transformers.ParakeetForTDT.from_pretrained.return_value.to.return_value

        def generate(**inputs):
            warnings.warn("Using the model-agnostic default `max_length` (=430) to control the generation length.")
            return SimpleNamespace(sequences="seq", durations="dur")

        model.generate.side_effect = generate

        backend.transcribe(audio_file, verbose=False)

        assert not [w for w in recwarn if "max_length" in str(w.message)]

    def test_translate_is_rejected(self, nvidia_modules, audio_file):
        with pytest.raises(ValueError, match="only transcribes"):
            self._backend().transcribe(audio_file, task="translate")

    def test_whitespace_token_between_words(self, nvidia_modules, audio_file):
        """The tokenizer can emit a word boundary as its own " " token; the words around it stay apart."""
        nvidia_modules.processor.return_value = FakeBatch()
        tokens = [
            {"token": "Hello", "start": 0.0, "end": 0.4},
            {"token": " ", "start": 0.4, "end": 0.5},
            {"token": "world", "start": 0.5, "end": 0.9},
            {"token": ".", "start": 0.9, "end": 0.9},
        ]
        nvidia_modules.processor.decode.return_value = ("Hello world.", [tokens])

        result = self._backend().transcribe(audio_file, verbose=False, word_timestamps=True)

        assert result.text == "Hello world."
        assert [word["word"] for word in result.segments[0]["words"]] == [" Hello", " world."]


class TestCanary:
    def _backend(self):
        backend = CanaryBackend()
        backend.load_model("canary-1b-v2")
        return backend

    def _prepare(self, nvidia_modules, texts):
        processor = nvidia_modules.processor
        processor.apply_transcription_request.side_effect = lambda **kw: FakeBatch(
            decoder_input_ids=SimpleNamespace(shape=(len(kw["audio"]), 4))
        )
        processor.batch_decode.side_effect = texts
        return processor

    def test_transcribe_in_batches(self, nvidia_modules, audio_file, capsys):
        nvidia_modules.load_audio.return_value = np.zeros(70 * RATE, dtype=np.float32)
        processor = self._prepare(nvidia_modules, [[" Hallo ", "Welt"], [""]])

        backend = self._backend()
        with patch.object(CanaryBackend, "BATCH_SIZE", 2):
            result = backend.transcribe(audio_file, language="de")

        first = processor.apply_transcription_request.call_args_list[0][1]
        assert (len(first["audio"]), first["source_language"], first["target_language"]) == (2, "de", None)
        model = nvidia_modules.transformers.CanaryForConditionalGeneration.from_pretrained.return_value.to.return_value
        _, generate_kwargs = model.generate.call_args
        assert generate_kwargs["max_new_tokens"] == CanaryBackend.MAX_NEW_TOKENS
        assert [s["text"] for s in result.segments] == ["Hallo", "Welt", ""]
        assert [s["start"] for s in result.segments] == pytest.approx([0.0, 25.05, 50.1])
        assert result.text == "Hallo Welt"
        assert result.language == "de"
        assert "3/3 chunks" in capsys.readouterr().err

    def test_translate_to_english(self, nvidia_modules, audio_file):
        processor = self._prepare(nvidia_modules, [["Hello"]])

        result = self._backend().transcribe(audio_file, language="fr", task="translate", verbose=False)

        kwargs = processor.apply_transcription_request.call_args[1]
        assert (kwargs["source_language"], kwargs["target_language"]) == ("fr", "en")
        assert (result.text, result.language) == ("Hello", "en")

    def test_no_language_warns_and_assumes_english(self, nvidia_modules, audio_file):
        processor = self._prepare(nvidia_modules, [["Hi"]])

        with pytest.warns(UserWarning, match="assumes English"):
            self._backend().transcribe(audio_file, verbose=False)

        assert processor.apply_transcription_request.call_args[1]["source_language"] == "en"

    def _generate(self, nvidia_modules, *batches):
        """Make generate() return these batches of new tokens, after the 4-token prompt that _prepare() sets up."""
        model = nvidia_modules.transformers.CanaryForConditionalGeneration.from_pretrained.return_value.to.return_value
        model.generation_config = SimpleNamespace(eos_token_id=CANARY_EOS, pad_token_id=CANARY_PAD)
        model.generate.side_effect = [np.array([[9, 9, 9, 9] + row for row in batch]) for batch in batches]

    def test_warns_when_a_chunk_reaches_the_token_limit(self, nvidia_modules, audio_file):
        nvidia_modules.load_audio.return_value = np.zeros(70 * RATE, dtype=np.float32)
        self._prepare(nvidia_modules, [["one", "two", "three"]])
        self._generate(
            nvidia_modules,
            [
                [5, CANARY_EOS, CANARY_PAD, CANARY_PAD],  # ended early, then padded
                [5, 6, 7, 8],  # still going at the limit
                [5, 6, 7, CANARY_EOS],  # ended right at the limit
            ],
        )

        backend = self._backend()
        with patch.object(CanaryBackend, "MAX_NEW_TOKENS", 4), pytest.warns(SpeechToolkitWarning) as warned:
            result = backend.transcribe(audio_file, language="en", verbose=False)

        assert [(warning.category, str(warning.message)) for warning in warned] == [
            (
                SpeechToolkitWarning,
                f"{audio_file}: Canary reached its limit of 4 tokens in the chunk from 0:25 to 0:50, "
                "so the text may be cut off.",
            )
        ]
        assert [s["text"] for s in result.segments] == ["one", "two", "three"]  # the text is kept

    def test_one_warning_for_several_cut_off_chunks(self, nvidia_modules, audio_file):
        nvidia_modules.load_audio.return_value = np.zeros(70 * RATE, dtype=np.float32)
        self._prepare(nvidia_modules, [["one", "two"], ["three"]])
        self._generate(nvidia_modules, [[5, CANARY_EOS], [5, 6]], [[7, 8]])

        backend = self._backend()
        with patch.multiple(CanaryBackend, BATCH_SIZE=2, MAX_NEW_TOKENS=2):
            with pytest.warns(SpeechToolkitWarning) as warned:
                backend.transcribe(audio_file, language="en", verbose=False)

        assert [(warning.category, str(warning.message)) for warning in warned] == [
            (
                SpeechToolkitWarning,
                f"{audio_file}: Canary reached its limit of 2 tokens in 2 chunks, the first from 0:25 to 0:50, "
                "so the text may be cut off.",
            )
        ]


class TestWordHelpers:
    def test_words_from_tokens(self):
        tokens = [
            {"token": "Wie", "start": 0.0, "end": 0.2},
            {"token": " ", "start": 0.2, "end": 0.2},  # whitespace-only pieces are skipped
            {"token": " geht", "start": 0.3, "end": 0.5},
            {"token": "'s", "start": 0.5, "end": 0.6},
            {"token": "?", "start": 0.6, "end": 0.6},
        ]

        assert _words_from_tokens(tokens, offset=10.0) == [
            {"word": " Wie", "start": 10.0, "end": 10.2},
            {"word": " geht's?", "start": 10.3, "end": 10.6},
        ]

    def test_whitespace_only_token_starts_a_word(self):
        tokens = [
            {"token": "Hello", "start": 0.0, "end": 0.4},
            {"token": " ", "start": 0.4, "end": 0.5},
            {"token": "world", "start": 0.5, "end": 0.9},
        ]

        assert _words_from_tokens(tokens, offset=0.0) == [
            {"word": " Hello", "start": 0.0, "end": 0.4},
            {"word": " world", "start": 0.5, "end": 0.9},
        ]

    @pytest.mark.parametrize("pieces", [[" 3", ".5"], [" 3", ".", "5"]])
    def test_decimal_tokens_stay_one_word(self, pieces):
        tokens = [{"token": piece, "start": 0.1 * i, "end": 0.1 * (i + 1)} for i, piece in enumerate(pieces)]

        assert [word["word"] for word in _words_from_tokens(tokens, offset=0.0)] == [" 3.5"]

    def test_segments_break_on_pauses_and_length(self):
        words = [{"word": f" w{i}", "start": float(i), "end": i + 0.5} for i in range(40)]
        words.append({"word": " late", "start": 45.0, "end": 45.5})  # after a 5 s pause

        segments = _segments_from_words(words)

        assert [(s["id"], s["start"], s["end"]) for s in segments] == [(0, 0.0, 29.5), (1, 30.0, 39.5), (2, 45.0, 45.5)]
        assert segments[2]["text"] == " late"


def _sentences(text):
    """The segments that the words of *text*, spoken without pauses, are grouped into."""
    words = [{"word": " " + word, "start": i * 0.5, "end": i * 0.5 + 0.4} for i, word in enumerate(text.split())]
    return [segment["text"].strip() for segment in _segments_from_words(words)]


class TestSentenceBreaks:
    @pytest.mark.parametrize(
        "text",
        [
            "Dr. Smith is here.",
            "It costs 3.5 dollars.",
            "It costs 3. 5 dollars.",  # a decimal point cut between two words
            "Mr. and Mrs. Smith met Ms. Jones on St. Mark's Square.",
            "Real Madrid vs. Barcelona starts soon.",
            "J. R. R. Tolkien wrote it.",
            "The U.S. economy grew, e.g. Ohio, i.e. the Midwest.",
            "Pens, paper, etc. and more.",
            "She has a Ph.D. in physics.",  # not in the list, but a lowercase word follows
            "¡Dr. García está aquí!",  # leading punctuation
        ],
    )
    def test_no_break_inside_a_sentence(self, text):
        assert _sentences(text) == [text]

    @pytest.mark.parametrize(
        "text, sentences",
        [
            ("Hello world. How are you? Fine!", ["Hello world.", "How are you?", "Fine!"]),
            ("Pens, paper, etc. Then we left.", ["Pens, paper, etc.", "Then we left."]),
            ("So did I. Then we left.", ["So did I.", "Then we left."]),
            ("We counted to 3. Then we left.", ["We counted to 3.", "Then we left."]),
            ("It costs 3.5. Then we paid.", ["It costs 3.5.", "Then we paid."]),
            ("Wait... Go!", ["Wait...", "Go!"]),
        ],
    )
    def test_sentence_ends(self, text, sentences):
        assert _sentences(text) == sentences

    def test_abbreviation_at_the_end_or_before_a_pause(self):
        words = [
            {"word": " Ask", "start": 0.0, "end": 0.3},
            {"word": " Dr.", "start": 0.4, "end": 0.8},
            {"word": " Smith", "start": 3.0, "end": 3.4},  # after a pause
            {"word": " Dr.", "start": 3.5, "end": 3.9},  # the last word
        ]

        assert [s["text"] for s in _segments_from_words(words)] == [" Ask Dr.", " Smith Dr."]


class TestTokenLimit:
    @pytest.mark.parametrize(
        "eos, pad, ended",
        [
            (3, 2, [[5, 3, 2], [5, 6, 3]]),  # padded after the end token, and ended right at the limit
            ([3, 4], None, [[5, 3, 3], [5, 6, 4]]),  # no padding token: generate() pads with the first end token
        ],
    )
    def test_rows_at_limit(self, eos, pad, ended):
        config = SimpleNamespace(eos_token_id=eos, pad_token_id=pad)
        generated = np.array([ended[0], [5, 6, 7], ended[1]])

        assert _rows_at_limit(generated, config, limit=3) == [1]
        assert _rows_at_limit(generated, config, limit=4) == []  # every row ended before 4 tokens

    @pytest.mark.parametrize(
        "seconds, clock", [(0.0, "0:00"), (25.05, "0:25"), (59.6, "1:00"), (754.2, "12:34"), (3725.0, "1:02:05")]
    )
    def test_clock(self, seconds, clock):
        assert _clock(seconds) == clock
