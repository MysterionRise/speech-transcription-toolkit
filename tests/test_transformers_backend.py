"""Tests for the generic transformers backend (-b transformers): torch, transformers and its ASR pipeline are mocked.

The mocked pipeline stands in for the model families the backend tells apart: multilingual and English-only Whisper
models, CTC models such as wav2vec2, and text-only sequence-to-sequence models.
"""

from __future__ import annotations

import pathlib
import sys
import warnings
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from speech_toolkit import ModelLoadError, ModelNotFoundError, Transcriber, UnsupportedOptionError
from speech_toolkit.backends import get_backend, get_backend_class
from speech_toolkit.backends.transformers_backend import (
    CTC,
    TEXT,
    WHISPER,
    ModelTraits,
    TransformersBackend,
    _language_code,
    _timed,
    read_traits,
)
from speech_toolkit.errors import BackendUnavailableError, SpeechToolkitWarning
from speech_toolkit.media import split_audio

RATE = 16000
WHISPER_LANGUAGES = {"english": "en", "german": "de", "french": "fr"}  # a part of Whisper's TO_LANGUAGE_CODE
# The cut split_audio() makes in 40 s of silence: the start of the last 5 s before the 30 s limit, plus half a window.
SECOND_CHUNK = 25.05


def whisper_config(multilingual: bool = True, **settings: Any) -> SimpleNamespace:
    """A Whisper generation config: whisper-small's (multilingual) or whisper-small.en's, with *settings* changed."""
    config: Dict[str, Any] = {"no_timestamps_token_id": 50363, "alignment_heads": [[3, 1], [4, 2]]}
    if multilingual:
        config.update(
            is_multilingual=True,
            lang_to_id={"<|en|>": 50259, "<|de|>": 50261, "<|fr|>": 50265},
            task_to_id={"transcribe": 50359, "translate": 50358},
        )
    else:
        config.update(is_multilingual=False)
    config.update(settings)
    return SimpleNamespace(**{key: value for key, value in config.items() if value is not None})


def whisper_pipeline(asr: MagicMock, config: Optional[SimpleNamespace] = None) -> None:
    asr.type = "seq2seq_whisper"
    asr.model.generation_config = config or whisper_config()


def ctc_pipeline(asr: MagicMock) -> None:
    asr.type = "ctc"
    asr.tokenizer = SimpleNamespace(_get_word_offsets=lambda offsets, delimiter: offsets)  # like wav2vec2's tokenizer


def text_pipeline(asr: MagicMock) -> None:
    asr.type = "seq2seq"  # e.g. Moonshine, Granite Speech
    asr.model.generation_config = SimpleNamespace(max_length=194)


@pytest.fixture
def modules():
    """Stand-ins for torch (no GPU), transformers with its pipeline, Whisper's language names, and the audio loader
    (5 s of silence)."""
    torch, transformers = MagicMock(), MagicMock()
    torch.cuda.is_available.return_value = False
    tokenization = SimpleNamespace(TO_LANGUAGE_CODE=WHISPER_LANGUAGES)
    stand_ins = {
        "torch": torch,
        "transformers": transformers,
        "transformers.models.whisper.tokenization_whisper": tokenization,
    }
    with patch.dict(sys.modules, stand_ins):
        with patch("speech_toolkit.backends.transformers_backend.load_audio") as load_audio:
            load_audio.return_value = np.zeros(5 * RATE, dtype=np.float32)
            asr = transformers.pipeline.return_value
            whisper_pipeline(asr)
            asr.return_value = {"text": "", "chunks": []}
            asr.model.detect_language.return_value = [50259]  # <|en|>
            yield SimpleNamespace(torch=torch, transformers=transformers, asr=asr, load_audio=load_audio)


@pytest.fixture
def audio_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "talk.wav"
    path.touch()
    return path


def loaded(model: str = "openai/whisper-small", **kwargs: Any) -> TransformersBackend:
    backend = TransformersBackend()
    backend.load_model(model, **kwargs)
    return backend


def pipeline_calls(asr: MagicMock) -> List[Dict[str, Any]]:
    """The keyword arguments of each pipeline call (one per chunk), without the audio."""
    return [kwargs for _, kwargs in asr.call_args_list]


def pipeline_audio(asr: MagicMock) -> List[Dict[str, Any]]:
    """The audio of each pipeline call: ``{"raw": samples, "sampling_rate": rate}``."""
    return [args[0] for args, _ in asr.call_args_list]


def chunk(text: str, start: Optional[float], end: Optional[float], **extra: Any) -> Dict[str, Any]:
    return {"text": text, "timestamp": (start, end), **extra}


###############################################################################
# Loading
###############################################################################


class TestLoading:
    def test_registry_and_defaults(self):
        assert get_backend_class("transformers") is TransformersBackend
        assert isinstance(get_backend("transformers"), TransformersBackend)
        assert TransformersBackend.default_model() == "openai/whisper-large-v3-turbo"
        assert "facebook/wav2vec2-base-960h" in TransformersBackend.available_models()
        # What some models can do: the checks made before a model loads let these through.
        assert TransformersBackend.capabilities == {"translate", "language_detection", "word_timestamps", "waveform"}
        assert TransformersBackend.requires == ("torch", "transformers")

    def test_pipeline_gets_the_model_device_and_dtype(self, modules):
        backend = loaded("openai/whisper-small")

        modules.transformers.pipeline.assert_called_once_with(
            "automatic-speech-recognition", model="openai/whisper-small", device="cpu", dtype=modules.torch.float32
        )
        assert (backend.is_loaded, backend.model_name, backend.device) == (True, "openai/whisper-small", "cpu")
        assert backend.options == {"compute_type": "auto"}

    def test_gpu_and_precision(self, modules):
        modules.torch.cuda.is_available.return_value = True
        modules.torch.cuda.is_bf16_supported.return_value = True

        loaded("org/whisper-fine-tune")
        loaded("org/whisper-fine-tune", device="cuda:1", options={"compute_type": "float16"})

        assert [kwargs for _, kwargs in modules.transformers.pipeline.call_args_list] == [
            {"model": "org/whisper-fine-tune", "device": "cuda", "dtype": modules.torch.bfloat16},
            {"model": "org/whisper-fine-tune", "device": "cuda:1", "dtype": modules.torch.float16},
        ]

    def test_local_folder(self, modules, tmp_path):
        backend = loaded(str(tmp_path))

        assert modules.transformers.pipeline.call_args.kwargs["model"] == str(tmp_path)
        assert backend.model_name == str(tmp_path)

    def test_transcriber_checks_and_passes_options(self, modules):
        transcriber = Transcriber("transformers", "openai/whisper-small", device="cpu", compute_type="bfloat16")

        assert modules.transformers.pipeline.call_args.kwargs["dtype"] is modules.torch.bfloat16
        assert transcriber.backend.options == {"compute_type": "bfloat16"}
        with pytest.raises(UnsupportedOptionError, match="max_new_tokens"):
            Transcriber("transformers", "openai/whisper-small", options={"max_new_tokens": 10})

    def test_a_name_that_is_no_model(self):
        with pytest.raises(ModelNotFoundError, match="the Hugging Face repo id .* of a speech recognition model"):
            TransformersBackend().load_model("tiny")

    def test_missing_packages(self):
        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            with pytest.raises(BackendUnavailableError, match=r"speech-transcription-toolkit\[transformers\]"):
                TransformersBackend().load_model("openai/whisper-small")

    def test_load_failure(self, modules):
        modules.transformers.pipeline.side_effect = ValueError("Unrecognized configuration class BertConfig")
        backend = TransformersBackend()

        with pytest.raises(ModelLoadError, match="Failed to load transformers model 'bert/base': Unrecognized"):
            backend.load_model("bert/base")
        assert not backend.is_loaded
        assert backend.capabilities == TransformersBackend.capabilities

    def test_a_failed_load_forgets_the_previous_model(self, modules):
        backend = loaded("facebook/wav2vec2-base-960h")
        modules.transformers.pipeline.side_effect = OSError("no such repo")

        with pytest.raises(ModelLoadError):
            backend.load_model("org/missing")
        with pytest.raises(ModelLoadError, match="No model loaded"):
            backend.transcribe(np.zeros(RATE, np.float32))

    def test_transcribe_without_model(self):
        with pytest.raises(ModelLoadError, match="No model loaded"):
            TransformersBackend().transcribe(pathlib.Path("talk.wav"))

    def test_missing_file(self, modules, tmp_path):
        with pytest.raises(FileNotFoundError):
            loaded().transcribe(tmp_path / "missing.wav")


###############################################################################
# What a loaded model can do
###############################################################################


class TestTraits:
    def test_multilingual_whisper(self, modules):
        backend = loaded()

        assert backend.traits == ModelTraits(
            WHISPER,
            languages=frozenset({"en", "de", "fr"}),
            translate=True,
            detects_language=True,
            segment_timestamps=True,
            word_timestamps=True,
        )
        assert backend.capabilities == {"translate", "language_detection", "word_timestamps", "waveform"}

    def test_english_only_whisper(self, modules):
        whisper_pipeline(modules.asr, whisper_config(multilingual=False))

        backend = loaded("openai/whisper-small.en")

        assert backend.traits.language == "en" and not backend.traits.languages
        assert backend.capabilities == {"word_timestamps", "waveform"}

    @pytest.mark.parametrize("language", ["<|en|>", "english", "en"])
    def test_whisper_set_to_one_language(self, modules, language):
        """distil-whisper's English models are multilingual Whisper models whose generation config sets English."""
        whisper_pipeline(modules.asr, whisper_config(language=language))

        backend = loaded("distil-whisper/distil-large-v3.5")

        assert backend.traits.language == "en"
        assert backend.capabilities == {"translate", "word_timestamps", "waveform"}  # it doesn't detect the language

    def test_whisper_without_alignment_heads_has_no_word_timestamps(self, modules):
        whisper_pipeline(modules.asr, whisper_config(alignment_heads=None))

        assert loaded("org/fine-tune").capabilities == {"translate", "language_detection", "waveform"}

    def test_whisper_without_timestamp_tokens(self, modules):
        whisper_pipeline(modules.asr, whisper_config(no_timestamps_token_id=None))

        backend = loaded("org/fine-tune")

        assert not backend.traits.segment_timestamps and not backend.traits.word_timestamps

    def test_ctc(self, modules):
        ctc_pipeline(modules.asr)

        backend = loaded("facebook/wav2vec2-base-960h")

        assert backend.traits == ModelTraits(CTC, segment_timestamps=True, word_timestamps=True)
        assert backend.capabilities == {"word_timestamps", "waveform"}

    def test_ctc_with_a_language_model(self):
        asr = SimpleNamespace(type="ctc_with_lm", tokenizer=SimpleNamespace())

        assert read_traits(asr).word_timestamps

    def test_ctc_whose_tokenizer_has_no_word_offsets_gives_text(self):
        asr = SimpleNamespace(type="ctc", tokenizer=SimpleNamespace())  # e.g. Parakeet CTC

        assert read_traits(asr) == ModelTraits(TEXT)

    @pytest.mark.parametrize("kind", ["seq2seq", "tdt", "something-new"])
    def test_other_models_give_text(self, kind):
        assert read_traits(SimpleNamespace(type=kind)) == ModelTraits(TEXT)
        assert ModelTraits(TEXT).capabilities == {"waveform"}

    def test_the_class_keeps_what_some_models_can_do(self, modules):
        text_pipeline(modules.asr)

        backend = loaded("moonshine-ai/moonshine-tiny")

        assert backend.capabilities == {"waveform"}
        assert TransformersBackend.capabilities == {"translate", "language_detection", "word_timestamps", "waveform"}


###############################################################################
# Requests the model can't do: refused before the audio is read
###############################################################################


class TestUnsupportedRequests:
    def test_word_timestamps_from_a_model_without_them_error_before_transcribing(self, modules, audio_file):
        text_pipeline(modules.asr)
        backend = loaded("moonshine-ai/moonshine-tiny")

        with pytest.raises(UnsupportedOptionError, match="moonshine-tiny can't give word timestamps"):
            backend.transcribe(audio_file, word_timestamps=True)
        modules.load_audio.assert_not_called()
        modules.asr.assert_not_called()

    @pytest.mark.parametrize("setup", [ctc_pipeline, text_pipeline], ids=["ctc", "text"])
    def test_translation_by_a_model_that_cant_errors_before_transcribing(self, modules, audio_file, setup):
        setup(modules.asr)
        backend = loaded("org/model")

        with pytest.raises(UnsupportedOptionError, match="org/model can't translate; multilingual Whisper models"):
            backend.transcribe(audio_file, task="translate")
        modules.load_audio.assert_not_called()

    def test_english_only_whisper_cant_translate(self, modules, audio_file):
        whisper_pipeline(modules.asr, whisper_config(multilingual=False))

        with pytest.raises(UnsupportedOptionError, match="can't translate"):
            loaded("openai/whisper-small.en").transcribe(audio_file, task="translate")

    def test_unknown_task(self, modules, audio_file):
        with pytest.raises(UnsupportedOptionError, match="task must be 'transcribe' or 'translate', not 'summarize'"):
            loaded().transcribe(audio_file, task="summarize")

    def test_language_whisper_doesnt_know(self, modules, audio_file):
        with pytest.raises(UnsupportedOptionError, match="doesn't know the language 'xx'; give a language code"):
            loaded().transcribe(audio_file, language="xx")
        modules.load_audio.assert_not_called()

    def test_transcriber_refuses_a_translation_before_decoding(self, modules):
        ctc_pipeline(modules.asr)
        transcriber = Transcriber("transformers", "facebook/wav2vec2-base-960h")

        with pytest.raises(UnsupportedOptionError, match="the transformers backend can't translate"):
            transcriber.transcribe(b"not decoded", task="translate")
        modules.asr.assert_not_called()

    def test_transcriber_sees_the_models_capabilities(self, modules, audio_file):
        """Word timestamps the model can't give are skipped with a warning, as for any backend without them, so
        diarization and subtitle line splitting don't ask for them."""
        text_pipeline(modules.asr)
        modules.asr.return_value = {"text": " Hello."}
        transcriber = Transcriber("transformers", "moonshine-ai/moonshine-tiny")

        assert not transcriber.supports("word_timestamps") and transcriber.supports("waveform")
        with pytest.warns(SpeechToolkitWarning, match="doesn't support word timestamps; ignoring it"):
            result = transcriber.transcribe(audio_file, word_timestamps=True)
        assert result.text == "Hello."


###############################################################################
# Whisper models
###############################################################################


class TestWhisper:
    def test_pipeline_receives_the_chunks_and_timestamps(self, modules, audio_file):
        """40 s of audio is two chunks, cut in a pause; each goes to the pipeline whole, with segment timestamps, the
        task, and the language detected in the first chunk."""
        audio = np.zeros(40 * RATE, dtype=np.float32)
        modules.load_audio.return_value = audio
        modules.asr.side_effect = [
            {"text": " Hello.", "chunks": [chunk(" Hello.", 0.0, 1.5)]},
            {"text": " Bye.", "chunks": [chunk(" Bye.", 2.0, 3.0)]},
        ]

        result = loaded().transcribe(audio_file, verbose=False)

        chunks = split_audio(audio, 30)
        assert [start for start, _ in chunks] == [0.0, SECOND_CHUNK]
        sent = pipeline_audio(modules.asr)
        assert [entry["sampling_rate"] for entry in sent] == [RATE, RATE]
        assert [len(entry["raw"]) for entry in sent] == [len(samples) for _, samples in chunks]
        assert pipeline_calls(modules.asr) == [
            {"generate_kwargs": {"task": "transcribe", "language": "en"}, "return_timestamps": True},
            {"generate_kwargs": {"task": "transcribe", "language": "en"}, "return_timestamps": True},
        ]
        assert result.segments == [
            {"id": 0, "start": 0.0, "end": 1.5, "text": " Hello."},
            {"id": 1, "start": 27.05, "end": 28.05, "text": " Bye."},
        ]
        assert (result.text, result.language, result.duration) == ("Hello. Bye.", "en", 40.0)

    def test_given_language_and_translation(self, modules):
        modules.asr.return_value = {"text": " Hello.", "chunks": [chunk(" Hello.", 0.0, 1.0)]}

        result = loaded().transcribe(np.zeros(RATE, np.float32), language="DE", task="translate", verbose=False)

        assert pipeline_calls(modules.asr) == [
            {"generate_kwargs": {"task": "translate", "language": "de"}, "return_timestamps": True}
        ]
        assert result.language == "de"  # the language spoken, as the Whisper backends report it

    def test_language_by_name(self, modules):
        loaded().transcribe(np.zeros(RATE, np.float32), language="French", verbose=False)

        assert pipeline_calls(modules.asr)[0]["generate_kwargs"] == {"task": "transcribe", "language": "fr"}

    def test_turbo_translation_warns(self, modules):
        backend = loaded("openai/whisper-large-v3-turbo")

        with pytest.warns(SpeechToolkitWarning, match="isn't trained for translation"):
            backend.transcribe(np.zeros(RATE, np.float32), task="translate", verbose=False)

    def test_english_only_model_gets_neither_task_nor_language(self, modules):
        whisper_pipeline(modules.asr, whisper_config(multilingual=False))
        modules.asr.return_value = {"text": " Hi.", "chunks": [chunk(" Hi.", 0.0, 1.0)]}
        backend = loaded("openai/whisper-small.en")

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # "en" is what it transcribes anyway: no warning
            result = backend.transcribe(np.zeros(RATE, np.float32), language="en", verbose=False)

        assert pipeline_calls(modules.asr) == [{"return_timestamps": True}]
        assert result.language == "en"

    def test_english_only_model_ignores_another_language(self, modules):
        whisper_pipeline(modules.asr, whisper_config(multilingual=False))
        backend = loaded("openai/whisper-small.en")

        with pytest.warns(SpeechToolkitWarning, match="can't be told the language; ignoring language 'de'"):
            backend.transcribe(np.zeros(RATE, np.float32), language="de", verbose=False)
        assert pipeline_calls(modules.asr) == [{"return_timestamps": True}]

    def test_model_set_to_one_language(self, modules):
        """distil-whisper's English models: no language detection, and the result has the language they're set to."""
        whisper_pipeline(modules.asr, whisper_config(language="english"))

        result = loaded("distil-whisper/distil-large-v3.5").transcribe(np.zeros(RATE, np.float32), verbose=False)

        assert pipeline_calls(modules.asr) == [{"generate_kwargs": {"task": "transcribe"}, "return_timestamps": True}]
        assert result.language == "en"

    def test_model_without_timestamp_tokens_gives_a_segment_per_chunk(self, modules):
        whisper_pipeline(modules.asr, whisper_config(no_timestamps_token_id=None))
        modules.asr.return_value = {"text": " Hallo."}
        modules.asr.model.detect_language.return_value = [50261]  # <|de|>

        result = loaded("org/fine-tune").transcribe(np.zeros(2 * RATE, np.float32), verbose=False)

        assert pipeline_calls(modules.asr) == [{"generate_kwargs": {"task": "transcribe", "language": "de"}}]
        assert (result.segments, result.language) == ([{"id": 0, "start": 0.0, "end": 2.0, "text": "Hallo."}], "de")

    def test_language_is_detected_once_in_the_first_chunk(self, modules):
        """As Whisper does: the first 30 s, as the model's feature extractor makes them, on the model's device."""
        asr = modules.asr
        asr.model.detect_language.return_value = [50261]  # <|de|>
        audio = np.zeros(40 * RATE, np.float32)

        result = loaded().transcribe(audio, verbose=False)

        [(samples,), options] = asr.feature_extractor.call_args
        assert len(samples) == len(split_audio(audio, 30)[0][1])
        assert options == {"sampling_rate": RATE, "return_tensors": "pt"}
        features = asr.feature_extractor.return_value.input_features
        features.to.assert_called_once_with(asr.model.device, dtype=asr.model.dtype)
        asr.model.detect_language.assert_called_once_with(input_features=features.to.return_value)
        assert [kwargs["generate_kwargs"]["language"] for kwargs in pipeline_calls(asr)] == ["de", "de"]
        assert result.language == "de"

    def test_a_language_token_it_doesnt_list(self, modules):
        modules.asr.model.detect_language.return_value = [1]

        result = loaded().transcribe(np.zeros(RATE, np.float32), verbose=False)

        assert pipeline_calls(modules.asr) == [{"generate_kwargs": {"task": "transcribe"}, "return_timestamps": True}]
        assert result.language is None

    def test_missing_and_stray_timestamps(self, modules):
        """Whisper's last segment can lack its end: it ends with the chunk. Times stay inside the chunk."""
        modules.asr.return_value = {
            "text": " One. Two. Three.",
            "chunks": [chunk(" One.", None, 1.0), chunk(" Two.", 0.5, 7.0), chunk(" Three.", 1.5, None)],
        }

        result = loaded().transcribe(np.zeros(4 * RATE, np.float32), language="en", verbose=False)

        assert [(segment["start"], segment["end"]) for segment in result.segments] == [
            (0.0, 1.0),
            (0.5, 4.0),
            (1.5, 4.0),
        ]

    def test_text_without_timed_chunks(self, modules):
        modules.asr.return_value = {"text": " Hello there."}

        result = loaded().transcribe(np.zeros(3 * RATE, np.float32), language="en", verbose=False)

        assert result.segments == [{"id": 0, "start": 0.0, "end": 3.0, "text": "Hello there."}]

    def test_word_timestamps_make_the_segments(self, modules):
        """With word timestamps, Whisper's words are grouped into segments at sentence ends and pauses."""
        modules.asr.return_value = {
            "text": " Hello world. How are you?",
            "chunks": [
                chunk(" Hello", 0.0, 0.4),
                chunk(" world.", 0.4, 0.9),
                chunk(" How", 1.2, 1.4),
                chunk(" are", 1.4, 1.6),
                chunk(" you?", 1.6, 2.0),
            ],
        }

        result = loaded().transcribe(np.zeros(3 * RATE, np.float32), language="en", word_timestamps=True, verbose=False)

        assert pipeline_calls(modules.asr)[0]["return_timestamps"] == "word"
        assert [(s["id"], s["start"], s["end"], s["text"]) for s in result.segments] == [
            (0, 0.0, 0.9, " Hello world."),
            (1, 1.2, 2.0, " How are you?"),
        ]
        assert result.segments[1]["words"] == [
            {"word": " How", "start": 1.2, "end": 1.4},
            {"word": " are", "start": 1.4, "end": 1.6},
            {"word": " you?", "start": 1.6, "end": 2.0},
        ]

    def test_progress_and_logging(self, modules, caplog):
        modules.asr.side_effect = [
            {"text": " One.", "chunks": [chunk(" One.", 0.0, 1.0)]},
            {"text": " Two.", "chunks": [chunk(" Two.", 0.0, 1.0)]},
        ]
        transcriber = Transcriber("transformers", "openai/whisper-small")
        events: List[Any] = []

        with caplog.at_level("INFO", logger="speech_toolkit"):
            transcriber.transcribe(np.zeros(40 * RATE, np.float32), language="en", verbose=True, progress=events.append)

        reported = [event.segment["text"] for event in events if event.segment is not None]
        assert reported == [" One.", " Two."]  # each segment once, as its chunk is done
        assert [event.done for event in events][-1] == 40.0
        assert "Transformers: transcribed 2/2 chunks" in caplog.text


###############################################################################
# CTC and text-only models
###############################################################################


class TestCtc:
    OUTPUT = {
        "text": "HELLO WORLD GOOD MORNING",
        "chunks": [
            chunk("HELLO", 0.1, 0.5),
            chunk("WORLD", 0.6, 1.0),
            chunk("GOOD", 2.5, 2.8),  # after a pause: a new segment
            chunk("MORNING", 2.9, 3.4),
        ],
    }

    def test_words_make_the_segments(self, modules, audio_file):
        ctc_pipeline(modules.asr)
        modules.asr.return_value = self.OUTPUT

        result = loaded("facebook/wav2vec2-base-960h").transcribe(audio_file, verbose=False)

        assert pipeline_calls(modules.asr) == [{"return_timestamps": "word"}]
        assert result.segments == [
            {"id": 0, "start": 0.1, "end": 1.0, "text": " HELLO WORLD"},
            {"id": 1, "start": 2.5, "end": 3.4, "text": " GOOD MORNING"},
        ]
        assert (result.text, result.language) == ("HELLO WORLD GOOD MORNING", None)

    def test_word_timestamps(self, modules, audio_file):
        ctc_pipeline(modules.asr)
        modules.asr.return_value = self.OUTPUT

        result = loaded("facebook/wav2vec2-base-960h").transcribe(audio_file, word_timestamps=True, verbose=False)

        assert result.segments[0]["words"] == [
            {"word": " HELLO", "start": 0.1, "end": 0.5},
            {"word": " WORLD", "start": 0.6, "end": 1.0},
        ]

    def test_language_is_ignored_with_a_warning(self, modules, audio_file):
        ctc_pipeline(modules.asr)
        backend = loaded("facebook/wav2vec2-base-960h")

        with pytest.warns(SpeechToolkitWarning, match="can't be told the language; ignoring language 'fr'"):
            result = backend.transcribe(audio_file, language="fr", verbose=False)
        assert pipeline_calls(modules.asr) == [{"return_timestamps": "word"}]
        assert result.language is None


class TestTextOnly:
    def test_one_segment_per_chunk(self, modules, audio_file):
        text_pipeline(modules.asr)
        modules.load_audio.return_value = np.zeros(40 * RATE, dtype=np.float32)
        modules.asr.side_effect = [{"text": " Hello there. "}, {"text": ""}]  # the second chunk is silent

        result = loaded("moonshine-ai/moonshine-tiny").transcribe(audio_file, verbose=False)

        assert pipeline_calls(modules.asr) == [{}, {}]  # neither timestamps nor generation settings
        assert result.segments == [{"id": 0, "start": 0.0, "end": SECOND_CHUNK, "text": "Hello there."}]
        assert result.text == "Hello there."

    def test_empty_audio(self, modules, audio_file):
        text_pipeline(modules.asr)
        modules.load_audio.return_value = np.zeros(0, dtype=np.float32)

        result = loaded("moonshine-ai/moonshine-tiny").transcribe(audio_file, verbose=False)

        assert (result.text, result.segments, result.duration) == ("", [], 0.0)
        modules.asr.assert_not_called()


###############################################################################
# Helpers
###############################################################################


class TestHelpers:
    def test_timed_keeps_times_in_the_chunk_and_in_order(self):
        items = [chunk("a", 0.5, 1.0), chunk("b", 0.2, 0.8), chunk("c", None, None), {"text": "d"}]

        assert _timed(items, offset=10.0, end=12.0) == [
            (10.5, 11.0, "a"),
            (10.5, 11.0, "b"),  # never before the previous one
            (11.0, 12.0, "c"),  # from the previous end to the chunk's end
            (12.0, 12.0, "d"),
        ]

    def test_language_codes(self):
        with patch.dict(sys.modules, {"transformers.models.whisper.tokenization_whisper": None}):
            assert _language_code("<|DE|>") == "de"  # without Whisper's language names, codes still work
            assert _language_code("english") == "english"
        tokenization = SimpleNamespace(TO_LANGUAGE_CODE=WHISPER_LANGUAGES)
        with patch.dict(sys.modules, {"transformers.models.whisper.tokenization_whisper": tokenization}):
            assert [_language_code(name) for name in ("<|en|>", " English ", "en", "xx")] == ["en", "en", "en", "xx"]
