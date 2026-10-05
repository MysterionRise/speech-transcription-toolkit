"""Tests for the pluggable backend system.

Heavy libraries (whisper, torch, transformers, faster_whisper) are replaced with mocks in
``sys.modules``: the backends import them lazily, so these tests run without them installed.
"""

import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from speech_toolkit.backends import (
    DEFAULT_BACKEND,
    TranscriptionBackend,
    TranscriptionResult,
    get_backend,
    get_backend_class,
    list_backends,
    register_backend,
)
from speech_toolkit.backends.base import TranscriptionBackend as BaseBackend
from speech_toolkit.backends.faster_whisper_backend import FasterWhisperBackend
from speech_toolkit.backends.voxtral_backend import VoxtralBackend
from speech_toolkit.backends.whisper_backend import WhisperBackend


@pytest.fixture
def audio_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """An (empty) audio file; the mocked models never read it."""
    path = tmp_path / "test.mp3"
    path.touch()
    return path


class TestTranscriptionResult:
    """Tests for TranscriptionResult class."""

    def test_basic_creation(self):
        """Test creating a basic TranscriptionResult."""
        result = TranscriptionResult(
            text="Hello world",
            segments=[{"start": 0.0, "end": 1.0, "text": "Hello world"}],
        )
        assert result.text == "Hello world"
        assert len(result.segments) == 1
        assert result.language is None
        assert result.raw == {}

    def test_with_language_and_raw(self):
        """Test creating TranscriptionResult with all fields."""
        result = TranscriptionResult(
            text="Bonjour",
            segments=[{"start": 0.0, "end": 0.5, "text": "Bonjour"}],
            language="fr",
            raw={"custom_field": "value"},
        )
        assert result.text == "Bonjour"
        assert result.language == "fr"
        assert result.raw["custom_field"] == "value"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = TranscriptionResult(
            text="Test",
            segments=[{"start": 0.0, "end": 1.0, "text": "Test"}],
            language="en",
            raw={"extra": "data"},
        )
        d = result.to_dict()
        assert d["text"] == "Test"
        assert d["segments"] == [{"start": 0.0, "end": 1.0, "text": "Test"}]
        assert d["language"] == "en"
        assert d["extra"] == "data"  # raw fields merged in

    def test_to_dict_empty_segments(self):
        """Test to_dict with empty segments."""
        result = TranscriptionResult(text="", segments=[])
        d = result.to_dict()
        assert d["text"] == ""
        assert d["segments"] == []
        assert d["language"] is None

    def test_to_dict_includes_speaker_segments_after_diarization(self):
        """speaker_segments appears in the dict (and JSON) only when diarization ran."""
        result = TranscriptionResult(text="Hi", segments=[{"start": 0.0, "end": 1.0, "text": "Hi"}])
        assert "speaker_segments" not in result.to_dict()

        result.speaker_segments = [{"start": 0.0, "end": 1.0, "text": "Hi", "speaker": "SPEAKER_00"}]
        assert result.to_dict()["speaker_segments"][0]["speaker"] == "SPEAKER_00"
        assert result.render("txt") == "[SPEAKER_00] Hi"

    def test_render_formats(self):
        """render() gives txt by default and any other output format on request."""
        result = TranscriptionResult(text=" Hi there.", segments=[{"start": 0.0, "end": 1.5, "text": " Hi there."}])
        assert result.render() == "Hi there."
        assert result.render("srt") == "1\n00:00:00,000 --> 00:00:01,500\nHi there.\n"

    def test_save_picks_format_from_extension(self, tmp_path: pathlib.Path):
        """save() writes the format named by the extension, or the one passed explicitly."""
        result = TranscriptionResult(text=" Hi.", segments=[{"start": 0.0, "end": 1.0, "text": " Hi."}])

        dest = result.save(tmp_path / "subs" / "talk.vtt")
        assert dest == tmp_path / "subs" / "talk.vtt"
        assert dest.read_text(encoding="utf-8").startswith("WEBVTT\n")

        result.save(str(tmp_path / "talk.out"), fmt="json")
        assert '"language": null' in (tmp_path / "talk.out").read_text(encoding="utf-8")


class TestBackendRegistry:
    """Tests for backend registry functions."""

    def test_list_backends(self):
        """Test listing available backends."""
        backends = list_backends()
        assert isinstance(backends, list)
        assert "whisper" in backends
        assert "faster-whisper" in backends
        assert "voxtral" in backends

    def test_get_backend_whisper(self):
        """Test getting Whisper backend."""
        backend = get_backend("whisper")
        assert isinstance(backend, WhisperBackend)
        assert backend.name == "whisper"

    def test_get_backend_faster_whisper(self):
        """Test getting faster-whisper backend."""
        backend = get_backend("faster-whisper")
        assert isinstance(backend, FasterWhisperBackend)
        assert backend.name == "faster-whisper"

    def test_get_backend_voxtral(self):
        """Test getting Voxtral backend."""
        backend = get_backend("voxtral")
        assert backend.name == "voxtral"

    def test_get_backend_invalid(self):
        """Test getting non-existent backend."""
        with pytest.raises(ValueError) as exc_info:
            get_backend("nonexistent")
        assert "Unknown backend" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)

    def test_get_backend_class(self):
        """Test getting backend class."""
        cls = get_backend_class("whisper")
        assert cls is WhisperBackend

    def test_get_backend_class_invalid(self):
        """Test getting non-existent backend class."""
        with pytest.raises(ValueError):
            get_backend_class("nonexistent")

    def test_default_backend(self):
        """Test that default backend is whisper."""
        assert DEFAULT_BACKEND == "whisper"

    def test_register_custom_backend(self, monkeypatch):
        """Test registering a custom backend."""
        from speech_toolkit.backends import _BACKENDS

        class CustomBackend(TranscriptionBackend):
            name = "custom"
            description = "Custom test backend"

            @classmethod
            def available_models(cls):
                return ["model1"]

            def load_model(self, model_name, device=None):
                pass

            def transcribe(self, audio_path, language=None, task="transcribe", verbose=True):
                return TranscriptionResult(text="custom", segments=[])

        monkeypatch.setattr("speech_toolkit.backends._BACKENDS", {**_BACKENDS})

        register_backend("custom_test", CustomBackend)
        assert "custom_test" in list_backends()

        backend = get_backend("custom_test")
        assert isinstance(backend, CustomBackend)

    def test_register_backend_rejects_non_subclass(self):
        """Test that registering a non-TranscriptionBackend class raises TypeError."""
        with pytest.raises(TypeError, match="must be a subclass"):
            register_backend("bad", str)  # type: ignore

    def test_register_backend_rejects_duplicate(self):
        """Test that registering over an existing backend raises ValueError."""
        with pytest.raises(ValueError, match="already registered"):
            register_backend("whisper", WhisperBackend)

    def test_register_backend_force_overwrite(self, monkeypatch):
        """Test that force=True allows overwriting an existing backend."""
        from speech_toolkit.backends import _BACKENDS

        monkeypatch.setattr("speech_toolkit.backends._BACKENDS", {**_BACKENDS})

        class AltWhisper(TranscriptionBackend):
            name = "alt_whisper"
            description = "Alternative"

            @classmethod
            def available_models(cls):
                return ["model1"]

            def load_model(self, model_name, device=None):
                pass

            def transcribe(self, audio_path, language=None, task="transcribe", verbose=True):
                return TranscriptionResult(text="", segments=[])

        register_backend("whisper", AltWhisper, force=True)
        assert get_backend_class("whisper") is AltWhisper


@pytest.fixture
def mock_whisper():
    """Stand-in for the openai-whisper module, whose model reports running on CPU."""
    module = MagicMock()
    param = MagicMock()
    param.device = "cpu"
    module.load_model.return_value.parameters.return_value = iter([param])
    with patch.dict(sys.modules, {"whisper": module}):
        yield module


class TestWhisperBackend:
    """Tests for WhisperBackend."""

    def test_available_models(self):
        """Test available models list, including every name openai-whisper accepts."""
        models = WhisperBackend.available_models()
        for name in ("tiny", "base", "small", "medium", "large", "turbo", "large-v3", "large-v3-turbo"):
            assert name in models

    def test_default_model(self):
        """Test default model is turbo."""
        assert WhisperBackend.default_model() == "turbo"

    def test_backend_properties_before_load(self):
        """Test backend properties before model is loaded."""
        backend = WhisperBackend()
        assert backend.is_loaded is False
        assert backend.model_name is None
        assert backend.device is None

    def test_load_model(self, mock_whisper):
        """Test loading a model."""
        backend = WhisperBackend()
        backend.load_model("tiny")

        mock_whisper.load_model.assert_called_once_with("tiny", device=None)
        assert backend.is_loaded is True
        assert backend.model_name == "tiny"
        assert backend.device == "cpu"

    def test_load_model_with_device(self, mock_whisper):
        """Test loading a model with specific device."""
        backend = WhisperBackend()
        backend.load_model("base", device="cuda")

        mock_whisper.load_model.assert_called_once_with("base", device="cuda")
        assert backend.device == "cuda"

    def test_load_model_invalid(self):
        """Test loading invalid model raises error."""
        backend = WhisperBackend()
        with pytest.raises(ValueError) as exc_info:
            backend.load_model("invalid_model")
        assert "Unknown Whisper model" in str(exc_info.value)

    def test_load_model_missing_dependency(self):
        """Test a helpful ImportError when openai-whisper isn't installed."""
        with patch.dict(sys.modules, {"whisper": None}):
            with pytest.raises(ImportError, match="pip install openai-whisper"):
                WhisperBackend().load_model("tiny")

    def test_transcribe_without_model(self):
        """Test transcribing without loading model raises error."""
        backend = WhisperBackend()
        with pytest.raises(RuntimeError) as exc_info:
            backend.transcribe(pathlib.Path("audio.mp3"))
        assert "No model loaded" in str(exc_info.value)

    def test_transcribe_missing_file(self, mock_whisper, tmp_path):
        """Test transcribing missing file raises error."""
        backend = WhisperBackend()
        backend.load_model("tiny")

        with pytest.raises(FileNotFoundError):
            backend.transcribe(tmp_path / "nonexistent.mp3")

    def test_transcribe_success(self, mock_whisper, audio_file):
        """Test successful transcription."""
        mock_whisper.load_model.return_value.transcribe.return_value = {
            "text": "Hello world",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": " Hello world",
                    "tokens": [1, 2, 3],
                    "temperature": 0.0,
                    "avg_logprob": -0.5,
                    "compression_ratio": 1.2,
                    "no_speech_prob": 0.01,
                }
            ],
            "language": "en",
        }

        backend = WhisperBackend()
        backend.load_model("tiny")
        result = backend.transcribe(audio_file)

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert len(result.segments) == 1
        assert result.language == "en"
        assert result.segments[0]["start"] == 0.0
        assert result.segments[0]["end"] == 1.0

    def test_transcribe_with_language(self, mock_whisper, audio_file):
        """Test transcription with specified language."""
        model = mock_whisper.load_model.return_value
        model.transcribe.return_value = {"text": "Bonjour", "segments": [], "language": "fr"}

        backend = WhisperBackend()
        backend.load_model("tiny")
        backend.transcribe(audio_file, language="fr")

        assert model.transcribe.call_args[1]["language"] == "fr"

    def test_transcribe_translate_task(self, mock_whisper, audio_file):
        """Test transcription with translate task."""
        model = mock_whisper.load_model.return_value
        model.transcribe.return_value = {"text": "Hello", "segments": []}

        backend = WhisperBackend()
        backend.load_model("tiny")
        backend.transcribe(audio_file, task="translate")

        assert model.transcribe.call_args[1]["task"] == "translate"

    @pytest.mark.parametrize("verbose, whisper_verbose", [(True, False), (False, None)])
    def test_transcribe_never_prints_segments(self, mock_whisper, audio_file, verbose, whisper_verbose):
        """Whisper's verbose=True prints every segment to stdout; progress bar (False) or silence (None) only."""
        model = mock_whisper.load_model.return_value
        model.transcribe.return_value = {"text": "", "segments": []}

        backend = WhisperBackend()
        backend.load_model("tiny")
        backend.transcribe(audio_file, verbose=verbose)

        assert model.transcribe.call_args[1]["verbose"] is whisper_verbose

    def test_transcribe_prompt_and_word_timestamps(self, mock_whisper, audio_file):
        """--prompt becomes initial_prompt; word timestamps come back as each segment's "words"."""
        model = mock_whisper.load_model.return_value
        word = {"word": " Hi", "start": 0.0, "end": 0.4, "probability": 0.9}
        model.transcribe.return_value = {
            "text": " Hi",
            "segments": [{"start": 0.0, "end": 0.4, "text": " Hi", "words": [word]}],
        }

        backend = WhisperBackend()
        backend.load_model("tiny")
        result = backend.transcribe(audio_file, prompt="Kubernetes", word_timestamps=True)

        kwargs = model.transcribe.call_args[1]
        assert kwargs["initial_prompt"] == "Kubernetes"
        assert kwargs["word_timestamps"] is True
        assert result.segments[0]["words"] == [word]
        assert "vad" not in WhisperBackend.capabilities

    def test_transcribe_defaults_leave_options_out(self, mock_whisper, audio_file):
        """Without the options, Whisper gets neither initial_prompt nor word_timestamps."""
        model = mock_whisper.load_model.return_value
        model.transcribe.return_value = {"text": "", "segments": [{"start": 0.0, "end": 1.0, "text": ""}]}

        backend = WhisperBackend()
        backend.load_model("tiny")
        result = backend.transcribe(audio_file)

        assert "initial_prompt" not in model.transcribe.call_args[1]
        assert "word_timestamps" not in model.transcribe.call_args[1]
        assert "words" not in result.segments[0]

    @pytest.mark.parametrize("device, fp16", [(None, False), ("cuda", True)])
    def test_transcribe_fp16_only_off_cpu(self, mock_whisper, audio_file, device, fp16):
        """FP16 is requested only on GPU (Whisper warns and falls back to FP32 on CPU)."""
        model = mock_whisper.load_model.return_value
        model.transcribe.return_value = {"text": "", "segments": []}

        backend = WhisperBackend()
        backend.load_model("tiny", device=device)
        backend.transcribe(audio_file)

        assert model.transcribe.call_args[1]["fp16"] is fp16


@pytest.fixture
def mock_faster_whisper():
    """Stand-ins for the faster_whisper and ctranslate2 modules (no GPU)."""
    faster_whisper = MagicMock()
    ctranslate2 = MagicMock()
    ctranslate2.get_cuda_device_count.return_value = 0
    with patch.dict(sys.modules, {"faster_whisper": faster_whisper, "ctranslate2": ctranslate2}):
        yield faster_whisper


class TestFasterWhisperBackend:
    """Tests for FasterWhisperBackend."""

    def test_available_models(self):
        """Test available models list."""
        models = FasterWhisperBackend.available_models()
        for name in ("tiny", "small", "large-v3", "turbo", "distil-large-v3"):
            assert name in models

    def test_default_model(self):
        """Test default model is turbo."""
        assert FasterWhisperBackend.default_model() == "turbo"

    def test_load_model_invalid(self):
        """Test loading invalid model raises error."""
        with pytest.raises(ValueError, match="Unknown faster-whisper model"):
            FasterWhisperBackend().load_model("invalid_model")

    def test_load_model_missing_dependency(self):
        """Test a helpful ImportError when faster-whisper isn't installed."""
        with patch.dict(sys.modules, {"faster_whisper": None, "ctranslate2": None}):
            with pytest.raises(ImportError, match=r"speech-transcription-toolkit\[faster-whisper\]"):
                FasterWhisperBackend().load_model("tiny")

    def test_load_model_cpu_uses_int8(self, mock_faster_whisper):
        """Without a GPU the model runs on CPU with int8 weights."""
        backend = FasterWhisperBackend()
        backend.load_model("tiny")

        mock_faster_whisper.WhisperModel.assert_called_once_with("tiny", device="cpu", compute_type="int8")
        assert backend.is_loaded is True
        assert backend.model_name == "tiny"
        assert backend.device == "cpu"

    def test_load_model_cuda_uses_float16(self, mock_faster_whisper):
        """On CUDA the model runs in float16."""
        backend = FasterWhisperBackend()
        backend.load_model("small", device="cuda")

        mock_faster_whisper.WhisperModel.assert_called_once_with("small", device="cuda", compute_type="float16")

    def test_load_model_failure(self, mock_faster_whisper):
        """Test download/load errors surface as RuntimeError."""
        mock_faster_whisper.WhisperModel.side_effect = OSError("network down")

        backend = FasterWhisperBackend()
        with pytest.raises(RuntimeError, match="network down"):
            backend.load_model("tiny")
        assert backend.is_loaded is False

    def test_transcribe_without_model(self):
        """Test transcribing without loading model raises error."""
        with pytest.raises(RuntimeError, match="No model loaded"):
            FasterWhisperBackend().transcribe(pathlib.Path("audio.mp3"))

    def test_transcribe_missing_file(self, mock_faster_whisper, tmp_path):
        """Test transcribing missing file raises error."""
        backend = FasterWhisperBackend()
        backend.load_model("tiny")
        with pytest.raises(FileNotFoundError):
            backend.transcribe(tmp_path / "nonexistent.mp3")

    def test_transcribe_success(self, mock_faster_whisper, audio_file):
        """Segments and language are mapped to the standard result format."""

        def segment(i, start, end, text):
            return SimpleNamespace(
                id=i,
                start=start,
                end=end,
                text=text,
                tokens=[1],
                temperature=0.0,
                avg_logprob=-0.2,
                compression_ratio=1.1,
                no_speech_prob=0.01,
            )

        model = mock_faster_whisper.WhisperModel.return_value
        info = SimpleNamespace(language="de", duration=3.0, language_probability=0.98)
        model.transcribe.return_value = (iter([segment(1, 0.0, 1.5, " Hallo"), segment(2, 1.5, 3.0, " Welt")]), info)

        backend = FasterWhisperBackend()
        backend.load_model("tiny")
        result = backend.transcribe(audio_file, language="de", task="transcribe", verbose=False)

        model.transcribe.assert_called_once_with(
            str(audio_file),
            language="de",
            task="transcribe",
            log_progress=False,
            initial_prompt=None,
            vad_filter=False,
            word_timestamps=False,
        )
        assert result.text == "Hallo Welt"
        assert result.language == "de"
        assert [(s["id"], s["start"], s["end"], s["text"]) for s in result.segments] == [
            (1, 0.0, 1.5, " Hallo"),
            (2, 1.5, 3.0, " Welt"),
        ]
        assert result.to_dict()["duration"] == 3.0
        assert "words" not in result.segments[0]

    def test_transcribe_accuracy_options(self, mock_faster_whisper, audio_file):
        """--prompt, --vad and word timestamps map to initial_prompt, vad_filter and Segment.words."""
        model = mock_faster_whisper.WhisperModel.return_value
        words = [SimpleNamespace(word=" Hi", start=0.0, end=0.4, probability=0.9)]
        seg = SimpleNamespace(
            id=0,
            start=0.0,
            end=0.4,
            text=" Hi",
            tokens=[1],
            temperature=0.0,
            avg_logprob=-0.1,
            compression_ratio=1.0,
            no_speech_prob=0.0,
            words=words,
        )
        info = SimpleNamespace(language="en", duration=0.4, language_probability=0.99)
        model.transcribe.return_value = (iter([seg]), info)

        backend = FasterWhisperBackend()
        backend.load_model("tiny")
        result = backend.transcribe(audio_file, prompt="Kubernetes", vad=True, word_timestamps=True)

        kwargs = model.transcribe.call_args[1]
        assert (kwargs["initial_prompt"], kwargs["vad_filter"], kwargs["word_timestamps"]) == ("Kubernetes", True, True)
        assert result.segments[0]["words"] == [{"word": " Hi", "start": 0.0, "end": 0.4, "probability": 0.9}]
        assert FasterWhisperBackend.capabilities == {"prompt", "vad", "word_timestamps"}


class FakeInputs(dict):
    """Minimal stand-in for the BatchFeature returned by the Voxtral processor."""

    def __init__(self, prompt_len: int):
        super().__init__(input_ids="ids")
        self.input_ids = SimpleNamespace(shape=(1, prompt_len))

    def to(self, *args, **kwargs):
        return self


@pytest.fixture
def voxtral_modules():
    """Stand-ins for torch, transformers and whisper (audio decoding) as used by the Voxtral backend."""
    torch = MagicMock()
    torch.cuda.is_available.return_value = False
    transformers = MagicMock()
    whisper = MagicMock()
    with patch.dict(sys.modules, {"torch": torch, "transformers": transformers, "whisper": whisper}):
        yield SimpleNamespace(torch=torch, transformers=transformers, whisper=whisper)


class TestVoxtralBackend:
    """Tests for VoxtralBackend."""

    def test_available_models(self):
        """Test available models list."""
        models = VoxtralBackend.available_models()
        assert "voxtral-mini" in models
        assert "voxtral-small" in models

    def test_model_repo_ids(self):
        """Test the Hugging Face repo ids are the published ones."""
        assert VoxtralBackend.MODELS["voxtral-mini"] == "mistralai/Voxtral-Mini-3B-2507"
        assert VoxtralBackend.MODELS["voxtral-small"] == "mistralai/Voxtral-Small-24B-2507"

    def test_default_model(self):
        """Test default model is voxtral-mini."""
        assert VoxtralBackend.default_model() == "voxtral-mini"

    def test_backend_properties_before_load(self):
        """Test backend properties before model is loaded."""
        backend = VoxtralBackend()
        assert backend.is_loaded is False
        assert backend.model_name is None
        assert backend.device is None

    def test_load_model_invalid(self):
        """Test loading invalid model raises error."""
        with pytest.raises(ValueError, match="Unknown Voxtral model"):
            VoxtralBackend().load_model("invalid_model")

    def test_load_model_missing_dependency(self):
        """Test a helpful ImportError when transformers/torch aren't installed."""
        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            with pytest.raises(ImportError, match=r"speech-transcription-toolkit\[voxtral\]"):
                VoxtralBackend().load_model("voxtral-mini")

    def test_load_model_cpu(self, voxtral_modules):
        """Test loading on CPU uses float32 and the right repo id."""
        backend = VoxtralBackend()
        backend.load_model("voxtral-small")

        tf = voxtral_modules.transformers
        tf.AutoProcessor.from_pretrained.assert_called_once_with("mistralai/Voxtral-Small-24B-2507")
        tf.VoxtralForConditionalGeneration.from_pretrained.assert_called_once_with(
            "mistralai/Voxtral-Small-24B-2507", dtype=voxtral_modules.torch.float32
        )
        tf.VoxtralForConditionalGeneration.from_pretrained.return_value.to.assert_called_once_with("cpu")
        assert backend.is_loaded is True
        assert backend.device == "cpu"
        assert backend.model_name == "voxtral-small"

    def test_load_model_cuda_uses_bfloat16(self, voxtral_modules):
        """Test loading on a bf16-capable GPU uses bfloat16."""
        voxtral_modules.torch.cuda.is_available.return_value = True
        voxtral_modules.torch.cuda.is_bf16_supported.return_value = True

        backend = VoxtralBackend()
        backend.load_model("voxtral-mini")

        _, kwargs = voxtral_modules.transformers.VoxtralForConditionalGeneration.from_pretrained.call_args
        assert kwargs["dtype"] is voxtral_modules.torch.bfloat16
        assert backend.device == "cuda"

    def test_load_model_failure(self, voxtral_modules):
        """Test download/auth errors surface as RuntimeError and leave nothing loaded."""
        voxtral_modules.transformers.AutoProcessor.from_pretrained.side_effect = OSError("401 gated repo")

        backend = VoxtralBackend()
        with pytest.raises(RuntimeError, match="401 gated repo"):
            backend.load_model("voxtral-mini")
        assert backend.is_loaded is False

    def test_transcribe_without_model(self):
        """Test transcribing without loading model raises error."""
        with pytest.raises(RuntimeError, match="No model loaded"):
            VoxtralBackend().transcribe(pathlib.Path("audio.mp3"))

    def test_transcribe_translate_unsupported(self, voxtral_modules, audio_file):
        """Test that translation is rejected with a clear error."""
        backend = VoxtralBackend()
        backend.load_model("voxtral-mini")
        with pytest.raises(ValueError, match="only supports --task transcribe"):
            backend.transcribe(audio_file, task="translate")

    def test_transcribe_missing_file(self, voxtral_modules, tmp_path):
        """Test transcribing missing file raises error."""
        backend = VoxtralBackend()
        backend.load_model("voxtral-mini")
        with pytest.raises(FileNotFoundError):
            backend.transcribe(tmp_path / "nonexistent.mp3")

    def test_transcribe_chunks_become_segments(self, voxtral_modules, audio_file, capsys):
        """70 s of audio -> 30 s chunks, transcribed in batches, one timestamped segment each."""
        rate = 16000
        voxtral_modules.whisper.load_audio.return_value = [0.0] * (70 * rate)
        processor = voxtral_modules.transformers.AutoProcessor.from_pretrained.return_value
        processor.apply_transcription_request.side_effect = [FakeInputs(prompt_len=5), FakeInputs(prompt_len=5)]
        processor.batch_decode.side_effect = [[" one ", "two"], [""]]

        backend = VoxtralBackend()
        backend.load_model("voxtral-mini")
        with patch("speech_toolkit.backends.voxtral_backend.BATCH_SIZE", 2):
            result = backend.transcribe(audio_file, language="en")

        first_call = processor.apply_transcription_request.call_args_list[0][1]
        assert [len(chunk) for chunk in first_call["audio"]] == [30 * rate, 30 * rate]
        assert first_call["model_id"] == "mistralai/Voxtral-Mini-3B-2507"
        assert first_call["language"] == "en"
        assert first_call["sampling_rate"] == rate
        assert first_call["format"] == "wav"
        assert [(s["id"], s["start"], s["end"], s["text"]) for s in result.segments] == [
            (0, 0.0, 30.0, "one"),
            (1, 30.0, 60.0, "two"),
            (2, 60.0, 70.0, ""),
        ]
        assert result.text == "one two"
        assert result.language == "en"
        assert "3/3 chunks" in capsys.readouterr().err

    def test_transcribe_empty_audio(self, voxtral_modules, audio_file):
        """Test that empty audio produces an empty result without calling the model."""
        voxtral_modules.whisper.load_audio.return_value = []

        backend = VoxtralBackend()
        backend.load_model("voxtral-mini")
        result = backend.transcribe(audio_file, verbose=False)

        assert result.text == ""
        assert result.segments == []
        voxtral_modules.transformers.AutoProcessor.from_pretrained.return_value.apply_transcription_request.assert_not_called()


class TestBaseBackend:
    """Tests for base TranscriptionBackend class."""

    def test_cannot_instantiate_abstract(self):
        """Test that abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseBackend()

    def test_default_model_empty_list(self):
        """Test default_model with empty available_models."""

        class EmptyBackend(BaseBackend):
            name = "empty"
            description = "Empty test backend"

            @classmethod
            def available_models(cls):
                return []

            def load_model(self, model_name, device=None):
                pass

            def transcribe(self, audio_path, language=None, task="transcribe", verbose=True):
                return TranscriptionResult(text="", segments=[])

        assert EmptyBackend.default_model() == ""

    def test_to_dict_standard_fields_override_raw(self):
        """Test that standardized fields take priority over raw data in to_dict()."""
        result = TranscriptionResult(
            text="standardized text",
            segments=[{"start": 0.0, "end": 1.0, "text": "clean"}],
            language="en",
            raw={"text": "RAW", "segments": [{"raw": True}], "language": "xx", "extra": "data"},
        )
        d = result.to_dict()
        assert d["text"] == "standardized text"
        assert d["segments"] == [{"start": 0.0, "end": 1.0, "text": "clean"}]
        assert d["language"] == "en"
        assert d["extra"] == "data"
