"""What a model loads with: custom models (repo ids, folders, files), devices (auto, cpu, cuda, cuda:N, mps), the
precision (compute_type) and the options a backend declares (-O key=value).

faster_whisper, ctranslate2, whisper, torch, transformers and pyannote are mocks in sys.modules, as everywhere else.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from speech_toolkit import (
    ModelLoadError,
    ModelNotFoundError,
    SpeechToolkitWarning,
    Transcriber,
    TranscriptionBackend,
    TranscriptionResult,
    UnsupportedOptionError,
    cli,
    register_backend,
    transcribe,
)
from speech_toolkit.api import check_load_arguments
from speech_toolkit.backends import (
    BackendOption,
    CanaryBackend,
    Device,
    FasterWhisperBackend,
    ParakeetBackend,
    VoxtralBackend,
    WhisperBackend,
    backends_on,
    backends_with_option,
    faster_whisper_backend,
    parse_device,
)
from speech_toolkit.backends._transformers import torch_dtype
from speech_toolkit.backends.config import check_device, describe_devices, model_source, parse_options
from speech_toolkit.diarization import load_diarization_pipeline
from speech_toolkit.server import parse_args as server_parse_args

TRANSFORMERS_BACKENDS = [
    (ParakeetBackend, "ParakeetForTDT"),
    (CanaryBackend, "CanaryForConditionalGeneration"),
    (VoxtralBackend, "VoxtralForConditionalGeneration"),
]
REPO_ID = "someone/fine-tuned-model"


@pytest.fixture(autouse=True)
def no_color(monkeypatch):
    """Python 3.14's argparse can colour help and errors; these tests match the plain text."""
    monkeypatch.setenv("PYTHON_COLORS", "0")


@pytest.fixture
def ct2(monkeypatch):
    """Mocked faster_whisper and ctranslate2, on a machine without a GPU. The CUDA library check finds everything;
    ``checked`` records each time it runs."""
    faster_whisper, ctranslate2 = MagicMock(), MagicMock()
    ctranslate2.__version__ = "4.8.3"
    ctranslate2.get_cuda_device_count.return_value = 0
    info = MagicMock(language="en", duration=1.0, language_probability=0.99)
    faster_whisper.WhisperModel.return_value.transcribe.return_value = (iter([]), info)
    checked: List[str] = []
    monkeypatch.setattr(faster_whisper_backend, "_cuda_library_problem", lambda version: checked.append(version))
    with patch.dict(sys.modules, {"faster_whisper": faster_whisper, "ctranslate2": ctranslate2}):
        yield SimpleNamespace(ctranslate2=ctranslate2, WhisperModel=faster_whisper.WhisperModel, checked=checked)


@pytest.fixture
def torch_modules():
    """Mocked torch and transformers, without a GPU."""
    torch, transformers = MagicMock(), MagicMock()
    torch.cuda.is_available.return_value = False
    with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
        yield SimpleNamespace(torch=torch, transformers=transformers)


@pytest.fixture
def whisper_module():
    """Mocked openai-whisper whose models land on the CPU."""
    module = MagicMock()
    module.load_model.return_value.parameters.side_effect = lambda: iter([MagicMock(device="cpu")])
    with patch.dict(sys.modules, {"whisper": module}):
        yield module


@pytest.fixture
def audio(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "talk.wav"
    path.touch()
    return path


@pytest.fixture
def diarization(monkeypatch) -> List[Any]:
    """Replace the pyannote loader; the list collects the device each load got."""
    devices: List[Any] = []
    monkeypatch.setattr("speech_toolkit.api.load_diarization_pipeline", lambda device: devices.append(device))
    return devices


class OldStyleBackend(TranscriptionBackend):
    """A custom backend written before devices and options were declared: load_model(model_name, device)."""

    name = "old-style"
    calls: List[Any] = []

    @classmethod
    def available_models(cls):
        return ["base"]

    def load_model(self, model_name, device=None):
        OldStyleBackend.calls.append((model_name, device))
        self._model = object()

    def transcribe(self, audio_path, language=None, task="transcribe", verbose=True):
        return TranscriptionResult(text="", segments=[])


class OptionsBackend(OldStyleBackend):
    """A custom backend that declares custom models, devices and options, and records what load_model() got."""

    name = "with-options"
    MODEL_SOURCES = {"directory": ""}
    DEVICES = frozenset({"cpu", "mps"})
    OPTIONS = {
        "beam_width": BackendOption(int, 5, "Beams", minimum=1),
        "greedy": BackendOption(bool, False, "Decode greedily"),
    }

    def load_model(self, model_name, device=None, *, options=None):
        OptionsBackend.calls.append((model_name, device, options))
        self._options = self._resolve_options(options)
        self._model = object()


@pytest.fixture
def custom_backends(fake_backend):
    """Register OldStyleBackend and OptionsBackend, in the fake_backend fixture's copy of the registry."""
    register_backend("old-style", OldStyleBackend)
    register_backend("with-options", OptionsBackend)
    OldStyleBackend.calls = []
    OptionsBackend.calls = []


def listing(out: str) -> Dict[str, List[str]]:
    """--list-backends output as {backend: its lines after the description}."""
    backends = {}
    for block in out.split("\n\n")[1:]:
        lines = [line.strip() for line in block.strip().splitlines()]
        if lines:
            backends[lines[0].split()[0]] = lines[2:]
    return backends


###############################################################################
# Models: names, Hugging Face repo ids, local folders and files
###############################################################################


class TestModelSource:
    def test_local_paths(self, tmp_path):
        model_file = tmp_path / "model.pt"
        model_file.touch()

        assert model_source(str(tmp_path)) == "directory"
        assert model_source(str(model_file)) == "file"

    @pytest.mark.parametrize(
        "name", ["Systran/faster-whisper-tiny", "nvidia/parakeet-tdt-0.6b-v3", "org_1/model.v2", "a/b", "_x/y_"]
    )
    def test_repo_ids(self, name):
        assert model_source(name) == "hub"

    @pytest.mark.parametrize(
        "name",
        [
            "turbbo",  # a typo of a model name: no slash
            "a/b/c",
            "-org/name",
            "org/name-",
            "org--x/name",
            "org/na..me",
            "./models/tiny",
            "/no/such/folder",
            "C:\\models\\tiny",
            "org/",
            "",
        ],
    )
    def test_neither(self, name):
        assert model_source(name) is None


class TestResolveModel:
    @pytest.mark.parametrize(
        "backend_class, name, expected",
        [
            (FasterWhisperBackend, "tiny", "tiny"),
            (WhisperBackend, "turbo", "turbo"),
            (ParakeetBackend, "parakeet-tdt-0.6b-v2", "nvidia/parakeet-tdt-0.6b-v2"),
            (CanaryBackend, "canary-1b-v2", "nvidia/canary-1b-v2"),
            (VoxtralBackend, "voxtral-small", "mistralai/Voxtral-Small-24B-2507"),
        ],
    )
    def test_names_from_the_list(self, backend_class, name, expected):
        assert backend_class.resolve_model(name) == expected

    @pytest.mark.parametrize("backend_class", [FasterWhisperBackend, ParakeetBackend, CanaryBackend, VoxtralBackend])
    def test_repo_ids_and_folders_unchanged(self, backend_class, tmp_path):
        assert backend_class.resolve_model(REPO_ID) == REPO_ID
        assert backend_class.resolve_model(str(tmp_path)) == str(tmp_path)

    def test_whisper_takes_checkpoint_files_only(self, tmp_path):
        """openai-whisper loads custom models from .pt files, never from the Hub or a folder."""
        checkpoint = tmp_path / "fine-tuned.pt"
        checkpoint.touch()

        assert WhisperBackend.resolve_model(str(checkpoint)) == str(checkpoint)
        with pytest.raises(ModelNotFoundError, match=r"^The whisper backend can't load a model from a Hugging Face"):
            WhisperBackend.resolve_model("openai/whisper-tiny")
        with pytest.raises(ModelNotFoundError, match="can't load a model from a folder") as error:
            WhisperBackend.resolve_model(str(tmp_path))
        assert str(error.value).endswith("large-v3-turbo, turbo; or a local checkpoint file (.pt)")

    def test_faster_whisper_takes_no_files(self, tmp_path):
        checkpoint = tmp_path / "model.bin"
        checkpoint.touch()

        with pytest.raises(ModelNotFoundError, match=re.escape(f"can't load a model from a file: {checkpoint}.")):
            FasterWhisperBackend.resolve_model(str(checkpoint))

    def test_unknown_name_lists_the_models_and_custom_ones(self):
        with pytest.raises(ModelNotFoundError) as error:
            FasterWhisperBackend.resolve_model("turbbo")

        assert str(error.value) == (
            f"Unknown faster-whisper model: turbbo. Available: {', '.join(FasterWhisperBackend.MODELS)}; or a Hugging "
            "Face repo id (org/name) of a CTranslate2 model or a local CTranslate2 model folder"
        )
        assert isinstance(error.value, ValueError)  # as ModelNotFoundError always was

    def test_usual_words_for_custom_models(self):
        assert ParakeetBackend.describe_custom_models() == "a Hugging Face repo id (org/name) or a local model folder"
        assert OldStyleBackend.describe_custom_models() == ""
        with pytest.raises(ModelNotFoundError, match=r"Available: base; or a local model folder$"):
            OptionsBackend.resolve_model("org/name")

    def test_undeclared_backends_check_names_themselves(self):
        """A backend without MODEL_SOURCES gets any name, as before: its load_model() checks it."""
        assert OldStyleBackend.resolve_model("anything at all") == "anything at all"

    def test_backend_without_custom_models(self):
        """An empty MODEL_SOURCES: the names in the list only."""

        class NamesOnly(OldStyleBackend):
            name = "names-only"
            MODEL_SOURCES: Dict[str, str] = {}

        with pytest.raises(ModelNotFoundError, match=r"^Unknown names-only model: org/name\. Available: base$"):
            NamesOnly.resolve_model("org/name")


class TestLoadersGetCustomModels:
    """A Hugging Face repo id and a local folder reach each loader unchanged."""

    @pytest.mark.parametrize("kind", ["repo id", "folder"])
    def test_faster_whisper(self, ct2, tmp_path, kind):
        model = REPO_ID if kind == "repo id" else str(tmp_path)

        backend = FasterWhisperBackend()
        backend.load_model(model)

        ct2.WhisperModel.assert_called_once_with(model, device="cpu", compute_type="int8")
        assert backend.model_name == model

    @pytest.mark.parametrize("backend_class, model_class", TRANSFORMERS_BACKENDS)
    @pytest.mark.parametrize("kind", ["repo id", "folder"])
    def test_transformers(self, torch_modules, tmp_path, backend_class, model_class, kind):
        model = REPO_ID if kind == "repo id" else str(tmp_path)

        backend = backend_class()
        backend.load_model(model)

        transformers = torch_modules.transformers
        transformers.AutoProcessor.from_pretrained.assert_called_once_with(model)
        getattr(transformers, model_class).from_pretrained.assert_called_once_with(
            model, dtype=torch_modules.torch.float32
        )
        assert (backend.model_name, backend._repo_id) == (model, model)

    def test_voxtral_asks_for_the_transcription_of_the_model_it_loaded(self, torch_modules, tmp_path):
        processor = torch_modules.transformers.AutoProcessor.from_pretrained.return_value
        processor.apply_transcription_request.side_effect = RuntimeError("stop here")

        backend = VoxtralBackend()
        backend.load_model(str(tmp_path))
        with pytest.raises(RuntimeError, match="stop here"):
            backend.transcribe(np.zeros(16000, np.float32), verbose=False)

        assert processor.apply_transcription_request.call_args[1]["model_id"] == str(tmp_path)

    def test_whisper_checkpoint_file(self, whisper_module, tmp_path):
        checkpoint = tmp_path / "fine-tuned.pt"
        checkpoint.touch()

        WhisperBackend().load_model(str(checkpoint))

        whisper_module.load_model.assert_called_once_with(str(checkpoint), device=None)

    def test_command_line(self, ct2, audio, capsys):
        cli.main([str(audio), "-m", "Systran/faster-whisper-small"])

        ct2.WhisperModel.assert_called_once_with("Systran/faster-whisper-small", device="cpu", compute_type="int8")
        assert "Loading faster-whisper model 'Systran/faster-whisper-small'..." in capsys.readouterr().err

    def test_unknown_model_fails_before_anything_loads(self, ct2, diarization, audio, capsys):
        """A typo fails before the diarization pipeline, and on the command line before "Loading"."""
        with pytest.raises(ModelNotFoundError, match="Unknown faster-whisper model: turbbo"):
            Transcriber(model="turbbo", diarize=True)
        assert diarization == []

        with pytest.raises(SystemExit, match="^Error: Unknown faster-whisper model: turbbo. Available: tiny,"):
            cli.main([str(audio), "-m", "turbbo"])
        assert "Loading" not in capsys.readouterr().err
        ct2.WhisperModel.assert_not_called()


class TestWhisperLoadErrors:
    @pytest.mark.parametrize("error", [RuntimeError("SHA256 checksum does not match"), OSError("network down")])
    def test_wrapped_in_model_load_error(self, whisper_module, error):
        whisper_module.load_model.side_effect = error

        backend = WhisperBackend()
        with pytest.raises(ModelLoadError, match=f"^Failed to load whisper model 'tiny': {error}$") as caught:
            backend.load_model("tiny")

        assert caught.value.__cause__ is error
        assert isinstance(caught.value, RuntimeError)
        assert backend.is_loaded is False


###############################################################################
# Devices
###############################################################################


class TestParseDevice:
    @pytest.mark.parametrize(
        "text, kind, index, shown",
        [
            (None, "auto", None, "auto"),
            ("auto", "auto", None, "auto"),
            (" AUTO ", "auto", None, "auto"),
            ("cpu", "cpu", None, "cpu"),
            ("cuda", "cuda", None, "cuda"),
            ("cuda:0", "cuda", 0, "cuda:0"),
            ("CUDA:12", "cuda", 12, "cuda:12"),
            ("mps", "mps", None, "mps"),
        ],
    )
    def test_devices(self, text, kind, index, shown):
        device = parse_device(text)

        assert device == Device(kind, index)
        assert str(device) == shown
        assert device.auto is (kind == "auto")

    def test_torch_device(self):
        """Anything whose text is a device works, such as a torch.device."""

        class TorchDevice:
            def __str__(self):
                return "cuda:1"

        assert parse_device(TorchDevice()) == Device("cuda", 1)

    @pytest.mark.parametrize("text", ["tpu", "gpu", "cuda:", "cuda:-1", "cuda:x", "cpu:1", "mps:0", "", "cuda 1"])
    def test_others_are_refused(self, text):
        with pytest.raises(UnsupportedOptionError, match=r"use auto, cpu, cuda, cuda:N \(N = 0, 1, \.\.\.\) or mps"):
            parse_device(text)

    def test_describe_and_check(self):
        assert describe_devices({"cpu", "cuda"}) == "auto, cpu, cuda or cuda:N"
        assert describe_devices({"mps", "cpu"}) == "auto, cpu or mps"
        assert check_device("cuda:3", {"cuda"}, "x") == Device("cuda", 3)
        assert check_device("auto", (), "x").auto
        with pytest.raises(UnsupportedOptionError, match="^the x backend can't run on cuda; the devices it takes: auto"):
            check_device("cuda:3", {"cpu"}, "x")

    def test_backends_on(self):
        assert backends_on("mps") == ["whisper", "voxtral", "parakeet", "canary"]
        assert backends_on("cuda") == ["whisper", "faster-whisper", "voxtral", "parakeet", "canary"]


class TestFasterWhisperDevices:
    def test_no_mps(self):
        """faster-whisper on mps fails before anything is imported or loaded."""
        with patch.dict(sys.modules, {"faster_whisper": None, "ctranslate2": None}):  # importing them would fail
            with pytest.raises(UnsupportedOptionError) as error:
                FasterWhisperBackend().load_model("tiny", device="mps")

        assert str(error.value) == "the faster-whisper backend can't run on mps; the devices it takes: auto, cpu, cuda or cuda:N"

    def test_cuda_n_becomes_device_index(self, ct2):
        ct2.ctranslate2.get_cuda_device_count.return_value = 2

        backend = FasterWhisperBackend()
        backend.load_model("tiny", device="cuda:1")

        ct2.WhisperModel.assert_called_once_with("tiny", device="cuda", compute_type="float16", device_index=1)
        assert backend.device == "cuda:1"
        assert ct2.checked == ["4.8.3"]  # the CUDA library check ran
        ct2.ctranslate2.get_supported_compute_types.assert_called_once_with("cuda", 1)

    def test_cuda_n_with_missing_libraries_raises(self, ct2, monkeypatch):
        ct2.ctranslate2.get_cuda_device_count.return_value = 2
        monkeypatch.setattr(faster_whisper_backend, "_cuda_library_problem", lambda version: "cuBLAS is missing.")

        with pytest.raises(ModelLoadError, match=r"^faster-whisper can't run on CUDA\. cuBLAS is missing\.$"):
            FasterWhisperBackend().load_model("tiny", device="cuda:1")
        ct2.WhisperModel.assert_not_called()

    def test_auto_falls_back_like_none(self, ct2, monkeypatch):
        """'auto' is the same as None: with a GPU but no CUDA libraries, a warning and the CPU."""
        ct2.ctranslate2.get_cuda_device_count.return_value = 1
        monkeypatch.setattr(faster_whisper_backend, "_cuda_library_problem", lambda version: "cuBLAS is missing.")

        with pytest.warns(SpeechToolkitWarning, match="faster-whisper runs on the CPU instead of the GPU"):
            backend = FasterWhisperBackend()
            backend.load_model("tiny", device="auto")

        ct2.WhisperModel.assert_called_once_with("tiny", device="cpu", compute_type="int8")
        assert backend.device == "cpu"

    def test_command_line_cuda_1(self, ct2, audio):
        ct2.ctranslate2.get_cuda_device_count.return_value = 2

        cli.main([str(audio), "-q", "--device", "cuda:1"])

        ct2.WhisperModel.assert_called_once_with("turbo", device="cuda", compute_type="float16", device_index=1)

    def test_command_line_mps_fails_before_loading(self, audio, capsys, diarization):
        with patch.dict(sys.modules, {"faster_whisper": None, "ctranslate2": None}):
            with pytest.raises(SystemExit, match="^Error: the faster-whisper backend can't run on mps; the devices"):
                cli.main([str(audio), "-b", "faster-whisper", "--device", "mps", "--diarize"])

        out, err = capsys.readouterr()
        assert out == "" and "Loading" not in err
        assert diarization == []  # nothing loaded, not even the diarization pipeline


class TestTransformersDevices:
    @pytest.mark.parametrize("backend_class, model_class", TRANSFORMERS_BACKENDS)
    @pytest.mark.parametrize("bf16, dtype", [(True, "bfloat16"), (False, "float16")])
    def test_cuda_n_uses_half_precision(self, torch_modules, backend_class, model_class, bf16, dtype):
        """Half precision on any CUDA GPU, asked of the GPU the model runs on."""
        torch = torch_modules.torch
        torch.cuda.is_bf16_supported.return_value = bf16

        backend = backend_class()
        backend.load_model(backend_class.default_model(), device="cuda:1")

        torch.cuda.device.assert_called_once_with(1)
        loaded = getattr(torch_modules.transformers, model_class).from_pretrained
        assert loaded.call_args[1]["dtype"] is getattr(torch, dtype)
        loaded.return_value.to.assert_called_once_with("cuda:1")
        assert backend.device == "cuda:1"

    def test_cuda_without_index_asks_the_current_gpu(self, torch_modules):
        torch = torch_modules.torch
        torch.cuda.is_bf16_supported.return_value = True

        ParakeetBackend().load_model("parakeet-tdt-0.6b-v3", device="cuda")

        torch.cuda.device.assert_not_called()
        assert torch_modules.transformers.ParakeetForTDT.from_pretrained.call_args[1]["dtype"] is torch.bfloat16

    def test_mps(self, torch_modules):
        backend = CanaryBackend()
        backend.load_model("canary-1b-v2", device="mps")

        loaded = torch_modules.transformers.CanaryForConditionalGeneration.from_pretrained
        assert loaded.call_args[1]["dtype"] is torch_modules.torch.float16
        loaded.return_value.to.assert_called_once_with("mps")
        assert backend.device == "mps"

    def test_auto_never_picks_mps(self, torch_modules):
        """Automatic choice is CUDA or the CPU, as before: an Apple GPU is used only when asked for."""
        torch_modules.torch.backends.mps.is_available.return_value = True

        backend = VoxtralBackend()
        backend.load_model("voxtral-mini", device="auto")

        assert backend.device == "cpu"

    def test_unknown_device(self):
        with pytest.raises(UnsupportedOptionError, match="unknown device 'tpu'"):
            ParakeetBackend().load_model("parakeet-tdt-0.6b-v3", device="tpu")

    def test_cuda_error_while_choosing_the_precision(self, torch_modules):
        """Asking a GPU that isn't there is a ModelLoadError, like any other failure to load."""
        torch_modules.torch.cuda.device.side_effect = RuntimeError("invalid device ordinal")

        backend = ParakeetBackend()
        with pytest.raises(ModelLoadError, match="invalid device ordinal"):
            backend.load_model("parakeet-tdt-0.6b-v3", device="cuda:7")
        assert backend.is_loaded is False


class TestWhisperDevices:
    @pytest.mark.parametrize("device, passed", [("cuda:1", "cuda:1"), ("mps", "mps"), ("auto", None), ("CPU", "cpu")])
    def test_devices(self, whisper_module, device, passed):
        backend = WhisperBackend()
        backend.load_model("tiny", device=device)

        whisper_module.load_model.assert_called_once_with("tiny", device=passed)
        assert backend.device == (passed or "cpu")


class TestDiarizationDevice:
    @pytest.fixture
    def pyannote(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        pyannote_audio = MagicMock()
        with patch.dict(sys.modules, {"torch": torch, "pyannote": MagicMock(), "pyannote.audio": pyannote_audio}):
            yield torch

    @pytest.mark.parametrize("device, used", [("cuda:1", "cuda:1"), ("mps", "mps"), ("auto", "cpu"), (None, "cpu")])
    def test_devices(self, pyannote, monkeypatch, device, used):
        monkeypatch.setenv("PYANNOTE_METRICS_ENABLED", "false")

        load_diarization_pipeline(device)

        pyannote.device.assert_called_once_with(used)

    def test_unknown_device_fails_before_importing(self):
        with patch.dict(sys.modules, {"torch": None, "pyannote.audio": None}):
            with pytest.raises(UnsupportedOptionError, match="unknown device 'gpu'"):
                load_diarization_pipeline("gpu")


class TestTranscriberDevices:
    def test_parsed_device_reaches_the_backend_and_diarization(self, ct2, diarization):
        ct2.ctranslate2.get_cuda_device_count.return_value = 2

        transcriber = Transcriber(model="tiny", device=" CUDA:1 ", diarize=True)

        assert diarization == ["cuda:1"]
        assert transcriber.backend.device == "cuda:1"
        assert ct2.WhisperModel.call_args[1]["device_index"] == 1

    def test_auto_is_none(self, ct2, diarization):
        Transcriber(model="tiny", device="auto", diarize=True)

        assert diarization == [None]

    def test_unsupported_device_fails_before_diarization_loads(self, ct2, diarization):
        with pytest.raises(UnsupportedOptionError, match="can't run on mps"):
            Transcriber(device="mps", diarize=True)
        assert diarization == []
        ct2.WhisperModel.assert_not_called()

    def test_backends_without_devices_get_them_unchecked(self, custom_backends):
        """A custom backend that lists no DEVICES gets the device as given, as before, but "auto" as None."""
        Transcriber("old-style", device="xpu")
        Transcriber("old-style", device="auto")
        Transcriber("old-style")

        assert OldStyleBackend.calls == [("base", "xpu"), ("base", None), ("base", None)]

    def test_declared_devices_are_checked(self, custom_backends):
        Transcriber("with-options", device="MPS")

        assert OptionsBackend.calls == [("base", "mps", None)]
        with pytest.raises(UnsupportedOptionError, match="the with-options backend can't run on cuda"):
            Transcriber("with-options", device="cuda:0")


###############################################################################
# Precision
###############################################################################


class TestComputeType:
    @pytest.mark.parametrize("compute_type", ["int8_float16", "float32", "default", "bfloat16"])
    def test_faster_whisper_takes_ctranslate2_types(self, ct2, compute_type):
        transcriber = Transcriber(model="tiny", compute_type=compute_type)

        ct2.WhisperModel.assert_called_once_with("tiny", device="cpu", compute_type=compute_type)
        assert transcriber.backend.options["compute_type"] == compute_type
        ct2.ctranslate2.get_supported_compute_types.assert_not_called()

    @pytest.mark.parametrize(
        "gpus, supported, chosen",
        [
            (1, {"int8", "int8_float32", "float32"}, "int8"),  # a GPU without fast float16 (#56)
            (1, {"float16", "int8_float16", "int8", "float32"}, "float16"),
            (1, {"float32"}, "float32"),
            (0, {"int8", "float32"}, "int8"),
            (0, {"float32"}, "float32"),
        ],
    )
    def test_auto_picks_a_supported_type(self, ct2, gpus, supported, chosen):
        ct2.ctranslate2.get_cuda_device_count.return_value = gpus
        ct2.ctranslate2.get_supported_compute_types.return_value = supported

        backend = FasterWhisperBackend()
        backend.load_model("tiny", options={"compute_type": "auto"})

        device = "cuda" if gpus else "cpu"
        ct2.ctranslate2.get_supported_compute_types.assert_called_once_with(device, 0)
        ct2.WhisperModel.assert_called_once_with("tiny", device=device, compute_type=chosen)
        assert backend.options == {"compute_type": chosen, "cpu_threads": 0}

    def test_auto_when_ctranslate2_cant_tell(self, ct2):
        ct2.ctranslate2.get_cuda_device_count.return_value = 1
        ct2.ctranslate2.get_supported_compute_types.side_effect = RuntimeError("CUDA driver version is insufficient")

        FasterWhisperBackend().load_model("tiny")

        ct2.WhisperModel.assert_called_once_with("tiny", device="cuda", compute_type="float16")

    @pytest.mark.parametrize("backend_class, model_class", TRANSFORMERS_BACKENDS)
    @pytest.mark.parametrize("compute_type", ["float16", "bfloat16", "float32"])
    def test_transformers(self, torch_modules, backend_class, model_class, compute_type):
        torch_modules.torch.cuda.is_available.return_value = True

        backend_class().load_model(backend_class.default_model(), options={"compute_type": compute_type})

        dtype = getattr(torch_modules.transformers, model_class).from_pretrained.call_args[1]["dtype"]
        assert dtype is getattr(torch_modules.torch, compute_type)

    def test_torch_dtype(self):
        torch = MagicMock()
        assert torch_dtype(torch, "cpu") is torch.float32
        assert torch_dtype(torch, "mps") is torch.float16
        assert torch_dtype(torch, "cpu", "bfloat16") is torch.bfloat16

    def test_unknown_type_lists_the_choices(self):
        with patch.dict(sys.modules, {"faster_whisper": None, "ctranslate2": None}):
            with pytest.raises(UnsupportedOptionError) as error:
                Transcriber(compute_type="float64")

        assert str(error.value) == (
            "option compute_type takes auto|default|int8|int8_float32|int8_float16|int8_bfloat16|int16|float16|"
            "bfloat16|float32, not 'float64'"
        )
        with pytest.raises(UnsupportedOptionError, match=r"takes auto\|float32\|float16\|bfloat16, not 'int8'"):
            Transcriber("parakeet", compute_type="int8")

    def test_backends_without_a_precision(self, whisper_module, custom_backends):
        """'auto' is what every backend does, so it is accepted everywhere; another type isn't."""
        Transcriber("whisper", "tiny", compute_type="auto")
        Transcriber("old-style", compute_type="auto", options={"compute-type": "auto"})

        whisper_module.load_model.assert_called_once_with("tiny", device=None)
        assert OldStyleBackend.calls == [("base", None)]
        with pytest.raises(UnsupportedOptionError, match="^the whisper backend has no option 'compute_type'; it takes no"):
            Transcriber("whisper", "tiny", compute_type="float16")

    def test_argument_wins_over_options(self, ct2):
        Transcriber(model="tiny", compute_type="int8", options={"compute_type": "float32"})

        assert ct2.WhisperModel.call_args[1]["compute_type"] == "int8"

    def test_command_line(self, ct2, audio):
        cli.main([str(audio), "-q", "-m", "tiny", "--compute-type", "int8_float32"])

        ct2.WhisperModel.assert_called_once_with("tiny", device="cpu", compute_type="int8_float32")

    def test_one_shot_transcribe(self, ct2, audio):
        transcribe(audio, model="tiny", compute_type="float32", options={"cpu_threads": 2})

        ct2.WhisperModel.assert_called_once_with("tiny", device="cpu", compute_type="float32", cpu_threads=2)


###############################################################################
# Declared options
###############################################################################


class TestBackendOption:
    def test_conversions(self):
        assert BackendOption(int, 0, "").parse(" 4 ") == 4
        assert BackendOption(int, 0, "").parse(4) == 4
        assert BackendOption(float, 0.5, "").parse("0.25") == 0.25
        assert BackendOption(float, 0.5, "").parse(2) == 2.0
        assert BackendOption(str, "a", "").parse(" b ") == " b "  # text is kept as given
        for text, value in [("true", True), ("Yes", True), ("on", True), ("1", True), ("false", False), ("0", False)]:
            assert BackendOption(bool, False, "").parse(text) is value

    @pytest.mark.parametrize(
        "option, value, message",
        [
            (BackendOption(int, 0, ""), "four", "option x takes INT, not 'four'"),
            (BackendOption(int, 0, ""), 2.5, "option x takes INT, not 2.5"),
            (BackendOption(int, 0, ""), True, "option x takes INT, not True"),
            (BackendOption(float, 0.0, ""), "fast", "option x takes NUMBER, not 'fast'"),
            (BackendOption(bool, False, ""), "maybe", "option x takes true or false, not 'maybe'"),
            (BackendOption(bool, False, ""), 1, "option x takes true|false, not 1"),
            (BackendOption(str, "", ""), 3, "option x takes TEXT, not 3"),
            (BackendOption(str, "a", "", choices=("a", "b")), "c", "option x takes a|b, not 'c'"),
            (BackendOption(int, 1, "", minimum=1), "0", "option x must be at least 1, not 0"),
            (BackendOption(float, 1.0, "", minimum=0.5), 0.25, "option x must be at least 0.5, not 0.25"),
        ],
    )
    def test_refused(self, option, value, message):
        with pytest.raises(UnsupportedOptionError) as error:
            option.parse(value, "x")
        assert str(error.value) == message

    def test_help(self):
        assert BackendOption(bool, False, "Decode greedily").describe() == "Decode greedily (default: false)"
        assert BackendOption(int, 3, "Beams").metavar == "INT"
        assert BackendOption(str, "a", "", choices=("a", "b")).metavar == "a|b"

    def test_types(self):
        with pytest.raises(TypeError, match="must be bool, int, float or str"):
            BackendOption(list, [], "")


class TestOptions:
    def test_cpu_threads_reaches_faster_whisper(self, ct2, audio):
        """-O cpu_threads=4 on the command line reaches WhisperModel."""
        cli.main([str(audio), "-q", "-O", "cpu_threads=4"])

        ct2.WhisperModel.assert_called_once_with("turbo", device="cpu", compute_type="int8", cpu_threads=4)

    def test_unknown_option_lists_the_valid_ones(self, ct2, audio, capsys):
        with pytest.raises(SystemExit) as error:
            cli.main([str(audio), "-O", "beam_size=5"])

        assert error.value.code == (
            "Error: the faster-whisper backend has no option 'beam_size'; its options: compute_type, cpu_threads"
        )
        assert "Loading" not in capsys.readouterr().err
        ct2.WhisperModel.assert_not_called()

    def test_bad_value(self, ct2, audio):
        with pytest.raises(SystemExit, match="^Error: option cpu_threads must be at least 0, not -1$"):
            cli.main([str(audio), "-O", "cpu_threads=-1"])

    @pytest.mark.parametrize("value", ["cpu_threads", "=4"])
    def test_not_key_value(self, value, capsys):
        with pytest.raises(SystemExit) as error:
            cli.parse_args(["talk.wav", "-O", value])

        assert error.value.code == 2
        assert f"expected KEY=VALUE, such as cpu_threads=4, got '{value}'" in capsys.readouterr().err

    def test_repeated_and_dashed(self, ct2, audio):
        """Several -O; the last one for a key wins, and dashes in names work as underscores."""
        args = cli.parse_args([str(audio), "-O", "cpu_threads=2", "-O", "cpu-threads=6", "-O", "compute_type=int8"])
        assert args.options == [("cpu_threads", "2"), ("cpu-threads", "6"), ("compute_type", "int8")]

        transcriber = cli.load_transcriber(args)

        assert transcriber.backend.options == {"compute_type": "int8", "cpu_threads": 6}

    def test_python_api_converts_text(self, ct2):
        transcriber = Transcriber(model="tiny", options={"cpu_threads": "3"})

        assert transcriber.backend.options == {"compute_type": "int8", "cpu_threads": 3}

    def test_max_new_tokens(self, torch_modules):
        transcriber = Transcriber("canary", options={"max_new_tokens": "800"})
        assert transcriber.backend.options == {"compute_type": "auto", "max_new_tokens": 800}

        with pytest.raises(UnsupportedOptionError, match="max_new_tokens must be at least 1"):
            Transcriber("voxtral", options={"max_new_tokens": 0})
        with pytest.raises(UnsupportedOptionError, match="its options: compute_type$"):
            Transcriber("parakeet", options={"max_new_tokens": 800})

    def test_defaults_before_loading(self):
        assert CanaryBackend().options == {"compute_type": "auto", "max_new_tokens": 500}
        assert WhisperBackend().options == {}
        backend = FasterWhisperBackend()
        backend.options["cpu_threads"] = 9  # a copy
        assert backend.options["cpu_threads"] == 0

    def test_custom_backend_with_options(self, custom_backends):
        """Only the options given reach load_model(), checked and converted; the backend adds the defaults."""
        transcriber = Transcriber("with-options", options={"greedy": "yes"})

        assert OptionsBackend.calls == [("base", None, {"greedy": True})]
        assert transcriber.backend.options == {"beam_width": 5, "greedy": True}

    def test_old_style_backend(self, custom_backends):
        """Without options, load_model() is called as before: a backend without the options argument works."""
        Transcriber("old-style", "base")

        assert OldStyleBackend.calls == [("base", None)]
        with pytest.raises(UnsupportedOptionError, match="^the old-style backend has no option 'beam'; it takes no "):
            Transcriber("old-style", options={"beam": 2})

    def test_parse_options(self):
        declared = {"beam_width": BackendOption(int, 5, "")}
        assert parse_options(declared, {"beam-width": "2"}, "x") == {"beam_width": 2}
        assert parse_options(declared, {}, "x") == {}

    def test_check_load_arguments(self):
        assert check_load_arguments(FasterWhisperBackend, "tiny", "Cuda:2", "int8", {"cpu_threads": "1"}) == (
            "cuda:2",
            {"compute_type": "int8", "cpu_threads": 1},
        )
        assert check_load_arguments(FasterWhisperBackend, "tiny") == (None, {})

    def test_backends_with_option(self):
        assert backends_with_option("compute_type") == ["faster-whisper", "voxtral", "parakeet", "canary"]
        assert backends_with_option("max_new_tokens") == ["voxtral", "canary"]


###############################################################################
# Command line: flags and listings
###############################################################################


class TestCommandLine:
    @pytest.mark.parametrize("text, device", [("CUDA:1", "cuda:1"), ("auto", "auto"), ("mps", "mps")])
    def test_device_flag(self, text, device):
        assert cli.parse_args(["talk.wav", "--device", text]).device == device

    def test_unknown_device(self, capsys):
        with pytest.raises(SystemExit) as error:
            cli.parse_args(["talk.wav", "--device", "tpu"])

        assert error.value.code == 2
        assert "argument --device: unknown device 'tpu'; use auto, cpu, cuda, cuda:N" in capsys.readouterr().err

    def test_help(self, capsys, monkeypatch):
        monkeypatch.setenv("COLUMNS", "500")
        with pytest.raises(SystemExit):
            cli.parse_args(["--help"])
        model_group = capsys.readouterr().out.split("\nmodel:\n", 1)[1].split("\n\n", 1)[0]

        assert "--device DEVICE " in model_group
        assert "or mps for an Apple GPU (whisper, voxtral, parakeet, canary)." in model_group
        assert "--compute-type TYPE " in model_group
        assert "such as int8 or float16 (faster-whisper, voxtral, parakeet, canary)." in model_group
        assert "-O, --option KEY=VALUE\n" in model_group

    def test_server_takes_them_too(self):
        args = server_parse_args(["--device", "cuda:1", "--compute-type", "int8", "-O", "cpu_threads=2"])

        assert (args.device, args.compute_type, args.options) == ("cuda:1", "int8", [("cpu_threads", "2")])

    def test_load_transcriber_without_the_new_flags(self, ct2):
        """A namespace from an older parser, without compute_type and options, still loads."""
        args = argparse.Namespace(backend="faster-whisper", model="tiny", device=None, quiet=True)
        args.diarize, args.hf_token = False, None

        cli.load_transcriber(args)

        ct2.WhisperModel.assert_called_once_with("tiny", device="cpu", compute_type="int8")


class TestListings:
    def test_list_backends(self, capsys):
        cli.show_backends()
        backends = listing(capsys.readouterr().out)

        faster = backends["faster-whisper"]
        assert faster[0].startswith("Models: tiny, ")  # the lines that were there stay first, in their order
        assert faster[1] == (
            "Custom models: a Hugging Face repo id (org/name) of a CTranslate2 model or a local CTranslate2 model folder"
        )
        assert faster[2].startswith("Capabilities: translate, ")
        assert faster[3] == "Devices: auto, cpu, cuda or cuda:N"
        assert faster[4].startswith(
            "-O compute_type=auto|default|int8|int8_float32|int8_float16|int8_bfloat16|int16|float16|bfloat16|float32: "
            "Precision; auto: float16 on a GPU and int8 on the CPU"
        )
        assert faster[4].endswith("(default: auto)")
        assert faster[5].startswith("-O cpu_threads=INT: Threads on the CPU;")
        assert faster[5].endswith("(default: 0)")
        assert faster[6].startswith("Installed: ")

        assert backends["whisper"][1:4] == [
            "Custom models: a local checkpoint file (.pt)",
            "Capabilities: translate, language detection, prompt, word timestamps",
            "Devices: auto, cpu, cuda, cuda:N or mps",
        ]
        assert not any(line.startswith("-O") for line in backends["whisper"])
        canary = [line.split(":", 1)[0] for line in backends["canary"]]
        assert canary == [
            "Models",
            "Custom models",
            "Capabilities",
            "Devices",
            "-O compute_type=auto|float32|float16|bfloat16",
            "-O max_new_tokens=INT",
            "Installed",
        ]

    def test_custom_backend_lines_unchanged(self, custom_backends, capsys):
        cli.main(["--list-backends"])
        backends = listing(capsys.readouterr().out)

        assert [line.split(":")[0] for line in backends["old-style"]] == ["Models", "Capabilities", "Installed"]
        assert backends["with-options"][4:6] == [
            "-O beam_width=INT: Beams (default: 5)",
            "-O greedy=true|false: Decode greedily (default: false)",
        ]

    @pytest.mark.parametrize(
        "backend, custom",
        [
            ("faster-whisper", "a Hugging Face repo id (org/name) of a CTranslate2 model or a local CTranslate2 model folder"),
            ("whisper", "a local checkpoint file (.pt)"),
            ("voxtral", "a Hugging Face repo id (org/name) or a local model folder"),
        ],
    )
    def test_list_models(self, capsys, backend, custom):
        cli.main(["--list-models", "-b", backend])

        assert capsys.readouterr().out.splitlines()[-2:] == ["", f"Custom models: -m also takes {custom}."]

    def test_list_models_of_a_backend_without_custom_models(self, custom_backends, capsys):
        cli.main(["--list-models", "-b", "old-style"])

        assert capsys.readouterr().out.splitlines()[-1] == "  base (default)"
