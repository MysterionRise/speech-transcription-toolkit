"""faster-whisper on a GPU without NVIDIA's CUDA libraries: CPU fallback with a warning, or an error for device="cuda".

faster_whisper and ctranslate2 are mocks in sys.modules, and ctypes.CDLL is replaced, so these tests run the same on
Linux, macOS and Windows, with or without a GPU. The platform is set per test.
"""

import ctypes
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from speech_toolkit import ModelLoadError, SpeechToolkitWarning, cli
from speech_toolkit.backends import faster_whisper_backend
from speech_toolkit.backends.faster_whisper_backend import FasterWhisperBackend

CUBLAS = {"win32": "cublas64_12.dll", "linux": "libcublas.so.12"}
CUDNN_9 = {"win32": ["cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"], "linux": ["libcudnn_ops.so.9", "libcudnn_cnn.so.9"]}
CUDNN_8 = {
    "win32": ["cudnn_ops_infer64_8.dll", "cudnn_cnn_infer64_8.dll"],
    "linux": ["libcudnn_ops_infer.so.8", "libcudnn_cnn_infer.so.8"],
}


class FakeLoader:
    """Stands in for ctypes.CDLL: libraries named in *missing* (by file name) fail to load, as a missing one does.

    Full paths listed in *present* load even so, like a library found in a folder ctypes doesn't search by default.
    """

    def __init__(self, *missing: str, present=()):
        self.missing = set(missing)
        self.present = {str(path) for path in present}
        self.calls = []

    def __call__(self, path):
        self.calls.append(path)
        if path not in self.present and os.path.basename(path) in self.missing:
            raise OSError(f"Could not find module '{path}' (or one of its dependencies)")
        return MagicMock()

    @property
    def names(self):
        return [os.path.basename(path) for path in self.calls]


@pytest.fixture
def ct2():
    """Mocked faster_whisper and ctranslate2 4.8.3 on a machine with one NVIDIA GPU."""
    faster_whisper = MagicMock()
    ctranslate2 = MagicMock()
    ctranslate2.__version__ = "4.8.3"
    ctranslate2.get_cuda_device_count.return_value = 1
    with patch.dict(sys.modules, {"faster_whisper": faster_whisper, "ctranslate2": ctranslate2}):
        yield ctranslate2, faster_whisper.WhisperModel


@pytest.fixture
def no_cuda_folders(monkeypatch, tmp_path):
    """No CUDA_PATH, and a PATH with one empty folder: Windows finds nothing beyond ctypes' default search."""
    monkeypatch.delenv("CUDA_PATH", raising=False)
    empty = tmp_path / "empty-path-folder"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    return empty


@pytest.fixture(params=["win32", "linux"])
def platform(request, monkeypatch, no_cuda_folders):
    """Runs the test as on Windows and as on Linux."""
    monkeypatch.setattr(sys, "platform", request.param)
    return request.param


@pytest.fixture
def windows(monkeypatch, no_cuda_folders):
    """Runs the test as on Windows."""
    monkeypatch.setattr(sys, "platform", "win32")


def load(loader, device=None, model="tiny"):
    backend = FasterWhisperBackend()
    with patch.object(ctypes, "CDLL", loader):
        backend.load_model(model, device=device)
    return backend


class TestAutomaticDevice:
    """device=None: the GPU when CTranslate2 sees one and its libraries load, else the CPU."""

    def test_libraries_present_uses_gpu(self, ct2, platform):
        """With the libraries in place the model runs on CUDA in float16, without a warning."""
        _, whisper_model = ct2
        loader = FakeLoader()

        backend = load(loader)

        whisper_model.assert_called_once_with("tiny", device="cuda", compute_type="float16")
        assert backend.device == "cuda"
        assert loader.names == [CUBLAS[platform]]  # CTranslate2 4.8.3 needs only cuBLAS

    def test_missing_library_falls_back_to_cpu_with_one_warning(self, ct2, platform):
        """A missing cuBLAS gives one warning that names it and the fix, then an int8 model on the CPU."""
        _, whisper_model = ct2

        with pytest.warns(SpeechToolkitWarning) as record:
            backend = load(FakeLoader(CUBLAS[platform]))

        assert len(record) == 1
        message = str(record[0].message)
        assert message.startswith("faster-whisper runs on the CPU instead of the GPU.")
        assert f"CUDA libraries not found: {CUBLAS[platform]}." in message
        assert "pip install nvidia-cublas-cu12," in message
        assert "docs/install.md#gpu" in message
        assert message.endswith("Pass --device cpu to skip this check.")
        assert record[0].filename == __file__  # points at the caller of load_model()
        whisper_model.assert_called_once_with("tiny", device="cpu", compute_type="int8")
        assert backend.device == "cpu"

    def test_no_gpu_skips_the_check(self, ct2, platform):
        """Without a GPU nothing is loaded or checked: an int8 model on the CPU, as before."""
        ctranslate2, whisper_model = ct2
        ctranslate2.get_cuda_device_count.return_value = 0
        loader = FakeLoader(CUBLAS[platform])

        load(loader)

        assert loader.calls == []
        whisper_model.assert_called_once_with("tiny", device="cpu", compute_type="int8")

    def test_cpu_requested_skips_the_check(self, ct2, platform):
        """device='cpu' neither checks the libraries nor asks CTranslate2 about the GPU."""
        ctranslate2, whisper_model = ct2
        loader = FakeLoader(CUBLAS[platform])

        load(loader, device="cpu")

        assert loader.calls == []
        ctranslate2.get_cuda_device_count.assert_not_called()
        whisper_model.assert_called_once_with("tiny", device="cpu", compute_type="int8")

    def test_other_load_errors_still_raise(self, ct2, platform):
        """With the libraries present, a failure such as running out of memory is still a ModelLoadError."""
        _, whisper_model = ct2
        whisper_model.side_effect = RuntimeError("CUDA failed with error out of memory")

        backend = FasterWhisperBackend()
        with patch.object(ctypes, "CDLL", FakeLoader()), pytest.raises(ModelLoadError, match="out of memory"):
            backend.load_model("large-v3")
        assert backend.is_loaded is False
        whisper_model.assert_called_once_with("large-v3", device="cuda", compute_type="float16")


class TestCudaRequested:
    """device='cuda': an error with the same hint instead of a fallback."""

    def test_missing_library_raises_with_hint(self, ct2, platform):
        """No model is loaded; the error names the library and how to install it."""
        _, whisper_model = ct2

        with pytest.raises(ModelLoadError) as error:
            load(FakeLoader(CUBLAS[platform]), device="cuda")

        message = str(error.value)
        assert message.startswith(f"faster-whisper can't run on CUDA. CUDA libraries not found: {CUBLAS[platform]}.")
        assert "pip install nvidia-cublas-cu12," in message
        assert isinstance(error.value, RuntimeError)  # as ModelLoadError always was
        whisper_model.assert_not_called()

    def test_libraries_present_uses_gpu(self, ct2, platform):
        _, whisper_model = ct2

        backend = load(FakeLoader(), device="cuda")

        whisper_model.assert_called_once_with("tiny", device="cuda", compute_type="float16")
        assert backend.device == "cuda"

    def test_no_gpu_skips_the_check(self, ct2, platform):
        """Without a GPU, device='cuda' is left to CTranslate2 as before, which reports the problem itself."""
        ctranslate2, whisper_model = ct2
        ctranslate2.get_cuda_device_count.return_value = 0
        loader = FakeLoader(CUBLAS[platform])

        load(loader, device="cuda")

        assert loader.calls == []
        whisper_model.assert_called_once_with("tiny", device="cuda", compute_type="float16")


class TestCommandLine:
    """`transcribe` with the default backend on a GPU machine without cuBLAS."""

    @pytest.fixture
    def audio(self, ct2, tmp_path):
        """An audio file that the mocked model 'transcribes' as one sentence."""
        _, whisper_model = ct2
        segment = MagicMock(start=0.0, end=1.0, text=" Hello there.", tokens=[1], words=None)
        info = MagicMock(language="en", duration=1.0, language_probability=0.99)
        whisper_model.return_value.transcribe.return_value = (iter([segment]), info)
        path = tmp_path / "talk.wav"
        path.touch()
        return path

    @pytest.mark.filterwarnings("default:faster-whisper runs on the CPU:UserWarning")
    def test_falls_back_with_one_warning_line(self, ct2, platform, audio, capsys):
        """One 'Warning:' line on stderr, the transcript alone on stdout, and a CPU model."""
        _, whisper_model = ct2

        with patch.object(ctypes, "CDLL", FakeLoader(CUBLAS[platform])):
            cli.main([str(audio), "-q"])

        out, err = capsys.readouterr()
        assert out == "Hello there.\n"
        assert err.count("Warning: faster-whisper runs on the CPU instead of the GPU.") == 1
        assert f"CUDA libraries not found: {CUBLAS[platform]}." in err
        whisper_model.assert_called_once_with("turbo", device="cpu", compute_type="int8")

    def test_device_cuda_exits_with_the_hint(self, ct2, platform, audio, capsys):
        """--device cuda stops before transcribing, with the install hint."""
        with patch.object(ctypes, "CDLL", FakeLoader(CUBLAS[platform])):
            with pytest.raises(SystemExit, match=r"^Error: faster-whisper can't run on CUDA\. CUDA libraries not"):
                cli.main([str(audio), "-q", "--device", "cuda"])

        assert capsys.readouterr().out == ""


class TestLibrariesByVersion:
    """Which libraries are checked follows what each CTranslate2 release loads."""

    @pytest.mark.parametrize(
        "version, cudnn",
        [
            ("4.0.0", CUDNN_8),
            ("4.4.0", CUDNN_8),
            ("4.5.0", CUDNN_9),
            ("4.6.2", CUDNN_9),
            ("4.6.3", None),  # convolutions in plain CUDA from here on: no cuDNN
            ("4.8.3", None),
            ("4.10.0.dev1", None),
        ],
    )
    def test_checked_libraries(self, platform, version, cudnn):
        loader = FakeLoader()
        with patch.object(ctypes, "CDLL", loader):
            assert faster_whisper_backend._cuda_library_problem(version) is None
        assert loader.names == [CUBLAS[platform]] + (cudnn[platform] if cudnn else [])

    @pytest.mark.parametrize("version", ["3.24.0", "5.0.0", "", "unknown"])
    def test_unknown_requirements_are_not_checked(self, platform, version):
        """Versions whose CUDA needs aren't known (CUDA 11 builds, future releases) use the GPU as before."""
        loader = FakeLoader(CUBLAS[platform])
        with patch.object(ctypes, "CDLL", loader):
            assert faster_whisper_backend._cuda_library_problem(version) is None
        assert loader.calls == []

    def test_missing_version_attribute_is_not_checked(self, ct2, platform):
        """A ctranslate2 without __version__ loads on the GPU without a check."""
        ctranslate2, whisper_model = ct2
        del ctranslate2.__version__
        loader = FakeLoader(CUBLAS[platform])

        load(loader)

        assert loader.calls == []
        whisper_model.assert_called_once_with("tiny", device="cuda", compute_type="float16")

    def test_missing_cudnn_9(self, ct2, platform):
        """CTranslate2 4.6.2 without cuDNN: both missing files, the cuDNN 9 package and the upgrade are named."""
        ctranslate2, whisper_model = ct2
        ctranslate2.__version__ = "4.6.2"

        with pytest.warns(SpeechToolkitWarning) as record:
            load(FakeLoader(*CUDNN_9[platform]))

        message = str(record[0].message)
        assert f"CUDA libraries not found: {', '.join(CUDNN_9[platform])}." in message
        assert 'pip install "nvidia-cudnn-cu12==9.*",' in message
        assert "nvidia-cublas-cu12" not in message  # cuBLAS loaded
        assert "(or upgrade CTranslate2, which needs no cuDNN since 4.6.3: pip install -U ctranslate2)" in message
        whisper_model.assert_called_once_with("tiny", device="cpu", compute_type="int8")

    def test_missing_cublas_and_cudnn_8(self, platform):
        """Everything missing for CTranslate2 4.4.0: both packages, in one pip command."""
        with patch.object(ctypes, "CDLL", FakeLoader(CUBLAS[platform], *CUDNN_8[platform])):
            message = faster_whisper_backend._cuda_library_problem("4.4.0")

        assert message is not None
        assert f"CUDA libraries not found: {', '.join([CUBLAS[platform]] + CUDNN_8[platform])}." in message
        assert 'pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==8.*",' in message


class TestWhereLibrariesAreFound:
    """The check looks where CTranslate2 does, per platform."""

    def test_install_hint_windows(self, windows):
        """Windows: the pip packages' bin folders go on PATH."""
        with patch.object(ctypes, "CDLL", FakeLoader("cublas64_12.dll", "cudnn_ops64_9.dll")):
            message = faster_whisper_backend._cuda_library_problem("4.6.0")

        assert message is not None
        assert (
            r"then add site-packages\nvidia\cublas\bin and site-packages\nvidia\cudnn\bin to PATH (or upgrade"
            in message
        )
        assert "LD_LIBRARY_PATH" not in message

    def test_install_hint_linux(self, monkeypatch):
        """Linux: the pip packages' lib folders go on LD_LIBRARY_PATH, which is read when the process starts."""
        monkeypatch.setattr(sys, "platform", "linux")
        with patch.object(ctypes, "CDLL", FakeLoader("libcublas.so.12")):
            message = faster_whisper_backend._cuda_library_problem("4.8.3")

        assert message is not None
        assert "then add site-packages/nvidia/cublas/lib to LD_LIBRARY_PATH before starting Python; see " in message
        assert "upgrade CTranslate2" not in message

    def test_linux_uses_dlopen_by_name(self, monkeypatch):
        """Linux: one dlopen by name, which searches LD_LIBRARY_PATH and the system folders like CTranslate2's."""
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("CUDA_PATH", "/usr/local/cuda")  # CTranslate2 reads it on Windows only
        loader = FakeLoader("libcublas.so.12")
        with patch.object(ctypes, "CDLL", loader):
            faster_whisper_backend._cuda_library_problem("4.8.3")

        assert loader.calls == ["libcublas.so.12"]

    def test_windows_found_by_ctypes(self, windows):
        """Windows: a library ctypes finds by name (already loaded, in System32, in CTranslate2's folder) is used."""
        loader = FakeLoader()
        with patch.object(ctypes, "CDLL", loader):
            assert faster_whisper_backend._cuda_library_problem("4.8.3") is None

        assert loader.calls == ["cublas64_12.dll"]

    def test_windows_found_on_path(self, windows, monkeypatch, tmp_path):
        """Windows: a library in a PATH folder, where CTranslate2's LoadLibraryA finds it, is loaded by full path."""
        empty, cublas = tmp_path / "empty", tmp_path / "nvidia" / "cublas" / "bin"
        empty.mkdir()
        cublas.mkdir(parents=True)
        (cublas / "cublas64_12.dll").touch()
        monkeypatch.setenv("PATH", os.pathsep.join(["", str(empty), f'"{cublas}"']))  # quoted, as Windows allows
        loader = FakeLoader("cublas64_12.dll", present=[cublas / "cublas64_12.dll"])

        with patch.object(ctypes, "CDLL", loader):
            assert faster_whisper_backend._cuda_library_problem("4.8.3") is None

        assert loader.calls == ["cublas64_12.dll", str(cublas / "cublas64_12.dll")]  # no attempt in "empty"

    def test_windows_found_in_cuda_path(self, windows, monkeypatch, tmp_path):
        """Windows: %CUDA_PATH%\\bin, which CTranslate2 adds to its search, comes before PATH."""
        cuda_bin, on_path = tmp_path / "CUDA" / "v12.4" / "bin", tmp_path / "other"
        for folder in (cuda_bin, on_path):
            folder.mkdir(parents=True)
            (folder / "cublas64_12.dll").touch()
        monkeypatch.setenv("CUDA_PATH", str(tmp_path / "CUDA" / "v12.4"))
        monkeypatch.setenv("PATH", str(on_path))
        loader = FakeLoader("cublas64_12.dll", present=[cuda_bin / "cublas64_12.dll", on_path / "cublas64_12.dll"])

        with patch.object(ctypes, "CDLL", loader):
            assert faster_whisper_backend._cuda_library_problem("4.8.3") is None

        assert loader.calls == ["cublas64_12.dll", str(cuda_bin / "cublas64_12.dll")]

    def test_windows_current_folder_is_not_searched(self, windows, monkeypatch, tmp_path):
        """Windows: a DLL in the current folder isn't loaded, so one planted next to the audio can't run."""
        (tmp_path / "cublas64_12.dll").touch()
        monkeypatch.chdir(tmp_path)
        loader = FakeLoader("cublas64_12.dll")

        with patch.object(ctypes, "CDLL", loader):
            assert faster_whisper_backend._cuda_library_problem("4.8.3") is not None

        assert loader.calls == ["cublas64_12.dll"]  # ctypes' default search, which leaves out the current folder

    def test_windows_file_that_fails_to_load(self, windows, monkeypatch, tmp_path):
        """Windows: a cublas64_12.dll on PATH that doesn't load (a dependency is missing, say) counts as missing."""
        (tmp_path / "cublas64_12.dll").touch()
        monkeypatch.setenv("PATH", str(tmp_path))
        loader = FakeLoader("cublas64_12.dll")

        with patch.object(ctypes, "CDLL", loader):
            assert faster_whisper_backend._cuda_library_problem("4.8.3") is not None

        assert loader.calls == ["cublas64_12.dll", str(tmp_path / "cublas64_12.dll")]

    def test_other_systems_are_not_checked(self, monkeypatch):
        """macOS has no CUDA build of CTranslate2: nothing to check."""
        monkeypatch.setattr(sys, "platform", "darwin")
        loader = FakeLoader("libcublas.so.12")
        with patch.object(ctypes, "CDLL", loader):
            assert faster_whisper_backend._cuda_library_problem("4.8.3") is None
        assert loader.calls == []

    def test_real_ctypes_reports_a_missing_library(self):
        """With the real ctypes, on any OS, a library that doesn't exist is reported as such, not a crash."""
        assert faster_whisper_backend._loads("speech-toolkit-no-such-library-56.dll") is False
        assert (
            faster_whisper_backend._loads(os.path.join(os.getcwd(), "speech-toolkit-no-such-library-56.dll")) is False
        )
