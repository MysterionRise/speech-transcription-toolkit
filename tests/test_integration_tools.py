"""The integration suite's own machinery, without models: device settings and checks, the GPU probe, memory skips,
the instrumentation, the report and the long clip. Plain pytest runs these, on every OS CI tests."""

from __future__ import annotations

import json
import subprocess
import sys
import types
import wave
from unittest.mock import patch

import numpy as np
import pytest

from tests.integration import devices, instrument
from tests.integration.devices import GB, MEMORY_GB, Card, Gpu
from tests.integration.helpers import (
    FULL_MODELS,
    MAX_WER,
    PACKAGES,
    EventLog,
    Sample,
    Turn,
    assert_chunks_covered,
    assert_no_cut_off_warning,
    chunk_seconds,
    command_env,
    join_audio,
    write_wav,
)
from tests.integration.report import MIB, Report, Run

CPU_TORCH = {"version": "2.9.0+cpu", "cuda": None, "available": False}
CUDA_TORCH = {"version": "2.9.0+cu128", "cuda": "12.8", "available": True, "name": "RTX Test", "memory": 10 * GB}


@pytest.fixture
def device(monkeypatch):
    """Set INTEGRATION_DEVICE for the test: ``device("cuda")``."""
    return lambda value: monkeypatch.setattr(devices, "DEVICE", value)


@pytest.fixture
def found_gpu(monkeypatch):
    """Make the GPU probe find *gpu* instead of running: ``found_gpu(Gpu(...))``."""
    return lambda gpu: monkeypatch.setattr(devices, "_probed", gpu)


class TestSettings:
    def test_auto_passes_no_device(self, device):
        device("auto")
        assert devices.device_args() == ()
        assert devices.api_device() is None

    def test_a_device_is_passed_on(self, device):
        device("cuda")
        assert devices.device_args() == ("--device", "cuda")
        assert devices.api_device() == "cuda"

    @pytest.mark.parametrize("name, value", [("DEVICE", "gpu"), ("TIER", "huge")])
    def test_unknown_values_fail(self, monkeypatch, name, value):
        monkeypatch.setattr(devices, name, value)
        with pytest.raises(pytest.fail.Exception, match=f"INTEGRATION_{name} must be .*{value}"):
            devices.check_settings()


class TestCheckDevices:
    def test_auto_accepts_any_device(self, device):
        device("auto")
        devices.check_devices("whisper", {"model": "cpu"})

    def test_the_asked_device_passes(self, device):
        device("cuda")
        devices.check_devices("faster-whisper", {"model": "cuda", "diarization": "cuda"})

    def test_a_model_on_another_device_fails(self, device):
        device("cuda")
        with pytest.raises(
            pytest.fail.Exception, match="INTEGRATION_DEVICE=cuda, but the pyannote pipeline ran on cpu"
        ):
            devices.check_devices("faster-whisper", {"model": "cuda", "diarization": "cpu"})


class TestProbe:
    def test_parse_nvidia_smi(self):
        output = "NVIDIA GeForce RTX 3080, 10240, 566.14\nOdd, Name, 8192, 550.1\nNo devices were found\n"

        assert devices.parse_nvidia_smi(output) == [
            Card("NVIDIA GeForce RTX 3080", 10 * GB, "566.14"),
            Card("Odd, Name", 8 * GB, "550.1"),
        ]

    def test_parse_probe_finds_its_line_among_others(self):
        found = {"torch": CPU_TORCH, "ctranslate2": None}
        output = f"a library printed this\n{instrument.PROBE_PREFIX}{json.dumps(found)}\n"

        assert devices.parse_probe(output) == found
        assert devices.parse_probe("nothing\n") is None

    def test_the_probe_runs_without_the_libraries(self):
        """The real probe, in a subprocess: on CI's unit jobs torch and CTranslate2 aren't installed."""
        with patch.object(devices, "query_cards", return_value=None):
            gpu = devices.probe_gpu()

        assert gpu.problem == ""
        assert gpu.found is (gpu.usable("torch") or gpu.usable("ctranslate2"))

    def test_the_probe_reports_libraries(self):
        fake_torch = types.SimpleNamespace(
            __version__="2.9.0+cpu", version=types.SimpleNamespace(cuda=None), cuda=types.SimpleNamespace()
        )
        fake_torch.cuda.is_available = lambda: False
        fake_ct2 = types.SimpleNamespace(__version__="4.6.0", get_cuda_device_count=lambda: 1)
        with patch.dict(sys.modules, {"torch": fake_torch, "ctranslate2": fake_ct2}):
            assert instrument.probe_torch() == CPU_TORCH
            assert instrument.probe_ctranslate2() == {"version": "4.6.0", "cuda_devices": 1}

    def test_the_probe_tells_a_missing_library_from_a_broken_one(self):
        with patch.dict(sys.modules, {"torch": None, "ctranslate2": None}):  # None: as if not installed
            assert instrument.probe_torch() is None and instrument.probe_ctranslate2() is None
        broken = ModuleNotFoundError("No module named 'typing_extensions'", name="typing_extensions")
        with patch("builtins.__import__", side_effect=broken):
            assert instrument.probe_torch() == {"error": "ModuleNotFoundError: No module named 'typing_extensions'"}
        with patch("builtins.__import__", side_effect=OSError("[WinError 126] fbgemm.dll")):
            assert instrument.probe_ctranslate2() == {"error": "OSError: [WinError 126] fbgemm.dll"}


class TestGpu:
    def test_nothing_found(self):
        gpu = Gpu(None, None, None)

        assert not gpu.found
        assert gpu.summary() == "nvidia-smi: not found; torch: not installed; CTranslate2: not installed"

    def test_a_cpu_only_torch(self):
        gpu = Gpu(CPU_TORCH, {"version": "4.6.0", "cuda_devices": 1}, (Card("RTX Test", 8 * GB, "566.14"),))

        assert gpu.found and gpu.usable("ctranslate2") and not gpu.usable("torch")
        assert "CPU-only build" in gpu.why_not("torch")
        assert gpu.name == "RTX Test" and gpu.memory == 8 * GB
        assert gpu.summary() == (
            "RTX Test (8.0 GB, driver 566.14); torch 2.9.0+cpu (CPU-only build, no GPU); CTranslate2 4.6.0 (1 CUDA device)"
        )

    def test_a_cuda_torch_without_a_gpu(self):
        gpu = Gpu({**CUDA_TORCH, "available": False}, {"version": "4.6.0", "cuda_devices": 0}, ())

        assert not gpu.found
        assert "finds no usable GPU" in gpu.why_not("torch")
        assert "CTranslate2 4.6.0 finds no CUDA GPU" in gpu.why_not("ctranslate2")
        assert gpu.summary().startswith("nvidia-smi: no GPU listed")

    def test_torch_names_the_gpu(self):
        gpu = Gpu(CUDA_TORCH, None, None)

        assert gpu.name == "RTX Test" and gpu.memory == 10 * GB
        assert "torch 2.9.0+cu128 (CUDA 12.8, uses the GPU)" in gpu.summary()


class TestRequire:
    def test_cuda_without_a_gpu_fails_clearly(self, device, found_gpu):
        device("cuda")
        found_gpu(Gpu(None, None, None))

        with pytest.raises(pytest.fail.Exception, match="INTEGRATION_DEVICE=cuda, but there is no usable CUDA GPU"):
            devices.require_gpu()

    def test_other_devices_need_no_gpu(self, device, found_gpu):
        device("cpu")
        found_gpu(Gpu(None, None, None))
        devices.require_gpu()
        devices.require_device("whisper")

    def test_each_backend_needs_its_library_to_see_the_gpu(self, device, found_gpu):
        device("cuda")
        found_gpu(Gpu(CPU_TORCH, {"version": "4.6.0", "cuda_devices": 1}, ()))

        devices.require_gpu()
        devices.require_device("faster-whisper", "server")
        with pytest.raises(pytest.fail.Exception, match="whisper can't use the GPU: torch 2.9.0[+]cpu is a CPU-only"):
            devices.require_device("whisper")


class TestMemory:
    def test_every_full_tier_model_has_its_needs(self):
        assert set(FULL_MODELS) <= set(MEMORY_GB)
        assert {backend for backend, _ in FULL_MODELS} <= set(PACKAGES) & set(MAX_WER)

    def test_a_model_too_big_for_the_gpu_is_skipped(self, device, found_gpu):
        device("cuda")
        found_gpu(Gpu(CUDA_TORCH, None, None))  # 10 GB, 8.5 usable

        with pytest.raises(
            pytest.skip.Exception, match=r"voxtral voxtral-mini needs about 11 GB of GPU memory; RTX Test"
        ):
            devices.require_memory(("voxtral", "voxtral-mini"))
        with pytest.raises(pytest.skip.Exception, match="whisper large-v3 needs about 10 GB"):
            devices.require_memory(("whisper", "large-v3"))
        devices.require_memory(("faster-whisper", "large-v3"))
        devices.require_memory(("whisper", "turbo"))

    def test_models_sharing_a_process_add_up(self, device, found_gpu):
        device("cuda")
        found_gpu(Gpu({**CUDA_TORCH, "memory": 4 * GB}, None, None))  # 3.4 GB usable: turbo alone (3) fits

        devices.require_memory(("faster-whisper", "turbo"))
        with pytest.raises(pytest.skip.Exception, match="faster-whisper turbo with diarize pyannote needs about 4 GB"):
            devices.require_memory(("faster-whisper", "turbo"), ("diarize", "pyannote"))

    def test_on_the_cpu_ram_counts(self, device, monkeypatch):
        device("cpu")
        monkeypatch.setattr(devices, "total_ram", lambda: 16 * GB)

        with pytest.raises(pytest.skip.Exception, match="needs about 20 GB of RAM; this machine has 16.0 GB"):
            devices.require_memory(("voxtral", "voxtral-mini"))
        devices.require_memory(("whisper", "large-v3"))

    def test_auto_counts_the_gpu_only_for_backends_that_can_use_it(self, device, found_gpu):
        device("auto")
        found_gpu(Gpu(CPU_TORCH, {"version": "4.6.0", "cuda_devices": 1}, ()))

        assert devices.memory_device("faster-whisper") == "cuda"
        assert devices.memory_device("whisper") == "cpu"

    def test_unknown_memory_lets_the_model_try(self, device, found_gpu):
        device("cuda")
        found_gpu(Gpu(None, None, None))
        devices.require_memory(("voxtral", "voxtral-mini"))

    def test_total_ram(self):
        assert devices.total_ram() > GB  # the real call, on each OS CI runs


class FakeParameter:
    def __init__(self, device, dtype):
        self.device = types.SimpleNamespace(type=device)
        self.dtype = dtype


class FakeTorchModel:
    def __init__(self, device, dtype="torch.bfloat16"):
        self.parameter = FakeParameter(device, dtype)

    def parameters(self):
        return iter([self.parameter])


class TestModelDevice:
    def test_ctranslate2(self):
        model = types.SimpleNamespace(model=types.SimpleNamespace(device="cuda", compute_type="float16"))
        backend = types.SimpleNamespace(_model=model, name="faster-whisper")

        assert instrument.model_device(backend) == ("cuda", "float16")

    def test_transformers(self):
        backend = types.SimpleNamespace(_model=FakeTorchModel("cuda"), name="parakeet")

        assert instrument.model_device(backend) == ("cuda", "bfloat16")

    def test_whisper_decodes_in_float16_on_a_gpu(self):
        on_gpu = types.SimpleNamespace(_model=FakeTorchModel("cuda", "torch.float32"), name="whisper")
        on_cpu = types.SimpleNamespace(_model=FakeTorchModel("cpu", "torch.float32"), name="whisper")

        assert instrument.model_device(on_gpu) == ("cuda", "float16")
        assert instrument.model_device(on_cpu) == ("cpu", "float32")

    def test_without_a_model_object_the_backend_says(self):
        assert instrument.model_device(types.SimpleNamespace(_model=object(), device="cuda:0")) == ("cuda", "")
        assert instrument.model_device(types.SimpleNamespace(_model=None, device=None)) == ("unknown", "")

    def test_a_transcriber_with_diarization(self):
        pipeline = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"))
        backend = types.SimpleNamespace(_model=FakeTorchModel("cuda"), name="canary")
        transcriber = types.SimpleNamespace(backend=backend, _pipeline=pipeline)

        assert instrument.transcriber_devices(transcriber) == ({"model": "cuda", "diarization": "cuda"}, "bfloat16")

    def test_a_pipeline_without_a_device_uses_its_segmentation_model(self):
        segmentation = types.SimpleNamespace(device="cpu")

        assert instrument.pipeline_device(types.SimpleNamespace(_segmentation=segmentation)) == "cpu"
        assert instrument.pipeline_device(object()) == "unknown"


class TestPeakMemory:
    def test_torch_counts_once_cuda_is_used(self):
        cuda = types.SimpleNamespace(is_initialized=lambda: True, max_memory_reserved=lambda: 123)
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(cuda=cuda)}):
            assert instrument.torch_peak() == 123
        cuda.is_initialized = lambda: False
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(cuda=cuda)}):
            assert instrument.torch_peak() is None

    def test_no_torch_no_count(self):
        with patch.dict(sys.modules, {"torch": None}):
            assert instrument.torch_peak() is None

    def test_sampler(self):
        readings = iter(["1000\n", "1500\n", "[N/A]\n"])
        done = lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, next(readings), "")  # noqa: E731
        with patch.object(instrument.subprocess, "run", done):
            sampler = instrument.GpuSampler("nvidia-smi", interval=3600)  # the thread waits; we sample by hand
            sampler.stop()
            sampler.peak = max(sampler.peak, sampler.used())

            assert sampler.used() is None
        assert sampler.baseline == 1000 * MIB
        assert sampler.peak_above_baseline() == 500 * MIB

    def test_a_sampler_without_nvidia_smi(self):
        sampler = instrument.GpuSampler("a-command-that-does-not-exist-55")

        assert sampler.baseline is None and sampler.peak_above_baseline() is None


class TestInstrumentedRun:
    def test_install_records_loads_and_transcriptions(self, fake_backend, tmp_path):
        from speech_toolkit.api import Transcriber

        class Instrumented(Transcriber):
            pass

        events = tmp_path / "events.jsonl"
        instrument.install(events, target=Instrumented)
        audio = tmp_path / "talk.wav"
        audio.write_bytes(b"not audio")

        Instrumented("fake", "small").transcribe(audio)

        load, transcription = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
        assert load["event"] == "load" and load["backend"] == "fake" and load["model"] == "small"
        assert load["devices"] == {"model": "unknown"} and load["seconds"] >= 0
        assert transcription["event"] == "transcribe" and transcription["audio"] == str(audio)
        assert transcription["audio_seconds"] is None  # not decodable
        assert transcription["torch_peak_bytes"] is None and transcription["sampled_peak_bytes"] is None

    def test_the_command_runs_unchanged(self, tmp_path):
        """python -m tests.integration.instrument speech_toolkit ...: stdout, exit code and events, on every OS."""
        audio, events = tmp_path / "my talk.wav", tmp_path / "events.jsonl"
        audio.write_bytes(b"not audio")
        code = (
            "import sys; from tests.conftest import FakeBackend; from speech_toolkit import register_backend; "
            "register_backend('fake', FakeBackend); from tests.integration import instrument; "
            "instrument.main(sys.argv[1:])"
        )
        env = dict(command_env(), **{instrument.EVENTS_VARIABLE: str(events)})
        command = [sys.executable, "-c", code, "speech_toolkit", str(audio), "-b", "fake", "-q"]

        done = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, encoding="utf-8", timeout=120)

        assert done.returncode == 0, done.stderr
        assert done.stdout == "Hello from my talk.\n"  # the fake backend's stdout noise went to stderr
        kinds = [json.loads(line)["event"] for line in events.read_text(encoding="utf-8").splitlines()]
        assert kinds == ["load", "transcribe"]

        failed = subprocess.run(
            [*command[:4], "--no-such-option"], cwd=tmp_path, env=env, capture_output=True, timeout=120
        )
        assert failed.returncode == 2  # argparse's exit code comes through

    def test_probe_mode_prints_one_line(self):
        command = [sys.executable, "-m", "tests.integration.instrument", "--probe"]
        done = subprocess.run(command, env=command_env(), capture_output=True, encoding="utf-8", timeout=300)

        assert done.returncode == 0, done.stderr
        assert set(devices.parse_probe(done.stdout)) == {"torch", "ctranslate2"}


def load_event(**devices_found):
    return {
        "event": "load",
        "backend": "faster-whisper",
        "model": "tiny",
        "devices": devices_found or {"model": "cuda"},
        "precision": "float16",
        "seconds": 2.0,
    }


def transcribe_event(seconds=1.0, audio=10.0, torch=None, sampled=None):
    return {
        "event": "transcribe",
        "audio": "a.wav",
        "audio_seconds": audio,
        "seconds": seconds,
        "torch_peak_bytes": torch,
        "sampled_peak_bytes": sampled,
    }


class TestEventLog:
    def test_reads_new_complete_lines_and_checks_devices(self, device, tmp_path):
        device("cuda")
        report, path = Report(), tmp_path / "events.jsonl"
        report.start_test("test_a", tmp_path)
        log = EventLog(path, report)
        assert log.record() == 0  # no file yet

        path.write_text(json.dumps(load_event()) + "\n" + json.dumps(transcribe_event()) + '\n{"event', "utf-8")
        assert log.record() == 1
        assert len(report.runs) == 1 and report.runs[0].model == "tiny" and report.runs[0].test == "test_a"

        with path.open("a", encoding="utf-8") as out:  # the half-written line, finished, and a second file
            out.write('": "transcribe", "audio": "b.wav", "audio_seconds": 5.0, "seconds": 0.5}\n')
        assert log.record() == 0
        assert [run.audio_seconds for run in report.runs] == [10.0, 5.0]

    def test_a_model_on_the_wrong_device_fails(self, device, tmp_path):
        device("cuda")
        path = tmp_path / "events.jsonl"
        path.write_text(json.dumps(load_event(model="cpu")) + "\n", encoding="utf-8")

        with pytest.raises(pytest.fail.Exception, match="the faster-whisper model ran on cpu"):
            EventLog(path, Report()).record()


def run(test="t", backend="faster-whisper", model="tiny", devices_found=None, **fields):
    values = dict(precision="float16", load_seconds=2.0, audio_seconds=10.0, seconds=1.0)
    values.update(fields)
    return Run(test, backend, model, devices_found or {"model": "cuda"}, **values)


class TestReport:
    def test_rows_add_up_runs(self, tmp_path):
        report = Report()
        report.start_test("t", tmp_path)
        report.add(run(load_seconds=9.0, sampled_peak_bytes=500 * MIB, errors=1, words=10))
        report.add(run(audio_seconds=30.0, seconds=3.0, sampled_peak_bytes=700 * MIB, errors=0, words=30))
        report.add(run(audio_seconds=None, seconds=5.0))
        report.add(run(model="turbo", devices_found={"model": "cuda", "diarization": "cuda"}, torch_peak_bytes=MIB))

        rows = report.rows()

        assert [row.cells() for row in rows] == [
            ["faster-whisper", "tiny", "cuda", "float16", "3", "40.0", "2.0", "9.0", "0.100", "0.025", "~700"],
            ["faster-whisper + pyannote", "turbo", "cuda", "float16", "1", "10.0", "2.0", "1.0", "0.100", "-", "1"],
        ]

    def test_peak_memory_source(self):
        assert run(backend="whisper", torch_peak_bytes=MIB, sampled_peak_bytes=2 * MIB).peak == (MIB, False)
        assert run(torch_peak_bytes=MIB, sampled_peak_bytes=2 * MIB).peak == (2 * MIB, True)  # faster-whisper
        assert run(sampled_peak_bytes=2 * MIB, devices_found={"model": "cpu"}).peak is None

    def test_a_device_mix_shows(self):
        mixed = run(devices_found={"model": "cuda", "diarization": "cpu"})

        assert mixed.key == ("faster-whisper + pyannote", "tiny", "cuda (pyannote on cpu)")

    def test_wer_goes_with_the_tests_runs(self, tmp_path):
        report = Report()
        report.start_test("batch", tmp_path)
        report.add(run(test="batch"))
        report.add(run(test="batch"))
        report.note_wer(1, 10)
        report.note_wer(2, 10)
        report.note_wer(3, 10)  # both runs are counted already
        assert [(r.errors, r.words) for r in report.runs] == [(1, 10), (2, 10)]

        report.start_test("server", tmp_path)
        report.note_wer(4, 10)  # before the run is recorded
        report.add(run(test="server"))
        assert (report.runs[-1].errors, report.runs[-1].words) == (4, 10)

    def test_markdown(self, tmp_path):
        report = Report()
        report.start_test("t", tmp_path)
        report.add(run(errors=0, words=8))

        text = report.markdown(
            ["- Machine: test"], {"passed": 3, "skipped": 1}, {"full tier: set INTEGRATION_TIER=full": 1}, ["t::x"]
        )

        assert text.startswith("## Integration report\n\n- Machine: test\n- Tests: 3 passed, 1 skipped in ")
        assert "| Backend | Model | Device |" in text and "| faster-whisper | tiny | cuda | float16 | 1 |" in text
        assert "- full tier: set INTEGRATION_TIER=full (1 test)" in text and "- `t::x`" in text
        assert text.isascii()
        assert "No model ran." in Report().markdown([], {}, {}, [])

    def test_environment_lines(self, device):
        device("cpu")
        lines = devices.environment_lines()

        assert lines[0].startswith("- Machine: ") and "INTEGRATION_DEVICE=cpu" in lines[1]
        assert lines[2] == "- GPU: not checked" and lines[3].startswith("- Packages: ")


SPEECH = Sample("speech", None, 1.0, (Turn("slt", "Hello.", 0.0, 1.0),))
DIALOGUE = Sample("dialogue", None, 2.0, (Turn("slt", "Hi.", 0.0, 0.8), Turn("rms", "Bye.", 1.2, 2.0)))


class TestLongClip:
    def test_join_audio_moves_the_turns(self):
        rate = 10
        parts = [(DIALOGUE, np.ones(20)), (SPEECH, np.ones(10)), (DIALOGUE, np.ones(20))]

        audio, turns = join_audio(parts, gap=0.5, rate=rate)

        assert len(audio) == 20 + 5 + 10 + 5 + 20  # no silence after the last part
        assert audio.dtype == np.float32 and audio[20:25].sum() == 0
        assert [(turn.voice, turn.start, turn.end) for turn in turns] == [
            ("slt", 0.0, 0.8),
            ("rms", 1.2, 2.0),
            ("slt", 2.5, 3.5),
            ("slt", 4.0, 4.8),
            ("rms", 5.2, 6.0),
        ]

    def test_write_wav(self, tmp_path):
        path = tmp_path / "a folder with spaces" / "clip.wav"
        path.parent.mkdir()
        write_wav(path, np.array([0.0, 0.5, -1.0, 2.0], dtype=np.float32), 16000)

        with wave.open(str(path), "rb") as clip:
            assert (clip.getnchannels(), clip.getsampwidth(), clip.getframerate()) == (1, 2, 16000)
            frames = np.frombuffer(clip.readframes(4), "<i2")
        assert frames.tolist() == [0, 16383, -32767, 32767]

    def test_chunk_seconds(self):
        assert [chunk_seconds(b) for b in ("whisper", "faster-whisper", "canary", "voxtral", "parakeet")] == [
            30,
            30,
            30,
            30,
            120,
        ]

    def test_chunks_covered(self):
        chunks = [(0.0, 30.0), (30.0, 58.0)]
        assert_chunks_covered([(1.0, 4.0), (29.0, 31.0)], chunks, "segment")
        with pytest.raises(AssertionError, match="no segment in the chunk at 30.00-58.00 s"):
            assert_chunks_covered([(1.0, 4.0)], chunks, "segment")

    def test_cut_off_warning(self):
        assert_no_cut_off_warning("Loading canary model...\nCanary: transcribed 4/4 chunks\n")
        warning = "Warning: long.wav: Canary reached its limit of 500 tokens in chunk 2, so the text may be cut off."
        with pytest.raises(AssertionError, match="cut text off"):
            assert_no_cut_off_warning(f"Loading...\n{warning}\n")
