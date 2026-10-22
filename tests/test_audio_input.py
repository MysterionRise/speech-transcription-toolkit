"""In-memory audio, and decoding each recording once: Transcriber.transcribe() takes a path, bytes, a binary file object
or a waveform, and gives the same result for each; with speaker labels, ffmpeg still runs once per recording.

ffmpeg is mocked (subprocess.run in speech_toolkit.media): FakeFfmpeg "decodes" WAV files and the MP4 files mp4()
makes, from a path or from its stdin, as the real one would, and counts its runs. tests/integration/test_audio_input.py
runs the real ffmpeg and real models.
"""

from __future__ import annotations

import io
import os
import pathlib
import subprocess
import sys
import wave
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from speech_toolkit import AudioDecodeError, Transcriber
from speech_toolkit import api as api_module
from speech_toolkit import transcribe
from speech_toolkit.api import _index_at_end, _Rewound, _write_wav
from speech_toolkit.backends import CanaryBackend, ParakeetBackend, VoxtralBackend
from tests.conftest import FakeBackend, FakeWaveformBackend, FakeWordsBackend

RATE = 16000
SAMPLES = (np.sin(np.arange(RATE) * 2 * np.pi * 440 / RATE) * 12000).astype(np.int16)  # 1 s of a 440 Hz tone
PCM = SAMPLES.astype("<i2").tobytes()  # what ffmpeg writes for it
WAVEFORM = SAMPLES.astype(np.float32) / 32768  # what load_audio() returns for it
PARTIAL_FILE = b"[mov,mp4,m4a,3gp,3g2,mj2 @ 0x5] stream 0, offset 0x2c: partial file\n"  # ffmpeg 6, MP4 from a pipe


def wav(samples: np.ndarray = SAMPLES) -> bytes:
    """*samples* (int16) as a 16 kHz mono WAV file."""
    out = io.BytesIO()
    with wave.open(out, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(RATE)
        file.writeframes(samples.astype("<i2").tobytes())
    return out.getvalue()


def box(kind: bytes, payload: bytes = b"") -> bytes:
    """An MP4 box: its size, its type and its payload."""
    return (8 + len(payload)).to_bytes(4, "big") + kind + payload


def mp4(index_first: bool) -> bytes:
    """An M4A file whose ``mdat`` box holds PCM, with its index (``moov``) before or after it."""
    moov, mdat = box(b"moov", bytes(24)), box(b"mdat", PCM)
    return box(b"ftyp", b"M4A \x00\x00\x00\x00isomM4A ") + (moov + mdat if index_first else mdat + moov)


def boxes(data: bytes) -> List[Tuple[bytes, bytes]]:
    """The top-level boxes of an MP4 file, as (type, payload)."""
    found, offset = [], 0
    while offset + 8 <= len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        found.append((data[offset + 4 : offset + 8], data[offset + 8 : offset + size]))
        offset += size
    return found


def finished(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["ffmpeg"], returncode=returncode, stdout=stdout, stderr=stderr)


class FakeFfmpeg:
    """subprocess.run as ffmpeg: decodes WAV and mp4() files from a path, bytes or its stdin, and records each run.

    Like the real one, it can't decode an MP4 whose index comes last from a pipe: it then exits successfully having
    decoded nothing, and complains.
    """

    def __init__(self) -> None:
        self.runs: List[Tuple[str, bytes]] = []  # (the input's name, its content)

    def __call__(self, cmd: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
        url = cmd[cmd.index("-i") + 1]
        if url.startswith("file:"):
            data = pathlib.Path(url[len("file:") :]).read_bytes()
        elif kwargs.get("input") is not None:
            data = bytes(kwargs["input"])
        else:
            data = b""
            while chunk := os.read(kwargs["stdin"], 65536):
                data += chunk
        self.runs.append((url, data))
        if data.startswith(b"RIFF"):
            with wave.open(io.BytesIO(data)) as file:
                return finished(file.readframes(file.getnframes()))
        kinds = [kind for kind, _ in boxes(data)]
        if data[4:8] == b"ftyp" and b"mdat" in kinds:
            if url == "pipe:0" and kinds.index(b"mdat") < kinds.index(b"moov"):
                return finished(b"", PARTIAL_FILE)
            return finished(dict(boxes(data))[b"mdat"])
        return finished(b"", b"pipe:0: Invalid data found when processing input\n", returncode=183)

    @property
    def inputs(self) -> List[str]:
        return [url for url, _ in self.runs]


class Unseekable(io.RawIOBase):
    """A binary stream that can only be read, like a pipe or a socket: reads may also return less than asked for."""

    def __init__(self, data: bytes, most: int = 5000) -> None:
        self._data, self._most = data, most

    def readable(self) -> bool:
        return True

    def read(self, size: Optional[int] = -1) -> bytes:
        size = len(self._data) if size is None or size < 0 else size
        chunk, self._data = self._data[: min(size, self._most)], self._data[min(size, self._most) :]
        return chunk


@pytest.fixture
def ffmpeg():
    fake = FakeFfmpeg()
    with patch("speech_toolkit.media.subprocess.run", side_effect=fake):
        yield fake


@pytest.fixture
def wav_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "talk.wav"
    path.write_bytes(wav())
    return path


# Every kind of input, made from the WAV file.
INPUTS: Dict[str, Callable[[pathlib.Path], Any]] = {
    "path": lambda path: path,
    "str": str,
    "bytes": lambda path: path.read_bytes(),
    "bytearray": lambda path: bytearray(path.read_bytes()),
    "memoryview": lambda path: memoryview(path.read_bytes()),
    "BytesIO": lambda path: io.BytesIO(path.read_bytes()),
    "unseekable stream": lambda path: Unseekable(path.read_bytes()),
    "waveform": lambda path: WAVEFORM.copy(),
}
IN_MEMORY = [kind for kind in INPUTS if kind not in ("path", "str")]


def heard(samples: np.ndarray) -> str:
    """What FakeWaveformBackend transcribes *samples* as."""
    result = FakeWaveformBackend().transcribe(samples)
    return result.text


class WavReader(FakeBackend):
    """A backend without the "waveform" capability that reads the WAV file it gets; it records the path and whether
    that file existed then."""

    name = "wav-reader"
    capabilities = frozenset({"word_timestamps"})
    seen: List[Tuple[pathlib.Path, bool]] = []

    def transcribe(self, audio_path, language=None, task="transcribe", verbose=True, **options):
        WavReader.seen.append((audio_path, audio_path.exists()))
        with wave.open(str(audio_path)) as file:
            samples = np.frombuffer(file.readframes(file.getnframes()), "<i2").astype(np.float32) / 32768
        return FakeWaveformBackend().transcribe(samples)


@pytest.fixture
def wav_reader(fake_waveform_backend):
    from speech_toolkit.backends import register_backend

    register_backend("wav-reader", WavReader)
    WavReader.seen = []
    return WavReader


@pytest.fixture
def diarization(monkeypatch):
    """A mocked pyannote pipeline that hears SPEAKER_00 throughout, and mocked torch; diarize_audio() itself runs."""
    torch, pipeline = MagicMock(), MagicMock()
    turn = SimpleNamespace(start=0.0, end=1.0)
    pipeline.return_value.exclusive_speaker_diarization.itertracks.return_value = [(turn, "A", "SPEAKER_00")]
    monkeypatch.setattr("speech_toolkit.api.load_diarization_pipeline", lambda device: pipeline)
    with patch.dict(sys.modules, {"torch": torch}):
        yield SimpleNamespace(torch=torch, pipeline=pipeline)


###############################################################################
# Every input gives the same result
###############################################################################


class TestInputs:
    @pytest.mark.parametrize("kind", INPUTS)
    def test_same_result_from_every_input(self, fake_waveform_backend, ffmpeg, wav_file, kind):
        result = Transcriber("fake-waveform").transcribe(INPUTS[kind](wav_file))

        assert result.text == heard(WAVEFORM)
        assert result.segments == [{"id": 0, "start": 0.0, "end": 1.0, "text": heard(WAVEFORM)}]
        assert len(ffmpeg.runs) == (0 if kind == "waveform" else 1)

    @pytest.mark.parametrize("kind", INPUTS)
    def test_same_result_without_the_waveform_capability(self, wav_reader, ffmpeg, wav_file, kind):
        result = Transcriber("wav-reader").transcribe(INPUTS[kind](wav_file))

        assert result.text == heard(WAVEFORM)
        [(path, existed)] = WavReader.seen
        assert existed
        if kind in IN_MEMORY:  # a temporary WAV file, deleted afterwards
            assert path.suffix == ".wav" and path != wav_file and not path.exists()
        else:
            assert path == wav_file
            assert ffmpeg.runs == []  # the backend reads the file itself

    def test_waveform_backends_get_the_samples(self, fake_waveform_backend, ffmpeg, wav_file):
        Transcriber("fake-waveform").transcribe(wav_file.read_bytes())

        [received] = FakeWaveformBackend.received
        assert isinstance(received, np.ndarray) and received.dtype == np.float32
        np.testing.assert_array_equal(received, WAVEFORM)

    def test_a_path_alone_is_left_to_the_backend(self, fake_waveform_backend, ffmpeg, wav_file):
        """Without diarization nothing else needs the samples, so the backend reads the file the way it likes."""
        Transcriber("fake-waveform").transcribe(str(wav_file))

        assert FakeWaveformBackend.received == [wav_file]
        assert ffmpeg.inputs == [f"file:{wav_file}"]  # FakeWaveformBackend decoding it

    def test_a_file_object_is_read_from_its_position_and_left_open(self, fake_waveform_backend, ffmpeg, wav_file):
        file = io.BytesIO(b"junk" + wav_file.read_bytes())
        file.seek(4)

        assert Transcriber("fake-waveform").transcribe(file).text == heard(WAVEFORM)
        assert not file.closed and file.read() == b""

    def test_strided_arrays_are_made_contiguous(self, fake_waveform_backend):
        stereo = np.stack([WAVEFORM, np.zeros_like(WAVEFORM)], axis=1)  # the left channel is a strided view

        assert Transcriber("fake-waveform").transcribe(stereo[:, 0]).text == heard(WAVEFORM)
        assert FakeWaveformBackend.received[0].flags.c_contiguous

    def test_one_shot_transcribe(self, fake_waveform_backend, ffmpeg, wav_file):
        result = transcribe(wav_file.read_bytes(), backend="fake-waveform")

        assert result.text == heard(WAVEFORM)

    def test_temporary_wav_is_deleted_when_the_backend_fails(self, fake_backend, ffmpeg, wav_file, monkeypatch):
        """FakeBackend fails for a file named bad*: the temporary WAV file gets such a name."""
        monkeypatch.setattr(api_module, "_temporary_file", _named("bad", api_module._temporary_file))
        made: List[pathlib.Path] = []
        monkeypatch.setattr(api_module, "_write_wav", _recording(made, api_module._write_wav))

        with pytest.raises(RuntimeError, match="Failed to load audio"):
            Transcriber("fake").transcribe(wav_file.read_bytes())
        assert len(made) == 1 and not made[0].exists()


def _named(prefix: str, temporary_file: Callable[..., Any]) -> Callable[..., Any]:
    """_temporary_file(), with *prefix* at the start of the file's name."""
    import contextlib

    @contextlib.contextmanager
    def named(suffix: str):
        with temporary_file(suffix) as path:
            renamed = path.with_name(prefix + path.name)
            path.rename(renamed)
            try:
                yield renamed
            finally:
                renamed.unlink(missing_ok=True)

    return named


def _recording(made: List[pathlib.Path], write_wav: Callable[..., Any]) -> Callable[..., Any]:
    def record(path: pathlib.Path, samples: np.ndarray) -> None:
        made.append(path)
        write_wav(path, samples)

    return record


###############################################################################
# Bad input
###############################################################################


class TestInputChecks:
    @pytest.mark.parametrize("audio", [42, None, ["talk.wav"], 1.5])
    def test_other_types_raise_type_error(self, fake_waveform_backend, audio):
        with pytest.raises(TypeError, match="audio must be a path, bytes, a binary file object or a NumPy array"):
            Transcriber("fake-waveform").transcribe(audio)
        assert FakeWaveformBackend.received == []

    @pytest.mark.parametrize(
        "array, shown",
        [
            (SAMPLES, "1-D int16"),
            (WAVEFORM.astype(np.float64), "1-D float64"),
            (np.stack([WAVEFORM, WAVEFORM]), "2-D float32"),
            (np.float32(0.5), "0-D float32"),
        ],
        ids=["int16", "float64", "stereo", "scalar"],
    )
    def test_arrays_must_be_1d_float32(self, fake_waveform_backend, array, shown):
        with pytest.raises(ValueError, match=f"1-D float32 samples .*not a {shown} array"):
            Transcriber("fake-waveform").transcribe(np.asarray(array))
        assert FakeWaveformBackend.received == []

    def test_one_shot_checks_before_loading_the_model(self, fake_waveform_backend):
        with pytest.raises(TypeError):
            transcribe(42, backend="fake-waveform")
        with pytest.raises(ValueError):
            transcribe(SAMPLES, backend="fake-waveform")
        assert FakeBackend.loaded == 0

    def test_text_file_objects_are_refused(self, fake_waveform_backend):
        with pytest.raises(TypeError, match="needs a binary file object, but read.. returned str"):
            Transcriber("fake-waveform").transcribe(io.StringIO("RIFF"))

    def test_undecodable_bytes(self, fake_waveform_backend, ffmpeg):
        with pytest.raises(AudioDecodeError, match="Invalid data found"):
            Transcriber("fake-waveform").transcribe(b"not audio")

    def test_missing_file_when_decoding_for_diarization(self, fake_waveform_backend, diarization, ffmpeg, tmp_path):
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            Transcriber("fake-waveform", diarize=True).transcribe(tmp_path / "gone.wav")
        assert ffmpeg.runs == []


###############################################################################
# Decoding once, with diarization
###############################################################################


class TestDecodeOnce:
    @pytest.mark.parametrize("kind", INPUTS)
    def test_ffmpeg_runs_once_with_diarization(self, fake_waveform_backend, diarization, ffmpeg, wav_file, kind):
        result = Transcriber("fake-waveform", diarize=True).transcribe(INPUTS[kind](wav_file))

        assert len(ffmpeg.runs) == (0 if kind == "waveform" else 1)
        [received] = FakeWaveformBackend.received
        assert isinstance(received, np.ndarray)
        diarized = diarization.torch.from_numpy.call_args.args[0]
        assert diarized is received  # the same array, decoded once
        assert result.text == heard(WAVEFORM)
        assert [segment["speaker"] for segment in result.speaker_segments] == ["SPEAKER_00"]

    @pytest.mark.parametrize("kind", ["path", "bytes", "unseekable stream", "waveform"])
    def test_without_the_waveform_capability(self, fake_backend, diarization, ffmpeg, wav_file, kind):
        """The backend gets the file, or a temporary WAV; diarization decodes the file, or gets the samples."""
        result = Transcriber("fake-words", diarize=True).transcribe(INPUTS[kind](wav_file))

        assert len(ffmpeg.runs) == (0 if kind == "waveform" else 1)
        np.testing.assert_array_equal(diarization.torch.from_numpy.call_args.args[0], WAVEFORM)
        assert FakeWordsBackend.calls == [{"word_timestamps": True}]
        assert [segment["speaker"] for segment in result.speaker_segments] == ["SPEAKER_00"]

    def test_diarization_alone_decodes_a_path_for_other_backends(self, fake_backend, diarization, ffmpeg, wav_file):
        Transcriber("fake", diarize=True).transcribe(wav_file)

        assert ffmpeg.inputs == [f"file:{wav_file}"]


@pytest.fixture
def whisper_model():
    module = MagicMock()
    model = module.load_model.return_value
    model.parameters.return_value = iter([SimpleNamespace(device="cpu")])
    model.transcribe.return_value = {
        "text": " Hi.",
        "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": " Hi."}],
        "language": "en",
    }
    with patch.dict(sys.modules, {"whisper": module}):
        yield model


@pytest.fixture
def faster_whisper_model():
    faster_whisper, ctranslate2 = MagicMock(), MagicMock()
    ctranslate2.get_cuda_device_count.return_value = 0
    model = faster_whisper.WhisperModel.return_value
    segment = SimpleNamespace(
        id=1,
        start=0.0,
        end=1.0,
        text=" Hi.",
        tokens=[1],
        temperature=0.0,
        avg_logprob=-0.1,
        compression_ratio=1.0,
        no_speech_prob=0.0,
    )
    info = SimpleNamespace(language="en", duration=2.5, language_probability=0.9)  # not the samples' 1 s: kept
    model.transcribe.side_effect = lambda audio, **kwargs: (iter([segment]), info)
    with patch.dict(sys.modules, {"faster_whisper": faster_whisper, "ctranslate2": ctranslate2}):
        yield model


class TestWhisperBackends:
    """whisper and faster-whisper get the samples whenever they were decoded, and the file otherwise."""

    @pytest.mark.parametrize(
        "backend, model", [("whisper", "whisper_model"), ("faster-whisper", "faster_whisper_model")]
    )
    @pytest.mark.parametrize("kind", ["bytes", "unseekable stream", "waveform"])
    def test_in_memory_input(self, request, ffmpeg, wav_file, backend, model, kind):
        mocked = request.getfixturevalue(model)

        result = Transcriber(backend, "tiny").transcribe(INPUTS[kind](wav_file))

        audio = mocked.transcribe.call_args.args[0]
        assert isinstance(audio, np.ndarray)
        np.testing.assert_array_equal(audio, WAVEFORM)
        assert result.text.strip() == "Hi."
        assert result.duration == (2.5 if backend == "faster-whisper" else 1.0)  # a backend's own duration is kept

    @pytest.mark.parametrize(
        "backend, model", [("whisper", "whisper_model"), ("faster-whisper", "faster_whisper_model")]
    )
    def test_path_with_diarization(self, request, diarization, ffmpeg, wav_file, backend, model):
        mocked = request.getfixturevalue(model)

        Transcriber(backend, "tiny", diarize=True).transcribe(wav_file)

        audio = mocked.transcribe.call_args.args[0]
        assert audio is diarization.torch.from_numpy.call_args.args[0]
        np.testing.assert_array_equal(audio, WAVEFORM)
        assert ffmpeg.inputs == [f"file:{wav_file}"]

    @pytest.mark.parametrize(
        "backend, model", [("whisper", "whisper_model"), ("faster-whisper", "faster_whisper_model")]
    )
    def test_path_alone(self, request, ffmpeg, wav_file, backend, model):
        mocked = request.getfixturevalue(model)

        Transcriber(backend, "tiny").transcribe(wav_file)

        assert mocked.transcribe.call_args.args[0] == str(wav_file)
        assert ffmpeg.runs == []


###############################################################################
# Parakeet, Canary and Voxtral: samples skip their own decoding
###############################################################################


class FakeBatch(dict):
    def to(self, *args: Any, **kwargs: Any) -> "FakeBatch":
        return self


@pytest.fixture
def transformers_modules():
    """Mocked torch and transformers: Parakeet hears " Hi." and Canary and Voxtral "Hi." in each chunk."""
    torch, transformers = MagicMock(), MagicMock()
    torch.cuda.is_available.return_value = False
    processor = transformers.AutoProcessor.from_pretrained.return_value
    processor.return_value = FakeBatch()
    processor.decode.return_value = ("Hi.", [[{"token": " Hi.", "start": 0.0, "end": 0.5}]])
    processor.batch_decode.side_effect = lambda outputs, **kwargs: ["Hi."]
    with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
        yield


@pytest.mark.parametrize(
    "backend_class, module",
    [(ParakeetBackend, "nvidia_backend"), (CanaryBackend, "nvidia_backend"), (VoxtralBackend, "voxtral_backend")],
)
class TestTransformersBackends:
    def _run(self, backend_class, audio):
        backend = backend_class()
        backend.load_model(backend.default_model())
        return backend.transcribe(audio, language="en" if backend_class is CanaryBackend else None, verbose=False)

    def test_samples_are_not_decoded_again(self, transformers_modules, backend_class, module):
        with patch(f"speech_toolkit.backends.{module}.load_audio") as load_audio:
            result = self._run(backend_class, WAVEFORM)

        load_audio.assert_not_called()
        assert result.text == "Hi."
        assert result.duration == 1.0

    def test_a_path_is_decoded_and_its_duration_reported(self, transformers_modules, backend_class, module, wav_file):
        """130 s: several chunks for each backend, and the duration is the whole recording's, not the last chunk's."""
        silence = np.zeros(130 * RATE, np.float32)
        with patch(f"speech_toolkit.backends.{module}.load_audio", return_value=silence) as load:
            result = self._run(backend_class, wav_file)

        load.assert_called_once_with(wav_file)
        assert result.duration == 130.0

    def test_samples_give_the_whole_duration(self, transformers_modules, backend_class, module):
        assert self._run(backend_class, np.zeros(130 * RATE, np.float32)).duration == 130.0

    def test_a_missing_file(self, transformers_modules, backend_class, module, tmp_path):
        with pytest.raises(FileNotFoundError):
            self._run(backend_class, tmp_path / "gone.wav")


class FakeVoxtralInputs(dict):
    """Stand-in for Voxtral's processor output: a prompt of 3 tokens per chunk."""

    def __init__(self, batch_size: int) -> None:
        super().__init__(input_ids="ids")
        self.input_ids = SimpleNamespace(shape=(batch_size, 3))

    def to(self, *args: Any, **kwargs: Any) -> "FakeVoxtralInputs":
        return self


def test_cut_off_warning_names_samples_audio():
    """The warning about text cut off at MAX_NEW_TOKENS names the file; samples have none, so they are "audio"."""
    from speech_toolkit import SpeechToolkitWarning

    torch, transformers = MagicMock(), MagicMock()
    torch.cuda.is_available.return_value = False
    eos, pad = 2, 11
    with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
        backend = VoxtralBackend()
        backend.load_model("voxtral-mini")
        processor = transformers.AutoProcessor.from_pretrained.return_value
        processor.apply_transcription_request.side_effect = lambda **kwargs: FakeVoxtralInputs(len(kwargs["audio"]))
        processor.batch_decode.side_effect = [["one", "two", "three"]]
        model = transformers.VoxtralForConditionalGeneration.from_pretrained.return_value.to.return_value
        model.generation_config = SimpleNamespace(eos_token_id=eos, pad_token_id=pad)
        # 70 s: three chunks; the second one's text reaches the limit of 3 tokens without ending.
        model.generate.return_value = np.array([[1, 1, 1, 5, eos, pad], [1, 1, 1, 5, 6, 7], [1, 1, 1, 5, 6, eos]])
        with patch("speech_toolkit.backends.voxtral_backend.MAX_NEW_TOKENS", 3):
            with pytest.warns(SpeechToolkitWarning) as warned:
                result = backend.transcribe(np.zeros(70 * RATE, np.float32), verbose=False)

    assert [str(warning.message) for warning in warned] == [
        "audio: Voxtral reached its limit of 3 tokens in the chunk from 0:25 to 0:50, so the text may be cut off."
    ]
    assert result.duration == 70.0


###############################################################################
# MP4, M4A and MOV from memory
###############################################################################


class TestMp4FromMemory:
    @pytest.mark.parametrize("kind", ["bytes", "BytesIO", "unseekable stream"])
    def test_index_at_the_end_goes_through_a_temporary_file(self, fake_waveform_backend, ffmpeg, kind):
        data = mp4(index_first=False)
        source = {"bytes": data, "BytesIO": io.BytesIO(data), "unseekable stream": Unseekable(data)}[kind]

        result = Transcriber("fake-waveform").transcribe(source)

        assert result.text == heard(WAVEFORM)
        [(url, received)] = ffmpeg.runs
        assert url.startswith("file:") and url.endswith(".mp4")
        assert received == data
        assert not pathlib.Path(url[len("file:") :]).exists()  # deleted afterwards

    @pytest.mark.parametrize("kind", ["bytes", "BytesIO", "unseekable stream"])
    def test_index_first_goes_through_the_pipe(self, fake_waveform_backend, ffmpeg, kind):
        data = mp4(index_first=True)
        source = {"bytes": data, "BytesIO": io.BytesIO(data), "unseekable stream": Unseekable(data)}[kind]

        assert Transcriber("fake-waveform").transcribe(source).text == heard(WAVEFORM)
        assert ffmpeg.runs == [("pipe:0", data)]  # all of it, the bytes looked at first included

    def test_the_mocked_ffmpeg_fails_like_the_real_one(self, ffmpeg):
        """What the temporary file avoids: load_audio() itself can't decode the file from bytes."""
        from speech_toolkit.media import load_audio

        with pytest.raises(AudioDecodeError, match="index at the end"):
            load_audio(mp4(index_first=False))

    def test_temporary_file_is_deleted_when_decoding_fails(self, fake_waveform_backend, ffmpeg):
        data = box(b"ftyp", b"M4A ") + box(b"free")  # no audio at all

        with pytest.raises(AudioDecodeError):
            Transcriber("fake-waveform").transcribe(data)
        [(url, _)] = ffmpeg.runs
        assert url.startswith("file:") and not pathlib.Path(url[len("file:") :]).exists()


class TestIndexAtEnd:
    @pytest.mark.parametrize(
        "head, expected",
        [
            (wav(), False),
            (b"ID3\x04\x00\x00\x00\x00\x00\x00", False),  # MP3
            (b"fLaC", False),  # too short to be an MP4
            (b"", False),
            (mp4(index_first=True), False),
            (mp4(index_first=False), True),
            (box(b"wide") + box(b"mdat", PCM) + box(b"moov"), True),  # QuickTime without ftyp
            (box(b"moov") + box(b"mdat", PCM), False),
            (box(b"ftyp", b"M4A ") + box(b"free", bytes(100)) + box(b"moov") + box(b"mdat"), False),
            (box(b"ftyp", b"M4A "), True),  # ends before either box: assume the index comes last
            (box(b"ftyp", b"M4A ")[:-2], True),
            (box(b"ftyp", b"M4A ") + bytes(4) + b"mdat", True),  # size 0: the box runs to the end of the file
            (box(b"ftyp", b"M4A ") + (3).to_bytes(4, "big") + b"free", True),  # a damaged size
            # A 64-bit size: 1, the type, then the size.
            ((1).to_bytes(4, "big") + b"ftyp" + (24).to_bytes(8, "big") + bytes(8) + box(b"moov"), False),
            ((1).to_bytes(4, "big") + b"ftyp" + (24).to_bytes(8, "big") + bytes(8) + box(b"mdat"), True),
            ((1).to_bytes(4, "big") + b"ftyp" + bytes(4), True),  # its 64-bit size cut off
        ],
        # Short ids: pytest puts the test's id in an environment variable, which Windows limits to 32767 characters.
        ids=[
            "wav",
            "mp3",
            "short",
            "empty",
            "index-first",
            "index-at-end",
            "quicktime-wide",
            "quicktime-moov",
            "free-box-first",
            "only-ftyp",
            "cut-ftyp",
            "size-0",
            "damaged-size",
            "64-bit-index-first",
            "64-bit-index-at-end",
            "64-bit-size-cut",
        ],
    )
    def test_boxes(self, head, expected):
        assert _index_at_end(head) is expected
        assert _index_at_end(memoryview(head)) is expected

    def test_index_beyond_the_bytes_looked_at(self, fake_waveform_backend, ffmpeg):
        """A file object is looked at up to 64 KiB in: an index further in is taken to come last."""
        data = box(b"ftyp", b"M4A ") + box(b"free", bytes(100_000)) + box(b"moov") + box(b"mdat", PCM)

        assert Transcriber("fake-waveform").transcribe(Unseekable(data)).text == heard(WAVEFORM)
        assert ffmpeg.inputs[0].startswith("file:")


class TestRewound:
    def test_reads_the_head_again_then_the_rest(self):
        stream = io.BytesIO(b"0123456789")
        head = stream.read(4)
        rewound = _Rewound(head, stream)

        assert [rewound.read(3), rewound.read(3), rewound.read(3), rewound.read(3)] == [b"012", b"3", b"456", b"789"]
        assert rewound.read(3) == b""

    @pytest.mark.parametrize("size", [-1, None])
    def test_read_all(self, size):
        stream = io.BytesIO(b"0123456789")
        rewound = _Rewound(stream.read(4), stream)

        assert rewound.read(size) == b"0123456789"
        assert rewound.read(size) == b""


###############################################################################
# The temporary WAV file
###############################################################################


class TestTemporaryWav:
    def test_decoded_samples_are_written_exactly(self, tmp_path, monkeypatch):
        monkeypatch.setattr(api_module, "_WAV_PART_SAMPLES", 3000)  # several parts
        path = tmp_path / "out.wav"

        _write_wav(path, WAVEFORM)

        assert path.read_bytes() == wav()

    def test_samples_out_of_range_are_clipped(self, tmp_path):
        path = tmp_path / "out.wav"

        _write_wav(path, np.array([1.5, -2.0, 0.25, -0.25], dtype=np.float32))

        with wave.open(str(path)) as file:
            assert np.frombuffer(file.readframes(4), "<i2").tolist() == [32767, -32768, 8192, -8192]
