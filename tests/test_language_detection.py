"""Language detection: Transcriber.detect_language(), the whisper and faster-whisper backends' detect_language(), the
language_probability of whisper's results, and ``transcribe --detect-language``.

The model libraries are mocked with ``patch.dict(sys.modules, ...)``, and so is decoding, which would need ffmpeg: the
tests check how much of each input is decoded, and that nothing is transcribed.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import wave
from types import SimpleNamespace
from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from speech_toolkit import Transcriber
from speech_toolkit.api import LANGUAGE_DETECTION_SECONDS, check_language_detection, language_detectors
from speech_toolkit.backends import TranscriptionBackend
from speech_toolkit.backends.faster_whisper_backend import FasterWhisperBackend
from speech_toolkit.backends.whisper_backend import WhisperBackend
from speech_toolkit.cli import main, parse_args
from speech_toolkit.errors import AudioDecodeError, BackendNotFoundError, ModelLoadError, UnsupportedOptionError
from speech_toolkit.media import SAMPLE_RATE
from tests.conftest import FakeBackend

RATE = SAMPLE_RATE
WINDOW = int(LANGUAGE_DETECTION_SECONDS * RATE)  # the samples of the first 30 s
FOUND = [("en", 0.02), ("de", 0.97), ("fr", 0.01)]  # in no particular order: Transcriber sorts them
RANKED = [
    {"language": "de", "probability": 0.97},
    {"language": "en", "probability": 0.02},
    {"language": "fr", "probability": 0.01},
]


class FakeDetector(TranscriptionBackend):
    """Hears German in anything, records what it was given, and fails the test if asked to transcribe."""

    name = "fake-detector"
    description = "Fake language detector for tests"
    capabilities = frozenset({"language_detection", "waveform"})
    received: List[Any] = []
    loaded = 0

    @classmethod
    def available_models(cls) -> List[str]:
        return ["small"]

    def load_model(self, model_name, device=None):
        FakeDetector.loaded += 1
        self._model = object()
        self._model_name = model_name

    def transcribe(self, audio_path, language=None, task="transcribe", verbose=True):
        raise AssertionError("language detection must not transcribe")

    def detect_language(self, audio):
        print("library noise on stdout")
        FakeDetector.received.append(audio)
        return list(FOUND)


class FakeFileDetector(FakeDetector):
    """FakeDetector without the "waveform" capability: it gets a WAV file, and records how many samples it holds."""

    name = "fake-file-detector"
    capabilities = frozenset({"language_detection"})

    def detect_language(self, audio):
        with wave.open(str(audio)) as wav:
            FakeDetector.received.append((audio, wav.getnframes()))
        return list(FOUND)


class FakeMethodOnly(FakeDetector):
    """Has a detect_language() method, but not the "language_detection" capability."""

    name = "fake-method-only"
    capabilities = frozenset({"waveform"})


@pytest.fixture
def detectors(monkeypatch):
    """Register the fake detectors, and FakeBackend as "fake" (the capability without the method)."""
    from speech_toolkit.backends import _BACKENDS

    fakes = {"fake": FakeBackend, "fake-detector": FakeDetector, "fake-file-detector": FakeFileDetector}
    monkeypatch.setattr("speech_toolkit.backends._BACKENDS", {**_BACKENDS, **fakes, "fake-method-only": FakeMethodOnly})
    FakeDetector.received = []
    FakeDetector.loaded = 0
    FakeBackend.loaded = 0


@pytest.fixture
def decoded(monkeypatch) -> List[Tuple[Any, Any]]:
    """speech_toolkit.api's load_audio(), which records (source, max_seconds) and returns that many seconds of a tone
    (a minute without max_seconds); a path with "bad" in its name fails to decode."""
    calls: List[Tuple[Any, Any]] = []

    def load_audio(source, sr=RATE, *, max_seconds=None, **options):
        calls.append((source, max_seconds))
        # Only paths: str() of a file object holds its memory address, whose hex digits can spell "bad".
        if isinstance(source, (str, pathlib.PurePath)) and "bad" in str(source):
            raise AudioDecodeError("Failed to load audio: Invalid data found when processing input")
        return np.full(int((60 if max_seconds is None else max_seconds) * RATE), 0.1, np.float32)

    monkeypatch.setattr("speech_toolkit.api.load_audio", load_audio)
    return calls


@pytest.fixture
def audio(tmp_path) -> pathlib.Path:
    path = tmp_path / "talk.mp3"
    path.touch()
    return path


# An MP4 whose index (moov) comes after its audio (mdat): ffmpeg can't decode it from a pipe.
MP4_INDEX_LAST = (16).to_bytes(4, "big") + b"ftypisom\0\0\0\0" + (16).to_bytes(4, "big") + b"mdat" + bytes(8)


###############################################################################
# Transcriber.detect_language()
###############################################################################


class TestTranscriber:
    def test_a_path_is_decoded_only_30_seconds_far(self, detectors, decoded, audio):
        detection = Transcriber("fake-detector").detect_language(audio)

        assert detection == {"language": "de", "probability": 0.97, "candidates": RANKED}
        assert decoded == [(audio, 30.0)]
        [samples] = FakeDetector.received
        assert isinstance(samples, np.ndarray) and len(samples) == WINDOW

    def test_a_str_path(self, detectors, decoded, audio):
        assert Transcriber("fake-detector").detect_language(str(audio))["language"] == "de"
        assert decoded == [(audio, 30.0)]

    @pytest.mark.parametrize("top, languages", [(1, ["de"]), (2, ["de", "en"]), (5, ["de", "en", "fr"])])
    def test_top_candidates(self, detectors, decoded, audio, top, languages):
        detection = Transcriber("fake-detector").detect_language(audio, top=top)

        assert [candidate["language"] for candidate in detection["candidates"]] == languages
        assert (detection["language"], detection["probability"]) == ("de", 0.97)

    def test_every_candidate(self, detectors, decoded, audio, monkeypatch):
        many = [(f"l{number}", number / 1000) for number in range(99)]
        monkeypatch.setattr(FakeDetector, "detect_language", lambda self, audio: many)

        detection = Transcriber("fake-detector").detect_language(audio, top=None)

        assert len(detection["candidates"]) == 99
        assert detection["candidates"][0] == {"language": "l98", "probability": 0.098}

    def test_probabilities_are_floats_and_codes_strings(self, detectors, decoded, audio, monkeypatch):
        monkeypatch.setattr(FakeDetector, "detect_language", lambda self, audio: [(np.str_("de"), np.float32(0.5))])

        detection = Transcriber("fake-detector").detect_language(audio)

        assert type(detection["probability"]) is float and type(detection["language"]) is str
        json.dumps(detection)  # what --detect-language -f json writes

    @pytest.mark.parametrize("top", [0, -1, 1.5, True, "2"])
    def test_a_bad_top_is_refused_before_decoding(self, detectors, decoded, audio, top):
        with pytest.raises(ValueError, match="top must be a whole number from 1 up, or None"):
            Transcriber("fake-detector").detect_language(audio, top=top)
        assert decoded == []

    def test_bytes(self, detectors, decoded):
        data = b"ID3" + bytes(100)

        assert Transcriber("fake-detector").detect_language(data)["language"] == "de"
        assert decoded == [(data, 30.0)]

    def test_a_file_object(self, detectors, decoded):
        file = io.BytesIO(b"OggS" + bytes(100))

        Transcriber("fake-detector").detect_language(file)

        [(source, seconds)] = decoded
        assert seconds == 30.0 and source.read() == b"OggS" + bytes(100)  # its start, read already, comes again

    @pytest.mark.parametrize("make", [bytes, io.BytesIO], ids=["bytes", "file object"])
    def test_an_mp4_with_its_index_last_goes_through_a_temporary_file(self, detectors, decoded, make):
        Transcriber("fake-detector").detect_language(make(MP4_INDEX_LAST))

        [(path, seconds)] = decoded
        assert isinstance(path, pathlib.Path) and path.suffix == ".mp4" and seconds == 30.0
        assert not path.exists()  # deleted afterwards

    @pytest.mark.parametrize("seconds", [45, 10])
    def test_samples_are_cut_to_30_seconds_not_decoded(self, detectors, decoded, seconds):
        samples = np.full(seconds * RATE, 0.2, np.float32)

        Transcriber("fake-detector").detect_language(samples)

        assert decoded == []
        [given] = FakeDetector.received
        assert len(given) == min(seconds * RATE, WINDOW) and given.flags["C_CONTIGUOUS"]

    def test_input_it_cant_take(self, detectors, decoded):
        with pytest.raises(TypeError, match="audio must be a path, bytes"):
            Transcriber("fake-detector").detect_language(42)
        with pytest.raises(ValueError, match="1-D float32"):
            Transcriber("fake-detector").detect_language(np.zeros((2, RATE), np.float32))
        assert decoded == []

    def test_a_missing_file(self, detectors, decoded, tmp_path):
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            Transcriber("fake-detector").detect_language(tmp_path / "missing.mp3")
        assert decoded == []

    def test_no_sound_at_all(self, detectors, monkeypatch, audio):
        monkeypatch.setattr("speech_toolkit.api.load_audio", lambda source, max_seconds: np.zeros(0, np.float32))

        with pytest.raises(AudioDecodeError, match="talk.mp3 holds no sound to detect the language of"):
            Transcriber("fake-detector").detect_language(audio)
        assert FakeDetector.received == []

    def test_a_backend_that_finds_no_language(self, detectors, decoded, monkeypatch):
        monkeypatch.setattr(FakeDetector, "detect_language", lambda self, audio: [])

        with pytest.raises(RuntimeError, match="the fake-detector backend found no language in the audio"):
            Transcriber("fake-detector").detect_language(b"data")

    def test_a_backend_without_waveform_gets_a_temporary_wav_file(self, detectors):
        samples = np.full(2 * RATE, 0.1, np.float32)

        Transcriber("fake-file-detector").detect_language(samples)

        [(path, frames)] = FakeDetector.received
        assert path.suffix == ".wav" and frames == 2 * RATE
        assert not path.exists()

    @pytest.mark.parametrize("backend", ["fake", "fake-method-only"])
    def test_a_backend_that_cant_detect_is_refused_before_decoding(self, detectors, decoded, audio, backend):
        transcriber = Transcriber(backend, "small")
        with pytest.raises(UnsupportedOptionError, match=f"the {backend} backend can't detect the language on its own"):
            transcriber.detect_language(audio)
        assert decoded == []

    def test_logs_what_it_found(self, detectors, decoded, audio, caplog):
        with caplog.at_level("DEBUG", logger="speech_toolkit"):
            Transcriber("fake-detector").detect_language(audio)
        assert any("Detected de (0.97) in " in message and "talk.mp3" in message for message in caplog.messages)


class TestWhichBackends:
    def test_whisper_and_faster_whisper_detect(self):
        assert language_detectors() == ["whisper", "faster-whisper"]

    @pytest.mark.parametrize("backend", ["whisper", "faster-whisper"])
    def test_check_passes(self, backend):
        check_language_detection(backend)  # nothing loads

    @pytest.mark.parametrize("backend", ["voxtral", "parakeet", "canary"])
    def test_check_names_the_backends_that_can(self, backend):
        message = f"the {backend} backend can't detect the language on its own; backends that can: whisper, faster"
        with pytest.raises(UnsupportedOptionError, match=message):
            check_language_detection(backend)

    def test_unknown_backend(self):
        with pytest.raises(BackendNotFoundError, match="Unknown backend"):
            check_language_detection("nope")

    def test_registered_detectors_count(self, detectors):
        assert language_detectors() == ["whisper", "faster-whisper", "fake-detector", "fake-file-detector"]


###############################################################################
# whisper
###############################################################################


@pytest.fixture
def whisper():
    """Stand-in for openai-whisper: its constants, and a multilingual model with 80 Mel bands on the CPU."""
    module = MagicMock()
    param = MagicMock()
    param.device = "cpu"
    model = module.load_model.return_value
    model.parameters.return_value = iter([param])
    model.is_multilingual = True
    model.dims.n_mels = 80
    module.audio = SimpleNamespace(CHUNK_LENGTH=30, N_SAMPLES=480000, N_FRAMES=3000)
    with patch.dict(sys.modules, {"whisper": module}):
        yield module


def whisper_backend(device=None) -> WhisperBackend:
    backend = WhisperBackend()
    backend.load_model("tiny", device=device)
    return backend


class TestWhisperDetection:
    def test_detects_as_transcribe_does(self, whisper):
        model = whisper.load_model.return_value
        model.detect_language.return_value = ("tokens", {"en": 0.1, "de": 0.85, "fr": 0.05})
        samples = np.zeros(45 * RATE, np.float32)

        found = whisper_backend().detect_language(samples)

        assert found == [("de", 0.85), ("en", 0.1), ("fr", 0.05)]
        mel = whisper.log_mel_spectrogram.call_args
        assert len(mel.args[0]) == WINDOW and mel.args[1] == 80 and mel.kwargs == {"padding": 480000}
        whisper.pad_or_trim.assert_called_once_with(whisper.log_mel_spectrogram.return_value, 3000)
        whisper.pad_or_trim.return_value.to.assert_called_once_with(model.device)
        window = whisper.pad_or_trim.return_value.to.return_value
        model.detect_language.assert_called_once_with(window)  # float32 on the CPU
        window.half.assert_not_called()

    def test_float16_on_a_gpu(self, whisper):
        model = whisper.load_model.return_value
        model.detect_language.return_value = ("tokens", {"en": 1.0})

        whisper_backend("cuda").detect_language(np.zeros(RATE, np.float32))

        window = whisper.pad_or_trim.return_value.to.return_value
        model.detect_language.assert_called_once_with(window.half.return_value)

    def test_a_path_is_decoded_only_30_seconds_far(self, whisper, audio):
        model = whisper.load_model.return_value
        model.detect_language.return_value = ("tokens", {"en": 0.9, "de": 0.1})
        samples = np.zeros(RATE, np.float32)

        with patch("speech_toolkit.media.load_audio", return_value=samples) as load_audio:
            found = whisper_backend().detect_language(audio)

        load_audio.assert_called_once_with(audio, max_seconds=30)
        assert whisper.log_mel_spectrogram.call_args.args[0] is samples
        assert found == [("en", 0.9), ("de", 0.1)]

    def test_an_english_only_model(self, whisper):
        whisper.load_model.return_value.is_multilingual = False

        assert whisper_backend().detect_language(np.zeros(RATE, np.float32)) == [("en", 1.0)]
        whisper.log_mel_spectrogram.assert_not_called()

    def test_without_a_model(self):
        with pytest.raises(ModelLoadError, match="No model loaded"):
            WhisperBackend().detect_language(np.zeros(RATE, np.float32))

    def test_a_missing_file(self, whisper, tmp_path):
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            whisper_backend().detect_language(tmp_path / "missing.mp3")


def whisper_transcribing(module: MagicMock, probabilities: Any) -> MagicMock:
    """The mocked model, whose transcribe() detects the language as openai-whisper's does: with detect_language(),
    keeping only the most likely language."""
    model = module.load_model.return_value
    model.detect_language.return_value = ("tokens", probabilities)

    def transcribe(audio, **options):
        language = options.get("language")
        if language is None:
            _, found = model.detect_language("mel")
            language = max(found, key=found.get) if isinstance(found, dict) else "en"
        return {"text": " Hallo.", "segments": [], "language": language}

    model.transcribe.side_effect = transcribe
    return model


class TestWhisperLanguageProbability:
    def test_the_probability_of_the_language_it_detected(self, whisper, audio):
        whisper_transcribing(whisper, {"en": 0.07, "de": 0.9})

        result = whisper_backend().transcribe(audio)

        assert (result.language, result.language_probability) == ("de", 0.9)
        assert result.to_dict()["language_probability"] == 0.9

    def test_one_for_a_language_given(self, whisper, audio):
        detect = whisper_transcribing(whisper, {"en": 0.07, "de": 0.9}).detect_language

        result = whisper_backend().transcribe(audio, language="fr")

        assert (result.language, result.language_probability) == ("fr", 1.0)
        detect.assert_not_called()

    def test_one_for_an_english_only_model(self, whisper, audio):
        whisper.load_model.return_value.is_multilingual = False
        model = whisper_transcribing(whisper, {"en": 0.5})
        model.transcribe.side_effect = None
        model.transcribe.return_value = {"text": " Hi.", "segments": [], "language": "en"}

        assert whisper_backend().transcribe(audio).language_probability == 1.0

    def test_none_when_the_model_detected_nothing(self, whisper, audio):
        model = whisper.load_model.return_value
        model.transcribe.return_value = {"text": " Hi.", "segments": [], "language": "en"}

        assert whisper_backend().transcribe(audio).language_probability is None

    def test_a_batch_of_recordings_is_not_recorded(self, whisper, audio):
        whisper_transcribing(whisper, [{"en": 0.9}, {"de": 0.9}])  # what detect_language() gives for several

        assert whisper_backend().transcribe(audio).language_probability is None

    def test_each_transcription_has_its_own(self, whisper, audio):
        model = whisper_transcribing(whisper, {"en": 0.07, "de": 0.9})
        backend = whisper_backend()
        original = model.detect_language

        first = backend.transcribe(audio)
        recorder = model.detect_language
        model.transcribe.side_effect = None
        model.transcribe.return_value = {"text": " Hi.", "segments": [], "language": "en"}  # no detection this time
        second = backend.transcribe(audio)

        assert (first.language_probability, second.language_probability) == (0.9, None)
        assert model.detect_language is recorder is not original  # put in place once
        assert original.call_count == 1

    def test_calls_pass_through_unchanged(self, whisper, audio):
        model = whisper_transcribing(whisper, {"en": 0.07, "de": 0.9})
        backend = whisper_backend()
        original = model.detect_language
        backend.transcribe(audio)

        assert model.detect_language("mel", tokenizer="t") == ("tokens", {"en": 0.07, "de": 0.9})
        original.assert_called_with("mel", tokenizer="t")

    def test_detect_language_works_after_transcribing(self, whisper, audio):
        whisper_transcribing(whisper, {"en": 0.07, "de": 0.9})
        backend = whisper_backend()
        backend.transcribe(audio)

        assert backend.detect_language(np.zeros(RATE, np.float32)) == [("de", 0.9), ("en", 0.07)]


###############################################################################
# faster-whisper
###############################################################################


@pytest.fixture
def faster_whisper():
    """Stand-ins for faster_whisper and ctranslate2 (no GPU), with a multilingual model."""
    module, ctranslate2 = MagicMock(), MagicMock()
    ctranslate2.get_cuda_device_count.return_value = 0
    model = module.WhisperModel.return_value
    model.model.is_multilingual = True
    model.feature_extractor.chunk_length = 30
    model.detect_language.return_value = ("de", 0.9, [("de", np.float32(0.875)), ("en", 0.0625)])
    with patch.dict(sys.modules, {"faster_whisper": module, "ctranslate2": ctranslate2}):
        yield module


def faster_whisper_backend() -> FasterWhisperBackend:
    backend = FasterWhisperBackend()
    backend.load_model("tiny")
    return backend


class TestFasterWhisperDetection:
    def test_uses_its_detect_language(self, faster_whisper):
        samples = np.zeros(45 * RATE, np.float32)

        found = faster_whisper_backend().detect_language(samples)

        assert found == [("de", 0.875), ("en", 0.0625)] and all(type(chance) is float for _, chance in found)
        faster_whisper.WhisperModel.return_value.detect_language.assert_called_once_with(samples)  # it cuts 30 s

    def test_a_path_is_decoded_only_30_seconds_far(self, faster_whisper, audio):
        samples = np.zeros(RATE, np.float32)

        with patch("speech_toolkit.media.load_audio", return_value=samples) as load_audio:
            faster_whisper_backend().detect_language(audio)

        load_audio.assert_called_once_with(audio, max_seconds=30)
        faster_whisper.WhisperModel.return_value.detect_language.assert_called_once_with(samples)

    def test_an_english_only_model(self, faster_whisper):
        model = faster_whisper.WhisperModel.return_value
        model.model.is_multilingual = False

        assert faster_whisper_backend().detect_language(np.zeros(RATE, np.float32)) == [("en", 1.0)]
        model.detect_language.assert_not_called()

    def test_without_a_model(self):
        with pytest.raises(ModelLoadError, match="No model loaded"):
            FasterWhisperBackend().detect_language(np.zeros(RATE, np.float32))

    def test_a_missing_file(self, faster_whisper, tmp_path):
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            faster_whisper_backend().detect_language(tmp_path / "missing.mp3")

    def test_through_the_transcriber(self, faster_whisper, decoded, audio):
        detection = Transcriber("faster-whisper", "tiny").detect_language(audio, top=1)

        assert detection == {
            "language": "de",
            "probability": 0.875,
            "candidates": [{"language": "de", "probability": 0.875}],
        }
        assert decoded == [(audio, 30.0)]
        faster_whisper.WhisperModel.return_value.transcribe.assert_not_called()


###############################################################################
# transcribe --detect-language
###############################################################################


def media_files(folder: pathlib.Path, *names: str) -> List[pathlib.Path]:
    paths = [folder / name for name in names]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return paths


class TestCommand:
    def test_a_line_per_file_on_stdout(self, detectors, decoded, tmp_path, capsys):
        first, second = media_files(tmp_path, "a.mp3", "b.wav")

        main(["--detect-language", str(first), str(second), "-b", "fake-detector"])

        out, err = capsys.readouterr()
        assert out == f"{first}\tde\t0.97\n{second}\tde\t0.97\n"
        assert "Loading fake-detector model 'small'..." in err
        assert "library noise on stdout" in err  # the backend's prints go to stderr
        assert [seconds for _, seconds in decoded] == [30.0, 30.0]
        assert FakeDetector.loaded == 1  # once for every file

    @pytest.mark.parametrize("fmt", [[], ["-f", "txt"]])
    def test_txt_is_the_default(self, detectors, decoded, audio, capsys, fmt):
        main(["--detect-language", str(audio), "-b", "fake-detector", *fmt])
        assert capsys.readouterr().out == f"{audio}\tde\t0.97\n"

    def test_json_lines(self, detectors, decoded, tmp_path, capsys):
        paths = media_files(tmp_path, "a.mp3", "b.mp3")

        main(["--detect-language", *map(str, paths), "-b", "fake-detector", "-f", "json"])

        lines = capsys.readouterr().out.splitlines()
        assert [json.loads(line) for line in lines] == [
            {"file": str(path), "language": "de", "probability": 0.97, "candidates": RANKED} for path in paths
        ]

    def test_folders_are_searched_for_media_files(self, detectors, decoded, tmp_path, capsys):
        found = media_files(tmp_path / "recordings", "a.mp3", "sub/b.flac")
        media_files(tmp_path / "recordings", "notes.txt", ".hidden.mp3")

        main(["--detect-language", str(tmp_path / "recordings"), "-b", "fake-detector", "-q"])

        out, err = capsys.readouterr()
        assert out == "".join(f"{path}\tde\t0.97\n" for path in found)
        assert "Loading" not in err  # -q

    def test_standard_input(self, detectors, decoded, monkeypatch, capsys):
        stdin = SimpleNamespace(buffer=io.BytesIO(b"OggS" + bytes(100)), isatty=lambda: False)
        monkeypatch.setattr("sys.stdin", stdin)

        main(["--detect-language", "-", "-b", "fake-detector"])

        assert capsys.readouterr().out == "-\tde\t0.97\n"
        [(source, seconds)] = decoded
        assert seconds == 30.0 and source.read() == b"OggS" + bytes(100)

    def test_the_default_backend(self, faster_whisper, decoded, audio, capsys):
        main(["--detect-language", str(audio), "-m", "tiny"])

        out, err = capsys.readouterr()
        assert out == f"{audio}\tde\t0.88\n"
        assert "Loading faster-whisper model 'tiny'..." in err
        faster_whisper.WhisperModel.return_value.transcribe.assert_not_called()

    @pytest.mark.parametrize(
        "options, flags",
        [
            (["-l", "de"], "--language"),
            (["-t", "translate"], "--task"),
            (["--prompt", "Grafana"], "--prompt"),
            (["--vad"], "--vad"),
            (["--word-timestamps"], "--word-timestamps"),
            (["--diarize", "--num-speakers", "2"], "--diarize, --num-speakers"),
            (["--min-speakers", "1", "--max-speakers", "3"], "--min-speakers, --max-speakers"),
            (["-o", "langs.txt"], "--output"),
            (["--outdir", "out"], "--outdir"),
            (["--json", "talk.json"], "--json"),
            (["--skip-existing"], "--skip-existing"),
            (["--from-json"], "--from-json"),
            (["--max-line-width", "42"], "--max-line-width"),
            (["--max-line-width", "42", "--max-lines", "3"], "--max-line-width, --max-lines"),
            (["--max-cue-duration", "6"], "--max-cue-duration"),
            (["--speaker-names", "Alice,Bob"], "--speaker-names"),
            (["--timestamps"], "--timestamps"),
            (["--word-level"], "--word-level"),
        ],
    )
    def test_refuses_the_transcription_options(self, detectors, audio, capsys, options, flags):
        with pytest.raises(SystemExit) as exit_info:
            main(["--detect-language", str(audio), "-b", "fake-detector", *options])

        assert exit_info.value.code == 2
        assert f"--detect-language only detects the language, so it doesn't take {flags}\n" in capsys.readouterr().err
        assert FakeDetector.loaded == 0

    @pytest.mark.parametrize("fmt", ["srt", "txt,json", "all"])
    def test_formats_other_than_txt_and_json(self, detectors, audio, capsys, fmt):
        with pytest.raises(SystemExit):
            main(["--detect-language", str(audio), "-f", fmt])
        assert "-f takes txt or json" in capsys.readouterr().err

    def test_needs_an_input(self, capsys):
        with pytest.raises(SystemExit):
            parse_args(["--detect-language"])
        assert "the following arguments are required: audio" in capsys.readouterr().err

    def test_list_backends_wins(self, capsys):
        main(["--list-backends", "--detect-language", "--diarize"])
        assert capsys.readouterr().out.startswith("Available transcription backends:")

    @pytest.mark.parametrize("backend", ["fake", "voxtral"])
    def test_a_backend_that_cant_detect_fails_before_loading(self, detectors, decoded, audio, backend):
        message = f"Error: the {backend} backend can't detect the language on its own; backends that can: whisper"
        with pytest.raises(SystemExit, match=message):
            main(["--detect-language", str(audio), "-b", backend])
        assert FakeBackend.loaded == 0 and decoded == []

    def test_an_unknown_backend(self, audio):
        with pytest.raises(SystemExit, match="Error: Unknown backend: 'nope'"):
            main(["--detect-language", str(audio), "-b", "nope"])

    def test_a_missing_file_fails_before_loading(self, detectors, tmp_path):
        with pytest.raises(SystemExit, match="Error: file not found: .*missing.mp3"):
            main(["--detect-language", str(tmp_path / "missing.mp3"), "-b", "fake-detector"])
        assert FakeDetector.loaded == 0

    def test_a_folder_without_audio(self, detectors, tmp_path):
        with pytest.raises(SystemExit, match="Error: no audio files found."):
            main(["--detect-language", str(tmp_path), "-b", "fake-detector"])

    def test_a_file_that_fails_doesnt_stop_the_others(self, detectors, decoded, tmp_path, capsys):
        bad, good = media_files(tmp_path, "bad.mp3", "good.mp3")

        with pytest.raises(SystemExit, match="Error: 1 of 2 files failed."):
            main(["--detect-language", str(bad), str(good), "-b", "fake-detector"])

        out, err = capsys.readouterr()
        assert out == f"{good}\tde\t0.97\n"
        assert f"Error: {bad}: Failed to load audio: Invalid data found when processing input\n" in err

    def test_a_single_file_that_fails(self, detectors, decoded, tmp_path, capsys):
        [bad] = media_files(tmp_path, "bad.mp3")

        with pytest.raises(SystemExit) as exit_info:
            main(["--detect-language", str(bad), "-b", "fake-detector"])

        assert exit_info.value.code == 1
        assert capsys.readouterr().out == ""

    def test_other_errors_show_their_type(self, detectors, decoded, audio, monkeypatch, capsys):
        def fail(self, audio):
            raise KeyError("probabilities")

        monkeypatch.setattr(FakeDetector, "detect_language", fail)

        with pytest.raises(SystemExit):
            main(["--detect-language", str(audio), "-b", "fake-detector"])

        assert f"Error: {audio}: KeyError: 'probabilities'\n" in capsys.readouterr().err

    def test_ctrl_c(self, detectors, decoded, audio, monkeypatch):
        def interrupt(self, audio):
            raise KeyboardInterrupt

        monkeypatch.setattr(FakeDetector, "detect_language", interrupt)

        with pytest.raises(SystemExit) as exit_info:
            main(["--detect-language", str(audio), "-b", "fake-detector"])
        assert exit_info.value.code == 130

    @pytest.mark.filterwarnings("default:careful:UserWarning")
    def test_warnings_are_one_line(self, detectors, decoded, audio, monkeypatch, capsys):
        def warn(self, audio):
            import warnings

            from speech_toolkit import SpeechToolkitWarning

            warnings.warn("careful", SpeechToolkitWarning)
            return list(FOUND)

        monkeypatch.setattr(FakeDetector, "detect_language", warn)

        main(["--detect-language", str(audio), "-b", "fake-detector"])

        assert "Warning: careful\n" in capsys.readouterr().err

    def test_help_names_the_backends(self, capsys):
        with pytest.raises(SystemExit):
            parse_args(["--help"])
        help_text = " ".join(capsys.readouterr().out.split())
        assert "--detect-language Print the language of each input, from its first 30 s" in help_text
        assert "(whisper, faster-whisper)" in help_text
