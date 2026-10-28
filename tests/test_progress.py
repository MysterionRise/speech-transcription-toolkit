"""Progress: Transcriber.transcribe(progress=...) gets a ProgressEvent as each stage advances, with every segment, in
order; faster-whisper's as they are decoded. An exception the callback raises cancels the transcription.

Also the messages the built-in backends log as they go (they printed them before), and the name their truncation
warning gives the recording when they get its samples. Model libraries are mocked with patch.dict(sys.modules, ...).
"""

from __future__ import annotations

import dataclasses
import io
import itertools
import logging
import sys
from types import SimpleNamespace
from typing import Any, Iterator, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from speech_toolkit import ProgressEvent, SpeechToolkitWarning, Transcriber, TranscriptionResult, transcribe
from speech_toolkit.backends import CanaryBackend, ParakeetBackend, VoxtralBackend, report_progress
from speech_toolkit.backends.base import _CALL, STAGES, _Call, recording_name
from tests.conftest import FakeWaveformBackend

RATE = 16000
ONE_SECOND = np.zeros(RATE, np.float32)


def check(events: List[ProgressEvent], result: TranscriptionResult, stages: List[str]) -> None:
    """The promises of ProgressEvent: the *stages* in their order, each once; within a stage, done never decreases
    and never exceeds a known total, and the last event has done == total; every segment of the result is reported
    once, in order, in the "transcribe" stage."""
    assert [stage for stage, _ in itertools.groupby(event.stage for event in events)] == stages
    assert stages == [stage for stage in STAGES if stage in stages]
    for stage in stages:
        ours = [event for event in events if event.stage == stage]
        assert [event.done for event in ours] == sorted(event.done for event in ours), ours
        assert all(event.total is None or event.done <= event.total for event in ours), ours
        assert ours[-1].done == ours[-1].total, ours
    reported = [event for event in events if event.segment is not None]
    assert {event.stage for event in reported} <= {"transcribe"}
    assert [event.segment for event in reported] == result.segments
    assert all(event.segment is segment for event, segment in zip(reported, result.segments))


def segments(events: List[ProgressEvent]) -> List[Any]:
    return [event.segment for event in events if event.segment is not None]


###############################################################################
# Fake backends: the events the Transcriber reports itself
###############################################################################


class TestStages:
    def test_a_path_to_a_backend_that_reports_nothing(self, fake_backend, tmp_path):
        """No decoding here, and an unknown duration: the transcribe stage ends as far as its segments go."""
        audio = tmp_path / "talk.wav"
        audio.touch()
        events: List[ProgressEvent] = []

        result = Transcriber("fake", "small").transcribe(audio, progress=events.append)

        segment = result.segments[0]
        assert events == [
            ProgressEvent("transcribe", 0.0, None),
            ProgressEvent("transcribe", 1.5, None, segment),
            ProgressEvent("transcribe", 1.5, 1.5),
        ]
        check(events, result, ["transcribe"])

    def test_in_memory_input_is_decoded_first(self, fake_waveform_backend, monkeypatch):
        monkeypatch.setattr("speech_toolkit.api.load_audio", lambda audio: ONE_SECOND)
        events: List[ProgressEvent] = []

        result = Transcriber("fake-waveform").transcribe(b"RIFF....WAVE", progress=events.append)

        assert events == [
            ProgressEvent("decode", 0.0, None),
            ProgressEvent("decode", 1.0, 1.0),
            ProgressEvent("transcribe", 0.0, 1.0),
            ProgressEvent("transcribe", 1.0, 1.0, result.segments[0]),
            ProgressEvent("transcribe", 1.0, 1.0),
        ]
        check(events, result, ["decode", "transcribe"])

    def test_samples_need_no_decoding(self, fake_waveform_backend):
        events: List[ProgressEvent] = []
        result = Transcriber("fake-waveform").transcribe(ONE_SECOND, progress=events.append)
        check(events, result, ["transcribe"])

    def test_with_speaker_labels(self, fake_waveform_backend, diarization, tmp_path):
        audio = tmp_path / "talk.wav"
        audio.touch()
        events: List[ProgressEvent] = []

        result = Transcriber("fake-waveform", diarize=True).transcribe(audio, progress=events.append)

        assert [event for event in events if event.stage == "diarize"] == [
            ProgressEvent("diarize", 0.0, 1.0),
            ProgressEvent("diarize", 1.0, 1.0),
        ]
        check(events, result, ["decode", "transcribe", "diarize"])

    def test_speaker_labels_for_a_backend_that_takes_files(self, fake_backend, diarization, tmp_path):
        """diarize_audio() decodes the file itself, so there is no decode stage."""
        audio = tmp_path / "talk.wav"
        audio.touch()
        events: List[ProgressEvent] = []

        result = Transcriber("fake", "small", diarize=True).transcribe(audio, progress=events.append)

        check(events, result, ["transcribe", "diarize"])
        assert events[-1] == ProgressEvent("diarize", 1.0, 1.0)

    def test_too_short_to_label_speakers(self, fake_waveform_backend, diarization):
        events: List[ProgressEvent] = []
        result = Transcriber("fake-waveform", diarize=True).transcribe(ONE_SECOND[:100], progress=events.append)

        diarization.pipeline.assert_not_called()
        assert events[-2:] == [ProgressEvent("diarize", 0.0, 0.00625), ProgressEvent("diarize", 0.00625, 0.00625)]
        check(events, result, ["transcribe", "diarize"])

    def test_one_shot_transcribe(self, fake_waveform_backend):
        events: List[ProgressEvent] = []
        result = transcribe(ONE_SECOND, backend="fake-waveform", progress=events.append)
        check(events, result, ["transcribe"])

    def test_without_a_callback(self, fake_waveform_backend):
        assert Transcriber("fake-waveform").transcribe(ONE_SECOND).text  # no callback, no error


@pytest.fixture
def diarization(monkeypatch):
    """A mocked pyannote pipeline and torch; diarize_audio() runs, and paths decode to one second of silence."""
    torch, pipeline = MagicMock(), MagicMock()
    turn = SimpleNamespace(start=0.0, end=1.0)
    pipeline.return_value.exclusive_speaker_diarization.itertracks.return_value = [(turn, "A", "SPEAKER_00")]
    monkeypatch.setattr("speech_toolkit.api.load_diarization_pipeline", lambda device: pipeline)
    monkeypatch.setattr("speech_toolkit.api.load_audio", lambda audio: ONE_SECOND)
    monkeypatch.setattr("speech_toolkit.diarization.load_audio", lambda audio: ONE_SECOND)
    with patch.dict(sys.modules, {"torch": torch}):
        yield SimpleNamespace(torch=torch, pipeline=pipeline)


###############################################################################
# Cancelling
###############################################################################


class Cancelled(Exception):
    pass


def cancel_at(stage: str):
    def callback(event: ProgressEvent) -> None:
        if event.stage == stage:
            raise Cancelled(stage)

    return callback


class TestCancelling:
    def test_while_decoding(self, fake_waveform_backend, monkeypatch):
        monkeypatch.setattr("speech_toolkit.api.load_audio", lambda audio: ONE_SECOND)

        with pytest.raises(Cancelled):
            Transcriber("fake-waveform").transcribe(b"RIFF....WAVE", progress=cancel_at("decode"))

        assert FakeWaveformBackend.received == []
        assert _CALL.get() is None  # the call's context ends with it

    def test_while_labelling_speakers(self, fake_waveform_backend, diarization):
        with pytest.raises(Cancelled):
            Transcriber("fake-waveform", diarize=True).transcribe(ONE_SECOND, progress=cancel_at("diarize"))
        diarization.pipeline.assert_not_called()


###############################################################################
# Every built-in backend, with its model library mocked
###############################################################################


def whisper_segment(start: float, end: float, text: str) -> SimpleNamespace:
    """A faster-whisper Segment."""
    return SimpleNamespace(
        start=start,
        end=end,
        text=text,
        tokens=[1],
        temperature=0.0,
        avg_logprob=-0.1,
        compression_ratio=1.0,
        no_speech_prob=0.0,
    )


@pytest.fixture
def faster_whisper():
    """faster-whisper hearing three segments of 70 s of audio; ``log`` records each segment its generator yields."""
    faster_whisper, ctranslate2 = MagicMock(), MagicMock()
    ctranslate2.get_cuda_device_count.return_value = 0
    model = faster_whisper.WhisperModel.return_value
    log: List[Any] = []

    def decode() -> Iterator[SimpleNamespace]:
        for number, (start, end) in enumerate([(0.0, 20.0), (20.0, 45.0), (45.0, 70.2)]):
            log.append(("decoded", number))
            yield whisper_segment(start, end, f" Part {number}.")

    info = SimpleNamespace(language="en", duration=70.0, language_probability=0.9)
    model.transcribe.side_effect = lambda audio, **kwargs: (decode(), info)
    with patch.dict(sys.modules, {"faster_whisper": faster_whisper, "ctranslate2": ctranslate2}):
        yield SimpleNamespace(model=model, log=log)


@pytest.fixture
def whisper():
    module = MagicMock()
    model = module.load_model.return_value
    model.parameters.return_value = iter([SimpleNamespace(device="cpu")])
    model.transcribe.return_value = {
        "text": " One. Two.",
        "segments": [
            {"id": 0, "start": 0.0, "end": 30.0, "text": " One."},
            {"id": 1, "start": 30.0, "end": 70.4, "text": " Two."},  # past the end of the audio, as can happen
        ],
        "language": "en",
    }
    with patch.dict(sys.modules, {"whisper": module}):
        yield model


class FakeBatch(dict):
    """A processor's output: the number of chunks in the batch (its "count"), and Voxtral's prompt length."""

    def __init__(self, count: int = 1) -> None:
        super().__init__(count=count)
        self.input_ids = SimpleNamespace(shape=(count, 0))

    def to(self, *args: Any, **kwargs: Any) -> "FakeBatch":
        return self


@pytest.fixture
def transformers():
    """Mocked torch and transformers: Parakeet hears " Hi." at the start of each chunk, Canary and Voxtral "Hi.".

    ``limit`` sets the tokens Canary and Voxtral generate per chunk: 2, or 3 to reach a limit of 3 tokens.
    """
    torch, transformers = MagicMock(), MagicMock()
    torch.cuda.is_available.return_value = False
    state = SimpleNamespace(tokens=2)
    processor = transformers.AutoProcessor.from_pretrained.return_value
    processor.return_value = FakeBatch()
    processor.decode.return_value = ("Hi.", [[{"token": " Hi.", "start": 0.0, "end": 0.5}]])
    processor.apply_transcription_request.side_effect = lambda audio, **kwargs: FakeBatch(len(audio))
    processor.batch_decode.side_effect = lambda generated, **kwargs: ["Hi."] * len(generated)
    for name in ("CanaryForConditionalGeneration", "VoxtralForConditionalGeneration"):
        model = getattr(transformers, name).from_pretrained.return_value.to.return_value
        model.generation_config = SimpleNamespace(eos_token_id=2, pad_token_id=0)
        model.generate.side_effect = lambda count=1, **kwargs: np.full((count, state.tokens), 5)
    with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
        yield state


# backend: (its mocked library, model, language, seconds of audio)
BACKENDS = {
    "whisper": ("whisper", "tiny", None, 70),
    "faster-whisper": ("faster_whisper", "tiny", None, 70),
    "parakeet": ("transformers", None, None, 130),  # two chunks of up to 120 s
    "canary": ("transformers", None, "en", 70),  # three chunks of up to 30 s
    "voxtral": ("transformers", None, None, 70),
}


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_every_backend(request, backend):
    library, model, language, seconds = BACKENDS[backend]
    request.getfixturevalue(library)
    events: List[ProgressEvent] = []

    with patch.object(CanaryBackend, "BATCH_SIZE", 2), patch("speech_toolkit.backends.voxtral_backend.BATCH_SIZE", 2):
        result = Transcriber(backend, model).transcribe(
            np.zeros(seconds * RATE, np.float32), language, progress=events.append
        )

    check(events, result, ["transcribe"])
    assert len(result.segments) >= 2
    assert events[-1] == ProgressEvent("transcribe", seconds, seconds)


def test_faster_whisper_reports_each_segment_as_it_is_decoded(faster_whisper):
    log = faster_whisper.log

    Transcriber("faster-whisper", "tiny").transcribe(np.zeros(70 * RATE, np.float32), progress=log.append)

    # Events as (done, total, the segment's text), between what the generator yielded.
    reports = [
        (item.done, item.total, item.segment and item.segment["text"]) if isinstance(item, ProgressEvent) else item
        for item in log
    ]
    assert reports == [
        (0.0, 70.0, None),
        ("decoded", 0),
        (20.0, 70.0, " Part 0."),
        ("decoded", 1),
        (45.0, 70.0, " Part 1."),
        ("decoded", 2),
        (70.0, 70.0, " Part 2."),  # it ends at 70.2 s, past the audio's end: done stops at the total
        (70.0, 70.0, None),
    ]


def test_cancelling_stops_faster_whisper_decoding(faster_whisper):
    def stop_after_the_first_segment(event: ProgressEvent) -> None:
        if event.segment is not None:
            raise Cancelled()

    with pytest.raises(Cancelled):
        Transcriber("faster-whisper", "tiny").transcribe(ONE_SECOND, progress=stop_after_the_first_segment)

    assert faster_whisper.log == [("decoded", 0)]


@pytest.mark.parametrize("backend", ["canary", "voxtral"])
def test_segments_are_reported_after_each_batch(transformers, backend):
    batches: List[int] = []  # how many segments were reported when each batch was transcribed
    processor = sys.modules["transformers"].AutoProcessor.from_pretrained.return_value
    make_batch = processor.apply_transcription_request.side_effect
    events: List[ProgressEvent] = []

    def apply_transcription_request(audio, **kwargs):
        batches.append(len(segments(events)))
        return make_batch(audio, **kwargs)

    processor.apply_transcription_request.side_effect = apply_transcription_request
    with patch.object(CanaryBackend, "BATCH_SIZE", 2), patch("speech_toolkit.backends.voxtral_backend.BATCH_SIZE", 2):
        Transcriber(backend).transcribe(np.zeros(70 * RATE, np.float32), "en", progress=events.append)

    assert batches == [0, 2]  # the first batch's two segments were reported before the second batch ran


###############################################################################
# The built-in backends log their progress, and never print it
###############################################################################


@pytest.mark.parametrize(
    "backend_class, message",
    [
        (ParakeetBackend, "Parakeet: transcribed 1/2 chunks"),  # two chunks of up to 120 s
        (CanaryBackend, "Canary: transcribed 5/5 chunks"),  # 130 s: five chunks of up to 30 s, one batch
        (VoxtralBackend, "Voxtral: transcribed 5/5 chunks"),
    ],
)
@pytest.mark.parametrize("verbose, level", [(True, logging.INFO), (False, logging.DEBUG)])
def test_chunk_backends_log_their_progress(transformers, capsys, caplog, backend_class, message, verbose, level):
    """Before 0.5.0 they printed it to stderr with verbose=True. Used on their own, report_progress() does nothing."""
    caplog.set_level(logging.DEBUG, logger="speech_toolkit")
    backend = backend_class()
    backend.load_model(backend.default_model())

    backend.transcribe(
        np.zeros(130 * RATE, np.float32), "en" if backend_class is CanaryBackend else None, verbose=verbose
    )

    module = backend_class.__module__
    assert (module, level, message) in [(record.name, record.levelno, record.getMessage()) for record in caplog.records]
    assert capsys.readouterr() == ("", "")


###############################################################################
# Truncation warnings name the recording, also when the backend gets its samples
###############################################################################


@pytest.fixture
def cut_off(transformers, monkeypatch):
    """Canary and Voxtral write 3 tokens in every chunk, their limit with warned_about()'s max_new_tokens=3; any audio
    decodes to 70 s of silence."""
    transformers.tokens = 3
    monkeypatch.setattr("speech_toolkit.api.load_audio", lambda audio: np.zeros(70 * RATE, np.float32))


def warned_about(backend: str, audio: Any, diarize: bool = False) -> str:
    """The recording the truncation warning names, when *backend* transcribes *audio*."""
    with pytest.warns(SpeechToolkitWarning, match="reached its limit of 3 tokens") as warned:
        Transcriber(backend, diarize=diarize, options={"max_new_tokens": 3}).transcribe(audio, "en")
    [warning] = [warning for warning in warned if "limit" in str(warning.message)]
    return str(warning.message).split(": ")[0]


@pytest.mark.parametrize("backend", ["canary", "voxtral"])
class TestRecordingNames:
    def test_a_path_with_speaker_labels(self, cut_off, diarization_pipeline, tmp_path, backend):
        """With speaker labels, the backend gets the samples the Transcriber decoded, but the warning names the file."""
        audio = tmp_path / "talk.wav"
        audio.touch()
        assert warned_about(backend, audio, diarize=True) == str(audio)

    def test_bytes(self, cut_off, backend):
        assert warned_about(backend, b"RIFF....WAVE") == "audio"

    def test_a_file_object_with_a_name(self, cut_off, tmp_path, backend):
        audio = tmp_path / "talk.ogg"
        audio.write_bytes(b"OggS....")
        with open(audio, "rb") as file:
            assert warned_about(backend, file) == str(audio)

    def test_a_file_object_without_one(self, cut_off, backend):
        assert warned_about(backend, io.BytesIO(b"OggS....")) == "audio"


@pytest.fixture
def diarization_pipeline(monkeypatch):
    """A pyannote pipeline that finds no speaker turns (torch is mocked already)."""
    monkeypatch.setattr("speech_toolkit.api.load_diarization_pipeline", lambda device: MagicMock())


def test_recording_name_outside_a_transcriber(tmp_path):
    assert recording_name(tmp_path / "talk.wav") == str(tmp_path / "talk.wav")
    assert recording_name(ONE_SECOND) == "audio"


###############################################################################
# The event and its bookkeeping
###############################################################################


def test_report_progress_outside_a_transcriber_does_nothing():
    report_progress(1.0, 2.0, {"start": 0.0, "end": 1.0, "text": " Hi."})


def test_events_are_frozen():
    event = ProgressEvent("transcribe", 1.0, 2.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.done = 3.0  # type: ignore[misc]


class TestMonotonic:
    """_Call keeps what a backend reports in order, however it reports it."""

    def report(self, *reports: Any) -> List[ProgressEvent]:
        events: List[ProgressEvent] = []
        call = _Call("talk.wav", events.append)
        for stage, done, total in reports:
            call.report(stage, done, total)
        return events

    def test_done_never_decreases(self):
        events = self.report(("transcribe", 5.0, 10.0), ("transcribe", 3.0, None))
        assert [event.done for event in events] == [5.0, 5.0]

    def test_done_stops_at_the_total(self):
        assert self.report(("transcribe", 12.0, 10.0))[-1] == ProgressEvent("transcribe", 10.0, 10.0)

    def test_the_total_is_kept(self):
        events = self.report(("transcribe", 0.0, 10.0), ("transcribe", 4.0, None))
        assert events[-1] == ProgressEvent("transcribe", 4.0, 10.0)

    def test_a_total_below_what_is_done(self):
        events = self.report(("transcribe", 8.0, None), ("transcribe", 9.0, 6.0))
        assert events[-1] == ProgressEvent("transcribe", 8.0, 8.0)

    def test_each_stage_starts_over(self):
        events = self.report(("decode", 10.0, 10.0), ("transcribe", 0.0, 10.0))
        assert events[-1] == ProgressEvent("transcribe", 0.0, 10.0)

    def test_without_a_callback(self):
        call = _Call(None, None)
        call.report("transcribe", 1.0, 2.0, {"text": " Hi."})
        assert (call.done, call.total, call.segments) == (1.0, 2.0, 1)


def test_waveform_fake_still_decodes_paths_itself(fake_waveform_backend, monkeypatch, tmp_path):
    """Without speaker labels a path goes to the backend: no decode stage, and the total comes at the end."""
    monkeypatch.setattr("speech_toolkit.media.load_audio", lambda audio: ONE_SECOND)
    audio = tmp_path / "talk.wav"
    audio.touch()
    events: List[ProgressEvent] = []

    result = Transcriber("fake-waveform").transcribe(audio, progress=events.append)

    check(events, result, ["transcribe"])
    assert events[0] == ProgressEvent("transcribe", 0.0, None)
