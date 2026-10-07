"""Merging speaker turns into transcript segments (smoothing, ids, words in pauses, speed), and the checks made before
diarizing: speaker-count hints, and audio too short to diarize."""

from __future__ import annotations

import pathlib
import random
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from speech_toolkit import Transcriber, UnsupportedOptionError, diarization, transcribe
from speech_toolkit.cli import parse_args
from speech_toolkit.diarization import (
    MIN_DIARIZATION_SECONDS,
    WHOLE_SEGMENT_FIELDS,
    _TurnIndex,
    check_speaker_hints,
    diarize_audio,
    merge_diarization,
)
from speech_toolkit.media import SAMPLE_RATE
from tests.conftest import FakeWordsBackend

Turn = Tuple[float, float, str]


def make_words(*spec: Tuple[str, float, float]) -> List[Dict[str, Any]]:
    """Word dicts in Whisper's format from ``(text, start, end)``: ``("Hi", 0.0, 0.4)`` is ``" Hi"``."""
    return [{"word": f" {text}", "start": start, "end": end} for text, start, end in spec]


def make_segment(words: List[Dict[str, Any]], **fields: Any) -> Dict[str, Any]:
    """A transcript segment of *words*, with any other *fields*."""
    text = "".join(w["word"] for w in words)
    return {**fields, "start": words[0]["start"], "end": words[-1]["end"], "text": text, "words": words}


def lines(speaker_segments: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """(speaker, text) of each speaker segment: the lines of the txt transcript."""
    return [(s["speaker"], s["text"]) for s in speaker_segments]


class TestSmoothing:
    """A speaker change too short to be real, between two stretches of one speaker, doesn't split a segment."""

    # "[A] I think / [B] we / [A] should go.": diarization gives one word to another speaker.
    FLIP = make_words(("I", 0.0, 0.2), ("think", 0.2, 0.5), ("we", 0.5, 0.7), ("should", 0.7, 1.0), ("go.", 1.0, 1.3))
    FLIP_TURNS = [(0.0, 0.5, "A"), (0.5, 0.7, "B"), (0.7, 1.3, "A")]

    def test_a_one_word_flip_no_longer_splits_the_segment(self):
        segment = make_segment(self.FLIP, id=7, tokens=[50364, 286, 519], avg_logprob=-0.2)

        result = merge_diarization({"segments": [segment]}, self.FLIP_TURNS)

        # One line, and the segment isn't split, so it keeps every field.
        assert result == [{**segment, "id": 0, "segment_id": 7, "speaker": "A"}]

    def test_without_smoothing_the_flip_splits_the_segment(self, monkeypatch):
        """The thresholds are module constants; at 0 every speaker change is kept."""
        monkeypatch.setattr(diarization, "MIN_RUN_WORDS", 0)
        monkeypatch.setattr(diarization, "MIN_RUN_SECONDS", 0.0)

        result = merge_diarization({"segments": [make_segment(self.FLIP)]}, self.FLIP_TURNS)

        assert lines(result) == [("A", " I think"), ("B", " we"), ("A", " should go.")]

    def test_a_run_of_several_words_shorter_than_the_minimum_is_smoothed(self):
        words = make_words(
            ("I", 0.0, 0.3), ("think", 0.3, 0.6), ("we", 0.6, 0.7), ("all", 0.7, 0.85), ("should", 0.85, 1.2)
        )
        turns = [(0.0, 0.6, "A"), (0.6, 0.85, "B"), (0.85, 1.5, "A")]  # B: two words in 0.25 s

        result = merge_diarization({"segments": [make_segment(words)]}, turns)

        assert lines(result) == [("A", " I think we all should")]

    def test_a_real_reply_is_kept(self):
        """Two words over 0.5 s or more make a speaker change of their own; so does a word at either end."""
        words = make_words(
            ("Are", 0.0, 0.2), ("you", 0.2, 0.4), ("coming?", 0.4, 0.9), ("Yes,", 1.0, 1.3), ("sure.", 1.3, 1.7)
        ) + make_words(("Great.", 1.9, 2.3))
        turns = [(0.0, 0.95, "A"), (0.95, 1.8, "B"), (1.8, 2.4, "A")]

        result = merge_diarization({"segments": [make_segment(words)]}, turns)

        assert lines(result) == [("A", " Are you coming?"), ("B", " Yes, sure."), ("A", " Great.")]

    def test_a_one_word_change_at_the_start_is_kept(self):
        words = make_words(("So", 0.0, 0.2), ("I", 0.3, 0.5), ("think", 0.5, 0.9))

        result = merge_diarization({"segments": [make_segment(words)]}, [(0.0, 0.25, "B"), (0.25, 1.0, "A")])

        assert lines(result) == [("B", " So"), ("A", " I think")]

    def test_the_speakers_around_the_run_must_agree(self):
        words = make_words(("One,", 0.0, 0.4), ("two,", 0.4, 0.6), ("three.", 0.6, 1.0))

        result = merge_diarization({"segments": [make_segment(words)]}, [(0, 0.4, "A"), (0.4, 0.6, "B"), (0.6, 1, "C")])

        assert lines(result) == [("A", " One,"), ("B", " two,"), ("C", " three.")]

    def test_every_flip_in_a_row_is_smoothed(self):
        words = make_words(*((f"w{i}", i * 0.3, i * 0.3 + 0.3) for i in range(5)))
        turns = [(i * 0.3, i * 0.3 + 0.3, "AB"[i % 2]) for i in range(5)]  # A B A B A, one word each

        result = merge_diarization({"segments": [make_segment(words)]}, turns)

        assert lines(result) == [("A", " w0 w1 w2 w3 w4")]


class TestSpeakerSegmentIds:
    """Speaker segments have unique ids, and pieces of a split segment don't keep its decoding details."""

    # One segment that splits ("Are you sure?" / "Yes.") and one that doesn't ("Good.").
    TURNS = [(0.0, 1.0, "A"), (1.1, 1.6, "B"), (1.6, 3.0, "A")]
    SPLIT = make_words(("Are", 0.0, 0.3), ("you", 0.3, 0.5), ("sure?", 0.5, 0.9), ("Yes.", 1.2, 1.5))
    WHOLE = make_words(("Good.", 2.0, 2.5))
    DECODING = {
        "seek": 0,
        "tokens": [50364, 2014, 291],
        "temperature": 0.0,
        "avg_logprob": -0.31,
        "compression_ratio": 1.1,
        "no_speech_prob": 0.02,
    }

    def test_ids_are_unique_and_segment_id_names_the_transcript_segment(self):
        segments = [make_segment(self.SPLIT, id=0), make_segment(self.WHOLE, id=1)]

        result = merge_diarization({"segments": segments}, self.TURNS)

        assert [(s["id"], s["segment_id"], s["speaker"], s["text"]) for s in result] == [
            (0, 0, "A", " Are you sure?"),
            (1, 0, "B", " Yes."),
            (2, 1, "A", " Good."),
        ]

    def test_segments_without_an_id_are_named_by_position(self):
        segments = [make_segment(self.SPLIT), make_segment(self.WHOLE)]

        result = merge_diarization({"segments": segments}, self.TURNS)

        assert [(s["id"], s["segment_id"]) for s in result] == [(0, 0), (1, 0), (2, 1)]

    def test_split_pieces_leave_out_the_whole_segment_fields(self):
        assert set(self.DECODING) == WHOLE_SEGMENT_FIELDS
        segments = [
            make_segment(self.SPLIT, id=0, **self.DECODING, language="en"),
            make_segment(self.WHOLE, id=1, **self.DECODING),
        ]

        result = merge_diarization({"segments": segments}, self.TURNS)

        for piece in result[:2]:
            assert not WHOLE_SEGMENT_FIELDS & piece.keys(), piece
            assert piece["language"] == "en"  # other fields are kept
        assert result[0]["words"] == self.SPLIT[:3]
        assert result[2] == {**segments[1], "id": 2, "segment_id": 1, "speaker": "A"}  # not split: every field kept

    def test_rendered_json_has_unique_ids(self, fake_backend, tmp_path, monkeypatch):
        monkeypatch.setattr("speech_toolkit.api.load_diarization_pipeline", lambda device: "pipeline")
        monkeypatch.setattr(
            "speech_toolkit.api.diarize_audio", lambda path, pipeline, **hints: [(0.0, 0.9, "S0"), (0.9, 1.5, "S1")]
        )

        result = Transcriber("fake-words", diarize=True).transcribe(tmp_path / "talk.mp3")

        assert [(s["id"], s["segment_id"]) for s in result.speaker_segments] == [(0, 0), (1, 0)]
        assert '"segment_id": 0' in result.render("json")


class TestWordsInPauses:
    """A word between two turns goes to the nearer one."""

    TURNS = [(0.0, 0.5, "A"), (1.6, 3.0, "B")]

    @pytest.mark.parametrize(
        "start, end, expected",
        [
            (0.6, 0.7, [("A", " Okay. so"), ("B", " anyway, let's go.")]),  # 0.1 s after A, 0.9 s before B
            (1.4, 1.5, [("A", " Okay."), ("B", " so anyway, let's go.")]),  # 0.9 s after A, 0.1 s before B
        ],
    )
    def test_a_word_in_a_pause_goes_to_the_nearer_turn(self, start, end, expected):
        words = make_words(("Okay.", 0.0, 0.4), ("so", start, end))
        words += make_words(("anyway,", 1.6, 2.0), ("let's", 2.0, 2.3), ("go.", 2.3, 2.6))

        result = merge_diarization({"segments": [make_segment(words)]}, self.TURNS)

        assert lines(result) == expected

    def test_a_tie_goes_to_the_segments_speaker(self):
        turns = [(0.0, 1.0, "A"), (2.0, 3.0, "B")]
        um = ("um", 1.25, 1.75)  # 0.25 s from either turn
        mostly_b = make_words(("Hi", 0.6, 0.9), um, ("there", 2.0, 2.5), ("friend.", 2.5, 3.0))
        mostly_a = make_words(("Hello", 0.0, 0.5), ("there", 0.5, 1.0), um, ("Bye.", 2.0, 2.3))

        result = merge_diarization({"segments": [make_segment(mostly_b), make_segment(mostly_a)]}, turns)

        assert lines(result) == [("A", " Hi"), ("B", " um there friend."), ("A", " Hello there um"), ("B", " Bye.")]

    def test_a_tie_between_other_speakers_goes_to_the_turn_before(self):
        turns = _TurnIndex([(0.0, 1.0, "A"), (2.0, 3.0, "B")])

        assert turns.nearest(1.25, 1.75, preferred="C") == "A"
        assert turns.nearest(1.25, 1.75, preferred="B") == "B"

    def test_without_turns_every_word_is_unknown(self):
        segment = make_segment(make_words(("Hi", 0.0, 0.4), ("there.", 0.5, 0.9)))

        result = merge_diarization({"segments": [segment]}, [])

        assert result == [{**segment, "id": 0, "segment_id": 0, "speaker": "unknown"}]


def brute_force_talk_time(turns: Sequence[Turn], start: float, end: float) -> Dict[str, float]:
    """Seconds each speaker talks during [start, end], trying every turn (what the merge did before #26)."""
    talk: Dict[str, float] = {}
    for turn_start, turn_end, speaker in sorted(turns, key=lambda turn: turn[0]):
        shared = min(end, turn_end) - max(start, turn_start)
        if shared > 0:
            talk[speaker] = talk.get(speaker, 0.0) + shared
    return talk


def brute_force_speaker(turns: Sequence[Turn], start: float, end: float) -> Optional[str]:
    """The most overlap, else the first turn that contains the midpoint, trying every turn."""
    talk = brute_force_talk_time(turns, start, end)
    if talk:
        return max(talk, key=lambda speaker: talk[speaker])
    middle = (start + end) / 2.0
    ordered = sorted(turns, key=lambda turn: turn[0])
    return next((speaker for turn_start, turn_end, speaker in ordered if turn_start <= middle <= turn_end), None)


class TestTurnLookups:
    """The bisecting lookups give what trying every turn gives, also with overlapping and zero-length turns."""

    @pytest.mark.parametrize("seed", range(5))
    def test_lookups_match_trying_every_turn(self, seed):
        rng = random.Random(seed)
        turns: List[Turn] = []
        for i in range(60):  # times on a 0.1 s grid, so turns and words often touch or tie
            start = round(rng.uniform(0, 100), 1)
            length = rng.choice([0.0, rng.uniform(0, 3), rng.uniform(0, 30)])  # some zero-length, many overlapping
            turns.append((start, round(start + length, 1), "ABC"[i % 3]))
        index = _TurnIndex(turns)

        for _ in range(500):
            start = round(rng.uniform(-5, 105), 1)
            end = round(start + rng.choice([0.0, rng.uniform(0, 2), rng.uniform(0, 20)]), 1)
            talk = list(index.talk_time(start, end).items())
            assert talk == list(brute_force_talk_time(turns, start, end).items()), (start, end)
            assert index.speaker(start, end) == brute_force_speaker(turns, start, end), (start, end)
            # The nearest speaker has a turn at the smallest gap from [start, end].
            gaps = [(max(turn_start - end, start - turn_end, 0.0), speaker) for turn_start, turn_end, speaker in turns]
            nearest = index.nearest(start, end, preferred="?")
            assert min(gap for gap, speaker in gaps if speaker == nearest) == min(gap for gap, _ in gaps), (start, end)


class TestMergeSpeed:
    def test_20k_words_and_2k_turns_merge_quickly(self):
        """Issue #26's acceptance criterion is 1 s; trying every turn for every word took about 4 s.

        The bound leaves room for slow CI machines and coverage tracing; the old merge was several times slower than it.
        """
        words = [{"word": f" w{i}", "start": i * 0.3, "end": i * 0.3 + 0.25} for i in range(20_000)]
        segments = [make_segment(words[i : i + 10], id=i // 10) for i in range(0, 20_000, 10)]
        # 2,000 turns of 3 s from three speakers, a second out of step with the segments, so most segments split.
        turns = [(1.0 + i * 3.0, 4.0 + i * 3.0, f"SPEAKER_{i % 3:02d}") for i in range(2_000)]

        timings = []
        for _ in range(3):
            began = time.perf_counter()
            result = merge_diarization({"segments": segments}, turns)
            timings.append(time.perf_counter() - began)

        assert len(result) > len(segments)  # the segments were split where the speaker changes
        assert min(timings) < 2.5, f"merging took {min(timings):.2f} s"


class TestCheckSpeakerHints:
    """Speaker-count hints are checked in one place, before anything loads."""

    @pytest.mark.parametrize(
        "hints",
        [
            {},
            {"num_speakers": 2},
            {"min_speakers": 2},
            {"max_speakers": 5},
            {"min_speakers": 1, "max_speakers": 3},
            {"min_speakers": 2, "max_speakers": 2},
            {"num_speakers": 3, "min_speakers": 1, "max_speakers": 4},  # consistent: pyannote uses num_speakers
            {"num_speakers": np.int64(2)},  # NumPy integers are integers
        ],
    )
    def test_hints_that_can_be_honoured(self, hints):
        check_speaker_hints(**hints, diarize=True)

    def test_no_hints_need_no_diarization(self):
        check_speaker_hints(diarize=False)

    @pytest.mark.parametrize(
        "hints, message",
        [
            ({"num_speakers": 2, "diarize": False}, "^num_speakers, min_speakers and max_speakers need diarize=True$"),
            ({"min_speakers": 0, "diarize": False}, "need diarize=True"),  # 0 is a hint too, though it is falsy
            ({"num_speakers": 0}, "^num_speakers must be a positive integer, got 0$"),
            ({"max_speakers": -1}, "^max_speakers must be a positive integer, got -1$"),
            ({"num_speakers": 2.5}, "got 2.5$"),
            ({"num_speakers": "2"}, "got '2'$"),
            ({"min_speakers": True}, "got True$"),
            ({"min_speakers": 3, "max_speakers": 2}, r"^min_speakers \(3\) can't be more than max_speakers \(2\)$"),
            ({"num_speakers": 1, "min_speakers": 2}, r"^min_speakers \(2\) can't be more than num_speakers \(1\)$"),
            ({"num_speakers": 4, "max_speakers": 3}, r"^num_speakers \(4\) can't be more than max_speakers \(3\)$"),
            ({"num_speakers": 5, "min_speakers": 1, "max_speakers": 3}, r"num_speakers \(5\) can't be more than"),
        ],
    )
    def test_hints_that_cant_be_honoured(self, hints, message):
        with pytest.raises(UnsupportedOptionError, match=message) as caught:
            check_speaker_hints(**{"diarize": True, **hints})

        assert type(caught.value) is UnsupportedOptionError
        assert isinstance(caught.value, ValueError)  # what the library raised before

    def test_flags_name_the_command_line_options(self):
        with pytest.raises(UnsupportedOptionError, match="^--num-speakers, --min-speakers and --max-speakers need --d"):
            check_speaker_hints(2, diarize=False, flags=True)
        with pytest.raises(UnsupportedOptionError, match=r"^--min-speakers \(3\) can't be more than --max-speakers"):
            check_speaker_hints(None, 3, 2, diarize=True, flags=True)

    @pytest.mark.parametrize(
        "argv, message",
        [
            (["--num-speakers", "2"], "--num-speakers, --min-speakers and --max-speakers need --diarize"),
            (["--diarize", "--min-speakers", "3", "--max-speakers", "2"], "--min-speakers (3) can't be more than"),
            (["--diarize", "--num-speakers", "5", "--max-speakers", "3"], "--num-speakers (5) can't be more than"),
        ],
    )
    def test_the_command_rejects_them_while_parsing(self, argv, message, capsys):
        with pytest.raises(SystemExit) as caught:
            parse_args(["talk.mp3", *argv])

        assert caught.value.code == 2  # argparse's usage error
        assert f"error: {message}" in capsys.readouterr().err

    def test_the_command_accepts_consistent_hints(self):
        args = parse_args(
            ["talk.mp3", "--diarize", "--num-speakers", "2", "--min-speakers", "1", "--max-speakers", "3"]
        )

        assert (args.num_speakers, args.min_speakers, args.max_speakers) == (2, 1, 3)

    def test_transcribe_rejects_them_before_loading_any_model(self, fake_backend, tmp_path, monkeypatch):
        load_pipeline = MagicMock()
        monkeypatch.setattr("speech_toolkit.api.load_diarization_pipeline", load_pipeline)

        with pytest.raises(UnsupportedOptionError, match=r"min_speakers \(3\) can't be more than max_speakers \(2\)"):
            transcribe(tmp_path / "talk.mp3", backend="fake", diarize=True, min_speakers=3, max_speakers=2)

        load_pipeline.assert_not_called()
        assert fake_backend.loaded == 0

    def test_a_transcriber_rejects_them_before_running_the_models(self, fake_backend, tmp_path, monkeypatch):
        monkeypatch.setattr("speech_toolkit.api.load_diarization_pipeline", lambda device: "pipeline")
        diarize = MagicMock()
        monkeypatch.setattr("speech_toolkit.api.diarize_audio", diarize)
        transcriber = Transcriber("fake-words", diarize=True)

        with pytest.raises(UnsupportedOptionError, match=r"num_speakers \(5\) can't be more than max_speakers \(3\)"):
            transcriber.transcribe(tmp_path / "talk.mp3", num_speakers=5, max_speakers=3)

        diarize.assert_not_called()
        assert FakeWordsBackend.calls == []


class TestShortAudio:
    """Audio shorter than MIN_DIARIZATION_SECONDS gets no speaker turns, and pyannote doesn't run on it."""

    MIN_SAMPLES = int(MIN_DIARIZATION_SECONDS * SAMPLE_RATE)

    @pytest.fixture(autouse=True)
    def torch(self):
        with patch.dict(sys.modules, {"torch": MagicMock()}):
            yield

    @staticmethod
    def audio(samples: int) -> Any:
        """Patch the ffmpeg loader to return *samples* samples of silence."""
        return patch("speech_toolkit.diarization.load_audio", return_value=np.zeros(samples, dtype=np.float32))

    @pytest.mark.parametrize("samples", [0, 1, MIN_SAMPLES - 1])
    def test_short_audio_gets_no_turns(self, samples, tmp_path: pathlib.Path):
        pipeline = MagicMock()

        with self.audio(samples):
            assert diarize_audio(tmp_path / "short.wav", pipeline, num_speakers=2) == []

        pipeline.assert_not_called()

    def test_audio_of_the_minimum_length_is_diarized(self, tmp_path: pathlib.Path):
        pipeline = MagicMock()
        turn = SimpleNamespace(start=0.0, end=0.5)
        pipeline.return_value.exclusive_speaker_diarization.itertracks.return_value = [(turn, None, "SPEAKER_00")]

        with self.audio(self.MIN_SAMPLES):
            assert diarize_audio(tmp_path / "short.wav", pipeline) == [(0.0, 0.5, "SPEAKER_00")]

        pipeline.assert_called_once()

    def test_its_transcript_is_labelled_unknown(self, fake_backend, tmp_path: pathlib.Path, monkeypatch):
        pipeline = MagicMock()
        monkeypatch.setattr("speech_toolkit.api.load_diarization_pipeline", lambda device: pipeline)

        with self.audio(100):
            result = Transcriber("fake-words", diarize=True).transcribe(tmp_path / "talk.mp3")

        pipeline.assert_not_called()
        assert result.render() == "[unknown] Hello from talk."
