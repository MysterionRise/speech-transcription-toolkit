"""Tests for speech_toolkit.formats (txt/srt/vtt/json rendering)."""

from __future__ import annotations

import json
import pathlib

import pytest

from speech_toolkit.formats import (
    FORMATS,
    _timestamp,
    format_for_path,
    render,
    split_cues,
    to_json,
    to_srt,
    to_txt,
    to_vtt,
    write_text,
)

RESULT = {
    "text": " Hello there. Hi!",
    "language": "en",
    "segments": [
        {"start": 0.0, "end": 1.25, "text": " Hello there."},
        {"start": 1.25, "end": 1.3, "text": "  "},  # empty segments are skipped
        {"start": 3661.5, "end": 3662.0, "text": " Hi!"},
    ],
}

DIARIZED = {
    "text": "Hello there. How are you? Fine.",
    "segments": [],
    "speaker_segments": [
        {"start": 0.0, "end": 1.0, "text": " Hello there.", "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "text": " How are you?", "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 3.0, "text": " Fine.", "speaker": "SPEAKER_01"},
    ],
}


class TestFormatForPath:
    @pytest.mark.parametrize(
        "path, expected",
        [("out.srt", "srt"), ("OUT.VTT", "vtt"), ("a/b.json", "json"), ("notes.md", "txt"), ("transcript", "txt")],
    )
    def test_extension_picks_format(self, path, expected):
        assert format_for_path(pathlib.Path(path)) == expected

    def test_no_path_uses_default(self):
        assert format_for_path(None) == "txt"
        assert format_for_path(None, default="srt") == "srt"


class TestTimestamp:
    @pytest.mark.parametrize(
        "seconds, expected",
        [
            (0, "00:00:00,000"),
            (1.25, "00:00:01,250"),
            (3661.5, "01:01:01,500"),
            (59.9996, "00:01:00,000"),
            (-1, "00:00:00,000"),
        ],
    )
    def test_srt_timestamps(self, seconds, expected):
        assert _timestamp(seconds, ",") == expected

    def test_vtt_separator(self):
        assert _timestamp(1.25, ".") == "00:00:01.250"


class TestTxt:
    def test_plain_transcript_is_stripped(self):
        assert to_txt(RESULT) == "Hello there. Hi!"

    def test_diarized_one_line_per_speaker_turn(self):
        assert to_txt(DIARIZED) == "[SPEAKER_00] Hello there. How are you?\n[SPEAKER_01] Fine."

    def test_diarized_without_segments(self):
        assert to_txt({"text": "", "speaker_segments": []}) == ""


class TestSrt:
    def test_cues_are_numbered_and_skip_empty_text(self):
        expected = "1\n00:00:00,000 --> 00:00:01,250\nHello there.\n\n2\n01:01:01,500 --> 01:01:02,000\nHi!\n"
        assert to_srt(RESULT) == expected

    def test_speaker_labels(self):
        assert to_srt(DIARIZED).splitlines()[2] == "[SPEAKER_00] Hello there."

    def test_no_segments(self):
        assert to_srt({"text": "", "segments": []}) == ""


class TestVtt:
    def test_header_and_cues(self):
        expected = "WEBVTT\n\n00:00:00.000 --> 00:00:01.250\nHello there.\n\n01:01:01.500 --> 01:01:02.000\nHi!\n"
        assert to_vtt(RESULT) == expected

    def test_speakers_become_voice_tags(self):
        assert "<v SPEAKER_01>Fine." in to_vtt(DIARIZED)


class TestJson:
    def test_round_trip_keeps_unicode(self):
        result = {"text": "Grüße, 世界", "segments": [{"start": 0.0, "end": 1.0, "text": "Grüße, 世界"}]}
        rendered = to_json(result)
        assert "世界" in rendered
        assert json.loads(rendered) == result


class TestRender:
    @pytest.mark.parametrize("fmt", FORMATS)
    def test_every_format_renders(self, fmt):
        assert isinstance(render(RESULT, fmt), str)

    def test_render_dispatches(self):
        assert render(RESULT, "srt") == to_srt(RESULT)

    def test_unknown_format_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown format"):
            render(RESULT, "docx")


class TestWriteText:
    def test_creates_folders_and_ends_with_newline(self, tmp_path: pathlib.Path):
        dest = tmp_path / "a" / "b" / "out.txt"
        write_text(dest, "hello")
        assert dest.read_text(encoding="utf-8") == "hello\n"

    def test_keeps_existing_final_newline(self, tmp_path: pathlib.Path):
        dest = tmp_path / "out.srt"
        write_text(dest, "1\n")
        assert dest.read_text(encoding="utf-8") == "1\n"


WORDS = [
    {"word": " Hello", "start": 0.0, "end": 0.5},
    {"word": " there,", "start": 0.5, "end": 1.0},
    {"word": " how", "start": 1.0, "end": 1.3},
    {"word": " are", "start": 1.3, "end": 1.6},
    {"word": " you", "start": 1.6, "end": 1.9},
    {"word": " today?", "start": 1.9, "end": 2.5},
]


class TestSplitCues:
    def test_word_timings_set_cue_times(self):
        cue = {"start": 0.0, "end": 2.5, "text": " Hello there, how are you today?", "words": WORDS}

        pieces = split_cues([cue], max_width=10)

        assert pieces == [
            {"start": 0.0, "end": 1.3, "text": "Hello\nthere, how"},
            {"start": 1.3, "end": 2.5, "text": "are you\ntoday?"},
        ]

    def test_without_words_time_is_shared_by_length(self):
        cue = {"start": 0.0, "end": 8.0, "text": " aaa bbb ccc ddd", "speaker": "SPEAKER_01"}

        pieces = split_cues([cue], max_width=7, max_lines=1)

        assert pieces == [
            {"start": 0.0, "end": 4.0, "text": "aaa bbb", "speaker": "SPEAKER_01"},
            {"start": 4.0, "end": 8.0, "text": "ccc ddd", "speaker": "SPEAKER_01"},
        ]

    def test_long_word_gets_its_own_line(self):
        pieces = split_cues([{"start": 0.0, "end": 1.0, "text": "a supercalifragilistic b"}], max_width=5)

        assert [p["text"] for p in pieces] == ["a\nsupercalifragilistic", "b"]

    def test_words_without_times_fall_back_to_text(self):
        cue = {"start": 0.0, "end": 2.0, "text": "one two", "words": [{"word": " one"}, {"word": " two"}]}

        assert split_cues([cue], max_width=3) == [{"start": 0.0, "end": 2.0, "text": "one\ntwo"}]

    def test_render_wraps_only_subtitles(self):
        result = {
            **RESULT,
            "segments": [{"start": 0.0, "end": 2.5, "text": " Hello there, how are you today?", "words": WORDS}],
        }

        srt = render(result, "srt", max_line_width=10)
        assert srt.startswith("1\n00:00:00,000 --> 00:00:01,300\nHello\nthere, how\n\n2\n")
        assert render(result, "vtt", max_line_width=10).count("-->") == 2
        assert render(result, "txt", max_line_width=10) == render(result, "txt")
        assert render(result, "json", max_line_width=10) == render(result, "json")

    def test_speaker_label_on_first_line(self):
        result = {**DIARIZED, "speaker_segments": [{"start": 0.0, "end": 1.0, "text": "aaa bbb", "speaker": "S1"}]}
        assert "[S1] aaa\nbbb" in render(result, "srt", max_line_width=3)
        assert "<v S1>aaa\nbbb" in render(result, "vtt", max_line_width=3)
