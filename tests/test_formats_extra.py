"""The formats and options of #38: tsv, rttm, timestamped txt and word-level vtt; cue limits (max_lines,
max_cue_duration); speaker names; and re-rendering a saved result with ``transcribe --from-json``.

Each new format has a golden file in tests/data/golden/. After a deliberate change to the output, write them again with
``UPDATE_GOLDEN=1 pytest tests/test_formats_extra.py`` and review the diff.
"""

from __future__ import annotations

import copy
import json
import os
import pathlib
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import pytest

from speech_toolkit import cli
from speech_toolkit.backends import TranscriptionBackend, TranscriptionResult
from speech_toolkit.cli import expand_formats, parse_args, positive_float, recording_id, speaker_names_arg
from speech_toolkit.errors import SpeechToolkitWarning, UnsupportedOptionError
from speech_toolkit.formats import (
    FORMATS,
    MIN_CUE_SECONDS,
    rename_speakers,
    render,
    speaker_mapping,
    split_cues,
    to_rttm,
    to_tsv,
    to_txt,
    to_vtt,
)
from tests.test_formats_subtitles import check_timing, parse_srt, parse_vtt, shown, vtt_millis

GOLDEN = pathlib.Path(__file__).parent / "data" / "golden"
UPDATE_GOLDEN = os.getenv("UPDATE_GOLDEN", "") not in ("", "0")


def words(*timed: Tuple[str, float, float]) -> List[Dict[str, Any]]:
    """Whisper-style words from ``(word, start, end)`` tuples."""
    return [{"word": word, "start": start, "end": end} for word, start, end in timed]


# A transcript without speakers: English with word timings, an empty segment, and Japanese an hour in.
TALK_SEGMENTS = [
    {
        "id": 0,
        "start": 0.0,
        "end": 2.2,
        "text": " Hello and welcome.",
        "words": words((" Hello", 0.0, 0.6), (" and", 0.6, 0.9), (" welcome.", 0.9, 2.2)),
    },
    {
        "id": 1,
        "start": 2.5,
        "end": 9.0,
        "text": " Today we talk about subtitles, timing & <tags>.",
        "words": words(
            (" Today", 2.5, 2.9),
            (" we", 2.9, 3.1),
            (" talk", 3.1, 3.5),
            (" about", 3.5, 4.0),
            (" subtitles,", 4.0, 5.2),
            (" timing", 5.4, 6.0),
            (" &", 6.0, 6.2),
            (" <tags>.", 6.3, 9.0),
        ),
    },
    {"id": 2, "start": 9.0, "end": 9.5, "text": " "},
    {
        "id": 3,
        "start": 3661.0,
        "end": 3664.5,
        "text": "字幕は便利です。",
        "words": words(
            ("字幕", 3661.0, 3661.8), ("は", 3661.8, 3662.0), ("便利", 3662.0, 3662.9), ("です。", 3662.9, 3664.5)
        ),
    },
]
TALK = TranscriptionResult(
    text="".join(seg["text"] for seg in TALK_SEGMENTS), segments=TALK_SEGMENTS, language="en", duration=3665.0
)


def speaker_segment(index: int, speaker: str, text: str, *timed: Tuple[str, float, float]) -> Dict[str, Any]:
    """A speaker segment of *text*, whose times are its words' times: ``(word, start, end)`` tuples."""
    start, end = timed[0][1], timed[-1][2]
    return {"id": index, "start": start, "end": end, "text": text, "words": words(*timed), "speaker": speaker}


# A diarized meeting: SPEAKER_01 speaks first, then SPEAKER_00, a stretch no speaker turn covers ("unknown"), Chinese
# from SPEAKER_02 in two segments that touch, and SPEAKER_00 answering before SPEAKER_02 has finished.
MEETING_SPEAKER_SEGMENTS = [
    {**segment, "segment_id": segment["id"]}
    for segment in (
        speaker_segment(
            0,
            "SPEAKER_01",
            " Good morning, everyone.",
            (" Good", 0.0, 0.4),
            (" morning,", 0.4, 0.9),
            (" everyone.", 0.9, 1.5),
        ),
        speaker_segment(
            1, "SPEAKER_01", " Let's get started.", (" Let's", 1.5, 1.9), (" get", 1.9, 2.2), (" started.", 2.2, 3.0)
        ),
        speaker_segment(
            2,
            "SPEAKER_00",
            " Thanks. The numbers are up.",
            (" Thanks.", 3.2, 3.7),
            (" The", 3.9, 4.1),
            (" numbers", 4.1, 4.5),
            (" are", 4.5, 4.7),
            (" up.", 4.7, 5.0),
        ),
        {"id": 3, "start": 5.0, "end": 5.4, "text": " Hmm.", "speaker": "unknown"},
        speaker_segment(4, "SPEAKER_02", "我同意。", ("我", 5.6, 5.9), ("同意。", 5.9, 7.0)),
        speaker_segment(5, "SPEAKER_02", "我们走吧。", ("我们", 7.0, 7.4), ("走吧。", 7.4, 8.0)),
        speaker_segment(
            6, "SPEAKER_00", " Great, see you.", (" Great,", 7.8, 8.2), (" see", 8.2, 8.5), (" you.", 8.5, 9.0)
        ),
    )
]
MEETING = TranscriptionResult(
    text="".join(segment["text"] for segment in MEETING_SPEAKER_SEGMENTS),
    segments=[
        {key: value for key, value in segment.items() if key not in ("speaker", "segment_id")}
        for segment in MEETING_SPEAKER_SEGMENTS
    ],
    language="en",
    speaker_segments=MEETING_SPEAKER_SEGMENTS,
)

# Golden files: (file name, result, render options). The extension picks the format.
GOLDEN_CASES: List[Tuple[str, TranscriptionResult, Dict[str, Any]]] = [
    ("talk.tsv", TALK, {}),
    ("meeting.tsv", MEETING, {}),
    ("meeting.rttm", MEETING, {}),
    ("talk-timestamps.txt", TALK, {"timestamps": True}),
    ("meeting-timestamps.txt", MEETING, {"timestamps": True}),
    ("talk-words.vtt", TALK, {"word_level": True}),
    ("meeting-words.vtt", MEETING, {"word_level": True, "max_line_width": 16}),
    ("talk-limits.srt", TALK, {"max_line_width": 20, "max_lines": 1, "max_cue_duration": 1.5}),
    ("meeting-limits.vtt", MEETING, {"max_line_width": 12, "max_lines": 3, "max_cue_duration": 1.0}),
    ("meeting-names.srt", MEETING, {"speaker_names": "Alice,Bob"}),
    ("meeting-names.txt", MEETING, {"speaker_names": "Alice,Bob"}),
    ("meeting-names.rttm", MEETING, {"speaker_names": ["Alice", "Bob"]}),
]


@pytest.mark.parametrize("name, result, options", GOLDEN_CASES, ids=[case[0] for case in GOLDEN_CASES])
def test_golden_files(name, result, options, tmp_path):
    written = result.save(tmp_path / name, **options).read_text(encoding="utf-8")
    golden = GOLDEN / name
    if UPDATE_GOLDEN:
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(written, encoding="utf-8", newline="\n")
    assert written == golden.read_text(encoding="utf-8"), f"{name} differs from its golden file"
    # Golden subtitles are valid too.
    if name.endswith(".vtt"):
        check_timestamp_tags(parse_vtt(written))
    elif name.endswith(".srt"):
        check_timing(parse_srt(written))


def test_every_new_format_has_a_golden_file():
    names = [name for name, _, _ in GOLDEN_CASES]
    for suffix in (".tsv", ".rttm", "-timestamps.txt", "-words.vtt"):
        assert any(name.endswith(suffix) for name in names), suffix
    assert sorted(path.name for path in GOLDEN.iterdir()) == sorted(names), "a golden file without a test case"


###############################################################################
# WebVTT timestamp tags
###############################################################################

STAMP = re.compile(r"<((?:\d{2,}:)?[0-5]\d:[0-5]\d\.\d{3})>")


def check_timestamp_tags(cues: List[Tuple[int, int, List[str]]]) -> List[List[int]]:
    """Every ``<00:00:01.200>`` tag is later than its cue's start and the tag before it, and earlier than its cue's end,
    as WebVTT requires; returns each cue's tag times in milliseconds."""
    tags = []
    for start, end, lines in cues:
        times = [vtt_millis(stamp) for stamp in STAMP.findall("\n".join(lines))]
        previous = start
        for time in times:
            assert previous < time < end, f"timestamp tag {time} out of order in cue {start}-{end}: {lines}"
            previous = time
        tags.append(times)
    return tags


class TestWordLevelVtt:
    def test_a_tag_where_each_word_starts(self):
        result = {"text": "", "segments": [{"start": 0.0, "end": 1.5, "text": " Hi there you", "words": WORDS}]}

        assert to_vtt(result, word_level=True) == (
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHi <00:00:00.500>there <00:00:01.000>you\n"
        )

    def test_without_word_timings_cues_are_unchanged(self):
        result = {"text": "", "segments": [{"start": 0.0, "end": 1.5, "text": " Hi there you"}]}

        assert to_vtt(result, word_level=True) == to_vtt(result)

    def test_off_by_default(self):
        result = {"text": "", "segments": [{"start": 0.0, "end": 1.5, "text": " Hi there you", "words": WORDS}]}

        assert "<00:" not in to_vtt(result) and "<00:" not in to_vtt(result, max_line_width=5)

    def test_words_that_start_together_share_a_tag(self):
        timed = words((" a", 0.0, 0.5), (" b", 0.5, 0.5), (" c", 0.5, 1.0), (" d", 0.2, 1.2))  # d starts too early
        result = {"text": "", "segments": [{"start": 0.0, "end": 1.2, "text": " a b c d", "words": timed}]}

        assert to_vtt(result, word_level=True).splitlines()[-1] == "a <00:00:00.500>b c d"

    def test_tags_stay_inside_the_cue_after_timing_fixes(self):
        """The second cue starts when the first ends, so its first word's own time is before the cue starts."""
        first = {"start": 0.0, "end": 2.0, "text": " one two", "words": words((" one", 0.0, 1.0), (" two", 1.0, 2.0))}
        second = {
            "start": 1.5,
            "end": 3.0,
            "text": " three four",
            "words": words((" three", 1.5, 2.6), (" four", 2.6, 3.0)),
        }

        cues = parse_vtt(to_vtt({"text": "", "segments": [first, second]}, word_level=True))

        assert check_timestamp_tags(cues) == [[1000], [2600]]
        assert cues[1][2] == ["three <00:00:02.600>four"]

    def test_escaped_text_and_voice_tags(self):
        segment = {
            "start": 0.0,
            "end": 1.0,
            "text": " Tom & <Jerry>",
            "speaker": "A&B",
            "words": words((" Tom", 0.0, 0.3), (" &", 0.3, 0.5), (" <Jerry>", 0.5, 1.0)),
        }

        vtt = to_vtt({"text": "", "segments": [], "speaker_segments": [segment]}, word_level=True)

        assert vtt.splitlines()[-1] == "<v A&amp;B>Tom <00:00:00.300>&amp; <00:00:00.500>&lt;Jerry&gt;"
        assert [shown(line) for line in parse_vtt(vtt)[0][2]] == ["Tom & <Jerry>"]

    def test_wrapped_lines_and_cjk_words(self):
        """Lines break between the characters of a CJK word, but tags mark only where words start."""
        cue = TALK_SEGMENTS[3]

        vtt = to_vtt({"text": "", "segments": [cue]}, word_level=True, max_line_width=6)

        assert [lines for _, _, lines in parse_vtt(vtt)] == [
            ["字幕<01:01:01.800>は", "<01:01:02.000>便利<01:01:02.900>で"],
            ["す。"],  # the rest of a word: no tag
        ]

    def test_render_and_save(self, tmp_path):
        assert TALK.render("vtt", word_level=True) == render(TALK.to_dict(), "vtt", word_level=True)
        assert "<00:00:00.600>" in TALK.save(tmp_path / "talk.vtt", word_level=True).read_text(encoding="utf-8")
        assert "<00:" not in TALK.render("srt", word_level=True)  # vtt only


WORDS = words((" Hi", 0.0, 0.5), (" there", 0.5, 1.0), (" you", 1.0, 1.5))

###############################################################################
# tsv
###############################################################################


class TestTsv:
    def test_whisper_layout(self):
        result = {
            "text": "",
            "segments": [{"start": 0.0, "end": 1.2345, "text": " Hello."}, {"start": 61.5, "end": 62.0, "text": "Bye"}],
        }

        assert to_tsv(result) == "start\tend\ttext\n0\t1234\tHello.\n61500\t62000\tBye"

    def test_text_on_one_line(self):
        result = {"text": "", "segments": [{"start": 0.0, "end": 1.0, "text": " one\ttwo\n\nthree  four "}]}

        assert to_tsv(result).splitlines()[1] == "0\t1000\tone two three four"

    def test_speakers_get_a_column(self):
        result = {
            "text": "",
            "segments": [],
            "speaker_segments": [
                {"start": 0.0, "end": 1.0, "text": " Hi.", "speaker": "Ann\tLee"},
                {"start": 1.0, "end": 2.0, "text": " Hello.", "speaker": None},
                {"start": 2.0, "end": 3.0, "text": " ", "speaker": "SPEAKER_00"},  # empty: left out
            ],
        }

        assert to_tsv(result) == "start\tend\ttext\tspeaker\n0\t1000\tHi.\tAnn Lee\n1000\t2000\tHello.\t"

    def test_no_segments(self):
        assert to_tsv({"text": "", "segments": []}) == "start\tend\ttext"
        assert render({"text": "", "segments": []}, "tsv") == "start\tend\ttext"

    def test_extension_picks_it(self, tmp_path):
        assert TALK.save(tmp_path / "talk.tsv").read_text(encoding="utf-8").startswith("start\tend\ttext\n0\t2200\t")


###############################################################################
# rttm
###############################################################################


def diarized(*segments: Tuple[float, float, str, Any]) -> Dict[str, Any]:
    """A diarized result from ``(start, end, text, speaker)`` tuples."""
    return {
        "text": "",
        "segments": [],
        "speaker_segments": [{"start": s, "end": e, "text": t, "speaker": who} for s, e, t, who in segments],
    }


class TestRttm:
    def test_every_line_has_the_ten_fields(self):
        rttm = MEETING.render("rttm", file_id="meeting")

        lines = rttm.splitlines()
        assert len(lines) == 4
        for line in lines:
            fields = line.split(" ")
            assert len(fields) == 10, line
            kind, file_id, channel, onset, duration, ortho, stype, name, conf, lookahead = fields
            assert (kind, file_id, channel, ortho, stype, conf, lookahead) == (
                "SPEAKER",
                "meeting",
                "1",
                "<NA>",
                "<NA>",
                "<NA>",
                "<NA>",
            )
            assert re.fullmatch(r"\d+\.\d{3}", onset) and re.fullmatch(r"\d+\.\d{3}", duration)
            assert float(duration) > 0 and name.startswith("SPEAKER_")

    def test_touching_turns_of_a_speaker_merge(self):
        result = diarized((0.0, 1.0, "a", "A"), (1.0, 2.0, "b", "A"), (1.5, 2.5, "c", "A"), (3.0, 4.0, "d", "A"))

        assert to_rttm(result, "x").splitlines() == [
            "SPEAKER x 1 0.000 2.500 <NA> <NA> A <NA> <NA>",
            "SPEAKER x 1 3.000 1.000 <NA> <NA> A <NA> <NA>",  # after a pause: a turn of its own
        ]

    def test_overlapping_speakers_and_time_order(self):
        result = diarized((2.0, 3.0, "b", "B"), (0.0, 2.5, "a", "A"))

        assert to_rttm(result).splitlines() == [
            "SPEAKER audio 1 0.000 2.500 <NA> <NA> A <NA> <NA>",
            "SPEAKER audio 1 2.000 1.000 <NA> <NA> B <NA> <NA>",
        ]

    def test_speech_without_a_speaker_and_empty_turns_are_left_out(self):
        result = diarized(
            (0.0, 1.0, "a", "unknown"),
            (1.0, 2.0, "b", None),
            (2.0, 2.0, "c", "A"),
            (3.0, 4.0, " ", "A"),
            (4.0, 5.0, "d", ""),
        )

        assert to_rttm(result) == ""

    def test_spaces_in_names_and_file_ids(self):
        result = diarized((0.0, 1.0, "a", "Ann  Lee"))

        assert to_rttm(result, "my talk") == "SPEAKER my_talk 1 0.000 1.000 <NA> <NA> Ann_Lee <NA> <NA>"
        assert to_rttm(result, "  ").split(" ")[1] == "audio"

    def test_needs_speaker_labels(self):
        with pytest.raises(UnsupportedOptionError, match="no speaker labels"):
            TALK.render("rttm")
        with pytest.raises(ValueError, match="diarize"):
            render({"text": "", "segments": []}, "rttm")

    def test_save_names_the_file(self, tmp_path):
        assert (
            MEETING.save(tmp_path / "board meeting.rttm")
            .read_text(encoding="utf-8")
            .startswith("SPEAKER board_meeting 1 0.000 3.000")
        )
        assert MEETING.save(tmp_path / "a.rttm", file_id="b").read_text(encoding="utf-8").startswith("SPEAKER b 1")
        assert MEETING.render("rttm").startswith("SPEAKER audio 1")


###############################################################################
# Timestamped txt, and the diarized txt join
###############################################################################


class TestTimestampedTxt:
    def test_one_line_per_segment(self):
        assert to_txt(TALK.to_dict(), timestamps=True).splitlines() == [
            "[00:00:00] Hello and welcome.",
            "[00:00:02] Today we talk about subtitles, timing & <tags>.",
            "[01:01:01] 字幕は便利です。",
        ]

    def test_times_are_rounded_down_to_the_second(self):
        result = {
            "text": "",
            "segments": [{"start": 59.9994, "end": 61.0, "text": "a"}, {"start": 59.9996, "end": 61, "text": "b"}],
        }

        assert to_txt(result, timestamps=True) == "[00:00:59] a\n[00:01:00] b"

    def test_speaker_labels(self):
        result = diarized((0.0, 1.0, " Hi.", "A"), (1.0, 2.0, " Again\non two lines.", "A"), (2.0, 3.0, " Bye.", None))

        assert (
            to_txt(result, timestamps=True) == "[00:00:00] [A] Hi.\n[00:00:01] [A] Again on two lines.\n[00:00:02] Bye."
        )

    def test_off_by_default(self):
        assert TALK.render("txt") == TALK.text.strip()
        assert TALK.render("srt", timestamps=True) == TALK.render("srt")  # txt only


class TestCjkJoin:
    """Diarized txt joins one speaker's segments with a space, but none next to Chinese or Japanese text."""

    @pytest.mark.parametrize(
        "first, second, joined",
        [
            ("我同意。", "我们走吧。", "我同意。我们走吧。"),
            ("今日は", "いい天気です。", "今日はいい天気です。"),
            ("Hello.", "你好。", "Hello.你好。"),
            ("你好", "OK", "你好OK"),
            ("안녕하세요.", "반갑습니다.", "안녕하세요. 반갑습니다."),  # Korean has spaces between words
            (" Hello.", " Bye.", "Hello. Bye."),
        ],
    )
    def test_join(self, first, second, joined):
        assert to_txt(diarized((0.0, 1.0, first, "A"), (1.0, 2.0, second, "A"))) == f"[A] {joined}"

    def test_meeting(self):
        assert MEETING.render("txt").splitlines()[3] == "[SPEAKER_02] 我同意。我们走吧。"


###############################################################################
# Cue limits
###############################################################################


class TestMaxLines:
    def test_lines_per_cue(self):
        result = {"text": "", "segments": [{"start": 0.0, "end": 3.0, "text": " aa bb cc dd ee ff"}]}

        assert [lines for _, _, lines in parse_srt(render(result, "srt", 5, max_lines=3))] == [
            ["aa bb", "cc dd", "ee ff"]
        ]
        assert [lines for _, _, lines in parse_srt(render(result, "srt", 5, max_lines=1))] == [
            ["aa bb"],
            ["cc dd"],
            ["ee ff"],
        ]
        assert render(result, "vtt", 5) == render(result, "vtt", 5, max_lines=2)  # the default

    def test_needs_a_line_width(self):
        with pytest.raises(UnsupportedOptionError, match="max_lines needs max_line_width"):
            TALK.render("srt", max_lines=1)

    @pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
    def test_bad_values(self, value):
        with pytest.raises(UnsupportedOptionError, match="max_lines must be a whole number"):
            TALK.render("srt", 10, max_lines=value)


class TestMaxCueDuration:
    def test_long_cues_split_at_word_timings(self):
        cues = parse_srt(TALK.render("srt", max_cue_duration=2.0))

        check_timing(cues)
        assert cues == [
            (0, 900, ["Hello and"]),
            (900, 2200, ["welcome."]),
            (2500, 4000, ["Today we talk about"]),
            (4000, 6000, ["subtitles, timing"]),
            (6000, 6300, ["&"]),  # lengthened towards half a second, up to the next cue
            (6300, 8300, ["<tags>."]),  # one word of 2.7 s: cut to 2 s
            (3661000, 3662900, ["字幕は便利"]),
            (3662900, 3664500, ["です。"]),
        ]

    def test_short_cues_are_unchanged(self):
        assert TALK.render("srt", max_cue_duration=10.0) == TALK.render("srt")

    def test_a_word_longer_than_the_limit_is_cut_short(self):
        result = {"text": "", "segments": [{"start": 0.0, "end": 30.0, "text": " Hello world"}]}

        assert [(start, end, lines) for start, end, lines in parse_srt(render(result, "srt", max_cue_duration=5))] == [
            (0, 5000, ["Hello"]),
            (15000, 20000, ["world"]),
        ]

    def test_with_a_line_width(self):
        cues = parse_vtt(TALK.render("vtt", 40, max_cue_duration=1.5))

        check_timing(cues)
        assert all(end - start <= 1500 for start, end, _ in cues)
        assert all(len(lines) == 1 for _, _, lines in cues[:3])  # 1.5 s of words fit on a line

    def test_shorter_than_the_minimum_cue_length(self):
        result = {
            "text": "",
            "segments": [{"start": 0.0, "end": 0.1, "text": "a"}, {"start": 5.0, "end": 6.0, "text": "b"}],
        }

        assert [(start, end) for start, end, _ in parse_srt(render(result, "srt", max_cue_duration=0.3))] == [
            (0, 300),  # lengthened, but only up to the limit
            (5000, 5300),
        ]
        assert MIN_CUE_SECONDS > 0.3

    @pytest.mark.parametrize("value", [0, -2.0, float("nan"), float("inf"), True, "5"])
    def test_bad_values(self, value):
        with pytest.raises(UnsupportedOptionError, match="max_cue_duration must be a number of seconds"):
            TALK.render("srt", max_cue_duration=value)

    def test_split_cues(self):
        cue = {"start": 0.0, "end": 8.0, "text": " aaa bbb ccc ddd"}

        assert split_cues([cue], 20, max_duration=4.0) == [
            {"start": 0.0, "end": 4.0, "text": "aaa bbb"},
            {"start": 4.0, "end": 8.0, "text": "ccc ddd"},
        ]


@pytest.mark.parametrize("seed", range(10))
def test_random_cue_limits(seed):
    """Whatever the transcript and limits, cues keep the text, fit the limits and stay valid."""
    rng = random.Random(seed)
    for _ in range(10):
        segments = []
        for _ in range(rng.randint(1, 6)):
            start = round(rng.uniform(0, 20), 2)
            vocabulary = ["a", "word", "longer", "東京", "は", "&", "<3"]
            text_words = [rng.choice(vocabulary) for _ in range(rng.randint(1, 12))]
            end = start + rng.choice([0.0, 0.3, 2.0, 7.0, 15.0])
            segment: Dict[str, Any] = {"start": start, "end": end, "text": " " + " ".join(text_words)}
            if rng.random() < 0.5:
                step = (end - start) / len(text_words)
                segment["words"] = words(
                    *((f" {w}", start + i * step, start + (i + 1) * step) for i, w in enumerate(text_words))
                )
            if rng.random() < 0.5:
                segment["speaker"] = rng.choice(["A", "B"])
            segments.append(segment)
        result = {"text": "", "segments": segments}
        width: Optional[int] = rng.choice([None, 6, 12, 30])
        max_lines = rng.choice([1, 2, 3]) if width else None
        duration = rng.choice([None, 0.4, 1.0, 3.0])
        word_level = rng.random() < 0.5
        source = sorted("".join(seg["text"] for seg in segments).replace(" ", ""))

        vtt = parse_vtt(to_vtt(result, width, max_lines=max_lines, max_cue_duration=duration, word_level=word_level))
        srt = parse_srt(render(result, "srt", width, max_lines=max_lines, max_cue_duration=duration))
        for cues in (vtt, srt):
            check_timing(cues)
            for start, end, lines in cues:
                if duration is not None:
                    assert end - start <= round(duration * 1000), (start, end, lines)
                if max_lines:
                    assert len(lines) <= max_lines + (lines[0].startswith("[") and lines[0].endswith("]"))
        check_timestamp_tags(vtt)
        text = "".join(shown(line) for _, _, lines in vtt for line in lines).replace(" ", "")
        assert sorted(text) == source, "text lost or added"


###############################################################################
# Speaker names
###############################################################################


class TestSpeakerNames:
    def test_order_of_first_appearance(self):
        assert speaker_mapping(MEETING.to_dict(), ["Alice", "Bob"]) == {"SPEAKER_01": "Alice", "SPEAKER_00": "Bob"}
        assert speaker_mapping(MEETING.to_dict(), "Alice, Bob,Carol,Dan") == {
            "SPEAKER_01": "Alice",
            "SPEAKER_00": "Bob",
            "SPEAKER_02": "Carol",
        }

    def test_empty_names_keep_the_label(self):
        assert speaker_mapping(MEETING.to_dict(), ",Bob") == {"SPEAKER_00": "Bob"}

    def test_first_appearance_is_in_time_order(self):
        result = diarized((5.0, 6.0, "later", "A"), (1.0, 2.0, "first", "B"), (0.5, 0.8, " ", "C"))  # C says nothing

        assert speaker_mapping(result, ["one", "two", "three"]) == {"B": "one", "A": "two"}

    def test_a_mapping(self):
        assert speaker_mapping(MEETING.to_dict(), {"SPEAKER_00": "Bob", "SPEAKER_02": " ", "unknown": "?"}) == {
            "SPEAKER_00": "Bob",
            "unknown": "?",
        }

    def test_unknown_is_not_a_speaker(self):
        result = diarized((0.0, 1.0, "a", "unknown"), (1.0, 2.0, "b", "SPEAKER_00"))

        assert speaker_mapping(result, ["Alice"]) == {"SPEAKER_00": "Alice"}

    @pytest.mark.parametrize("fmt", FORMATS)
    def test_every_format_renames_consistently(self, fmt):
        """Renaming gives the output of a result whose labels were the names all along."""
        renamed = copy.deepcopy(MEETING.to_dict())
        names = {"SPEAKER_01": "Alice", "SPEAKER_00": "Bob"}
        for seg in renamed["speaker_segments"]:
            seg["speaker"] = names.get(seg["speaker"], seg["speaker"])

        output = MEETING.render(fmt, speaker_names="Alice,Bob")

        assert output == render(renamed, fmt)
        assert "SPEAKER_00" not in output and "SPEAKER_01" not in output
        assert "Alice" in output and "Bob" in output and "SPEAKER_02" in output

    def test_the_same_name_twice_makes_one_speaker(self):
        result = diarized((0.0, 1.0, "a", "A"), (1.0, 2.0, "b", "B"))

        assert to_rttm(rename_speakers(result, "Ann,Ann")) == "SPEAKER audio 1 0.000 2.000 <NA> <NA> Ann <NA> <NA>"
        assert render(result, "txt", speaker_names="Ann,Ann") == "[Ann] a b"

    def test_the_result_is_not_changed(self):
        before = copy.deepcopy(MEETING.to_dict())

        MEETING.render("json", speaker_names="Alice,Bob")

        assert MEETING.to_dict() == before

    def test_without_speakers_nothing_changes(self):
        assert rename_speakers(TALK.to_dict(), "Alice") == TALK.to_dict()
        assert TALK.render("srt", speaker_names="Alice") == TALK.render("srt")
        assert MEETING.render("srt", speaker_names="") == MEETING.render("srt")

    def test_names_in_vtt_voice_tags_are_escaped(self):
        assert "<v Tom &amp; Jerry>" in MEETING.render("vtt", speaker_names=["Tom & Jerry"])


def test_unknown_format_is_an_unsupported_option():
    with pytest.raises(
        UnsupportedOptionError, match=r"Unknown format: 'docx'\. Available: txt, srt, vtt, json, tsv, rttm"
    ):
        TALK.render("docx")


###############################################################################
# The command line
###############################################################################


class ScriptBackend(TranscriptionBackend):
    """A backend that 'transcribes' every file as TALK, with word timings when asked for them."""

    name = "script"
    description = "Scripted backend for tests"
    capabilities = frozenset({"language_detection", "word_timestamps"})
    loaded = 0
    calls: List[bool] = []

    @classmethod
    def available_models(cls):
        return ["base"]

    def load_model(self, model_name, device=None):
        ScriptBackend.loaded += 1
        self._model = object()
        self._model_name = model_name

    def transcribe(self, audio_path, language=None, task="transcribe", verbose=True, word_timestamps=False):
        ScriptBackend.calls.append(word_timestamps)
        segments = copy.deepcopy(TALK_SEGMENTS)
        if not word_timestamps:
            for segment in segments:
                segment.pop("words", None)
        return TranscriptionResult(TALK.text, segments, language="en", duration=3665.0)


# The speaker turns the mocked diarization finds in TALK.
TURNS = [(0.0, 2.2, "SPEAKER_01"), (2.5, 9.0, "SPEAKER_00"), (3661.0, 3665.0, "SPEAKER_01")]


@pytest.fixture
def script(fake_backend, monkeypatch):
    """Register ScriptBackend as 'script', next to the fake backends; it starts with no loads and no calls."""
    from speech_toolkit.backends import _BACKENDS  # with the fake backends already in

    monkeypatch.setattr("speech_toolkit.backends._BACKENDS", {**_BACKENDS, "script": ScriptBackend})
    ScriptBackend.loaded = 0
    ScriptBackend.calls = []
    return ScriptBackend


@pytest.fixture
def diarization(monkeypatch):
    """Mock diarization: it finds TURNS."""
    monkeypatch.setattr("speech_toolkit.api.load_diarization_pipeline", lambda device: object())
    monkeypatch.setattr("speech_toolkit.api.diarize_audio", lambda path, pipeline, **hints: TURNS)


@pytest.fixture
def no_model(monkeypatch):
    """Fail the test if the command loads a model."""

    def refuse(args):
        raise AssertionError("a model was loaded")

    monkeypatch.setattr(cli, "load_transcriber", refuse)


def run(*argv: object) -> None:
    cli.main([str(arg) for arg in argv])


def files(folder: pathlib.Path) -> Dict[str, str]:
    """The files in *folder*, by name, with their text."""
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(folder.iterdir())}


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "talk.mp3"
    path.touch()
    return path


class TestFromJson:
    # Every output option at once: what --from-json writes must be what the transcription wrote.
    OPTIONS = ("--max-line-width", "16", "--max-lines", "3", "--max-cue-duration", "2", "--word-level", "--timestamps")

    def test_same_output_as_a_direct_run_and_no_model(self, script, audio, tmp_path, monkeypatch):
        run(audio, "-b", "script", "-q", "--outdir", tmp_path / "direct", "-f", "all", *self.OPTIONS)
        assert script.loaded == 1 and script.calls == [True]  # --word-level turned word timestamps on
        monkeypatch.setattr(cli, "load_transcriber", lambda args: pytest.fail("a model was loaded"))

        run(
            "--from-json",
            tmp_path / "direct" / "talk.json",
            "-q",
            "--outdir",
            tmp_path / "again",
            "-f",
            "all",
            *self.OPTIONS,
        )

        direct = files(tmp_path / "direct")
        assert sorted(direct) == ["talk.json", "talk.srt", "talk.tsv", "talk.txt", "talk.vtt"]
        assert files(tmp_path / "again") == direct
        assert script.loaded == 1
        assert "<00:00:00.600>" in direct["talk.vtt"] and direct["talk.txt"].startswith("[00:00:00] Hello")

    def test_with_speakers_and_names(self, script, diarization, audio, tmp_path):
        names = ("--speaker-names", "Alice")
        run(audio, "-b", "script", "-q", "--diarize", *names, "--outdir", tmp_path / "direct", "-f", "all")
        saved = tmp_path / "direct" / "talk.json"

        run("--from-json", saved, "-q", "--outdir", tmp_path / "again", "-f", "all,rttm", *names)

        assert script.loaded == 1
        direct = files(tmp_path / "direct")
        assert sorted(direct) == ["talk.json", "talk.rttm", "talk.srt", "talk.tsv", "talk.txt", "talk.vtt"]
        assert files(tmp_path / "again") == direct
        assert direct["talk.rttm"].splitlines() == [
            "SPEAKER talk 1 0.000 2.200 <NA> <NA> Alice <NA> <NA>",
            "SPEAKER talk 1 2.500 6.500 <NA> <NA> SPEAKER_00 <NA> <NA>",
            "SPEAKER talk 1 3661.000 3.500 <NA> <NA> Alice <NA> <NA>",
        ]
        assert "SPEAKER_01" not in "".join(direct.values())
        assert json.loads(direct["talk.json"])["speaker_segments"][0]["speaker"] == "Alice"

    def test_to_stdout(self, script, audio, tmp_path, capsys):
        run(audio, "-b", "script", "-q", "--json", tmp_path / "talk.json", "-f", "vtt", "--word-level")
        direct = capsys.readouterr().out

        run("--from-json", tmp_path / "talk.json", "-f", "vtt", "--word-level")

        captured = capsys.readouterr()
        assert captured.out == direct and "<00:00:00.600>" in direct
        assert captured.err == ""  # no loading message, no progress

    def test_a_folder_of_results(self, script, tmp_path, no_model, capsys):
        (tmp_path / "in" / "sub").mkdir(parents=True)
        TALK.save(tmp_path / "in" / "a.json")
        MEETING.save(tmp_path / "in" / "sub" / "b.json")
        (tmp_path / "in" / "c.mp3").touch()  # not a result: left out

        run("--from-json", tmp_path / "in", "--outdir", tmp_path / "out", "-f", "srt")

        assert sorted(p.relative_to(tmp_path / "out").as_posix() for p in (tmp_path / "out").rglob("*")) == [
            "a.srt",
            "sub",
            "sub/b.srt",
        ]
        assert (tmp_path / "out" / "sub" / "b.srt").read_text(encoding="utf-8") == MEETING.render("srt")
        err = capsys.readouterr().err
        assert "[1/2]" in err and "Finished in" in err and "Loading" not in err

    def test_no_results_in_the_folder(self, script, tmp_path, no_model):
        (tmp_path / "talk.mp3").touch()

        with pytest.raises(SystemExit, match="no JSON files found"):
            run("--from-json", tmp_path, "-f", "srt")

    def test_a_file_that_is_not_a_result(self, script, tmp_path, no_model, capsys):
        (tmp_path / "talk.json").write_text('{"text": 1}', encoding="utf-8")

        with pytest.raises(SystemExit) as exit_info:
            run("--from-json", tmp_path / "talk.json", "-f", "srt")

        assert exit_info.value.code == 1
        assert "talk.json: 'text' must be a string, not a number" in capsys.readouterr().err

    def test_a_missing_file(self, script, tmp_path, no_model):
        with pytest.raises(SystemExit, match="file not found"):
            run("--from-json", tmp_path / "talk.json")

    def test_rttm_of_a_result_without_speakers_fails(self, script, tmp_path, no_model, capsys):
        TALK.save(tmp_path / "talk.json")

        with pytest.raises(SystemExit):
            run("--from-json", tmp_path / "talk.json", "-f", "rttm")

        assert "no speaker labels" in capsys.readouterr().err

    def test_does_not_replace_its_input(self, script, tmp_path, no_model):
        TALK.save(tmp_path / "talk.json")
        before = (tmp_path / "talk.json").read_text(encoding="utf-8")

        with pytest.raises(SystemExit, match="would replace a result it reads: .*talk.json"):
            run("--from-json", tmp_path / "talk.json", "-f", "srt,json")  # written next to the input
        with pytest.raises(SystemExit, match="would replace"):
            run("--from-json", tmp_path / "talk.json", "-q", "--json", tmp_path / "talk.json")

        assert (tmp_path / "talk.json").read_text(encoding="utf-8") == before
        assert not (tmp_path / "talk.srt").exists()

    def test_skip_existing_keeps_the_input(self, script, tmp_path, no_model):
        TALK.save(tmp_path / "talk.json")
        before = (tmp_path / "talk.json").read_text(encoding="utf-8")

        run("--from-json", tmp_path / "talk.json", "-q", "-f", "all", "--skip-existing")

        assert sorted(files(tmp_path)) == ["talk.json", "talk.srt", "talk.tsv", "talk.txt", "talk.vtt"]
        assert (tmp_path / "talk.json").read_text(encoding="utf-8") == before

    @pytest.mark.parametrize(
        "extra, flags",
        [
            (["-m", "base"], "--model"),
            (["--diarize"], "--diarize"),
            (["-l", "en", "--vad", "-O", "a=1"], "-O, --language, --vad"),
            (["--num-speakers", "2"], "--num-speakers"),
            (["-t", "translate"], "--task"),
        ],
    )
    def test_refuses_transcription_options(self, tmp_path, extra, flags, capsys):
        with pytest.raises(SystemExit):
            parse_args(["--from-json", str(tmp_path / "talk.json"), *extra])

        assert (
            f"--from-json writes saved results without transcribing, so it doesn't take {flags}"
            in capsys.readouterr().err
        )

    def test_refuses_standard_input(self, capsys):
        with pytest.raises(SystemExit):
            parse_args(["--from-json", "-"])

        assert "--from-json reads results saved in files, not standard input" in capsys.readouterr().err


class TestOutputOptions:
    def test_parsed(self):
        args = parse_args(
            [
                "a.mp3",
                "--max-line-width",
                "30",
                "--max-lines",
                "3",
                "--max-cue-duration",
                "4.5",
                "--timestamps",
                "--word-level",
            ]
        )

        assert (args.max_line_width, args.max_lines, args.max_cue_duration) == (30, 3, 4.5)
        assert args.timestamps and args.word_level and not args.from_json and args.speaker_names is None

    @pytest.mark.parametrize(
        "argv, message",
        [
            (["--max-lines", "3"], "--max-lines needs --max-line-width"),
            (["--speaker-names", "Alice"], "--speaker-names renames the speakers --diarize labels"),
            (["--max-cue-duration", "0"], "must be a positive number"),
            (["--max-lines", "0", "--max-line-width", "9"], "must be a positive integer"),
            (["--speaker-names", " , "], "expected names separated by commas"),
        ],
    )
    def test_refused(self, argv, message, capsys):
        with pytest.raises(SystemExit):
            parse_args(["a.mp3", *argv])

        assert message in capsys.readouterr().err

    def test_speaker_names(self):
        assert speaker_names_arg("Alice, Bob ,") == ("Alice", "Bob", "")
        assert parse_args(["a.mp3", "--diarize", "--speaker-names", ",Bob"]).speaker_names == ("", "Bob")

    @pytest.mark.parametrize("value, number", [("2", 2.0), ("0.25", 0.25), ("1e1", 10.0)])
    def test_positive_float(self, value, number):
        assert positive_float(value) == number

    @pytest.mark.parametrize("value", ["-1", "inf", "nan"])
    def test_positive_float_refuses(self, value):
        with pytest.raises(Exception, match="must be a positive number"):
            positive_float(value)

    def test_recording_id(self):
        assert recording_id(pathlib.Path("recordings/board meeting.mp3")) == "board meeting"
        assert recording_id(cli.STDIN) == "stdin"


class TestAllFormats:
    def test_rttm_only_with_diarize(self):
        everything = tuple(fmt for fmt in FORMATS if fmt != "rttm")
        assert parse_args(["a.mp3", "-f", "all"]).format == everything
        assert parse_args(["a.mp3", "-f", "all", "--diarize"]).format == FORMATS
        assert parse_args(["a.mp3", "-f", "all,rttm", "--diarize"]).format == FORMATS
        assert parse_args(["a.mp3", "-f", "rttm,srt", "--diarize"]).format == ("rttm", "srt")

    def test_expand_formats(self):
        assert expand_formats(None, diarize=True) is None
        assert expand_formats(["all", "rttm"], diarize=False) == FORMATS
        assert expand_formats(["srt"], diarize=False) == ("srt",)

    @pytest.mark.parametrize("argv", [["-f", "rttm"], ["-f", "srt,rttm", "--outdir", "out"], ["-o", "talk.rttm"]])
    def test_rttm_needs_diarize(self, script, audio, argv, monkeypatch):
        monkeypatch.chdir(audio.parent)

        with pytest.raises(SystemExit, match="rttm lists the speaker turns --diarize finds, so it needs --diarize"):
            run(audio, "-b", "script", "-q", *argv)

        assert script.loaded == 0  # refused before the model loads

    def test_diarized_run(self, script, diarization, audio, tmp_path):
        run(audio, "-b", "script", "-q", "--diarize", "-f", "all", "--outdir", tmp_path / "out")

        assert sorted(files(tmp_path / "out")) == [f"talk.{fmt}" for fmt in sorted(FORMATS)]


class TestWordTimestamps:
    """--word-level and --max-cue-duration ask the backend for word timings, as --max-line-width does."""

    @pytest.mark.parametrize(
        "argv, asked",
        [
            (["-f", "vtt", "--word-level"], True),
            (["-f", "srt", "--word-level"], False),  # word-level timings are for vtt only
            (["-f", "srt", "--max-cue-duration", "3"], True),
            (["-f", "txt", "--max-cue-duration", "3"], False),
            (["-f", "srt"], False),
        ],
    )
    def test_asked_for(self, script, audio, argv, asked):
        run(audio, "-b", "script", "-q", *argv)

        assert script.calls == [asked]

    @pytest.mark.filterwarnings("default::speech_toolkit.SpeechToolkitWarning")
    def test_a_backend_without_word_timings_warns(self, fake_backend, audio, capsys):
        run(audio, "-b", "fake", "-q", "-f", "vtt", "--word-level")

        captured = capsys.readouterr()
        assert "Warning: the fake backend doesn't support word timestamps; ignoring it." in captured.err
        assert captured.out == "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello from talk.\n\n"

    def test_max_cue_duration_without_word_timings(self, fake_backend, audio, capsys):
        run(audio, "-b", "fake", "-q", "-f", "srt", "--max-cue-duration", "0.75")  # no warning: they're optional

        # The words' times are shared out by length: "Hello" 0-0.54 s, "from" 0.54-0.96 s, "talk." 0.96-1.5 s.
        assert [lines for _, _, lines in parse_srt(capsys.readouterr().out[:-1])] == [["Hello"], ["from"], ["talk."]]
