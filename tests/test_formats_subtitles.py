"""Subtitle correctness (#19): valid WebVTT, line widths with speaker labels and CJK text, cue timing, txt speakers
and unknown output extensions."""

from __future__ import annotations

import html
import html.entities
import pathlib
import random
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import pytest

from speech_toolkit import cli
from speech_toolkit.backends import TranscriptionResult
from speech_toolkit.errors import SpeechToolkitWarning
from speech_toolkit.formats import (
    MIN_CUE_SECONDS,
    _width,
    format_for_path,
    render,
    split_cues,
    to_srt,
    to_txt,
    to_vtt,
)

Cue = Tuple[int, int, List[str]]  # start and end in milliseconds, text lines

###############################################################################
# A strict WebVTT check and an SRT parser (W3C WebVTT, https://www.w3.org/TR/webvtt1/)
###############################################################################

TIMESTAMP = r"(?:\d{2,}:)?[0-5]\d:[0-5]\d\.\d{3}"
CHAR_REF = r"&(?:[A-Za-z][A-Za-z\d]*|#\d+|#[xX][\dA-Fa-f]+);"
CLASSES = r"(?:\.[^\s.&<>]+)*"
TAG = (
    rf"<(?:[cibu]|ruby|rt){CLASSES}>"  # spans without an annotation
    rf"|<(?:v|lang){CLASSES}[ \t](?:[^&>\r\n]|{CHAR_REF})+>"  # voice and language spans need one
    rf"|</(?:[cibu]|ruby|rt|v|lang)>"
    rf"|<{TIMESTAMP}>"
)
PAYLOAD_LINE = re.compile(rf"(?:[^&<\r\n]|{CHAR_REF}|{TAG})+")
TIMING_LINE = re.compile(rf"({TIMESTAMP})[ \t]+-->[ \t]+({TIMESTAMP})(?:[ \t][^\r\n]*)?")
SRT_TIMING = re.compile(r"(\d{2,}):([0-5]\d):([0-5]\d),(\d{3}) --> (\d{2,}):([0-5]\d):([0-5]\d),(\d{3})")


def vtt_millis(stamp: str) -> int:
    *hours, minutes, seconds = stamp.split(":")
    secs, millis = seconds.split(".")
    return ((int(hours[0]) if hours else 0) * 60 + int(minutes)) * 60_000 + int(secs) * 1000 + int(millis)


def check_payload(lines: List[str]) -> None:
    """Cue text: no ``-->``, ``&`` only in known character references, ``<`` only in well-formed, nested tags."""
    for line in lines:
        assert "-->" not in line, f"'-->' in cue text: {line!r}"
        assert PAYLOAD_LINE.fullmatch(line), f"invalid cue text: {line!r}"
        for name in re.findall(r"&([A-Za-z][A-Za-z\d]*);", line):
            assert f"{name};" in html.entities.html5, f"unknown character reference &{name}; in {line!r}"
    payload = "\n".join(lines)
    open_tags: List[Tuple[str, int]] = []  # (name, position)
    for tag in re.finditer(TAG, payload):
        if tag.group().startswith("</"):
            assert open_tags and open_tags.pop()[0] == tag.group()[2:-1], f"unmatched end tag in {payload!r}"
        elif not re.fullmatch(f"<{TIMESTAMP}>", tag.group()):
            open_tags.append((tag.group()[1:].split(".")[0].split()[0].rstrip(">"), tag.start()))
    # Only a voice span holding the whole cue text may leave out its end tag.
    assert open_tags in ([], [("v", 0)]), f"unclosed tag in {payload!r}"


def parse_vtt(text: str) -> List[Cue]:
    """Check *text* strictly against the WebVTT syntax and return its cues.

    Cue start times must not decrease and every cue must end after it starts. Stricter than the format on two points:
    blocks are separated by exactly one blank line, and only a voice span holding the whole cue text may leave out its
    end tag.
    """
    assert "\r" not in text and "\0" not in text
    header, _, body = text.partition("\n")
    assert re.fullmatch(r"WEBVTT(?:[ \t][^\n]*)?", header), f"bad header: {header!r}"
    if not body:
        return []
    assert body.startswith("\n") and body.endswith("\n"), "a blank line after the header and a final newline"
    cues: List[Cue] = []
    for block in body[1:-1].split("\n\n"):
        lines = block.split("\n")
        assert all(lines), f"extra blank line near {block!r}"
        if "-->" not in lines[0]:
            lines = lines[1:]  # cue identifier
        assert lines, f"cue without timings: {block!r}"
        timing = TIMING_LINE.fullmatch(lines[0])
        assert timing, f"bad cue timings: {lines[0]!r}"
        start, end = vtt_millis(timing.group(1)), vtt_millis(timing.group(2))
        assert end > start, f"cue ends before it starts: {lines[0]!r}"
        assert not cues or start >= cues[-1][0], f"cue starts before the previous one: {lines[0]!r}"
        check_payload(lines[1:])
        cues.append((start, end, lines[1:]))
    return cues


def parse_srt(text: str) -> List[Cue]:
    """Check *text* against the SubRip format (numbered cues, no blank lines in a cue) and return its cues."""
    if not text:
        return []
    assert text.endswith("\n") and "\r" not in text
    cues: List[Cue] = []
    for number, block in enumerate(text[:-1].split("\n\n"), start=1):
        lines = block.split("\n")
        assert len(lines) >= 3 and all(lines), f"bad cue: {block!r}"
        assert lines[0] == str(number), f"cue number {lines[0]!r}, expected {number}"
        timing = SRT_TIMING.fullmatch(lines[1])
        assert timing, f"bad cue timings: {lines[1]!r}"
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, timing.groups())
        cues.append(((h1 * 60 + m1) * 60_000 + s1 * 1000 + ms1, (h2 * 60 + m2) * 60_000 + s2 * 1000 + ms2, lines[2:]))
    return cues


def check_timing(cues: List[Cue]) -> None:
    """Every cue ends after it starts, and no two cues overlap."""
    for k, (start, end, lines) in enumerate(cues):
        assert start < end, f"cue {k + 1} {lines} doesn't end after it starts"
        assert k + 1 == len(cues) or end <= cues[k + 1][0], f"cue {k + 1} {lines} overlaps the next one"


def columns(text: str) -> int:
    """Display width, worked out independently of formats._width: wide and fullwidth characters take two columns."""
    return sum(2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in text)


def shown(line: str) -> str:
    """What a player shows for a line of WebVTT cue text: tags removed, character references decoded."""
    return html.unescape(re.sub(TAG, "", line))


def diarized(*segments: Tuple[float, float, str, Any]) -> Dict[str, Any]:
    """A diarized result from ``(start, end, text, speaker)`` tuples."""
    return {
        "text": "",
        "segments": [],
        "speaker_segments": [{"start": s, "end": e, "text": t, "speaker": who} for s, e, t, who in segments],
    }


def plain(*segments: Tuple[float, float, str]) -> Dict[str, Any]:
    """A result from ``(start, end, text)`` tuples."""
    return {"text": "", "segments": [{"start": s, "end": e, "text": t} for s, e, t in segments]}


def srt_times(result: Dict[str, Any], max_line_width: Optional[int] = None) -> List[Tuple[int, int]]:
    return [(start, end) for start, end, _ in parse_srt(to_srt(result, max_line_width))]


VTT = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n{}\n"


class TestStrictWebVttCheck:
    """The check the other tests rely on accepts valid cue text and rejects what #19 reported."""

    @pytest.mark.parametrize(
        "payload",
        [
            "Tom &amp; Jerry &lt;3 --&gt; here",
            "<v A&lt;B>hi\nthere",
            "<v.loud Ann>hi</v> <i>there</i> &nbsp;&#169;&#xA9;",
            "<c.a.b>x</c><00:00:00.500>y",
        ],
    )
    def test_accepts(self, payload):
        assert parse_vtt(VTT.format(payload)) == [(0, 1000, payload.split("\n"))]

    @pytest.mark.parametrize(
        "payload",
        [
            "Tom & Jerry",  # bare ampersand
            "I <3 you",  # '<' that isn't a tag
            "a --> b",  # would start a new cue
            "&bogus;",  # unknown character reference
            "<v Tom & Jerry>hi",  # bare ampersand in a voice name
            "<v>hi",  # voice span without a name
            "<i>open",  # unclosed tag
            "</v>hi",  # end tag without a start tag
            "a\n\nb",  # blank line ends the cue
        ],
    )
    def test_rejects(self, payload):
        with pytest.raises(AssertionError):
            parse_vtt(VTT.format(payload))

    @pytest.mark.parametrize(
        "text",
        [
            "WEBVTT\n\n00:00:01.000 --> 00:00:01.000\nzero length\n",
            "WEBVTT\n\n00:00:02.000 --> 00:00:03.000\nb\n\n00:00:01.000 --> 00:00:04.000\na\n",
            "WEBVTT\n00:00:00.000 --> 00:00:01.000\nno blank line after the header\n",
            "WEBVTT\n\n0:00:00.000 --> 00:00:01.000\nbad timestamp\n",
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nno final newline",
            "WEBVTT FILE\n\n00:00:00.000 --> 00:00:01.000\nok\n\n\n00:00:01.000 --> 00:00:02.000\ntwo blank lines\n",
        ],
    )
    def test_rejects_files(self, text):
        with pytest.raises(AssertionError):
            parse_vtt(text)

    def test_timing_check(self):
        check_timing([(0, 1000, ["a"]), (1000, 2000, ["b"])])
        with pytest.raises(AssertionError, match="overlaps"):
            check_timing([(0, 1500, ["a"]), (1000, 2000, ["b"])])


###############################################################################
# WebVTT escaping and blank lines
###############################################################################


class TestVttEscaping:
    def test_text_and_voice_names_are_escaped(self):
        vtt = to_vtt(diarized((0.0, 1.0, " Tom & Jerry <3 --> here", "A<B")))

        assert vtt == "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n<v A&lt;B>Tom &amp; Jerry &lt;3 --&gt; here\n"
        [(_, _, [line])] = parse_vtt(vtt)
        assert shown(line) == "Tom & Jerry <3 --> here"

    @pytest.mark.parametrize(
        "speaker, voice",
        [("A<B", "A<B"), ("A>B", "A>B"), ("Tom & Jerry", "Tom & Jerry"), ("line\nbreak", "line break"), (" x ", "x")],
    )
    def test_voice_names_survive(self, speaker, voice):
        [(_, _, [line])] = parse_vtt(to_vtt(diarized((0.0, 1.0, "hi", speaker))))

        name = re.fullmatch(r"<v ([^>]*)>hi", line)
        assert name and html.unescape(name.group(1)) == voice

    @pytest.mark.parametrize("speaker", ["", "  ", None])
    def test_no_voice_tag_without_a_speaker(self, speaker):
        assert to_vtt(diarized((0.0, 1.0, "hi", speaker))).endswith("\nhi\n")

    @pytest.mark.parametrize("max_line_width", [None, 12])
    def test_escaped_split_cues_are_valid(self, max_line_width):
        result = diarized((0.0, 4.0, "if a < b && b > c --> done", "S&P"), (4.0, 6.0, "<i>not a tag</i>", "A>B"))

        cues = parse_vtt(to_vtt(result, max_line_width))

        assert " ".join(shown(line) for _, _, lines in cues for line in lines) == (
            "if a < b && b > c --> done <i>not a tag</i>"
        )


@pytest.mark.parametrize("fmt", ["srt", "vtt"])
def test_blank_lines_inside_a_segment_are_dropped(fmt):
    """A blank line would end the cue early."""
    assert render(plain((0.0, 2.0, " one\n\n  two \n")), fmt).endswith("\none\ntwo\n")


###############################################################################
# Line width
###############################################################################


class TestLineWidth:
    def test_srt_speaker_label_counts(self):
        """The example from #19: at width 20 the first line used to be 27 characters."""
        result = diarized((0.0, 3.0, "Hello there how are you doing today my friend", "SPEAKER_00"))

        cues = parse_srt(to_srt(result, max_line_width=20))

        assert [lines for _, _, lines in cues] == [
            ["[SPEAKER_00] Hello", "there how are you"],
            ["[SPEAKER_00] doing", "today my friend"],
        ]

    def test_label_gets_its_own_line_when_the_first_word_does_not_fit(self):
        cues = parse_srt(to_srt(diarized((0.0, 2.0, "transcription is fun", "SPEAKER_00")), max_line_width=20))

        assert [lines for _, _, lines in cues] == [["[SPEAKER_00]", "transcription is fun"]]

    def test_label_wider_than_the_line(self):
        """Like a long word, a label that can't fit gets a line of its own."""
        cues = parse_srt(to_srt(diarized((0.0, 2.0, "a b c", "SPEAKER_00")), max_line_width=4))

        assert [lines for _, _, lines in cues] == [["[SPEAKER_00]", "a b"], ["[SPEAKER_00]", "c"]]

    def test_one_line_cues_keep_the_label_with_the_text(self):
        cue = {"start": 0.0, "end": 1.0, "text": "transcription", "speaker": "SPEAKER_00"}

        pieces = split_cues([cue], max_width=10, max_lines=1, labels=True)

        assert pieces == [{**cue, "text": "[SPEAKER_00] transcription"}]

    def test_vtt_voice_tags_and_escapes_do_not_count(self):
        """Players don't show them, so the shown text fills the line."""
        cues = parse_vtt(to_vtt(diarized((0.0, 2.0, "Tom & Jerry <3 you", "SPEAKER_00")), max_line_width=11))

        assert [[shown(line) for line in lines] for _, _, lines in cues] == [["Tom & Jerry", "<3 you"]]


###############################################################################
# Chinese, Japanese and Korean
###############################################################################

JAPANESE = "今日はとても良い天気ですね。散歩に行きましょう。"
CHINESE = "我们今天去公园散步，然后一起吃午饭。"


class TestCjk:
    @pytest.mark.parametrize(
        "text, width",
        [("abc", 3), ("日本語", 6), ("한국어", 6), ("ＡＢ", 4), ("ｱｲ", 2), ("e\u0301", 1), ("a\u200db", 2)],
    )
    def test_width_counts_wide_characters_twice(self, text, width):
        assert _width(text) == width

    def test_japanese_wraps_without_spaces(self):
        pieces = split_cues([{"start": 0.0, "end": 3.0, "text": JAPANESE}], max_width=10)

        assert [p["text"] for p in pieces] == ["今日はとて\nも良い天気", "ですね。散\n歩に行きま", "しょう。"]

    @pytest.mark.parametrize("text", [JAPANESE, CHINESE])
    @pytest.mark.parametrize("width", [4, 7, 10, 16, 40])
    def test_lines_fit_and_keep_the_text(self, text, width):
        pieces = split_cues([{"start": 0.0, "end": 5.0, "text": text}], max_width=width)

        lines = [line for piece in pieces for line in piece["text"].split("\n")]
        assert "".join(lines) == text  # nothing added (no spaces) or lost
        assert all(columns(line) <= width for line in lines)
        assert not any(line.startswith(("。", "、", "，")) for line in lines)

    def test_no_line_starts_with_closing_or_ends_with_opening_punctuation(self):
        pieces = split_cues([{"start": 0.0, "end": 2.0, "text": "彼は「はい」と言った。"}], max_width=4, max_lines=9)

        assert [p["text"] for p in pieces] == ["彼は\n「は\nい」\nと言\nっ\nた。"]

    def test_timed_words_are_joined_without_spaces(self):
        timings = [("今日", 0.0, 0.5), ("は", 0.5, 0.75), ("とても", 0.75, 1.5), ("良い", 1.5, 2.0), ("天気", 2.0, 2.5)]
        words = [{"word": w, "start": s, "end": e} for w, s, e in timings + [("です。", 2.5, 3.25)]]
        cue = {"start": 0.0, "end": 3.25, "text": "今日はとても良い天気です。", "words": words}

        pieces = split_cues([cue], max_width=8)

        # Times come from the words; a word split across lines shares its time out by characters.
        assert pieces == [
            {"start": 0.0, "end": 2.0, "text": "今日はと\nても良い"},
            {"start": 2.0, "end": 3.25, "text": "天気で\nす。"},
        ]

    def test_spaces_in_mixed_text_are_kept(self):
        pieces = split_cues([{"start": 0.0, "end": 1.0, "text": "今日は Python を使います"}], max_width=13)

        assert [p["text"] for p in pieces] == ["今日は Python\nを使います"]

    def test_korean_breaks_at_spaces(self):
        """Korean puts spaces between words, so words stay whole; each syllable takes two columns."""
        pieces = split_cues([{"start": 0.0, "end": 1.0, "text": "안녕하세요 여러분 반갑습니다"}], max_width=12)

        assert [p["text"] for p in pieces] == ["안녕하세요\n여러분", "반갑습니다"]

    def test_srt_and_vtt(self):
        result = diarized((0.0, 3.0, JAPANESE, "SPEAKER_00"))

        srt = [lines for _, _, lines in parse_srt(to_srt(result, max_line_width=20))]
        vtt = [[shown(line) for line in lines] for _, _, lines in parse_vtt(to_vtt(result, max_line_width=20))]

        assert srt == [["[SPEAKER_00] 今日は", "とても良い天気です"], ["[SPEAKER_00] ね。散", "歩に行きましょう。"]]
        assert vtt == [["今日はとても良い天気", "ですね。散歩に行きま"], ["しょう。"]]


###############################################################################
# Cue timing
###############################################################################


class TestCueTiming:
    def test_minimum_duration(self):
        assert MIN_CUE_SECONDS == 0.5
        result = plain((1.0, 1.0, "zero"), (2.0, 2.2, "short"), (5.0, 6.0, "long enough"))

        assert srt_times(result) == [(1000, 1500), (2000, 2500), (5000, 6000)]

    def test_minimum_duration_stops_at_the_next_cue(self):
        assert srt_times(plain((1.0, 1.0, "zero"), (1.2, 2.0, "next"))) == [(1000, 1200), (1200, 2000)]

    def test_overlaps_are_trimmed(self):
        assert srt_times(plain((0.0, 2.5, "one"), (2.0, 4.0, "two"))) == [(0, 2000), (2000, 4000)]

    def test_cues_starting_together_are_staggered(self):
        result = plain((3.0, 3.0, "a"), (3.0, 3.0, "b"), (3.0, 4.0, "c"))

        assert srt_times(result) == [(3000, 3001), (3001, 3002), (3002, 4000)]

    def test_end_before_start(self):
        assert srt_times(plain((2.0, 1.0, "odd"))) == [(2000, 2500)]

    def test_times_are_fixed_after_rounding_to_milliseconds(self):
        assert srt_times(plain((1.0001, 1.0004, "x"), (1.0004, 1.8, "y"))) == [(1000, 1001), (1001, 1800)]

    def test_cues_are_written_in_time_order(self):
        cues = parse_srt(to_srt(plain((5.0, 6.0, "later"), (1.0, 2.0, "earlier"))))

        assert [(start, lines) for start, _, lines in cues] == [(1000, ["earlier"]), (5000, ["later"])]

    def test_split_cues_with_zero_length_words(self):
        words = [{"word": " one", "start": 0.0, "end": 0.5}]
        words += [{"word": f" {word}", "start": 0.5, "end": 0.5} for word in ("two", "six", "ten", "end")]
        result = {"text": "", "segments": [{"start": 0.0, "end": 0.5, "text": " one two six ten end", "words": words}]}

        # "one two\nsix ten" (0-0.5 s) and "end" (zero length, so lengthened)
        assert srt_times(result, max_line_width=7) == [(0, 500), (500, 1000)]
        # "one\ntwo", then "six\nten" and "end", both zero length at 0.5 s: the first gets a millisecond
        cues = parse_vtt(to_vtt(result, max_line_width=3))
        assert [(start, end) for start, end, _ in cues] == [(0, 500), (500, 501), (501, 1001)]


###############################################################################
# txt and output extensions
###############################################################################


class TestTxt:
    def test_segments_without_a_speaker(self):
        result = {
            "text": "",
            "speaker_segments": [
                {"start": 0.0, "end": 1.0, "text": " Hi."},
                {"start": 1.0, "end": 2.0, "text": " There.", "speaker": None},
                {"start": 2.0, "end": 3.0, "text": " Bye.", "speaker": "SPEAKER_00"},
                {"start": 3.0, "end": 4.0, "text": " Again.", "speaker": ""},
            ],
        }

        assert to_txt(result) == "Hi. There.\n[SPEAKER_00] Bye.\nAgain."

    def test_speaker_label_stays_on_one_line(self):
        assert to_txt(diarized((0.0, 1.0, "Hi", "Ann\nLee"))) == "[Ann Lee] Hi"


class TestUnknownExtension:
    @pytest.mark.parametrize("name, suffix", [("out.tsv", ".tsv"), ("notes.md", ".md"), ("OUT.Docx", ".Docx")])
    def test_warns_and_writes_txt(self, name, suffix):
        message = f"'{suffix}' isn't an output format (txt, srt, vtt, json); writing txt."
        with pytest.warns(SpeechToolkitWarning, match=re.escape(message)) as caught:
            assert format_for_path(pathlib.Path(name)) == "txt"

        assert [warning.category for warning in caught] == [SpeechToolkitWarning]

    def test_known_or_missing_extension_does_not_warn(self):
        """Warnings from speech_toolkit fail tests, so these would fail if they warned."""
        assert format_for_path(pathlib.Path("talk.SRT")) == "srt"
        assert format_for_path(pathlib.Path("transcript")) == "txt"
        assert format_for_path(None, default="vtt") == "vtt"

    def test_save(self, tmp_path):
        result = TranscriptionResult(" Hi.", [{"start": 0.0, "end": 1.0, "text": " Hi."}])

        with pytest.warns(SpeechToolkitWarning, match="'.tsv' isn't an output format"):
            result.save(tmp_path / "out.tsv")

        assert (tmp_path / "out.tsv").read_text(encoding="utf-8") == "Hi.\n"
        result.save(tmp_path / "out2.tsv", fmt="srt")  # an explicit format doesn't warn

    # Only SpeechToolkitWarning is let through: pyproject.toml turns any other warning from speech_toolkit into an error.
    @pytest.mark.filterwarnings("default::speech_toolkit.SpeechToolkitWarning")
    def test_cli_prints_one_warning_line(self, fake_backend, tmp_path, capsys):
        audio = tmp_path / "talk.mp3"
        audio.touch()

        cli.main([str(audio), "-b", "fake", "-q", "-o", str(tmp_path / "talk.tsv")])
        cli.main([str(audio), "-b", "fake", "-q", "-o", str(tmp_path / "talk2.tsv"), "-f", "txt"])  # no warning

        captured = capsys.readouterr()
        assert [line for line in captured.err.splitlines() if "Warning" in line] == [
            "Warning: '.tsv' isn't an output format (txt, srt, vtt, json); writing txt."
        ]
        assert captured.out == ""  # the transcripts went to the files
        assert (tmp_path / "talk.tsv").read_text(encoding="utf-8") == "Hello from talk.\n"
        assert (tmp_path / "talk2.tsv").read_text(encoding="utf-8") == "Hello from talk.\n"


###############################################################################
# Everything at once, on random transcripts
###############################################################################

LATIN = ["a", "to", "the", "word", "Hello,", "Tom", "&", "<3", "-->", "a<b>c", "x&amp;y", "supercalifragilistic"]
HANGUL = ["안녕하세요", "여러분"]
# Chinese and Japanese words never start with closing punctuation, so the longest unbreakable piece is 4 columns.
CJK = ["今日", "は", "天気。", "「東京」", "我们，", "（公园）", "とても", "です、"]
SPEAKERS = ["SPEAKER_00", "SPEAKER_01", "A<B", "Tom & Jerry", "张三", "", None]
IDEOGRAPHS = "".join(sorted({char for word in CJK for char in word if unicodedata.category(char) == "Lo"}))


def random_segment(rng: random.Random, diarize: bool) -> Dict[str, Any]:
    """Whisper-style words: Latin and Korean words start with a space (sometimes a blank line), CJK words don't."""
    words = []
    for _ in range(rng.randint(1, 10)):
        kind = rng.random()
        space = "\n\n" if rng.random() < 0.05 else " "
        if kind < 0.45:
            words.append(space + rng.choice(LATIN))
        elif kind < 0.55:
            words.append(space + rng.choice(HANGUL))
        else:
            words.append(rng.choice(CJK))
    start = round(rng.uniform(0, 8), rng.choice([1, 3, 4]))
    end = start + rng.choice([0.0, -0.3, 0.0004, 0.2, 1.0, 2.5, 4.0])
    segment: Dict[str, Any] = {"start": start, "end": end, "text": "".join(words)}
    if rng.random() < 0.5:  # word timings, some of them zero-length
        step = (end - start) / len(words)
        timed = [(start + step * i, start + step * (i + rng.choice([0, 1, 1]))) for i in range(len(words))]
        segment["words"] = [{"word": w, "start": s, "end": e} for w, (s, e) in zip(words, timed)]
    if diarize:
        segment["speaker"] = rng.choice(SPEAKERS)
    return segment


def output_text(lines: List[str], vtt: bool) -> List[str]:
    """Shown text of a cue's lines, without the srt speaker label."""
    if vtt:
        return [shown(line) for line in lines]
    first = re.sub(r"^\[[^\]]*\]", "", lines[0]).lstrip(" ")
    return ([first] if first else []) + lines[1:]


@pytest.mark.parametrize("seed", range(20))
def test_random_transcripts(seed):
    rng = random.Random(seed)
    for _ in range(10):
        diarize = rng.random() < 0.6
        segments = [random_segment(rng, diarize) for _ in range(rng.randint(1, 8))]
        result = {"text": "", "segments": segments, **({"speaker_segments": segments} if diarize else {})}
        width = rng.choice([None, 4, 5, 8, 13, 20, 42])
        atoms = {word.strip() for word in LATIN + HANGUL}
        atoms |= {f"[{name}]" for name in SPEAKERS if name}
        source = sorted("".join(seg["text"] for seg in segments).replace(" ", "").replace("\n", ""))

        for vtt, cues in ((False, parse_srt(to_srt(result, width))), (True, parse_vtt(to_vtt(result, width)))):
            check_timing(cues)
            text = [line for _, _, lines in cues for line in output_text(lines, vtt)]
            assert sorted("".join(text).replace(" ", "")) == source, "text lost or added"
            assert not re.search(f"[{IDEOGRAPHS}] [{IDEOGRAPHS}]", "\n".join(text)), "space between CJK characters"
            if width:
                for _, _, lines in cues:
                    assert len(lines) <= 2
                    for line in lines if not vtt else [shown(line) for line in lines]:
                        assert columns(line) <= width or line in atoms, f"line too wide at {width}: {line!r}"
