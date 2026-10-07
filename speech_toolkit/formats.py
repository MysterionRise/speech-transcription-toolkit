"""Render transcription results as plain text, SRT, WebVTT or JSON."""

from __future__ import annotations

import html
import json
import pathlib
import unicodedata
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

FORMATS = ("txt", "srt", "vtt", "json")

MIN_CUE_SECONDS = 0.5
"""Subtitle cues last at least this long, unless the next cue starts sooner."""

# Unicode categories a line may not break before (combining marks, ー and 々, dashes, closing punctuation such as
# 。 and 」) or after (opening brackets and quotes such as 「). Format characters such as the zero-width joiner are in
# both, so they stay with the characters on either side.
_NO_BREAK_BEFORE = frozenset({"Mn", "Mc", "Me", "Cf", "Lm", "Sk", "Pd", "Pe", "Pf", "Po"})
_NO_BREAK_AFTER = frozenset({"Cf", "Ps", "Pi"})


def format_for_path(path: Optional[pathlib.Path], default: str = "txt") -> str:
    """Pick the output format from a file extension (``out.srt`` -> ``srt``).

    Without a path or an extension the format is *default*. An extension that isn't a format also gives *default*,
    with a warning.
    """
    suffix = path.suffix if path else ""
    fmt = suffix.lower().lstrip(".")
    if fmt in FORMATS:
        return fmt
    if fmt:
        warnings.warn(f"'{suffix}' isn't an output format ({', '.join(FORMATS)}); writing {default}.", stacklevel=2)
    return default


def _millis(seconds: float) -> int:
    """Whole milliseconds, as timestamps show them (never negative)."""
    return max(0, round(seconds * 1000))


def _timestamp(seconds: float, separator: str) -> str:
    """Format seconds as ``HH:MM:SS<separator>mmm``."""
    hours, millis = divmod(_millis(seconds), 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _cues(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Segments to show as subtitles: speaker-labelled ones if diarization ran, skipping empty text."""
    segments = result["speaker_segments"] if "speaker_segments" in result else result.get("segments", [])
    return [seg for seg in segments if seg.get("text", "").strip()]


def _speaker(cue: Dict[str, Any]) -> str:
    """The cue's speaker on one line, or '' without one."""
    return " ".join(str(cue.get("speaker") or "").split())


def _label(cue: Dict[str, Any]) -> str:
    """The ``[SPEAKER_00]`` label that txt and srt show before a speaker's text, or '' without a speaker."""
    speaker = _speaker(cue)
    return f"[{speaker}]" if speaker else ""


###############################################################################
# Line splitting
###############################################################################


def _char_width(char: str) -> int:
    """Columns *char* takes on screen: 2 for wide and fullwidth (CJK) characters, 0 for combining marks."""
    if unicodedata.category(char) in ("Mn", "Me", "Cf"):
        return 0
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


def _width(text: str) -> int:
    """Columns *text* takes on screen, counting wide and fullwidth (CJK) characters as two."""
    return sum(map(_char_width, text))


def _ideographic(char: str) -> bool:
    """Whether *char* is CJK text that lines may break around: wide, but not Hangul (Korean has spaces between words)."""
    return _char_width(char) == 2 and not unicodedata.name(char, "").startswith("HANGUL")


def _breaks_between(before: str, after: str) -> bool:
    """Whether a line may break between two adjacent characters: next to CJK text, except before closing punctuation
    such as 。 and after opening punctuation such as 「."""
    return (
        (_ideographic(before) or _ideographic(after))
        and unicodedata.category(after) not in _NO_BREAK_BEFORE
        and unicodedata.category(before) not in _NO_BREAK_AFTER
    )


@dataclass
class _Unit:
    """Text that lines break around: a word, or a CJK character with the punctuation that sticks to it."""

    text: str
    space: bool  # whitespace came before it, so on the same line a space joins it to the unit before
    start: float
    end: float


def _timed_words(cue: Dict[str, Any]) -> List[Tuple[str, float, float]]:
    """The cue's ``(word, start, end)`` timings, or its whole text over its whole time when it has none."""
    words = [w for w in cue.get("words") or [] if str(w.get("word") or "").strip()]
    if words and all("start" in w and "end" in w for w in words):
        return [(str(w["word"]), w["start"], w["end"]) for w in words]
    return [(str(cue["text"]), cue.get("start", 0.0), cue.get("end", 0.0))]


def _share(start: float, end: float, done: int, total: int) -> float:
    """The time after *done* of *total* characters, spread evenly from *start* to *end*."""
    return end if done >= total else start + (end - start) * done / total


def _units(words: Sequence[Tuple[str, float, float]]) -> List[_Unit]:
    """Split timed words into units; a unit's time is its characters' share of their word's time.

    The spacing of the words is kept: Whisper-style words start with a space when the transcript has one there.
    """
    units: List[_Unit] = []
    space, previous = False, ""
    for text, start, end in words:
        total, done = sum(not char.isspace() for char in text), 0
        for char in text:
            if char.isspace():
                space, previous = True, ""
                continue
            if previous and not _breaks_between(previous, char):
                unit = units[-1]
                unit.text += char
            else:
                unit = _Unit(char, space, _share(start, end, done, total), end)
                units.append(unit)
            done += 1
            unit.end = _share(start, end, done, total)
            space, previous = False, char
    return units


def _wrap(units: Sequence[_Unit], max_width: int, max_lines: int, label: str = "") -> List[List[List[int]]]:
    """Fill lines of at most *max_width* columns with unit indices, *max_lines* lines per piece.

    A unit wider than a line gets a line of its own. Each piece's first line starts with *label* and a space; when the
    first unit doesn't fit beside it, the label gets a line of its own, shown as a line without units.
    """
    pieces: List[List[List[int]]] = []
    label_width = _width(label) + 1 if label else 0  # the label and the space after it
    used = 0  # columns taken on the current line
    for i, unit in enumerate(units):
        size = _width(unit.text)
        gap = 1 if unit.space else 0
        lines = pieces[-1] if pieces else []
        if lines and used + gap + size <= max_width:
            lines[-1].append(i)
            used += gap + size
        elif lines and len(lines) < max_lines:
            lines.append([i])
            used = size
        elif label and max_lines > 1 and label_width + size > max_width:
            pieces.append([[], [i]])
            used = size
        else:
            pieces.append([[i]])
            used = label_width + size
    return pieces


def _line(units: Sequence[_Unit], ids: List[int]) -> str:
    """The text of the units *ids*, with a space between them where the transcript has one."""
    return "".join((" " if n and units[i].space else "") + units[i].text for n, i in enumerate(ids))


def split_cues(
    cues: List[Dict[str, Any]], max_width: int, max_lines: int = 2, *, labels: bool = False
) -> List[Dict[str, Any]]:
    """Break subtitle cues into pieces of at most *max_lines* lines of *max_width* columns.

    Wide and fullwidth (CJK) characters take two columns. Lines break at spaces and, in Chinese and Japanese text,
    between characters, without adding spaces and not before closing punctuation such as 。. A word longer than
    *max_width* gets a line of its own. With *labels*, each piece starts with its speaker's ``[SPEAKER_00]`` label, as
    SRT shows it, and the label counts toward the first line; a label that doesn't fit beside the first word gets a
    line of its own.

    Times come from the cue's word timestamps when it has them, otherwise its duration is shared out in
    proportion to text length. Cues are never merged, so every piece keeps its cue's speaker.
    """
    pieces: List[Dict[str, Any]] = []
    for cue in cues:
        units = _units(_timed_words(cue))
        label = _label(cue) if labels else ""
        for lines in _wrap(units, max_width, max_lines, label):
            ids = [i for line in lines for i in line]
            text = "\n".join(_line(units, line) for line in lines)
            if label:  # an empty first line means the label has that line to itself
                text = f"{label} {text}" if lines[0] else f"{label}{text}"
            piece = {"start": units[ids[0]].start, "end": units[ids[-1]].end, "text": text}
            if "speaker" in cue:
                piece["speaker"] = cue["speaker"]
            pieces.append(piece)
    return pieces


###############################################################################
# Subtitles
###############################################################################


def _whole(cue: Dict[str, Any], labels: bool) -> Dict[str, Any]:
    """The cue unsplit, without blank lines (they would end it early), and with *labels* its speaker label."""
    lines = [line.strip() for line in str(cue["text"]).splitlines() if line.strip()]
    label = _label(cue) if labels else ""
    if label:
        lines[0] = f"{label} {lines[0]}"
    return {**cue, "text": "\n".join(lines)}


def _fix_timing(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cues in time order, each lasting at least MIN_CUE_SECONDS (or until the next one starts), none overlapping.

    Times are whole milliseconds, as written. A cue that overlaps the next one ends when the next one starts; when two
    cues start together, the first lasts a millisecond and the second starts after it.
    """
    ordered = sorted(cues, key=lambda cue: _millis(cue.get("start", 0.0)))
    starts = [_millis(cue.get("start", 0.0)) for cue in ordered]
    minimum = round(MIN_CUE_SECONDS * 1000)
    fixed: List[Dict[str, Any]] = []
    previous_end = 0
    for k, cue in enumerate(ordered):
        start = max(starts[k], previous_end)
        end = _millis(cue.get("end", 0.0))
        following = starts[k + 1] if k + 1 < len(starts) else None
        if following is not None:
            end = min(end, following)
        if end - start < minimum:
            end = start + minimum if following is None else min(start + minimum, following)
        end = previous_end = max(end, start + 1)
        fixed.append({**cue, "start": start / 1000, "end": end / 1000})
    return fixed


def _subtitles(result: Dict[str, Any], max_line_width: Optional[int], labels: bool = False) -> List[Dict[str, Any]]:
    """The cues to write, split to *max_line_width*, with valid times; *labels* puts speaker labels in the text."""
    cues = _cues(result)
    if max_line_width:
        cues = split_cues(cues, max_line_width, labels=labels)
    else:
        cues = [_whole(cue, labels) for cue in cues]
    return _fix_timing(cues)


def to_txt(result: Dict[str, Any]) -> str:
    """Plain transcript; with diarization, one ``[SPEAKER] text`` line per speaker turn."""
    if "speaker_segments" not in result:
        return str(result.get("text", "")).strip()

    lines: List[str] = []
    current: Optional[str] = None
    for seg in _cues(result):
        label, text = _label(seg), seg["text"].strip()
        if lines and label == current:
            lines[-1] += f" {text}"
        else:
            current = label
            lines.append(f"{label} {text}" if label else text)
    return "\n".join(lines)


def to_srt(result: Dict[str, Any], max_line_width: Optional[int] = None) -> str:
    """SubRip subtitles: one cue per segment, or lines of at most *max_line_width* columns, speaker label included."""
    blocks = []
    for number, cue in enumerate(_subtitles(result, max_line_width, labels=True), start=1):
        start, end = _timestamp(cue["start"], ","), _timestamp(cue["end"], ",")
        blocks.append(f"{number}\n{start} --> {end}\n{cue['text']}\n")
    return "\n".join(blocks)


def to_vtt(result: Dict[str, Any], max_line_width: Optional[int] = None) -> str:
    """WebVTT subtitles like :func:`to_srt`; speakers become ``<v>`` voice tags.

    ``&``, ``<`` and ``>`` are escaped in the text and in voice names, so ``-->`` in the text can't end a cue.
    """
    blocks = ["WEBVTT\n"]
    for cue in _subtitles(result, max_line_width):
        text = html.escape(cue["text"], quote=False)
        speaker = _speaker(cue)
        if speaker:
            text = f"<v {html.escape(speaker, quote=False)}>{text}"
        start, end = _timestamp(cue["start"], "."), _timestamp(cue["end"], ".")
        blocks.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def to_json(result: Dict[str, Any]) -> str:
    """The full result (text, segments, language, speakers) as pretty-printed JSON."""
    return json.dumps(result, ensure_ascii=False, indent=2)


def render(result: Dict[str, Any], fmt: str, max_line_width: Optional[int] = None) -> str:
    """Render *result* in one of :data:`FORMATS`; *max_line_width* wraps srt/vtt subtitle lines."""
    if fmt == "txt":
        return to_txt(result)
    if fmt == "srt":
        return to_srt(result, max_line_width)
    if fmt == "vtt":
        return to_vtt(result, max_line_width)
    if fmt == "json":
        return to_json(result)
    raise ValueError(f"Unknown format: {fmt!r}. Available: {', '.join(FORMATS)}")


def write_text(path: pathlib.Path, text: str) -> None:
    """Write *text* to *path* as UTF-8 with a final newline, creating missing folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
