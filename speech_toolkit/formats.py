"""Render transcription results as plain text, SRT, WebVTT, JSON, TSV or RTTM.

:func:`render` takes a result as a dict, as ``TranscriptionResult.to_dict()`` gives it; ``TranscriptionResult.render()``
and ``save()`` call it with the same options.
"""

from __future__ import annotations

import html
import json
import math
import numbers
import pathlib
import sys
import unicodedata
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .errors import SpeechToolkitWarning, UnsupportedOptionError

FORMATS = ("txt", "srt", "vtt", "json", "tsv", "rttm")

SPEAKER_FORMATS = frozenset({"rttm"})
"""The formats that need speaker labels (diarization): their lines are speaker turns."""

MIN_CUE_SECONDS = 0.5
"""Subtitle cues last at least this long, unless the next cue starts sooner (or ``max_cue_duration`` is shorter)."""

DEFAULT_MAX_LINES = 2
"""Lines per subtitle cue when ``max_line_width`` splits cues, unless ``max_lines`` says otherwise."""

UNKNOWN_SPEAKER = "unknown"
"""The label diarization gives a segment that no speaker turn covers: only a mapping renames it, and rttm leaves it
out."""

DEFAULT_FILE_ID = "audio"
"""The file ID of rttm lines when none is given (``TranscriptionResult.save()`` uses the file's name)."""

SpeakerNames = Union[str, Sequence[str], Mapping[str, str]]
"""What ``speaker_names`` takes: new names in order of first appearance, as a list or as one string separated by
commas (``"Alice,Bob"``), or a mapping from labels to names (``{"SPEAKER_01": "Bob"}``)."""

# Unicode categories a line may not break before (combining marks, ー and 々, dashes, closing punctuation such as
# 。 and 」) or after (opening brackets and quotes such as 「). Format characters such as the zero-width joiner are in
# both, so they stay with the characters on either side.
_NO_BREAK_BEFORE = frozenset({"Mn", "Mc", "Me", "Cf", "Lm", "Sk", "Pd", "Pe", "Pf", "Po"})
_NO_BREAK_AFTER = frozenset({"Cf", "Ps", "Pi"})

_UNLIMITED = sys.maxsize  # the line width of cues split only by time
_CHUNKS = "_chunks"  # the key of a split cue's timed pieces of text, for word-level vtt


def format_for_path(path: Optional[pathlib.Path], default: str = "txt") -> str:
    """Pick the output format from a file extension (``out.srt`` -> ``srt``).

    Without a path or an extension the format is *default*. An extension that isn't a format also gives *default*,
    with a :class:`~speech_toolkit.SpeechToolkitWarning`.
    """
    suffix = path.suffix if path else ""
    fmt = suffix.lower().lstrip(".")
    if fmt in FORMATS:
        return fmt
    if fmt:
        warnings.warn(
            f"'{suffix}' isn't an output format ({', '.join(FORMATS)}); writing {default}.",
            SpeechToolkitWarning,
            stacklevel=2,
        )
    return default


def _millis(seconds: float) -> int:
    """Whole milliseconds, as timestamps show them (never negative)."""
    return max(0, round(seconds * 1000))


def _clock(millis: int, separator: str) -> str:
    """Format milliseconds as ``HH:MM:SS<separator>mmm``."""
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _timestamp(seconds: float, separator: str) -> str:
    """Format seconds as ``HH:MM:SS<separator>mmm``."""
    return _clock(_millis(seconds), separator)


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


def _one_line(text: str) -> str:
    """*text* on one line: runs of whitespace, tabs and line breaks become one space."""
    return " ".join(text.split())


###############################################################################
# Speaker names
###############################################################################


def speaker_mapping(result: Dict[str, Any], names: SpeakerNames) -> Dict[str, str]:
    """The new name of each speaker label that *names* renames in *result*, such as ``{"SPEAKER_00": "Alice"}``.

    A list, or a string of names separated by commas, names the speakers in the order they first speak. An empty name
    keeps that speaker's label, and speakers beyond the list keep theirs; the same name twice makes those speakers
    one. A mapping renames the labels it names. ``unknown``, the label of speech no speaker turn covers, is renamed
    only by a mapping.
    """
    if isinstance(names, Mapping):
        return {str(label): str(name).strip() for label, name in names.items() if str(name).strip()}
    if isinstance(names, str):
        names = names.split(",")
    labels: List[str] = []  # in order of first appearance
    for cue in sorted(_cues(result), key=lambda cue: cue.get("start") or 0.0):
        label = cue.get("speaker")
        if label and label != UNKNOWN_SPEAKER and label not in labels:
            labels.append(label)
    return {label: str(name).strip() for label, name in zip(labels, names) if str(name).strip()}


def rename_speakers(result: Dict[str, Any], names: SpeakerNames) -> Dict[str, Any]:
    """*result* with its speakers renamed, as :func:`speaker_mapping` says; *result* itself isn't changed."""
    mapping = speaker_mapping(result, names)
    key = "speaker_segments" if "speaker_segments" in result else "segments"
    if not mapping or not result.get(key):
        return result
    renamed = [
        {**seg, "speaker": mapping.get(seg["speaker"], seg["speaker"])} if seg.get("speaker") in mapping else seg
        for seg in result[key]
    ]
    return {**result, key: renamed}


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


def _join(before: str, after: str) -> str:
    """Two pieces of text on one line: with a space between them, but none next to Chinese or Japanese text."""
    if _ideographic(before[-1]) or _ideographic(after[0]):
        return before + after
    return f"{before} {after}"


@dataclass
class _Unit:
    """Text that lines break around: a word, or a CJK character with the punctuation that sticks to it."""

    text: str
    space: bool  # whitespace came before it, so on the same line a space joins it to the unit before
    start: float
    end: float
    word: bool = False  # it starts one of the cue's timed words, so word-level vtt marks its start


def _word_times(cue: Dict[str, Any]) -> List[Tuple[str, float, float]]:
    """The cue's ``(word, start, end)`` timings, or [] when it has none (or a word has no times)."""
    words = [w for w in cue.get("words") or [] if str(w.get("word") or "").strip()]
    if words and all("start" in w and "end" in w for w in words):
        return [(str(w["word"]), w["start"], w["end"]) for w in words]
    return []


def _share(start: float, end: float, done: int, total: int) -> float:
    """The time after *done* of *total* characters, spread evenly from *start* to *end*."""
    return end if done >= total else start + (end - start) * done / total


def _units(cue: Dict[str, Any]) -> List[_Unit]:
    """Split the cue's timed words into units; a unit's time is its characters' share of their word's time. A cue
    without word timings is one word: its whole text over its whole time.

    The spacing of the words is kept: Whisper-style words start with a space when the transcript has one there.
    """
    words = _word_times(cue)
    timed = bool(words)
    if not timed:
        words = [(str(cue["text"]), cue.get("start", 0.0), cue.get("end", 0.0))]
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
                unit = _Unit(char, space, _share(start, end, done, total), end, word=timed and done == 0)
                units.append(unit)
            done += 1
            unit.end = _share(start, end, done, total)
            space, previous = False, char
    return units


def _wrap(
    units: Sequence[_Unit], max_width: int, max_lines: int, label: str = "", max_millis: Optional[int] = None
) -> List[List[List[int]]]:
    """Fill lines of at most *max_width* columns with unit indices, *max_lines* lines per piece, and with
    *max_millis* pieces that last at most that many milliseconds (a unit that lasts longer gets a piece of its own).

    A unit wider than a line gets a line of its own. Each piece's first line starts with *label* and a space; when the
    first unit doesn't fit beside it, the label gets a line of its own, shown as a line without units.
    """
    pieces: List[List[List[int]]] = []
    label_width = _width(label) + 1 if label else 0  # the label and the space after it
    used = 0  # columns taken on the current line
    first = 0  # when the current piece starts, in milliseconds
    for i, unit in enumerate(units):
        size = _width(unit.text)
        gap = 1 if unit.space else 0
        lines = pieces[-1] if pieces else []
        in_time = max_millis is None or _millis(unit.end) - first <= max_millis
        if lines and in_time and used + gap + size <= max_width:
            lines[-1].append(i)
            used += gap + size
        elif lines and in_time and len(lines) < max_lines:
            lines.append([i])
            used = size
        elif label and max_lines > 1 and label_width + size > max_width:
            pieces.append([[], [i]])
            used, first = size, _millis(unit.start)
        else:
            pieces.append([[i]])
            used, first = label_width + size, _millis(unit.start)
    return pieces


def _line(units: Sequence[_Unit], ids: List[int]) -> str:
    """The text of the units *ids*, with a space between them where the transcript has one."""
    return "".join((" " if n and units[i].space else "") + units[i].text for n, i in enumerate(ids))


def _chunks(units: Sequence[_Unit], lines: List[List[int]]) -> List[Tuple[Optional[float], str]]:
    """The text of a piece (*lines* of unit indices) cut where its timed words start, as ``(start, text)`` pairs: the
    start of the word the text begins with, or None. Each text but the first begins with the space or line break
    before it, so the texts joined give the piece's text."""
    chunks: List[Tuple[Optional[float], str]] = []
    for number, line in enumerate(lines):
        for n, i in enumerate(line):
            unit = units[i]
            before = (" " if unit.space else "") if n else ("\n" if number else "")
            if chunks and not unit.word:
                chunks[-1] = (chunks[-1][0], chunks[-1][1] + before + unit.text)
            else:
                chunks.append((unit.start if unit.word else None, before + unit.text))
    return chunks


def _split(
    cue: Dict[str, Any], max_width: int, max_lines: int, label: str = "", max_millis: Optional[int] = None
) -> List[Dict[str, Any]]:
    """The pieces :func:`split_cues` breaks *cue* into, each with its timed text under ``_CHUNKS``."""
    units = _units(cue)
    pieces: List[Dict[str, Any]] = []
    for lines in _wrap(units, max_width, max_lines, label, max_millis):
        ids = [i for line in lines for i in line]
        text = "\n".join(_line(units, line) for line in lines)
        if label:  # an empty first line means the label has that line to itself
            text = f"{label} {text}" if lines[0] else f"{label}{text}"
        piece = {"start": units[ids[0]].start, "end": units[ids[-1]].end, "text": text}
        if "speaker" in cue:
            piece["speaker"] = cue["speaker"]
        piece[_CHUNKS] = _chunks(units, lines)
        pieces.append(piece)
    return pieces


def split_cues(
    cues: List[Dict[str, Any]],
    max_width: int,
    max_lines: int = DEFAULT_MAX_LINES,
    *,
    labels: bool = False,
    max_duration: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Break subtitle cues into pieces of at most *max_lines* lines of *max_width* columns.

    Wide and fullwidth (CJK) characters take two columns. Lines break at spaces and, in Chinese and Japanese text,
    between characters, without adding spaces and not before closing punctuation such as 。. A word longer than
    *max_width* gets a line of its own. With *labels*, each piece starts with its speaker's ``[SPEAKER_00]`` label, as
    SRT shows it, and the label counts toward the first line; a label that doesn't fit beside the first word gets a
    line of its own. With *max_duration*, a piece also ends before it would last longer than that many seconds; a word
    that lasts longer gets a piece of its own.

    Times come from the cue's word timestamps when it has them, otherwise its duration is shared out in
    proportion to text length. Cues are never merged, so every piece keeps its cue's speaker.
    """
    max_millis = None if max_duration is None else _millis(max_duration)
    pieces: List[Dict[str, Any]] = []
    for cue in cues:
        for piece in _split(cue, max_width, max_lines, _label(cue) if labels else "", max_millis):
            del piece[_CHUNKS]
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


def _fix_timing(cues: List[Dict[str, Any]], max_millis: Optional[int] = None) -> List[Dict[str, Any]]:
    """Cues in time order, each lasting at least MIN_CUE_SECONDS (or until the next one starts), none overlapping, and
    with *max_millis* none lasting longer than that many milliseconds.

    Times are whole milliseconds, as written. A cue that overlaps the next one ends when the next one starts; when two
    cues start together, the first lasts a millisecond and the second starts after it.
    """
    ordered = sorted(cues, key=lambda cue: _millis(cue.get("start", 0.0)))
    starts = [_millis(cue.get("start", 0.0)) for cue in ordered]
    minimum = round(MIN_CUE_SECONDS * 1000) if max_millis is None else min(round(MIN_CUE_SECONDS * 1000), max_millis)
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
        if max_millis is not None:
            end = min(end, start + max_millis)
        end = previous_end = max(end, start + 1)
        fixed.append({**cue, "start": start / 1000, "end": end / 1000})
    return fixed


def _subtitles(
    result: Dict[str, Any],
    max_line_width: Optional[int],
    labels: bool = False,
    *,
    max_lines: Optional[int] = None,
    max_cue_duration: Optional[float] = None,
    word_level: bool = False,
) -> List[Dict[str, Any]]:
    """The cues to write, split to *max_line_width* and *max_cue_duration*, with valid times; *labels* puts speaker
    labels in the text. With *word_level*, the cues with word timings keep them under ``_CHUNKS``."""
    max_millis = None if max_cue_duration is None else max(1, _millis(max_cue_duration))
    pieces: List[Dict[str, Any]] = []
    for cue in _cues(result):
        label = _label(cue) if labels else ""
        if max_line_width:
            pieces += _split(cue, max_line_width, max_lines or DEFAULT_MAX_LINES, label, max_millis)
        elif (word_level and _word_times(cue)) or (
            max_millis is not None and _millis(cue.get("end", 0.0)) - _millis(cue.get("start", 0.0)) > max_millis
        ):
            pieces += _split(cue, _UNLIMITED, 1, label, max_millis)  # one line per piece, split only by time
        else:
            pieces.append(_whole(cue, labels))
    return _fix_timing(pieces, max_millis)


def _timed_text(chunks: Sequence[Tuple[Optional[float], str]], start: float, end: float) -> str:
    """WebVTT cue text with a ``<00:00:01.200>`` timestamp tag where each timed word starts, escaped.

    WebVTT needs each tag later than the cue's start and the tag before it, and earlier than the cue's end; a word
    whose time doesn't fit is shown with the words before it.
    """
    text: List[str] = []
    last, stop = _millis(start), _millis(end)
    for time, chunk in chunks:
        stamp = None if time is None else _millis(time)
        if stamp is not None and last < stamp < stop:
            words = chunk.lstrip(" \n")
            text.append(f"{chunk[: len(chunk) - len(words)]}<{_clock(stamp, '.')}>{html.escape(words, quote=False)}")
            last = stamp
        else:
            text.append(html.escape(chunk, quote=False))
    return "".join(text)


###############################################################################
# Formats
###############################################################################


def _hms(seconds: float) -> str:
    """The time at the start of the second *seconds* falls in, as ``HH:MM:SS``."""
    minutes, secs = divmod(_millis(seconds) // 1000, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def to_txt(result: Dict[str, Any], *, timestamps: bool = False) -> str:
    """Plain transcript; with diarization, one ``[SPEAKER] text`` line per speaker turn.

    With *timestamps*, one line per segment instead, starting with the time it starts: ``[00:01:02] text``, or
    ``[00:01:02] [SPEAKER_00] text`` with diarization.
    """
    if timestamps:
        return "\n".join(
            " ".join(part for part in (f"[{_hms(seg.get('start', 0.0))}]", _label(seg), _one_line(seg["text"])) if part)
            for seg in _cues(result)
        )
    if "speaker_segments" not in result:
        return str(result.get("text", "")).strip()

    lines: List[str] = []
    current: Optional[str] = None
    for seg in _cues(result):
        label, text = _label(seg), seg["text"].strip()
        if lines and label == current:
            lines[-1] = _join(lines[-1], text)
        else:
            current = label
            lines.append(f"{label} {text}" if label else text)
    return "\n".join(lines)


def to_srt(
    result: Dict[str, Any],
    max_line_width: Optional[int] = None,
    *,
    max_lines: Optional[int] = None,
    max_cue_duration: Optional[float] = None,
) -> str:
    """SubRip subtitles: one cue per segment, or lines of at most *max_line_width* columns, speaker label included.

    *max_lines* lines per cue (default 2) with *max_line_width*; cues that would last longer than *max_cue_duration*
    seconds are split.
    """
    cues = _subtitles(result, max_line_width, labels=True, max_lines=max_lines, max_cue_duration=max_cue_duration)
    blocks = []
    for number, cue in enumerate(cues, start=1):
        start, end = _timestamp(cue["start"], ","), _timestamp(cue["end"], ",")
        blocks.append(f"{number}\n{start} --> {end}\n{cue['text']}\n")
    return "\n".join(blocks)


def to_vtt(
    result: Dict[str, Any],
    max_line_width: Optional[int] = None,
    *,
    max_lines: Optional[int] = None,
    max_cue_duration: Optional[float] = None,
    word_level: bool = False,
) -> str:
    """WebVTT subtitles like :func:`to_srt`; speakers become ``<v>`` voice tags.

    ``&``, ``<`` and ``>`` are escaped in the text and in voice names, so ``-->`` in the text can't end a cue. With
    *word_level*, cues with word timings mark when each word starts with a timestamp tag, ``<00:00:01.200>``, which
    players use to highlight the words as they are spoken.
    """
    cues = _subtitles(
        result, max_line_width, max_lines=max_lines, max_cue_duration=max_cue_duration, word_level=word_level
    )
    blocks = ["WEBVTT\n"]
    for cue in cues:
        if word_level and _CHUNKS in cue:
            text = _timed_text(cue[_CHUNKS], cue["start"], cue["end"])
        else:
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


def to_tsv(result: Dict[str, Any]) -> str:
    """Tab-separated values, as openai-whisper writes them: a ``start``, ``end``, ``text`` header, then a row per
    segment, with its start and end in whole milliseconds and its text on one line.

    With diarization, the rows are the speaker-labelled segments, and a fourth column holds their ``speaker``.
    """
    diarized = "speaker_segments" in result
    rows = ["start\tend\ttext\tspeaker" if diarized else "start\tend\ttext"]
    for seg in _cues(result):
        row = [str(_millis(seg.get("start", 0.0))), str(_millis(seg.get("end", 0.0))), _one_line(seg["text"])]
        if diarized:
            row.append(_speaker(seg))
        rows.append("\t".join(row))
    return "\n".join(rows)


def _rttm_field(text: str) -> str:
    """*text* as one field of an rttm line, which fields are separated by spaces: spaces become underscores."""
    return "_".join(text.split())


def to_rttm(result: Dict[str, Any], file_id: Optional[str] = None) -> str:
    """The speaker turns in the Rich Transcription Time Marked format, one line of 10 fields per turn::

        SPEAKER talk 1 0.000 2.500 <NA> <NA> SPEAKER_00 <NA> <NA>

    The fields: the type, the *file_id* (default ``"audio"``), the channel, the start and duration in seconds, then
    the speaker. A turn is a speaker-labelled segment, merged with the next one of the same speaker when they touch or
    overlap. Speech without a speaker (``unknown``) is left out, and spaces in the file ID and names become ``_``.

    Raises:
        UnsupportedOptionError: If *result* has no speaker labels (it wasn't diarized).
    """
    if "speaker_segments" not in result:
        raise UnsupportedOptionError(
            "rttm lists speaker turns, and this result has no speaker labels: transcribe with diarization "
            "(diarize=True, or transcribe --diarize)"
        )
    file_field = _rttm_field(file_id or "") or DEFAULT_FILE_ID
    turns: List[List[Any]] = []  # [start, end, speaker], in milliseconds
    for seg in sorted(_cues(result), key=lambda seg: _millis(seg.get("start", 0.0))):
        speaker = _rttm_field(_speaker(seg))
        start, end = _millis(seg.get("start", 0.0)), _millis(seg.get("end", 0.0))
        if not speaker or speaker == UNKNOWN_SPEAKER or end <= start:
            continue
        if turns and turns[-1][2] == speaker and start <= turns[-1][1]:
            turns[-1][1] = max(turns[-1][1], end)
        else:
            turns.append([start, end, speaker])
    return "\n".join(
        f"SPEAKER {file_field} 1 {_seconds(start)} {_seconds(end - start)} <NA> <NA> {speaker} <NA> <NA>"
        for start, end, speaker in turns
    )


def _seconds(millis: int) -> str:
    """Milliseconds as seconds with three decimals: ``1.250``."""
    return f"{millis // 1000}.{millis % 1000:03d}"


def _check_cue_limits(max_line_width: Optional[int], max_lines: object, max_cue_duration: object) -> None:
    """Raise UnsupportedOptionError for a *max_lines* or *max_cue_duration* that can't be honoured (they are typed as
    objects: callers may pass anything)."""
    if max_lines is not None:
        if isinstance(max_lines, bool) or not isinstance(max_lines, numbers.Integral) or int(max_lines) < 1:
            raise UnsupportedOptionError(f"max_lines must be a whole number from 1 up, not {max_lines!r}")
        if not max_line_width:
            raise UnsupportedOptionError("max_lines needs max_line_width: cues are split into lines to a width")
    if max_cue_duration is not None and not (
        not isinstance(max_cue_duration, bool)
        and isinstance(max_cue_duration, numbers.Real)
        and math.isfinite(float(max_cue_duration))
        and float(max_cue_duration) > 0
    ):
        raise UnsupportedOptionError(f"max_cue_duration must be a number of seconds above 0, not {max_cue_duration!r}")


def render(
    result: Dict[str, Any],
    fmt: str,
    max_line_width: Optional[int] = None,
    *,
    max_lines: Optional[int] = None,
    max_cue_duration: Optional[float] = None,
    speaker_names: Optional[SpeakerNames] = None,
    timestamps: bool = False,
    word_level: bool = False,
    file_id: Optional[str] = None,
) -> str:
    """Render *result* in one of :data:`FORMATS`; each option applies to the formats it names, and the others ignore
    it.

    Args:
        result: The result as a dict, as ``TranscriptionResult.to_dict()`` gives it.
        fmt: ``txt``, ``srt``, ``vtt``, ``json``, ``tsv`` or ``rttm``.
        max_line_width: srt, vtt: split cues into lines of at most this many columns.
        max_lines: srt, vtt: with *max_line_width*, at most this many lines per cue (default 2).
        max_cue_duration: srt, vtt: split cues that would last longer than this many seconds, and end a single word
            that lasts longer after that time.
        speaker_names: Every format: rename the speakers, in the order they first speak (``["Alice", "Bob"]`` or
            ``"Alice,Bob"``), or with a mapping (``{"SPEAKER_01": "Bob"}``); see :func:`speaker_mapping`.
        timestamps: txt: one line per segment, starting with its time: ``[00:01:02] text``.
        word_level: vtt: mark when each word starts, with ``<00:00:01.200>`` timestamp tags, in cues with word
            timings.
        file_id: rttm: the file ID its lines start with (default: ``"audio"``).

    Raises:
        UnsupportedOptionError: For an unknown format, a *max_lines* without *max_line_width*, a value that doesn't
            fit, or rttm for a result without speaker labels.
    """
    if fmt not in FORMATS:
        raise UnsupportedOptionError(f"Unknown format: {fmt!r}. Available: {', '.join(FORMATS)}")
    _check_cue_limits(max_line_width, max_lines, max_cue_duration)
    if speaker_names:
        result = rename_speakers(result, speaker_names)
    if fmt == "txt":
        return to_txt(result, timestamps=timestamps)
    if fmt in ("srt", "vtt"):
        limits: Dict[str, Any] = {"max_lines": max_lines, "max_cue_duration": max_cue_duration}
        if fmt == "srt":
            return to_srt(result, max_line_width, **limits)
        return to_vtt(result, max_line_width, word_level=word_level, **limits)
    if fmt == "json":
        return to_json(result)
    if fmt == "tsv":
        return to_tsv(result)
    return to_rttm(result, file_id)


def write_text(path: pathlib.Path, text: str) -> None:
    """Write *text* to *path* as UTF-8 with a final newline, creating missing folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
