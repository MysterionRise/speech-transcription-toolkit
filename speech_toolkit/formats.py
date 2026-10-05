"""Render transcription results as plain text, SRT, WebVTT or JSON."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional

FORMATS = ("txt", "srt", "vtt", "json")


def format_for_path(path: Optional[pathlib.Path], default: str = "txt") -> str:
    """Pick the output format from a file extension (``out.srt`` -> ``srt``)."""
    suffix = path.suffix.lower().lstrip(".") if path else ""
    return suffix if suffix in FORMATS else default


def _timestamp(seconds: float, separator: str) -> str:
    """Format seconds as ``HH:MM:SS<separator>mmm``."""
    millis = max(0, round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _cues(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Segments to show as subtitles: speaker-labelled ones if diarization ran, skipping empty text."""
    segments = result["speaker_segments"] if "speaker_segments" in result else result.get("segments", [])
    return [seg for seg in segments if seg.get("text", "").strip()]


def split_cues(cues: List[Dict[str, Any]], max_width: int, max_lines: int = 2) -> List[Dict[str, Any]]:
    """Break subtitle cues into pieces of at most *max_lines* lines of *max_width* characters.

    A piece's lines are separated by newlines; a word longer than *max_width* gets a line of its own.
    Times come from the cue's word timestamps when it has them, otherwise its duration is shared out in
    proportion to text length. Cues are never merged, so every piece keeps its cue's speaker.
    """
    pieces: List[Dict[str, Any]] = []
    for cue in cues:
        words = [w for w in cue.get("words") or [] if str(w.get("word", "")).strip()]
        timed = bool(words) and all("start" in w and "end" in w for w in words)
        tokens = [w["word"].strip() for w in words] if timed else cue["text"].split()
        start, end = cue.get("start", 0.0), cue.get("end", 0.0)
        total, done = sum(map(len, tokens)) or 1, 0
        for lines in _wrap(tokens, max_width, max_lines):
            ids = [i for line in lines for i in line]
            if timed:
                piece_start, piece_end = words[ids[0]]["start"], words[ids[-1]]["end"]
            else:
                piece_start = start + (end - start) * done / total
                done += sum(len(tokens[i]) for i in ids)
                piece_end = start + (end - start) * done / total
            piece = {
                "start": piece_start,
                "end": piece_end,
                "text": "\n".join(" ".join(tokens[i] for i in line) for line in lines),
            }
            if "speaker" in cue:
                piece["speaker"] = cue["speaker"]
            pieces.append(piece)
    return pieces


def _wrap(tokens: List[str], max_width: int, max_lines: int) -> List[List[List[int]]]:
    """Fill lines of at most *max_width* characters with token indices, *max_lines* lines per group."""
    lines: List[List[int]] = []
    width = 0
    for i, token in enumerate(tokens):
        if lines and width + 1 + len(token) <= max_width:
            lines[-1].append(i)
            width += 1 + len(token)
        else:
            lines.append([i])
            width = len(token)
    return [lines[i : i + max_lines] for i in range(0, len(lines), max_lines)]


def _subtitles(result: Dict[str, Any], max_line_width: Optional[int]) -> List[Dict[str, Any]]:
    cues = _cues(result)
    return split_cues(cues, max_line_width) if max_line_width else cues


def to_txt(result: Dict[str, Any]) -> str:
    """Plain transcript; with diarization, one ``[SPEAKER] text`` line per speaker turn."""
    if "speaker_segments" not in result:
        return str(result.get("text", "")).strip()

    lines: List[str] = []
    current: Optional[str] = None
    for seg in _cues(result):
        text = seg["text"].strip()
        if seg.get("speaker") == current:
            lines[-1] += f" {text}"
        else:
            current = seg.get("speaker")
            lines.append(f"[{current}] {text}")
    return "\n".join(lines)


def to_srt(result: Dict[str, Any], max_line_width: Optional[int] = None) -> str:
    """SubRip subtitles: one cue per segment, or lines of at most *max_line_width* characters."""
    blocks = []
    for number, seg in enumerate(_subtitles(result, max_line_width), start=1):
        text = seg["text"].strip()
        if seg.get("speaker"):
            text = f"[{seg['speaker']}] {text}"
        start, end = _timestamp(seg.get("start", 0.0), ","), _timestamp(seg.get("end", 0.0), ",")
        blocks.append(f"{number}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def to_vtt(result: Dict[str, Any], max_line_width: Optional[int] = None) -> str:
    """WebVTT subtitles like :func:`to_srt`; speakers become ``<v>`` voice tags."""
    blocks = ["WEBVTT\n"]
    for seg in _subtitles(result, max_line_width):
        text = seg["text"].strip()
        if seg.get("speaker"):
            text = f"<v {seg['speaker']}>{text}"
        start, end = _timestamp(seg.get("start", 0.0), "."), _timestamp(seg.get("end", 0.0), ".")
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
