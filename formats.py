"""Render transcription results as plain text, SRT, WebVTT or JSON."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Callable, Dict, List, Optional

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


def to_srt(result: Dict[str, Any]) -> str:
    """SubRip subtitles, one cue per segment."""
    blocks = []
    for number, seg in enumerate(_cues(result), start=1):
        text = seg["text"].strip()
        if seg.get("speaker"):
            text = f"[{seg['speaker']}] {text}"
        start, end = _timestamp(seg.get("start", 0.0), ","), _timestamp(seg.get("end", 0.0), ",")
        blocks.append(f"{number}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def to_vtt(result: Dict[str, Any]) -> str:
    """WebVTT subtitles, one cue per segment; speakers become ``<v>`` voice tags."""
    blocks = ["WEBVTT\n"]
    for seg in _cues(result):
        text = seg["text"].strip()
        if seg.get("speaker"):
            text = f"<v {seg['speaker']}>{text}"
        start, end = _timestamp(seg.get("start", 0.0), "."), _timestamp(seg.get("end", 0.0), ".")
        blocks.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def to_json(result: Dict[str, Any]) -> str:
    """The full result (text, segments, language, speakers) as pretty-printed JSON."""
    return json.dumps(result, ensure_ascii=False, indent=2)


_RENDERERS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "txt": to_txt,
    "srt": to_srt,
    "vtt": to_vtt,
    "json": to_json,
}


def render(result: Dict[str, Any], fmt: str) -> str:
    """Render *result* in one of :data:`FORMATS`."""
    return _RENDERERS[fmt](result)
