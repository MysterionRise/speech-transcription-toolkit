"""Meeting notes in Markdown: who said what, and when, from a recording of the meeting.

    pip install "speech-transcription-toolkit[diarize]"
    export HF_TOKEN=hf_...   # after accepting the terms of pyannote/speaker-diarization-community-1, see below
    python meeting_notes.py meeting.m4a --num-speakers 3 -o notes.md
    python meeting_notes.py meeting.m4a --name SPEAKER_00=Ana --name SPEAKER_01=Ben -o notes.md

Speaker diarization labels the speakers SPEAKER_00, SPEAKER_01 and so on, in the order they first speak. Run once
without names to see who is who, then give each label a name with --name. The notes have a paragraph per turn:

    **Ana** (0:01:05): Let's start with the budget.

Speaker labels need the ``diarize`` extra and a Hugging Face token: accept the terms of
https://hf.co/pyannote/speaker-diarization-community-1 and set HF_TOKEN. The docs' "Speaker labels" page has more.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from speech_toolkit import SpeechToolkitError, Transcriber, TranscriptionResult


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write meeting notes in Markdown, with the speakers labelled.")
    parser.add_argument("audio", type=Path, help="the recording: any audio or video file ffmpeg can decode")
    parser.add_argument("-o", "--output", type=Path, help="the Markdown file to write (default: stdout)")
    parser.add_argument("--num-speakers", type=int, help="how many people speak, if you know")
    parser.add_argument(
        "--name", action="append", default=[], metavar="LABEL=NAME", help="a speaker's name, e.g. SPEAKER_00=Ana"
    )
    parser.add_argument("-b", "--backend", default="faster-whisper", help="default: faster-whisper")
    parser.add_argument("-m", "--model", help="default: the backend's default model")
    parser.add_argument("-l", "--language", help="a language code such as en (default: detect it)")
    args = parser.parse_args(argv)
    try:
        args.names = dict(name.split("=", 1) for name in args.name)
    except ValueError:
        parser.error("--name takes LABEL=NAME, such as SPEAKER_00=Ana")
    return args


def turns(segments: List[Dict[str, Any]]) -> List[Tuple[str, float, str]]:
    """``(speaker, start, text)`` for each turn: the consecutive segments of one speaker, joined."""
    result: List[Tuple[str, float, str]] = []
    for segment in segments:
        speaker, text = segment.get("speaker", "unknown"), segment["text"].strip()
        if result and result[-1][0] == speaker:
            result[-1] = (speaker, result[-1][1], f"{result[-1][2]} {text}")
        elif text:
            result.append((speaker, segment["start"], text))
    return result


def timestamp(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes // 60}:{minutes % 60:02}:{seconds:02}"


def to_markdown(result: TranscriptionResult, title: str, names: Mapping[str, str]) -> str:
    """The notes: a title, how long the meeting was and who spoke, then a paragraph per turn."""
    meeting = turns(result.speaker_segments or [])
    speakers = list(dict.fromkeys(names.get(speaker, speaker) for speaker, _, _ in meeting))
    summary = f"Speakers: {', '.join(speakers) or 'none'}."
    if result.duration is not None:
        summary = f"Length: {timestamp(result.duration)}. {summary}"
    lines = [f"# {title}", "", summary, ""]
    for speaker, start, text in meeting:
        lines += [f"**{names.get(speaker, speaker)}** ({timestamp(start)}): {text}", ""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        transcriber = Transcriber(args.backend, args.model, diarize=True)  # also loads the speaker diarization model
        result = transcriber.transcribe(args.audio, language=args.language, num_speakers=args.num_speakers)
    except (ImportError, OSError, SpeechToolkitError) as e:  # the diarize extra or HF_TOKEN missing, say
        sys.exit(f"Error: {e}")
    notes = to_markdown(result, f"Meeting notes: {args.audio.name}", args.names)
    if args.output:
        args.output.write_text(notes, encoding="utf-8")
        print(args.output)
    else:
        print(notes, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
