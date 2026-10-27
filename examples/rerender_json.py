"""Render a saved transcript again, as subtitles, without transcribing the recording again.

    transcribe talk.mp3 --json talk.json        # or result.save("talk.json") in Python
    python rerender_json.py talk.json           # writes talk.srt and talk.vtt next to it
    python rerender_json.py talk.json --max-line-width 42 --formats srt

TranscriptionResult.load() reads a result saved as JSON, speaker labels and word timings included, and save() writes
it in the format its file's extension names: here, subtitles with lines of at most --max-line-width characters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from speech_toolkit import TranscriptionResult

FORMATS = ("txt", "srt", "vtt")  # not json: that would replace the file it reads


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a transcript saved as JSON in other formats.")
    parser.add_argument("json", type=Path, help="a result saved as JSON")
    parser.add_argument("--formats", default="srt,vtt", help="formats to write, separated by commas (default: srt,vtt)")
    parser.add_argument("--max-line-width", type=int, help="split cues into lines of at most this many characters")
    args = parser.parse_args(argv)
    args.formats = args.formats.split(",")
    unknown = [fmt for fmt in args.formats if fmt not in FORMATS]
    if unknown:
        parser.error(f"unknown format {unknown[0]!r}; choose from {', '.join(FORMATS)}")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = TranscriptionResult.load(args.json)
    except (OSError, ValueError) as e:  # a missing file, or one that isn't a saved result
        sys.exit(f"Error: {e}")
    for fmt in args.formats:
        print(result.save(args.json.with_suffix(f".{fmt}"), max_line_width=args.max_line_width))
    return 0


if __name__ == "__main__":
    sys.exit(main())
