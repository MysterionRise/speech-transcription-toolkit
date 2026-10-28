"""Subtitle a folder of recordings: an .srt file for every audio or video file in it, with the model loaded once.

    pip install speech-transcription-toolkit
    python folder_to_srt.py recordings/                     # recordings/talk.mp3 -> recordings/talk.srt
    python folder_to_srt.py recordings/ --outdir subtitles  # the same folder tree, in subtitles/
    python folder_to_srt.py recordings/ -m small -l en --max-line-width 42

The folder is searched recursively, for the extensions the ``transcribe`` command looks for. A recording that has its
subtitles already is skipped, so a stopped run picks up where it left off, and one that fails is reported without
stopping the others. The paths of the files written go to stdout, progress and errors to stderr.

The command line does the same with ``transcribe recordings/ --outdir subtitles -f srt --skip-existing``. This
script is the Python API underneath, to adapt: other formats or names, or your own use of each result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from speech_toolkit import SpeechToolkitError, Transcriber
from speech_toolkit.media import MEDIA_EXTENSIONS, collect_files

# What one recording can fail with: it can't be read or decoded, say. Anything else stops the run.
FILE_ERRORS = (SpeechToolkitError, OSError, RuntimeError, ValueError)


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write an .srt file for every recording in a folder.")
    parser.add_argument("folder", type=Path, help="a folder of audio or video files, searched recursively")
    parser.add_argument("--outdir", type=Path, help="where to write the subtitles (default: next to each recording)")
    parser.add_argument("-b", "--backend", default="faster-whisper", help="default: faster-whisper")
    parser.add_argument("-m", "--model", help="default: the backend's default model")
    parser.add_argument("-l", "--language", help="a language code such as en (default: detect it)")
    parser.add_argument("--max-line-width", type=int, help="split cues into lines of at most this many characters")
    return parser.parse_args(argv)


def subtitle_paths(folder: Path, outdir: Optional[Path]) -> Dict[Path, Path]:
    """Each recording in *folder*, with the path of its subtitles: talk.mp3 gets talk.srt."""
    paths: Dict[Path, Path] = {}
    for recording, relative in collect_files([folder], MEDIA_EXTENSIONS):
        target = outdir / relative if outdir else recording
        subtitles = target.with_suffix(".srt")
        if subtitles in paths.values():  # talk.mp3 and talk.wav: the second keeps its whole name, talk.wav.srt
            subtitles = target.with_name(f"{target.name}.srt")
        paths[recording] = subtitles
    return paths


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not args.folder.is_dir():
        sys.exit(f"Error: {args.folder} is not a folder")

    todo = [(audio, srt) for audio, srt in subtitle_paths(args.folder, args.outdir).items() if not srt.exists()]
    if not todo:
        print("Every recording has its subtitles already.", file=sys.stderr)
        return 0

    transcriber = Transcriber(args.backend, args.model)  # loads the model once, for every recording
    failed = 0
    for number, (recording, subtitles) in enumerate(todo, 1):
        print(f"[{number}/{len(todo)}] {recording}", file=sys.stderr)
        try:
            result = transcriber.transcribe(recording, language=args.language)
        except FILE_ERRORS as e:
            print(f"Error: {recording}: {e}", file=sys.stderr)
            failed += 1
            continue
        print(result.save(subtitles, max_line_width=args.max_line_width))
    if failed:
        print(f"Error: {failed} of {len(todo)} recordings failed.", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
