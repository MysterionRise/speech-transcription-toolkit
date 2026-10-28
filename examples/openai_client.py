"""OpenAI's Python SDK against transcribe-server: code written for OpenAI's speech-to-text API, run offline.

Start the server in one terminal:

    pip install "speech-transcription-toolkit[server]" openai
    transcribe-server -m small          # faster-whisper's small model on http://127.0.0.1:8000/v1

and run this in another:

    python openai_client.py talk.mp3                       # the transcript
    python openai_client.py talk.mp3 --format srt          # subtitles: text, srt, vtt or verbose_json
    python openai_client.py talk.mp3 --format verbose_json # each segment with its times
    python openai_client.py talk.mp3 --translate           # to English, with a backend that translates

The only change from code that calls OpenAI is ``base_url``. The server runs the model it loaded and ignores the
request's model name, so "whisper-1" works. If the server has an API key (``--api-key``, or TRANSCRIBE_API_KEY), pass
the same key with --api-key or TRANSCRIBE_API_KEY.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

FORMATS = ("text", "srt", "vtt", "verbose_json")


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe a file with transcribe-server, through OpenAI's SDK.")
    parser.add_argument("audio", type=Path, help="the audio or video file to send")
    parser.add_argument("--format", choices=FORMATS, default="text", help="the response format (default: text)")
    parser.add_argument("--translate", action="store_true", help="translate to English")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="default: http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TRANSCRIBE_API_KEY") or "unused",  # the SDK needs one, even if the server doesn't
        help="the server's API key (default: TRANSCRIBE_API_KEY)",
    )
    return parser.parse_args(argv)


def show(response: Any) -> str:
    """The response as text: text, srt and vtt come as a string, verbose_json as an object with segments."""
    if isinstance(response, str):
        return response.rstrip("\n")
    lines = [f"Language: {response.language}, {response.duration:.1f} s"]
    lines += [f"[{segment.start:7.2f} - {segment.end:7.2f}] {segment.text.strip()}" for segment in response.segments]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    from openai import OpenAI  # pip install openai

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    endpoint = client.audio.translations if args.translate else client.audio.transcriptions
    with args.audio.open("rb") as file:
        response = endpoint.create(model="whisper-1", file=file, response_format=args.format)
    print(show(response))
    return 0


if __name__ == "__main__":
    sys.exit(main())
