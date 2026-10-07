"""Regenerate the speech samples in this folder from samples.json.

The integration tests (tests/integration) transcribe these samples and compare the transcripts with the text in
samples.json. The speech is synthesized with ffmpeg's flite filter (Ubuntu's ffmpeg has it), so the samples are
reproducible and hold no recorded voices. In samples.json, each sample is a list of turns, each spoken by a flite
voice (awb, kal, kal16, rms or slt); turns are separated by PAUSE_SECONDS of silence. The script writes the audio as
16 kHz mono FLAC, and each turn's start and end time and the sample's duration back into samples.json:

    python tests/data/make_samples.py

Keep each sample under 200 KB.
"""

from __future__ import annotations

import json
import pathlib
import subprocess  # nosec B404
import tempfile
from typing import Any, Dict

DATA = pathlib.Path(__file__).resolve().parent
MANIFEST = DATA / "samples.json"
SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # 16-bit mono PCM
PAUSE_SECONDS = 0.5
FFMPEG = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]


def synthesize(text: str, voice: str) -> bytes:
    """*text* spoken by a flite *voice*, as 16 kHz mono 16-bit PCM."""
    with tempfile.TemporaryDirectory() as tmp:
        # A text file spares escaping the text for ffmpeg's filter syntax.
        pathlib.Path(tmp, "text.txt").write_text(text, encoding="utf-8")
        cmd = FFMPEG + ["-f", "lavfi", "-i", f"flite=textfile=text.txt:voice={voice}"]
        cmd += ["-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
        return subprocess.run(cmd, cwd=tmp, capture_output=True, check=True).stdout  # nosec B603


def write_flac(pcm: bytes, dest: pathlib.Path) -> None:
    """Encode 16 kHz mono PCM as FLAC; without metadata, the same audio always gives the same file."""
    cmd = FFMPEG + ["-y", "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-i", "-", "-map_metadata", "-1"]
    cmd += ["-fflags", "+bitexact", "-flags:a", "+bitexact", "-compression_level", "12", str(dest)]
    subprocess.run(cmd, input=pcm, capture_output=True, check=True)  # nosec B603


def make_sample(path: pathlib.Path, sample: Dict[str, Any]) -> None:
    """Synthesize the sample's turns into *path*, recording their times (seconds) in *sample*."""
    pcm = b""
    for turn in sample["turns"]:
        if pcm:
            pcm += bytes(int(PAUSE_SECONDS * BYTES_PER_SECOND))
        turn["start"] = round(len(pcm) / BYTES_PER_SECOND, 3)
        pcm += synthesize(turn["text"], turn["voice"])
        turn["end"] = round(len(pcm) / BYTES_PER_SECOND, 3)
    sample["duration"] = round(len(pcm) / BYTES_PER_SECOND, 3)
    write_flac(pcm, path)


def main() -> None:
    samples = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name, sample in samples.items():
        make_sample(DATA / name, sample)
        print(f"{name}: {sample['duration']:.2f} s, {(DATA / name).stat().st_size / 1024:.0f} KB")
    MANIFEST.write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
