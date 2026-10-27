"""A custom backend: plug your own speech recognizer into Transcriber, and get its outputs, formats and speaker labels.

This one needs no model, so it runs anywhere: it finds the stretches where someone speaks, from how loud the audio is,
and writes "[speech]" for each. That is enough to show what a backend does:

- ``available_models()`` names its models, and ``load_model()`` loads one. Import heavy libraries there, not at the
  top of the module, so that listing backends stays fast.
- ``transcribe()`` returns a TranscriptionResult: the text, and segments with their start and end in seconds.
- ``capabilities`` declares what else it does. With "waveform", Transcriber may pass the decoded samples instead of a
  path, so a recording is decoded once, also for speaker labels.

    python custom_backend.py talk.mp3            # srt subtitles that show when someone speaks
    python custom_backend.py talk.mp3 -f json    # or txt, vtt

register_backend() makes the backend available by name to Transcriber and transcribe() in this process. To use your
own model, load it in load_model() and call it in transcribe(). The docs' "Python API" page, under "Custom backends",
lists the rest of the contract.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from speech_toolkit import (
    ModelNotFoundError,
    Segment,
    Transcriber,
    TranscriptionBackend,
    TranscriptionResult,
    register_backend,
)
from speech_toolkit.media import SAMPLE_RATE, load_audio

FRAME_SECONDS = 0.03  # loudness is measured over frames of 30 ms
QUIET_DB = 30.0  # a frame this much quieter than the loudest one is silence
SILENT_DB = -60.0  # and so is any frame below -60 dB (0 dB is the loudest possible), so silence alone is no speech
MIN_PAUSE_SECONDS = 0.3  # shorter pauses don't end a stretch of speech


class SpeechRegionsBackend(TranscriptionBackend):
    name = "speech-regions"
    description = "Marks where someone speaks, from the loudness (no model)"
    capabilities = frozenset({"waveform"})

    @classmethod
    def available_models(cls) -> List[str]:
        return ["loudness"]

    def load_model(self, model_name: str, device: Optional[str] = None) -> None:
        if model_name not in self.available_models():
            raise ModelNotFoundError(f"unknown model {model_name!r}; available: {', '.join(self.available_models())}")
        self._model = model_name  # a real backend keeps its loaded model here
        self._model_name = model_name
        self._device = "cpu"

    def transcribe(self, audio_path, language=None, task="transcribe", verbose=True) -> TranscriptionResult:
        # A path, or with the "waveform" capability the samples: a 1-D float32 array of mono audio at 16 kHz.
        samples = load_audio(audio_path) if isinstance(audio_path, Path) else audio_path
        segments: List[Segment] = [
            {"id": number, "start": start, "end": end, "text": " [speech]"}
            for number, (start, end) in enumerate(speech_regions(samples))
        ]
        text = "".join(segment["text"] for segment in segments)
        return TranscriptionResult(text, segments, language=language, duration=len(samples) / SAMPLE_RATE)


def speech_regions(samples: np.ndarray) -> List[Tuple[float, float]]:
    """``(start, end)`` in seconds of each stretch louder than silence, joined across short pauses."""
    frame = int(SAMPLE_RATE * FRAME_SECONDS)
    count = len(samples) // frame
    if count == 0:
        return []
    power = np.mean(samples[: count * frame].reshape(count, frame) ** 2, axis=1)
    loudness = 10 * np.log10(power + 1e-12)  # in decibels
    regions: List[List[float]] = []
    for index in np.flatnonzero(loudness > max(loudness.max() - QUIET_DB, SILENT_DB)):
        start, end = index * FRAME_SECONDS, (index + 1) * FRAME_SECONDS
        if regions and start - regions[-1][1] < MIN_PAUSE_SECONDS:
            regions[-1][1] = end
        else:
            regions.append([start, end])
    return [(round(start, 2), round(end, 2)) for start, end in regions]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Show when someone speaks in a recording, with a custom backend.")
    parser.add_argument("audio", type=Path, help="any audio or video file ffmpeg can decode")
    parser.add_argument("-f", "--format", choices=("txt", "srt", "vtt", "json"), default="srt", help="default: srt")
    args = parser.parse_args(argv)

    register_backend(SpeechRegionsBackend.name, SpeechRegionsBackend)
    transcriber = Transcriber(SpeechRegionsBackend.name)  # like any other backend: diarize=True works too
    print(transcriber.transcribe(args.audio).render(args.format).rstrip("\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
