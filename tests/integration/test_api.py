"""The Python API with a real model: one Transcriber for several files, and saving subtitles."""

from __future__ import annotations

import time

from tests.integration.devices import api_device, check_devices
from tests.integration.helpers import assert_timeline, assert_wer, parse_cues, require
from tests.integration.instrument import transcriber_devices
from tests.integration.report import REPORT, Run


def test_transcriber_reuses_its_model(samples, tmp_path):
    require("faster-whisper")
    from speech_toolkit import Transcriber

    started = time.perf_counter()
    transcriber = Transcriber("faster-whisper", "tiny", device=api_device())
    load_seconds = time.perf_counter() - started
    devices, precision = transcriber_devices(transcriber)
    check_devices("faster-whisper", devices)  # on INTEGRATION_DEVICE (with auto, wherever the backend put it)
    for sample in samples.values():
        started = time.perf_counter()
        result = transcriber.transcribe(sample.path)
        seconds = time.perf_counter() - started
        REPORT.add(
            Run(REPORT.test, "faster-whisper", "tiny", devices, precision, load_seconds, sample.duration, seconds)
        )

        assert_wer(result.text, sample, "faster-whisper")
        assert result.language == "en"
        assert_timeline(((seg["start"], seg["end"]) for seg in result.segments), sample.duration, "segments")
        vtt = result.save(tmp_path / f"{sample.name}.vtt").read_text(encoding="utf-8")
        assert vtt.startswith("WEBVTT\n")
        assert_timeline(((cue.start, cue.end) for cue in parse_cues(vtt)), sample.duration, "cues")
