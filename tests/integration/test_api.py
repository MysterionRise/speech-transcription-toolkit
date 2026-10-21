"""The Python API with a real model: one Transcriber for several files, and saving subtitles."""

from __future__ import annotations

from tests.integration.helpers import assert_timeline, assert_wer, parse_cues, require


def test_transcriber_reuses_its_model(samples, tmp_path):
    require("faster-whisper")
    from speech_toolkit import Transcriber

    transcriber = Transcriber("faster-whisper", "tiny")
    for sample in samples.values():
        result = transcriber.transcribe(sample.path)

        assert_wer(result.text, sample, "faster-whisper")
        assert result.language == "en"
        assert_timeline(((seg["start"], seg["end"]) for seg in result.segments), sample.duration, "segments")
        vtt = result.save(tmp_path / f"{sample.name}.vtt").read_text(encoding="utf-8")
        assert vtt.startswith("WEBVTT\n")
        assert_timeline(((cue.start, cue.end) for cue in parse_cues(vtt)), sample.duration, "cues")
