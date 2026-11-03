"""The output options of #38 on a real model's word timings: word-level WebVTT and cue limits stay valid, tsv and
timestamped txt hold the transcript, and ``transcribe --from-json`` writes the same files without loading a model."""

from __future__ import annotations

import pathlib
from typing import Dict

from tests.integration.helpers import BACKEND_ARGS, assert_wer, require, run_module
from tests.test_formats_extra import check_timestamp_tags
from tests.test_formats_subtitles import check_timing, parse_srt, parse_vtt, shown

BACKEND = "faster-whisper"  # a core dependency, so every install has it
# Every output option at once; --word-level and --max-cue-duration turn word timestamps on.
OPTIONS = ("--word-level", "--max-line-width", "20", "--max-lines", "1", "--max-cue-duration", "1.5", "--timestamps")


def files(folder: pathlib.Path) -> Dict[str, str]:
    """The files in *folder*, by name, with their text."""
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(folder.iterdir())}


def test_output_options_then_from_json(speech, transcribe, tmp_path):
    require(BACKEND)
    name = speech.path.stem

    run = transcribe(speech.path, *BACKEND_ARGS[BACKEND], "-q", "--outdir", tmp_path / "direct", "-f", "all", *OPTIONS)

    assert run.stdout == ""
    direct = files(tmp_path / "direct")
    assert sorted(direct) == sorted(f"{name}.{fmt}" for fmt in ("json", "srt", "tsv", "txt", "vtt"))
    cues = parse_vtt(direct[f"{name}.vtt"])  # the strict WebVTT check of the unit tests
    assert any(check_timestamp_tags(cues)), "no word got a timestamp tag"
    for start, end, lines in [*cues, *parse_srt(direct[f"{name}.srt"])]:
        assert end - start <= 1500 and len(lines) == 1, (start, end, lines)
    check_timing(cues)
    assert cues[0][0] < 1000 and cues[-1][0] < speech.duration * 1000
    assert_wer(" ".join(shown(line) for _, _, lines in cues for line in lines), speech, BACKEND)
    header, *rows = direct[f"{name}.tsv"].splitlines()
    assert header == "start\tend\ttext"
    assert_wer(" ".join(row.split("\t")[2] for row in rows), speech, BACKEND)
    assert direct[f"{name}.txt"].startswith("[00:00:0")

    saved = tmp_path / "direct" / f"{name}.json"
    again = run_module(
        "speech_toolkit", "--from-json", saved, "--outdir", tmp_path / "again", "-f", "all", *OPTIONS, cwd=tmp_path
    )

    assert "Loading" not in again.stderr
    assert files(tmp_path / "again") == direct
