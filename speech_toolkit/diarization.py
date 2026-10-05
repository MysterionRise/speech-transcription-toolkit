"""Speaker diarization with pyannote.audio, and merging speaker turns into transcript segments."""

from __future__ import annotations

import itertools
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple

from .media import SAMPLE_RATE, load_audio

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


def load_diarization_pipeline(device: Optional[str] = None) -> Any:
    """Load the pyannote speaker-diarization pipeline (imported lazily: it is slow to import)."""
    # pyannote.audio 4 sends usage metrics to pyannote.ai by default; stay offline unless the user opted in.
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise ImportError(
            f"speaker diarization needs pyannote.audio ({e}). "
            'Install with: pip install "speech-transcription-toolkit[diarize]"'
        ) from e

    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=os.getenv("HF_TOKEN"))
    if pipeline is None:  # pyannote returns None when the gated model can't be downloaded
        raise RuntimeError(
            f"could not download '{DIARIZATION_MODEL}'. Accept its terms at https://hf.co/{DIARIZATION_MODEL} "
            "and set HF_TOKEN (or pass --hf-token)."
        )
    pipeline.to(torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")))
    return pipeline


def diarize_audio(
    audio_path: pathlib.Path,
    pipeline: Any,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[Tuple[float, float, str]]:
    """Return list of (start, end, speaker_label)."""
    import torch

    # Decode with ffmpeg here so pyannote needs no audio I/O backend of its own, whatever the format.
    waveform = torch.from_numpy(load_audio(audio_path)).unsqueeze(0)
    output = pipeline(
        {"waveform": waveform, "sample_rate": SAMPLE_RATE},
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    # pyannote 4 returns both variants; the "exclusive" one has no overlapping turns, which suits transcripts.
    annotation = getattr(output, "exclusive_speaker_diarization", output)
    return [(turn.start, turn.end, speaker) for turn, _, speaker in annotation.itertracks(yield_label=True)]


def merge_diarization(
    transcription_result: Dict[str, Any],
    spk_segments: List[Tuple[float, float, str]],
) -> List[Dict[str, Any]]:
    """Attach to each segment the speaker who talks the most during it.

    Segments with word timings (``"words"``) are labelled word by word instead and split wherever
    the speaker changes, so a quick reply inside one segment gets its own line.
    """
    turns = sorted(spk_segments, key=lambda turn: turn[0])

    output: List[Dict[str, Any]] = []
    for seg in transcription_result.get("segments", []):
        if "start" not in seg or "end" not in seg:
            output.append({**seg, "speaker": "unknown"})
            continue
        speaker = _best_speaker(seg["start"], seg["end"], turns)
        words = seg.get("words") or []
        if not words or not all("start" in w and "end" in w for w in words):
            output.append({**seg, "speaker": speaker})
            continue
        labels = _fill_unknown([_best_speaker(w["start"], w["end"], turns) for w in words], speaker)
        for label, group in itertools.groupby(zip(words, labels), key=lambda pair: pair[1]):
            part = [word for word, _ in group]
            text = "".join(w["word"] for w in part)
            output.append(
                {
                    **seg,
                    "start": part[0]["start"],
                    "end": part[-1]["end"],
                    "text": text,
                    "words": part,
                    "speaker": label,
                }
            )
    return output


def _fill_unknown(labels: List[str], fallback: str) -> List[str]:
    """Give words in pauses between turns ("unknown") the speaker of the word before them (else after)."""
    known = [label for label in labels if label != "unknown"]
    if not known:
        return [fallback] * len(labels)
    filled, previous = [], known[0]
    for label in labels:
        previous = label if label != "unknown" else previous
        filled.append(previous)
    return filled


def _best_speaker(start: float, end: float, turns: List[Tuple[float, float, str]]) -> str:
    """Speaker with the most overlap with [start, end]; else the one whose turn contains its midpoint."""
    overlap: Dict[str, float] = {}
    for turn_start, turn_end, speaker in turns:
        shared = min(end, turn_end) - max(start, turn_start)
        if shared > 0:
            overlap[speaker] = overlap.get(speaker, 0.0) + shared
    if overlap:
        return max(overlap, key=lambda speaker: overlap[speaker])

    middle = (start + end) / 2.0
    return next((speaker for turn_start, turn_end, speaker in turns if turn_start <= middle <= turn_end), "unknown")
