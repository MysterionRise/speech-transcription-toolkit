# Speaker labels

Speaker diarization labels who speaks when, using
[pyannote.audio](https://github.com/pyannote/pyannote-audio) and the `pyannote/speaker-diarization-community-1` model.

1. `pip install "speech-transcription-toolkit[diarize]"`
2. Accept the terms of [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1)
   and create a [Hugging Face token](https://huggingface.co/settings/tokens).
3. Run:

```bash
export HF_TOKEN=hf_...
transcribe meeting.wav --diarize --num-speakers 3 -o meeting.txt
```

`--num-speakers` sets the exact number of speakers when you know it. Otherwise, `--min-speakers` and `--max-speakers`
bound the number.

The hints are checked before any model loads:
- each is a whole number above 0;
- `--min-speakers` can't be more than `--max-speakers`;
- `--num-speakers`, given with either, must lie between them.

In Python:

```python
from speech_toolkit import Transcriber

Transcriber(diarize=True).transcribe("meeting.wav", num_speakers=3)
```

Hints that break these rules raise `UnsupportedOptionError`, a `ValueError`.

## Output

Every format shows the speaker labels:
- **txt:** one line per speaker turn, like `[SPEAKER_00] Hello there.`
- **srt:** each cue starts with the label.
- **vtt:** labels are voice tags (`<v SPEAKER_00>`).
- **json:** `speaker_segments` holds the segments, each with a `speaker` field. Their `id` numbers them from 0, and
  `segment_id` is the `id` of the segment in `segments` each one comes from.

See [Output formats](formats.md).

## How speakers are matched to text

- **Word by word,** with whisper, faster-whisper and parakeet. Word timestamps are turned on automatically, each word
  gets the speaker who talks the most during it, and a segment is split where the speaker changes. So a quick reply
  mid-sentence gets its own line.
  - **Pauses:** a word between two speaker turns goes to the nearer turn. On a tie, it goes to the speaker who talks
    the most during the segment.
  - **Smoothing:** a speaker change inside a segment that lasts one word, or less than half a second, between two
    stretches of the same speaker, is taken for a diarization error. Its words get the speaker around them, so one
    mislabelled word no longer splits a line in three. A change at the start or end of a segment is kept.
  - **Thresholds:** they are constants in `speech_toolkit.diarization`. `MIN_RUN_WORDS = 2` means a run of fewer
    words is too short, and `MIN_RUN_SECONDS = 0.5` sets the minimum length. Set both to 0 to keep every change.
  - **Split segments:** each piece gets its own `id`. Pieces leave out the fields that describe the whole segment:
    `tokens`, `avg_logprob`, `compression_ratio`, `no_speech_prob`, `seek` and `temperature`. Segments that aren't
    split keep them.
- **One speaker per segment** for backends without word timestamps (voxtral, canary). Each segment gets the speaker who
  talks the most during it, and their segments are about 30 seconds long.

## Notes

- **Short audio:** audio shorter than half a second (`MIN_DIARIZATION_SECONDS`) isn't diarized, and its transcript is
  labelled `unknown`.
- **Device:** diarization runs on the same device as transcription: `--device`, or the GPU when available.
- **Telemetry:** pyannote's usage telemetry stays off unless you set `PYANNOTE_METRICS_ENABLED=true`.
- **"Could not download" error:** accept the model's terms (step 2) and set `HF_TOKEN`.
