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

In Python:

```python
from speech_toolkit import Transcriber

Transcriber(diarize=True).transcribe("meeting.wav", num_speakers=3)
```

## Output

Every format shows the speaker labels:
- **txt:** one line per speaker turn, like `[SPEAKER_00] Hello there.`
- **srt:** each cue starts with the label.
- **vtt:** labels are voice tags (`<v SPEAKER_00>`).
- **json:** `speaker_segments` holds the segments, each with a `speaker` field.

See [Output formats](formats.md).

## How speakers are matched to text

- **Word by word,** with whisper, faster-whisper and parakeet. Word timestamps are turned on automatically, each word
  gets the speaker who talks the most during it, and a segment is split where the speaker changes. So a quick reply
  mid-sentence gets its own line.
- **One speaker per segment** for backends without word timestamps (voxtral, canary). Each segment gets the speaker who
  talks the most during it, and their segments are about 30 seconds long.

## Notes

- **Device:** diarization runs on the same device as transcription: `--device`, or the GPU when available.
- **Telemetry:** pyannote's usage telemetry stays off unless you set `PYANNOTE_METRICS_ENABLED=true`.
- **"Could not download" error:** accept the model's terms (step 2) and set `HF_TOKEN`.
