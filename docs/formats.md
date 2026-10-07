# Output formats

How the format is chosen:
- `-f`/`--format` names it explicitly.
- Otherwise the `--output` file's extension picks it (`-o talk.srt`).
- With neither, `transcribe` prints plain text.

In Python, use `result.render(fmt)` or `result.save(path)`.

| Format | Contents | With speaker labels |
|---|---|---|
| `txt` | The plain transcript | One `[SPEAKER_00] text` line per speaker turn |
| `srt` | SubRip subtitles, one cue per segment | Each cue starts with `[SPEAKER_00]` |
| `vtt` | WebVTT subtitles, one cue per segment | Voice tags: `<v SPEAKER_00>text` |
| `json` | The full result: `text`, `segments`, `language` and backend-specific fields | Adds `speaker_segments` |

**Writing files:**
- Files are written as UTF-8 with a final newline.
- Missing folders are created.
- An extension `transcribe` doesn't know gives plain text.

`--json FILE` writes the JSON result in addition to the main output.

## Readable subtitle lines

`--max-line-width N` splits srt and vtt cues into lines of at most N characters, with two lines per cue. A word longer
than N gets a line of its own.

New cues get their timing from the word timestamps when the backend provides them. Otherwise each cue's duration is
shared out in proportion to text length. Cues are never merged, so each piece keeps its speaker.

## JSON

```json
{
  "schema_version": 1,
  "text": " Hello world. This is a speech recognition test.",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.2,
      "text": " Hello world. This is a speech recognition test.",
      "words": [{"word": " Hello", "start": 0.0, "end": 0.42, "probability": 0.98}]
    }
  ],
  "language": "en",
  "duration": 3.4,
  "language_probability": 0.99
}
```

- `schema_version` is the version of this layout. It goes up only when a change would make older versions misread a
  file, not when a field is added. Files without it come from earlier versions, which wrote the same layout with
  fewer fields.
- `id` numbers the segments from 0, with every backend. (faster-whisper's used to start at 1.)
- `duration` is the audio's length in seconds, and `language_probability` how likely the detected language is, from 0
  to 1. Both are `null` when the backend doesn't report them; faster-whisper reports both.
- `words` appears only with word timestamps (`--word-timestamps`, or automatically with `--diarize` on backends that
  support them).
- Fields vary by backend: whisper and faster-whisper add decoding details such as `avg_logprob` and `no_speech_prob`.
- With `--diarize`, `speaker_segments` holds the segments with a `speaker` label each.

In Python, `TranscriptionResult.load("talk.json")` reads the file back, to render it in another format; see
[Loading a saved result](python-api.md#loading-a-saved-result).
