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
  "text": " Hello world. This is a speech recognition test.",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.2,
      "text": " Hello world. This is a speech recognition test.",
      "words": [{"word": " Hello", "start": 0.0, "end": 0.42}]
    }
  ],
  "language": "en"
}
```

- `words` appears only with word timestamps (`--word-timestamps`, or automatically with `--diarize` on backends that
  support them).
- Fields vary by backend:
  - whisper and faster-whisper add decoding details such as `avg_logprob` and `no_speech_prob`;
  - faster-whisper also adds `duration` and `language_probability` at the top level.
