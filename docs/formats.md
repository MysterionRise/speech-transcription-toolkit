# Output formats

How the format is chosen:
- `-f`/`--format` names it explicitly.
- Otherwise the `--output` file's extension picks it (`-o talk.srt`). An extension that isn't a format, such as
  `.tsv` or `.md`, gives plain text and a warning; add `-f txt` to keep the extension without the warning.
- With neither, `transcribe` prints plain text.

In Python, use `result.render(fmt)` or `result.save(path)`. `save()` picks the format from the extension the same way,
warning about unknown ones; pass `fmt` to choose it.

| Format | Contents | With speaker labels |
|---|---|---|
| `txt` | The plain transcript | One `[SPEAKER_00] text` line per speaker turn |
| `srt` | SubRip subtitles, one cue per segment | Each cue starts with `[SPEAKER_00]` |
| `vtt` | WebVTT subtitles, one cue per segment | Voice tags: `<v SPEAKER_00>text` |
| `json` | The full result: `text`, `segments`, `language` and backend-specific fields | Adds `speaker_segments` |

With speaker labels, a segment that has no speaker gets no label.

**Writing files:**
- Files are written as UTF-8 with a final newline.
- Missing folders are created.

`--json FILE` writes the JSON result in addition to the main output.

## Subtitles

srt and vtt files are written so that players and validators accept them:
- **Timing:** cues are in time order, and no two overlap: a cue that runs into the next one ends when the next one
  starts. Each cue lasts at least half a second, unless the next cue starts sooner, so none has zero length. The JSON
  output keeps the original times.
- **WebVTT escaping:** `&`, `<` and `>` in the text and in speaker names are written as `&amp;`, `&lt;` and `&gt;`, and
  players show the original characters. So text like `Tom & Jerry <3`, or a `-->` in it, can't break a cue.
- **Line breaks:** blank lines inside a segment's text are dropped, since they would end the cue. Line breaks in speaker
  names become spaces.

## Readable subtitle lines

`--max-line-width N` splits srt and vtt cues into lines of at most N columns, with two lines per cue:
- **Width:** Chinese, Japanese and Korean characters, like other wide and fullwidth characters, take two columns, as
  they do on screen.
- **Speaker labels:** in srt, the `[SPEAKER_00]` label counts toward the first line of each cue; when the first word
  doesn't fit beside it, the label gets a line of its own. In vtt, voice tags and escapes like `&amp;` aren't shown, so
  they don't count.
- **Where lines break:**
  - Lines break at spaces.
  - Chinese and Japanese text, which has no spaces between words, also breaks between characters, and no spaces are
    added. A line doesn't start with closing punctuation such as `。` or `」`, or end with opening punctuation such as
    `「`.
  - Korean breaks at spaces, so words stay whole.
  - A word or speaker label longer than N gets a line of its own.

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
