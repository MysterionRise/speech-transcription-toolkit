# Output formats

How the format is chosen:
- `-f`/`--format` names it: one format, or several separated by commas (`-f srt,json`), each written to its own file
  from one transcription.
  - `-f all` writes txt, srt, vtt, json and tsv, and rttm too with `--diarize`: rttm lists speaker turns, so it needs
    speaker labels. Name it to add it anyway, as for a diarized result with `--from-json`: `-f all,rttm`.
- Otherwise the `--output` file's extension picks it (`-o talk.srt`). An extension that isn't a format, such as
  `.csv` or `.md`, gives plain text and a warning; add `-f txt` to keep the extension without the warning.
- With neither, `transcribe` prints plain text.

In Python, use `result.render(fmt)` or `result.save(path)`. `save()` picks the format from the extension the same way,
with a `SpeechToolkitWarning` for an unknown one; pass `fmt` to choose it.

| Format | Contents | With speaker labels |
|---|---|---|
| `txt` | The plain transcript; with `--timestamps`, one `[00:01:02] text` line per segment | One `[SPEAKER_00] text` line per speaker turn |
| `srt` | SubRip subtitles, one cue per segment | Each cue starts with `[SPEAKER_00]` |
| `vtt` | WebVTT subtitles, one cue per segment; with `--word-level`, a `<00:00:01.200>` tag where each word starts | Voice tags: `<v SPEAKER_00>text` |
| `json` | The full result: `text`, `segments`, `language` and backend-specific fields | Adds `speaker_segments` |
| `tsv` | Tab-separated `start`, `end` and `text` columns, as openai-whisper writes them | Adds a `speaker` column |
| `rttm` | Needs speaker labels | One line per speaker turn, the standard format of diarization tools |

With speaker labels, a segment that has no speaker gets no label.

**Writing files:**
- Files are written as UTF-8 with a final newline.
- Missing folders are created.

`--json FILE` writes the JSON result in addition to the main output.

## Output options

Each option changes the formats it names; the others ignore it. In Python, `render()` and `save()` take them as
keyword arguments.

| Option | Python | Formats | What it does |
|---|---|---|---|
| `--max-line-width N` | `max_line_width=N` | srt, vtt | Split cues into lines of at most N columns: [Readable subtitle lines](#readable-subtitle-lines). |
| `--max-lines N` | `max_lines=N` | srt, vtt | With `--max-line-width`, at most N lines per cue (default: 2). |
| `--max-cue-duration SECONDS` | `max_cue_duration=SECONDS` | srt, vtt | Split cues that would last longer: [Cue length](#cue-length). |
| `--word-level` | `word_level=True` | vtt | Mark when each word starts: [Word-level WebVTT](#word-level-webvtt). |
| `--timestamps` | `timestamps=True` | txt | One line per segment, starting with its time: [Timestamped text](#timestamped-text). |
| `--speaker-names NAMES` | `speaker_names=...` | all | Rename the speakers: [Speaker names](#speaker-names). |

```python
result.save("talk.srt", max_line_width=42, max_lines=1, max_cue_duration=6)
result.save("meeting.txt", speaker_names=["Alice", "Bob"], timestamps=True)
```

An option value that doesn't fit, such as `max_lines` without `max_line_width`, raises `UnsupportedOptionError` (a
`ValueError`), and so does an unknown format.

## Timestamped text

`--timestamps` writes one line per segment instead of the plain transcript, each starting with the time the segment
starts, rounded down to the second:

```text
[00:00:00] Hello and welcome.
[00:00:02] Today we talk about subtitles.
[01:01:01] 字幕は便利です。
```

With speaker labels, the label follows the time: `[00:00:03] [SPEAKER_00] Thanks.` Line breaks inside a segment become
spaces.

Without `--timestamps`, a diarized transcript has one line per speaker turn: a speaker's consecutive segments are
joined with a space, but without one next to Chinese or Japanese text (`我同意。我们走吧。`).

## TSV

`tsv` is the tab-separated format openai-whisper writes: a header, then one row per segment with its start and end in
whole milliseconds, and its text on one line (tabs and line breaks become spaces).

```text
start	end	text
0	2200	Hello and welcome.
2500	9000	Today we talk about subtitles.
```

With speaker labels, the rows are the speaker-labelled segments, and a fourth column, `speaker`, holds their label
(empty for a segment without one). The first three columns stay as openai-whisper writes them.

## RTTM

`rttm` (Rich Transcription Time Marked) is the format that diarization tools such as pyannote.audio, and scoring tools
such as pyannote.metrics and dscore, read and write. Each speaker turn is a line of 10 fields separated by spaces:

```text
SPEAKER meeting 1 0.000 3.000 <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER meeting 1 3.200 1.800 <NA> <NA> SPEAKER_00 <NA> <NA>
```

- **Fields:** the type (`SPEAKER`), the file ID, the channel (`1`), the start and the duration in seconds, two unused
  fields (`<NA>`), the speaker, and two more unused fields.
- **File ID:** the input's name without its extension (`meeting.wav` gives `meeting`, standard input `stdin`). In
  Python, `save()` uses the file's name (`result.save("meeting.rttm")` gives `meeting`), and `render()` takes
  `file_id=`, else uses `audio`. Spaces in the file ID and in speaker names become `_`, since they separate the fields.
- **Turns:** the speaker-labelled segments, in time order. Consecutive segments of one speaker that touch or overlap
  are one turn. Speech no speaker turn covers (labelled `unknown`) is left out. Turns of different speakers may
  overlap.
- **Speaker labels needed:** `-f rttm` and `-o talk.rttm` need `--diarize` (or `--from-json`), and are refused before
  the model loads. In Python, `render("rttm")` of a result without speaker labels raises `UnsupportedOptionError`.

## Subtitles

srt and vtt files are written so that players and validators accept them:
- **Timing:** cues are in time order, and no two overlap: a cue that runs into the next one ends when the next one
  starts. Each cue lasts at least half a second, unless the next cue starts sooner (or `--max-cue-duration` is
  shorter), so none has zero length. The JSON output keeps the original times.
- **WebVTT escaping:** `&`, `<` and `>` in the text and in speaker names are written as `&amp;`, `&lt;` and `&gt;`, and
  players show the original characters. So text like `Tom & Jerry <3`, or a `-->` in it, can't break a cue.
- **Line breaks:** blank lines inside a segment's text are dropped, since they would end the cue. Line breaks in speaker
  names become spaces.

## Readable subtitle lines

`--max-line-width N` splits srt and vtt cues into lines of at most N columns, with two lines per cue, or
`--max-lines` of them:
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

## Cue length

`--max-cue-duration SECONDS` splits srt and vtt cues that would last longer than SECONDS, so a cue doesn't stay on
screen through a pause or a long sentence:
- **Where cues split:** between words, at their word timestamps when the backend provides them (the option turns them
  on for a backend that has them). Otherwise each cue's duration is shared out in proportion to text length.
- **With `--max-line-width`:** a cue ends when its lines are full, or when the next word would take it past the limit.
- **A single long word** gets a cue of its own, which ends SECONDS after it starts.
- Cues that are short enough are written as they are. The minimum length of half a second never takes a cue past the
  limit.

## Word-level WebVTT

`--word-level` marks when each word starts with a WebVTT timestamp tag, for karaoke-style highlighting: browsers style
the words already spoken and those still to come with the `::cue(:past)` and `::cue(:future)` CSS selectors.

```text
00:00:00.000 --> 00:00:02.200
Hello <00:00:00.600>and <00:00:00.900>welcome.
```

- **Needs word timestamps:** `--word-level` turns them on, with a warning for a backend that has none. Cues without
  word timings are written as usual.
- **Valid WebVTT:** each tag is later than the cue's start and the tag before it, and earlier than the cue's end. A
  word whose time doesn't fit, such as one that starts with the word before it, gets no tag of its own.
- **Chinese and Japanese:** tags mark where words start, not every character.
- It works with `--max-line-width` and `--max-cue-duration`. The cues of segments with word timings take their text and
  times from the words.

## Speaker names

`--speaker-names "Alice,Bob"` renames the speakers in the order they first speak: the first to speak becomes Alice, the
next Bob. Every format shows the names, the JSON output's `speaker_segments` included.
- **Fewer names than speakers:** the others keep their labels, such as `SPEAKER_02`. Extra names are ignored.
- **Keeping a label:** an empty name keeps that speaker's: `--speaker-names ",Bob"` renames only the second speaker.
- **Merging:** the same name twice makes those speakers one, for when diarization splits one person in two.
- **`unknown`**, the label of speech no speaker turn covers, isn't a speaker and isn't renamed.
- Names can't contain commas. The option needs `--diarize` or `--from-json`.

In Python, `render()` and `save()` take `speaker_names=`: a list (`["Alice", "Bob"]`), a string separated by commas,
or a mapping from labels to names (`{"SPEAKER_01": "Bob"}`), which can rename `unknown` too. The result itself isn't
changed.

## Rendering a saved result again

`--from-json` takes results saved as JSON (by `--json FILE` or `-f json`) as its inputs, and writes them in the formats
asked for without transcribing: no model loads.

```bash
transcribe meeting.wav --diarize -f json                                   # meeting.json
transcribe --from-json meeting.json -f srt,rttm --speaker-names "Alice,Bob"
transcribe --from-json results/ --outdir subs -f vtt --word-level          # every .json in a folder
```

- **The same output:** a format written from the JSON is the one the transcription would have written with the same
  output options.
- **Inputs:** JSON files, or folders, which are searched for `.json` files; not standard input. Outputs are named and
  placed as in [batch mode](cli.md#several-files-and-formats).
- **Options:** the output options above all apply. Options that change the transcription, such as `--model`,
  `--language` or `--diarize`, are refused.
- **The inputs stay as they are:** an output that would replace a JSON file it reads, such as `-f json` next to it, is
  refused before anything is written; use `--outdir`. `--skip-existing` keeps existing files anyway, so
  `-f all --skip-existing` works.
- **Errors:** a file that isn't a saved result is reported with what is wrong in it, and the others carry on.

In Python, `TranscriptionResult.load("meeting.json")` reads a saved result; see
[Loading a saved result](python-api.md#loading-a-saved-result).

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
- With `--diarize`, `speaker_segments` holds the segments with a `speaker` label each, and a `segment_id`: the `id`
  of the segment in `segments` each one comes from.

In Python, `TranscriptionResult.load("talk.json")` reads the file back, to render it in another format; see
[Loading a saved result](python-api.md#loading-a-saved-result).
