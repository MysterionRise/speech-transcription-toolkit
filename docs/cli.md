# Command line

## transcribe

```bash
transcribe audio.mp3                           # transcript to stdout
transcribe audio.mp3 -o audio.srt              # subtitles; format from extension (txt, srt, vtt, json, tsv, rttm)
transcribe recordings/ --outdir out -f vtt     # every audio file in a folder
transcribe recordings/ -f srt,json --skip-existing  # two formats; resume a stopped run
transcribe --from-json audio.json -f vtt --word-level  # a saved result in another format, no model
cat audio.mp3 | transcribe -                   # audio from standard input
transcribe audio.mp3 -m large-v3 -l de         # another model, known language
transcribe audio.mp3 -t translate -m large-v3  # translate to English (turbo can't translate)
transcribe audio.mp3 -b whisper                # another backend: OpenAI's Whisper (the whisper extra)
transcribe --list-backends                     # backends, their models and capabilities
transcribe --detect-language recordings/       # each file's language, without transcribing
```

Progress goes to stderr, so `transcribe a.mp3 > a.txt` gives a clean file (`-q` hides progress, see
[Messages](#messages)). `python -m speech_toolkit` runs the same command.

**Better results:**
- `--prompt "Kubernetes, Grafana"` helps spell names and jargon.
- `--vad` skips silence, which stops made-up text in quiet parts (faster-whisper, the default backend).
- `--max-line-width 42` splits subtitles into readable lines, and `--max-cue-duration 6` keeps cues short.
- `--word-timestamps` adds per-word timings to JSON.

### Several files and formats

`transcribe` switches to batch mode, writing each input's transcript to files instead of stdout, when you pass any
of:
- several files;
- a folder;
- several formats;
- `--outdir`.

How batch mode works:
- **Folders** are searched recursively for audio and video files, by extension in any case:
  - audio: `.aac .aif .aiff .amr .caf .flac .m4a .m4b .mka .mp3 .oga .ogg .opus .wav .wma`;
  - video: `.3gp .avi .m2ts .m4v .mkv .mov .mp4 .mpeg .mpg .mts .webm`.

  Hidden files and folders, whose names start with a dot, are left out. So are the `._talk.mp3` files macOS writes on
  other disks to hold a file's metadata. `.ts` files aren't picked up, since most are TypeScript. To transcribe one of
  these, name it on the command line, like any other file ffmpeg can decode.
- **Formats:** `-f srt,json` (several, separated by commas) or `-f all` writes a file per format, all from one
  transcription of each input. `-f all` writes rttm only with `--diarize`; see [Output formats](formats.md).
- **Saved results:** with `--from-json`, the inputs are results saved as JSON, and folders are searched for `.json`
  files. They are written in the formats asked for without transcribing, so no model loads; see
  [Rendering a saved result again](formats.md#rendering-a-saved-result-again).
- **Output location:**
  - With `--outdir`, outputs go into that folder and mirror the input folder structure.
  - Without it, each output is written next to its input.
  - Outputs are named after their input, with the format as extension: `talk.mp3` gives `talk.srt` and `talk.json`.
- **Name clashes:** if two inputs share a name, such as `talk.mp3` and `talk.wav`, their outputs keep the full name
  (`talk.mp3.srt`).
- **Existing outputs** are overwritten. With `--skip-existing` they are kept:
  - an input whose outputs all exist is skipped without being transcribed, so a stopped run picks up where it left off;
  - an input with some outputs missing is transcribed, and only the missing ones are written.
- **Progress:** each input gets a line such as `[2/5] recordings/b.mp3` on stderr, and the run ends with a summary
  such as `Finished in 4:05: 3 done, 1 skipped, 1 failed.` `-q` hides them; errors are always shown.
- **Errors:** a file that fails is reported and the others carry on. The exit status is non-zero if any file failed.
- **Ctrl-C** stops the run, shows the summary and exits with status 130. Each file is written under a temporary name
  and renamed once complete, so an interrupted run leaves no half-written output. Run the command again with
  `--skip-existing` to carry on.

`--json FILE` takes a single input, also with `--outdir` or several formats. For a JSON file per input, add `json` to
`-f`. `-o` takes a single input and format.

### Pipes and redirection

`transcribe -` reads the audio piped into standard input:

```bash
cat talk.mp3 | transcribe -                          # transcript to stdout
transcribe - -o talk.srt < talk.mp3                  # to a file
transcribe - --outdir subs -f srt,json < talk.mp3    # subs/stdin.srt and subs/stdin.json
```

- `-` must be the only input.
- With `--outdir`, the outputs are named `stdin.<format>`. Several formats need `--outdir`.
- ffmpeg can't decode an MP4, M4A or MOV file whose index comes after the audio, as phones record them, from a pipe.
  Such a file is copied to a temporary file first, which is deleted afterwards.

stdout carries only the transcript. On Windows, a transcript redirected to a file or a pipe is written as UTF-8, so
text in any language can be saved; set `PYTHONIOENCODING` to choose another encoding.

### Language detection

`--detect-language` prints the language spoken in each input, and exits without transcribing anything:

```console
$ transcribe --detect-language talk.mp3 interview.m4a
talk.mp3	en	0.98
interview.m4a	de	0.97
$ transcribe --detect-language talk.mp3 -f json
{"file": "talk.mp3", "language": "en", "probability": 0.98, "candidates": [{"language": "en", "probability": 0.98}, ...]}
```

- **Output:** a line per input on stdout, as soon as it is detected: the file, the language code and its probability
  (from 0 to 1, with two decimals), separated by tabs. With `-f json`, a JSON object per line instead, with the five
  most likely languages in `candidates`, most likely first. Standard input is the file `-`.
- **Only the first 30 seconds** of each input are decoded. The Whisper models look at no more, also when they detect the
  language to transcribe. A recording that starts with music or a long silence may be misdetected: give its language
  with `-l` to transcribe it.
- **Backends:** whisper and faster-whisper. Another backend is an error, reported before the model loads. `-m`,
  `--device`, `--compute-type` and `-O` choose the model as usual: smaller models are faster, larger ones more accurate.
- **Inputs** are found as for transcription: files, folders (searched for audio and video files) or `-`.
- **Transcription options,** such as `--language`, `--diarize` or `--output`, are an error with `--detect-language`.
  To keep the lines, redirect stdout: `transcribe --detect-language recordings/ > languages.tsv`.
- **Errors:** a file that fails is reported on stderr and the others carry on. The exit status is non-zero if any file
  failed.

### Options

| Option | Description |
|---|---|
| `audio …` | Audio or video files, or folders, to transcribe; `-` reads standard input. Anything ffmpeg can decode works. |
| `-b`, `--backend` | `faster-whisper` (default), `whisper`, `voxtral`, `parakeet` or `canary`. See [Backends](backends.md). |
| `-m`, `--model` | Model name (default: the backend's default; see `--list-models`), or a Hugging Face repo id (`org/name`) or local path: see [Custom models](backends.md#custom-models). |
| `-l`, `--language` | Language code such as `en` (default: detect it). Parakeet ignores it, with a warning. |
| `-t`, `--task` | `transcribe` (default), or `translate` to English (whisper, faster-whisper, canary). With another backend, `translate` is an error, reported before the model loads. |
| `--device` | `auto` (default: the GPU when available), `cpu`, `cuda`, `cuda:N` (the N-th GPU, from 0) or `mps` (an Apple GPU; not faster-whisper). See [Devices and precision](backends.md#devices-and-precision). |
| `--compute-type TYPE` | Precision: `auto` (default) picks one the device supports; or a type such as `int8` or `float16`, see [Devices and precision](backends.md#devices-and-precision). |
| `-O`, `--option KEY=VALUE` | Set a backend option, such as `-O cpu_threads=4`; repeat for several. See [Backend options](backends.md#backend-options). |
| `--hf-token TOKEN` | Hugging Face token for model downloads. Defaults to the `HF_TOKEN` or `HUGGINGFACE_TOKEN` environment variable, which is safer than passing it on the command line. |
| `-q`, `--quiet` | Hide progress: the loading, `[i/N]` and `→ file` lines and the summary, progress bars (Hugging Face downloads and transformers' too) and info messages. Warnings and errors still show. |
| `-v`, `--verbose` | Also show debug messages: what is decoded, transcribed and labelled, and how long it took. Can't be combined with `-q`. |
| `--log-level LEVEL` | Show messages from this level up: `debug`, `info`, `warning` or `error` (default: `info`; `warning` with `-q`, `debug` with `-v`). |
| `-o`, `--output FILE` | Write the transcript to this file instead of stdout; its extension picks the format. |
| `-f`, `--format` | `txt`, `srt`, `vtt`, `json`, `tsv` or `rttm` (default: from the `--output` extension, else txt; rttm needs `--diarize`). Several separated by commas (`srt,json`), or `all` (rttm only with `--diarize`), write a file each from one transcription. See [Output formats](formats.md). |
| `--outdir DIR` | Write each input's output files into this folder. |
| `--json FILE` | Also write the full result as JSON to this file (a single input; for several, add `json` to `-f`). |
| `--skip-existing` | Keep output files that already exist: transcribe only the inputs with an output missing, and write just the missing ones (default: overwrite them). |
| `--from-json` | The inputs are results saved as JSON (by `--json` or `-f json`), not audio: write them in the formats asked for without transcribing, so no model loads. Folders are searched for `.json` files. See [Rendering a saved result again](formats.md#rendering-a-saved-result-again). |
| `--max-line-width N` | Split subtitles into lines of at most N columns, 2 lines per cue (srt, vtt). Chinese, Japanese and Korean characters take two columns. |
| `--max-lines N` | With `--max-line-width`, at most N lines per cue (default: 2). |
| `--max-cue-duration SECONDS` | Split subtitle cues that would last longer than SECONDS (srt, vtt). See [Cue length](formats.md#cue-length). |
| `--speaker-names NAMES` | Rename the speakers in the order they first speak, such as `"Alice,Bob"`, in every format; an empty name keeps that speaker's label (with `--diarize`, or `--from-json`). See [Speaker names](formats.md#speaker-names). |
| `--timestamps` | txt: one line per segment, starting with its time: `[00:01:02] text`. |
| `--word-level` | vtt: mark when each word starts, with `<00:00:01.200>` tags, so players can highlight the words as they are spoken. Turns on word timestamps. See [Word-level WebVTT](formats.md#word-level-webvtt). |
| `--prompt TEXT` | Names, terms or a sample sentence that guide spelling and style (whisper, faster-whisper). |
| `--vad` | Skip silence before transcribing (faster-whisper). |
| `--word-timestamps` | Add per-word timings to the JSON output (whisper, faster-whisper, parakeet). |
| `--diarize` | Label speakers with pyannote.audio. See [Speaker labels](diarization.md). |
| `--num-speakers N` | Exact number of speakers, if known (needs `--diarize`). |
| `--min-speakers N`, `--max-speakers N` | Bounds on the number of speakers (needs `--diarize`). |
| `--list-backends` | List the backends with their models, capabilities, devices and options, and whether their packages are installed, then exit. |
| `--list-models` | List the models of `--backend`, and the custom models it takes, then exit. |
| `--detect-language` | Print the language of each input and its probability, from its first 30 seconds, then exit without transcribing (whisper, faster-whisper); `-f json` prints JSON. See [Language detection](#language-detection). |
| `--version` | Print the version. |

### Messages

Everything but the transcript goes to stderr:

| Message | Shown |
|---|---|
| `Loading faster-whisper model 'turbo'...`, `[2/5] talk.mp3`, `→ talk.srt` and the summary | Unless `-q`. |
| Progress bars: faster-whisper's and whisper's while transcribing, Hugging Face's while downloading a model, transformers' `Loading weights` | Unless `-q`. faster-whisper hides its download bars itself. |
| Progress messages, such as `Parakeet: transcribed 1/3 chunks` | At `info` level: by default, not with `-q`. |
| `Debug: …` lines | At `debug` level: with `-v` or `--log-level debug`. |
| `Warning: …` and `Error: …` lines | Always, unless `--log-level error` hides the warnings that are logged. |

`-q` also sets `HF_HUB_DISABLE_PROGRESS_BARS=1` and `TRANSFORMERS_VERBOSITY=error` for the run, so transformers
shows only its errors. `--log-level` applies to speech_toolkit's own messages; other libraries' log messages aren't
shown. To log from Python instead, see [Logging](python-api.md#logging).

## ogg2wav

Converts `.ogg`, `.oga` and `.opus` files to 16-bit WAV with ffmpeg. Folders are searched recursively.

```bash
ogg2wav recordings/ --outdir wav --rate 16000 --channels 1
```

It prints the path of each WAV it writes, one per line; files it skips and errors go to stderr. Files whose names differ
only in the extension, such as `talk.ogg` and `talk.opus`, keep their full name (`talk.ogg.wav`, `talk.opus.wav`). The
exit status is non-zero if any file failed.

| Option | Description |
|---|---|
| `inputs …` | Files, or folders containing them. |
| `--outdir DIR` | Where to write the WAV files, mirroring sub-folders (default: next to each input). |
| `--rate HZ` | Sample rate (default: 16000). |
| `--channels {1,2}` | 1 for mono (default), 2 for stereo. |
| `--overwrite` | Replace WAV files that already exist (default: skip them). |
