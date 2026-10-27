# Command line

## transcribe

```bash
transcribe audio.mp3                           # transcript to stdout
transcribe audio.mp3 -o audio.srt              # subtitles; format from extension (txt, srt, vtt, json)
transcribe recordings/ --outdir out -f vtt     # every audio file in a folder
transcribe recordings/ -f srt,json --skip-existing  # two formats; resume a stopped run
cat audio.mp3 | transcribe -                   # audio from standard input
transcribe audio.mp3 -m large-v3 -l de         # another model, known language
transcribe audio.mp3 -t translate -m large-v3  # translate to English (turbo can't translate)
transcribe audio.mp3 -b whisper                # another backend: OpenAI's Whisper (the whisper extra)
transcribe --list-backends                     # backends, their models and capabilities
```

Progress goes to stderr, so `transcribe a.mp3 > a.txt` gives a clean file (`-q` hides progress). `python -m speech_toolkit`
runs the same command.

**Better results:**
- `--prompt "Kubernetes, Grafana"` helps spell names and jargon.
- `--vad` skips silence, which stops made-up text in quiet parts (faster-whisper, the default backend).
- `--max-line-width 42` splits subtitles into readable lines.
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
  transcription of each input.
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

### Options

| Option | Description |
|---|---|
| `audio …` | Audio or video files, or folders, to transcribe; `-` reads standard input. Anything ffmpeg can decode works. |
| `-b`, `--backend` | `faster-whisper` (default), `whisper`, `voxtral`, `parakeet` or `canary`. See [Backends](backends.md). |
| `-m`, `--model` | Model name (default: the backend's default; see `--list-models`). |
| `-l`, `--language` | Language code such as `en` (default: detect it). Parakeet ignores it, with a warning. |
| `-t`, `--task` | `transcribe` (default), or `translate` to English (whisper, faster-whisper, canary). With another backend, `translate` is an error, reported before the model loads. |
| `--device` | `cpu` or `cuda` (default: the GPU when available). |
| `--hf-token TOKEN` | Hugging Face token for model downloads. Defaults to the `HF_TOKEN` or `HUGGINGFACE_TOKEN` environment variable, which is safer than passing it on the command line. |
| `-q`, `--quiet` | Hide progress output. |
| `-o`, `--output FILE` | Write the transcript to this file instead of stdout; its extension picks the format. |
| `-f`, `--format` | `txt`, `srt`, `vtt` or `json` (default: from the `--output` extension, else txt). Several separated by commas (`srt,json`), or `all`, write a file each from one transcription. See [Output formats](formats.md). |
| `--outdir DIR` | Write each input's output files into this folder. |
| `--json FILE` | Also write the full result as JSON to this file (a single input; for several, add `json` to `-f`). |
| `--skip-existing` | Keep output files that already exist: transcribe only the inputs with an output missing, and write just the missing ones (default: overwrite them). |
| `--max-line-width N` | Split subtitles into lines of at most N columns, 2 lines per cue (srt, vtt). Chinese, Japanese and Korean characters take two columns. |
| `--prompt TEXT` | Names, terms or a sample sentence that guide spelling and style (whisper, faster-whisper). |
| `--vad` | Skip silence before transcribing (faster-whisper). |
| `--word-timestamps` | Add per-word timings to the JSON output (whisper, faster-whisper, parakeet). |
| `--diarize` | Label speakers with pyannote.audio. See [Speaker labels](diarization.md). |
| `--num-speakers N` | Exact number of speakers, if known (needs `--diarize`). |
| `--min-speakers N`, `--max-speakers N` | Bounds on the number of speakers (needs `--diarize`). |
| `--list-backends` | List the backends with their models and capabilities, and whether their packages are installed, then exit. |
| `--list-models` | List the models of `--backend`, then exit. |
| `--version` | Print the version. |

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
