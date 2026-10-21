# Command line

## transcribe

```bash
transcribe audio.mp3                           # transcript to stdout
transcribe audio.mp3 -o audio.srt              # subtitles; format from extension (txt, srt, vtt, json)
transcribe recordings/ --outdir out -f vtt     # every audio file in a folder
transcribe audio.mp3 -m large-v3 -l de         # another model, known language
transcribe audio.mp3 -t translate -m large-v3  # translate to English (turbo can't translate)
transcribe audio.mp3 -b faster-whisper         # faster, especially on CPU
transcribe --list-backends                     # backends, their models and capabilities
```

Progress goes to stderr, so `transcribe a.mp3 > a.txt` gives a clean file (`-q` hides progress). `python -m speech_toolkit`
runs the same command.

**Better results:**
- `--prompt "Kubernetes, Grafana"` helps spell names and jargon.
- `--vad` skips silence, which stops made-up text in quiet parts (faster-whisper).
- `--max-line-width 42` splits subtitles into readable lines.
- `--word-timestamps` adds per-word timings to JSON.

### Several files

`transcribe` switches to batch mode, writing one output file per input, when you pass any of:
- several files;
- a folder;
- `--outdir`.

How batch mode works:
- **Folders** are searched recursively for audio and video files.
- **Output location:**
  - With `--outdir`, outputs go into that folder and mirror the input folder structure.
  - Without it, each output is written next to its input.
- **Name clashes:** if two inputs share a name, such as `talk.mp3` and `talk.wav`, their outputs keep the full name
  (`talk.mp3.srt`).
- **Errors:** a file that fails is reported and the others carry on. The exit status is non-zero if any file failed.

### Options

| Option | Description |
|---|---|
| `audio …` | Audio or video files, or folders, to transcribe. Anything ffmpeg can decode works. |
| `-b`, `--backend` | `whisper` (default), `faster-whisper`, `voxtral`, `parakeet` or `canary`. See [Backends](backends.md). |
| `-m`, `--model` | Model name (default: the backend's default; see `--list-models`). |
| `-l`, `--language` | Language code such as `en` (default: detect it). Parakeet ignores it, with a warning. |
| `-t`, `--task` | `transcribe` (default), or `translate` to English (whisper, faster-whisper, canary). With another backend, `translate` is an error, reported before the model loads. |
| `--device` | `cpu` or `cuda` (default: the GPU when available). |
| `--hf-token TOKEN` | Hugging Face token for model downloads. Defaults to the `HF_TOKEN` or `HUGGINGFACE_TOKEN` environment variable, which is safer than passing it on the command line. |
| `-q`, `--quiet` | Hide progress output. |
| `-o`, `--output FILE` | Write the transcript to this file instead of stdout; its extension picks the format. |
| `-f`, `--format` | `txt`, `srt`, `vtt` or `json` (default: from the `--output` extension, else txt). See [Output formats](formats.md). |
| `--outdir DIR` | Write one file per input into this folder. |
| `--json FILE` | Also write the full result as JSON (single input only). |
| `--max-line-width N` | Split subtitles into lines of at most N characters, 2 lines per cue (srt, vtt). |
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

Converts `.ogg`, `.oga` and `.opus` files to 16-bit WAV. Folders are searched recursively.

```bash
ogg2wav recordings/ --outdir wav --rate 16000 --channels 1
```

| Option | Description |
|---|---|
| `inputs …` | Files, or folders containing them. |
| `--outdir DIR` | Where to write the WAV files, mirroring sub-folders (default: next to each input). |
| `--rate HZ` | Sample rate (default: 16000). |
| `--channels {1,2}` | 1 for mono (default), 2 for stereo. |
| `--overwrite` | Replace WAV files that already exist (default: skip them). |
