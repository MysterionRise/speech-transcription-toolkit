# Command line

## transcribe

```bash
transcribe audio.mp3                           # transcript to stdout
transcribe audio.mp3 -o audio.srt              # subtitles; format from extension (txt, srt, vtt, json)
transcribe recordings/ --outdir out -f vtt     # every audio file in a folder
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
| `-b`, `--backend` | `faster-whisper` (default), `whisper`, `voxtral`, `parakeet` or `canary`. See [Backends](backends.md). |
| `-m`, `--model` | Model name (default: the backend's default; see `--list-models`), or a Hugging Face repo id (`org/name`) or local path: see [Custom models](backends.md#custom-models). |
| `-l`, `--language` | Language code such as `en` (default: detect it). Parakeet ignores it, with a warning. |
| `-t`, `--task` | `transcribe` (default), or `translate` to English (whisper, faster-whisper, canary). With another backend, `translate` is an error, reported before the model loads. |
| `--device` | `auto` (default: the GPU when available), `cpu`, `cuda`, `cuda:N` (the N-th GPU, from 0) or `mps` (an Apple GPU; not faster-whisper). See [Devices and precision](backends.md#devices-and-precision). |
| `--compute-type TYPE` | Precision: `auto` (default) picks one the device supports; or a type such as `int8` or `float16`, see [Devices and precision](backends.md#devices-and-precision). |
| `-O`, `--option KEY=VALUE` | Set a backend option, such as `-O cpu_threads=4`; repeat for several. See [Backend options](backends.md#backend-options). |
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
| `--list-backends` | List the backends with their models, capabilities, devices and options, and whether their packages are installed, then exit. |
| `--list-models` | List the models of `--backend`, and the custom models it takes, then exit. |
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
