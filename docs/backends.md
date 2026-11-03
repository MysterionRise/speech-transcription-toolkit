# Backends

| Backend | Models (default in bold) | Notes |
|---|---|---|
| `faster-whisper` (default) | tiny … large-v3, distil-\*, **turbo** | Whisper's models, several times faster; runs on CTranslate2 instead of PyTorch |
| `whisper` | tiny … large-v3, **turbo** | OpenAI Whisper, on PyTorch |
| `voxtral` | **voxtral-mini**, voxtral-small | Mistral Voxtral (3B and 24B); transcription only, timestamps per ~30 s |
| `parakeet` | **parakeet-tdt-0.6b-v3**, parakeet-tdt-0.6b-v2 | NVIDIA Parakeet: fast and accurate, word timestamps. v3: English + 24 European languages, detected automatically; v2: English only |
| `canary` | **canary-1b-v2** | NVIDIA Canary: the same 25 languages; translates to English. Can't detect the language: give it with `-l` (default English). Timestamps per ~30 s |
| `transformers` | **openai/whisper-large-v3-turbo**, or any speech recognition model on the Hugging Face Hub | Hugging Face transformers' pipeline: Whisper fine-tunes, distil-whisper, wav2vec2, MMS, Moonshine, ... What it can do depends on the model: see [Any Hugging Face model](#any-hugging-face-model-transformers) |

`transcribe --list-backends` lists them with their models, capabilities, devices and [options](#backend-options), and
says whether each one's packages are installed. `transcribe --list-models -b <backend>` lists one backend's models.
faster-whisper comes with every install; each other backend needs its [extra](install.md#extras).

## Custom models

Besides the names in the table, `-m` (`model=` in Python) takes fine-tunes, other sizes and offline copies:

| Backend | Also takes | For example |
|---|---|---|
| `faster-whisper` | A Hugging Face repo id (`org/name`) of a CTranslate2 model, or a local CTranslate2 model folder | `-m Systran/faster-whisper-small`, `-m deepdml/faster-whisper-large-v3-turbo-ct2`, `-m ./models/my-ct2-model` |
| `whisper` | A local checkpoint file (`.pt`); openai-whisper loads no repo ids or folders | `-m ~/models/whisper-finetuned.pt` |
| `voxtral`, `parakeet`, `canary` | A Hugging Face repo id (`org/name`), or a local model folder, of a model of the same kind | `-m nvidia/parakeet-tdt-0.6b-v2` |
| `transformers` | The repo id (`org/name`), or a local folder, of any speech recognition model transformers' pipeline runs | `-m openai/whisper-small`, `-m facebook/wav2vec2-base-960h`, `-m ./models/my-fine-tune` |

- **Offline:** a local folder or file loads without the network. A folder saved by `huggingface-cli download` or
  `faster_whisper.download_model()` works as is.
- **Checked first:** the name is checked before anything loads, from its shape alone: a name that is none of these
  stops with `ModelNotFoundError`, which lists the backend's names and the custom models it takes. A repo id that
  doesn't exist, or that you have no access to, fails when it downloads, with `ModelLoadError`.
- **Gated repos** need a Hugging Face token: `--hf-token` or `HF_TOKEN`.

## What each backend supports

Each backend declares its capabilities; `--list-backends` shows them, and `--help` names the backends that support each
option.

| Capability | whisper | faster-whisper | voxtral | parakeet | canary |
|---|---|---|---|---|---|
| `translate`: translate to English (`-t translate`) | yes, except turbo | yes, except turbo | no | no | yes |
| `language_detection`: detect the language and report it | yes | yes | no | no | no (needs `-l`) |
| `prompt`: `--prompt` | yes | yes | no | no | no |
| `vad`: `--vad`, see [Voice activity detection](#voice-activity-detection) | yes | yes | yes | yes | yes |
| `word_timestamps`: `--word-timestamps` | yes | yes | no | yes | no |

The `transformers` backend's capabilities depend on the model it loads: see [Any Hugging Face model](#any-hugging-face-model-transformers).

**Translation:**
- `-t translate` with a backend that can't translate is an error, reported before the model loads.
- Whisper's turbo models weren't trained for translation and keep the original language, so translate with `medium` or
  `large-v3`.

**Language:** Parakeet detects the language itself (v2 is English only), so it ignores `-l` with a warning. It can't
report the language it detected, so the result's language is empty (`null` in JSON).

**Unsupported options:** passing `--prompt`, `--vad` or `--word-timestamps` to a backend that doesn't support it prints
a warning and skips the option. It doesn't fail the run.

## Devices and precision

The GPU is used when available. `--device` (`device=` in Python) picks one:

| Device | Means | Backends |
|---|---|---|
| `auto` (default) | An NVIDIA GPU when the backend's library sees one, else the CPU. Never an Apple GPU: ask for `mps`. | all |
| `cpu` | The CPU. | all |
| `cuda` | The first NVIDIA GPU. | all |
| `cuda:N` | The NVIDIA GPU numbered N, from 0: `cuda:1` is the second. | all |
| `mps` | An Apple Silicon GPU. | whisper, voxtral, parakeet, canary, transformers; not faster-whisper (CTranslate2 has no Apple GPU support) |

- A device the backend can't use, such as `-b faster-whisper --device mps`, stops before anything loads.
- [Speaker labels](diarization.md) run on the same device.
- An NVIDIA GPU needs NVIDIA's CUDA libraries, and on Windows a CUDA build of PyTorch: see
  [GPU setup](install.md#gpu).

**Precision:** `--compute-type` (`compute_type=` in Python) sets it; the default, `auto`, picks one that suits the
device:

| Backend | `auto` on an NVIDIA GPU | `auto` on an Apple GPU | `auto` on the CPU | `--compute-type` also takes |
|---|---|---|---|---|
| faster-whisper | float16, or the next of int8_float16, int8, float32 the GPU supports | (no Apple GPU) | int8, or float32 | CTranslate2's types: `int8`, `int8_float32`, `int8_float16`, `int8_bfloat16`, `int16`, `float16`, `bfloat16`, `float32`, and `default` (the type the model was saved in) |
| voxtral, parakeet, canary, transformers | bfloat16, or float16 on a GPU without bfloat16 | float16 | float32 | `float32`, `float16`, `bfloat16` |
| whisper | fp16 | fp16 | fp32 | (nothing else) |

faster-whisper asks CTranslate2 which types the device supports, so a GPU without fast float16 runs in int8 instead
of failing. A type the device can't run fails when the model loads.

Out of memory? Use a smaller model (`-m small`), a smaller type (`--compute-type int8` with faster-whisper), or
`-b faster-whisper`.

## Backend options

Each backend declares its own settings. Set them with `-O key=value`, repeated for several, or `options={...}` in
Python; `transcribe --list-backends` lists them, with their defaults:

| Backend | Option | Default | Meaning |
|---|---|---|---|
| faster-whisper | `compute_type` | `auto` | The precision, as `--compute-type` sets it. |
| faster-whisper | `cpu_threads` | `0` | Threads on the CPU; 0 uses `OMP_NUM_THREADS` if set, else 4. |
| faster-whisper | `num_workers` | `1` | Transcriptions the model runs at once when several threads of a program share it; see [Speed](#speed-batches-beams-and-threads). |
| faster-whisper | `batch_size` | `1` | From 2 up, the batched pipeline, as `--batch-size` sets it; see [Speed](#speed-batches-beams-and-threads). |
| voxtral, canary | `batch_size` | `8` | Chunks of ~30 s transcribed at once. |
| parakeet | `batch_size` | `2` | Chunks of up to 120 s transcribed at once. |
| faster-whisper | `beam_size` | `5` | Beams in beam search, as `--beam-size` sets it; 1 decodes greedily. |
| whisper | `beam_size` | `1` | The same; openai-whisper decodes greedily by default. |
| whisper, faster-whisper | `condition_on_previous_text` | `true` | Prompt each 30 s window with the text before it; see [Made-up text](#made-up-text). |
| whisper, faster-whisper | `no_speech_threshold` | `0.6` | Skip a window as silence when its no-speech probability is above this and its text is improbable. |
| whisper, faster-whisper | `hallucination_silence_threshold` | `0` (off) | Skip silences longer than this many seconds around text that looks made up. |
| voxtral, parakeet, canary, transformers | `compute_type` | `auto` | The precision, as `--compute-type` sets it. |
| voxtral, canary | `max_new_tokens` | `500` | Most tokens of text per ~30 s chunk; see [Text limit per chunk](#text-limit-per-chunk). |
| all | `vad_threshold`, `vad_min_silence`, `vad_padding` | `0.5`, `2.0`, `0.4` | Tune `--vad`; see [Voice activity detection](#voice-activity-detection). |

```bash
transcribe talk.mp3 -O cpu_threads=8
transcribe talk.mp3 -b canary -l de -O max_new_tokens=1000 --device cuda:1
transcribe talk.mp3 -O condition_on_previous_text=false
```

- Values are converted to the option's type: numbers, `true`/`false`, or one of the listed choices.
- An option the backend doesn't have stops before anything loads, with a message that lists the ones it has; so does
  a value that doesn't fit.
- `--batch-size N` and `--beam-size N` are shortcuts for `-O batch_size=N` and `-O beam_size=N`, and win over them.

## Speed: batches, beams and threads

`--batch-size` and `--beam-size` trade memory, or a little accuracy, for speed. Both stop before anything loads with a
backend that doesn't have them.

**faster-whisper's batched pipeline,** `--batch-size N` with N from 2 up: voice activity detection cuts the audio at
its pauses into pieces of speech of up to 30 s, and the model decodes N pieces at once. faster-whisper's own
benchmarks find it several times faster on a GPU, where 8 or 16 is a good start; on the CPU it gains less, mostly on
long recordings (see the numbers below). What changes:

- **Silence is skipped,** as with `--vad`, which the pipeline needs to cut the audio.
- **Each piece is one segment,** of up to 30 s, instead of a sentence or two. Word timings, with `--word-timestamps`
  or `--max-line-width` (which turns them on for subtitles), still split subtitles into short cues. (Asking the
  pipeline to predict timestamps would split the pieces, but loses text: it decodes each piece once.)
- **No temperature fallback and no conditioning:** each piece is decoded once, on its own. So
  `condition_on_previous_text`, `no_speech_threshold` and `hallucination_silence_threshold` don't apply, and setting
  them warns.

**Chunked backends:** Voxtral and Canary cut long audio into ~30 s chunks and Parakeet into chunks of up to 120 s, and
transcribe `batch_size` chunks at once: 8, or 2 for Parakeet, the same 4 minutes of audio. Lower it if a long
recording runs out of GPU memory, raise it on a large GPU. A batch pads its shorter chunks to the longest one, which
costs time on the CPU when the last chunk of a recording is short; `-O batch_size=1` turns batching off.

**Beam size:** faster-whisper keeps 5 candidate transcripts (beams) at each step and openai-whisper 1, which is greedy
decoding. `--beam-size 1` makes faster-whisper up to twice as fast on the CPU, usually at a small cost in accuracy;
`--beam-size 5` makes openai-whisper search like faster-whisper: slower, and often a little more accurate.

**Threads:** `-O cpu_threads=N` sets the threads faster-whisper uses on the CPU (default: `OMP_NUM_THREADS`, else 4).
`-O num_workers=N` lets N threads of a Python program transcribe with one faster-whisper model at the same time, each
worker taking memory of its own; `transcribe` and `transcribe-server` transcribe one file at a time, so they don't
gain from it.

### Measured

faster-whisper `tiny` in int8 on 4 CPUs (a cloud VM's Intel Xeon @ 2.1 GHz). First with the
[benchmark harness](#benchmarks), on the first 20 clips of each set (`--limit 20`: 7.3 minutes of short recordings,
2.5 to 43 s each, in English, German and Spanish), all in one run:

| Options | RTFx | WER LibriSpeech | WER FLEURS (de, es) |
|---|--:|--:|--:|
| (the defaults: 5 beams, no batches) | 11.3 | 0.076 | 0.215 (0.273, 0.153) |
| `--beam-size 1` | 13.1 | 0.100 | 0.244 (0.294, 0.191) |
| `--batch-size 4` | 14.1 | 0.074 | 0.173 (0.216, 0.128) |
| `--batch-size 8` | 14.8 | 0.074 | 0.173 (0.216, 0.128) |
| `--batch-size 4 --beam-size 1` | 19.3 | 0.089 | 0.421 (0.265, 0.583) |
| `-O condition_on_previous_text=false` | 11.8 | 0.076 | 0.215 (0.273, 0.153) |
| `-O hallucination_silence_threshold=2` | 11.9 | 0.076 | 0.215 (0.273, 0.153) |

Then on a longer recording, the integration tests' long clip (104 s: two voices, sentences with 1 s pauses):

| Options | RTFx | WER |
|---|--:|--:|
| (the defaults) | 18.6 | 0.000 |
| `--beam-size 1` | 35.2 | 0.000 |
| `--batch-size 8` | 41.0 | 0.019 |
| `--batch-size 8 --beam-size 1` | 48.0 | 0.000 |

- **Greedy decoding** is up to twice as fast, and a little less accurate on harder speech.
- **The batched pipeline** is faster on the CPU too, and skipping silence made its transcripts of the short clips a
  little more accurate here. Short recordings make one piece each, so the speed comes from long ones.
- **Batched and greedy together** were the fastest, but on one Spanish clip the model fell into a loop ("Ciudad de
  Ciudad de ..."), which the batched pipeline has no fallback to stop: 103 errors in that 23-word clip instead of 6.
  Keep the default beams with `--batch-size`, unless speed matters most.
- `condition_on_previous_text` and `hallucination_silence_threshold` cost next to nothing; on these clips, with no
  long silences, they change nothing either.

## Made-up text

Whisper models sometimes write text nobody said, mostly in silence and long pauses: a phrase over and over, or "Thank
you." at the end. Ways to avoid it, for whisper and faster-whisper:

| Option | What it does | Cost |
|---|---|---|
| `-O condition_on_previous_text=false` | Transcribes each 30 s window without the text before it as a prompt, so one made-up phrase doesn't set off the next. | Names may be spelled differently from one window to the next. |
| `--vad` (faster-whisper) | Skips silence before transcribing. | Very quiet speech may be skipped too. Pauses shorter than 2 s stay. |
| `--batch-size 4` (faster-whisper) | The batched pipeline skips silence too, cutting at shorter pauses, and decodes each piece on its own. | Segments of up to 30 s; see [Speed](#speed-batches-beams-and-threads). |
| `-O hallucination_silence_threshold=2` | Skips silences longer than 2 s around text that looks made up. | Needs word timings, which it computes; may drop real words next to a pause. |
| `-O no_speech_threshold=0.4` | Drops a window as silence when its no-speech probability is above this (default 0.6) and its text is improbable. Lower drops more. | Can drop real speech; the default is Whisper's own. |

faster-whisper `turbo` on the CPU, on the long clip (104 s, see above) followed by 1 or 3 s of silence. With the
defaults it adds text at the end, and the word error rate rises from 0 to about 0.2:

| Options | 1 s of silence: WER, and the made-up text | 3 s of silence | RTFx (3 s) |
|---|---|---|--:|
| (the defaults) | 0.210: "That was a speech recognition test. Yes, I sent it to the whole team and I asked you after lunch. Thank you. Thank you, Gemma." | 0.214: "All right, thank you to the text test. Thank you." | 1.1 |
| `-O condition_on_previous_text=false` | **0.000**, none | **0.000**, none | 3.3 |
| `--vad` | 0.190: "A reality test. Thank you." | 0.167, none (but 25 real words dropped) | 4.3 |
| `--batch-size 4` | **0.000**, none | **0.000**, none | 4.9 |
| `-O hallucination_silence_threshold=2` | 0.090, none (but 19 real words dropped) | 0.090, none (the same) | 2.9 |
| `-O no_speech_threshold=0.3` | (not measured) | 0.257: "We did a speech recognition test. Let's take down and do this text. ..." | 1.1 |

`condition_on_previous_text=false` and the batched pipeline kept every word and made up none. They were also faster:
Whisper's temperature fallback, which decodes a window again when its text looks wrong, had nothing to retry (the batched pipeline has none). `--vad`
and `hallucination_silence_threshold` dropped real words. Measured once, on one clip, so take it as a hint of what to
try first, not as a rule.

## Text limit per chunk

Canary and Voxtral transcribe ~30 s chunks and write at most 500 tokens for each, plenty for normal speech. A chunk
that reaches the limit, from very fast speech or the model repeating itself, may be cut off, so they warn once per file
with the chunk's time range. Check the transcript there, raise the limit with `-O max_new_tokens=1000`, or transcribe
the file with another backend.

## Voice activity detection

`--vad` (`vad=True` in Python) skips the silence and transcribes only the speech, which every backend supports. That
avoids the text Whisper models make up in long silences, and the time spent on silence; the timestamps stay those of
the recording.

| Backend | How it skips the silence |
|---|---|
| faster-whisper | Its built-in filter: the stretches of speech joined end to end, the timestamps moved back. The batched pipeline (`--batch-size` 2 or more) always filters, with the same settings. |
| whisper | The same, with the same detector and settings. |
| voxtral, parakeet, canary, transformers | Chunks cut at the edges of the speech, so no chunk holds the silence between two stretches of speech; a stretch longer than a chunk is cut in its quietest moment, as without `--vad`. Canary and Voxtral, which time each chunk as a whole, then give a segment per stretch of speech. |

- **No speech:** a recording without speech gives an empty transcript without running the model, and the command says
  `No speech found in …` on stderr.
- **The detector** is Silero VAD, the model faster-whisper ships, so it needs nothing more. It runs on the CPU, at
  about 10 seconds per hour of audio.
- **Decoding:** with `--vad`, the audio is decoded once with ffmpeg, as for speaker labels; whisper and faster-whisper
  otherwise decode files themselves.

**Tuning:** three [options](#backend-options), the same for every backend:

| Option | Default | Meaning |
|---|---|---|
| `vad_threshold` | `0.5` | How sure, from 0 to 1, the detector must be that a 32 ms frame is speech. Raise it to skip more noise; lower it if quiet speech goes missing. |
| `vad_min_silence` | `2.0` | Seconds of silence that end a stretch of speech; shorter pauses stay in it. Lower it, to `0.5` say, for shorter Canary and Voxtral segments. |
| `vad_padding` | `0.4` | Seconds of audio kept before and after each stretch of speech, so its first and last words stay whole. |

```bash
transcribe talk.mp3 -b whisper --vad
transcribe meeting.mp3 -b canary -l de --vad -O vad_min_silence=0.5 -O vad_threshold=0.6
```

In Python, `speech_toolkit.vad.detect_speech(samples)` gives the stretches of speech of decoded audio as `(start, end)`
pairs in seconds, and `speech_toolkit.media.split_audio(samples, 30, regions=...)` cuts chunks from them; see the
[API reference](reference/vad.md).

## Any Hugging Face model (`transformers`)

`-b transformers` runs most speech recognition models on the Hugging Face Hub through transformers' speech recognition
pipeline, with no code per model: Whisper fine-tunes and distil-whisper, CTC models such as wav2vec2, HuBERT and MMS,
and the other models the pipeline supports, such as Moonshine and Granite Speech. Install the
[`transformers` extra](install.md#extras) and give `-m` a repo id or a local model folder:

```bash
transcribe talk.mp3 -b transformers -m openai/whisper-small
transcribe talk.mp3 -b transformers -m facebook/wav2vec2-base-960h --word-timestamps -o talk.json
transcribe talk.mp3 -b transformers -m ./models/my-whisper-fine-tune -l de
```

In Python: `Transcriber("transformers", "openai/whisper-small")`. `transcribe --list-models -b transformers` lists a
few models that work well; without `-m`, the backend loads `openai/whisper-large-v3-turbo`.

**What a model can do** is read from it once it has loaded, from the kind of model the pipeline makes of it and, for
Whisper, from its generation config:

| Kind of model | For example | Timestamps | Word timestamps | `-l` | Detects the language | Translates |
|---|---|---|---|---|---|---|
| Whisper, multilingual | `openai/whisper-small`, `openai/whisper-large-v3-turbo`, fine-tunes | per segment | yes, if its generation config has `alignment_heads` (OpenAI's models do) | yes | yes, unless its generation config sets a language, as distil-whisper's English models do | yes (turbo isn't trained for it: a warning says so) |
| Whisper, English-only | `openai/whisper-small.en` | per segment | as above | no: English | no | no |
| CTC | `facebook/wav2vec2-base-960h`, HuBERT, MMS | per segment, made from the words | yes | no | no | no |
| Others: sequence-to-sequence models, speech LLMs | `moonshine-ai/moonshine-tiny`, Granite Speech | per ~30 s chunk | no | no | no | no |

- **Segments:** with word timestamps, a Whisper model's segments are made from its words, like a CTC model's: a segment
  ends after a sentence, before a pause of more than a second, or at 30 s.
- **Language:** a multilingual Whisper model takes a language code (`-l de`) or, as Whisper does, detects the
  language in the first 30 s and keeps it for the rest. Other models can't be told the language, so they ignore `-l`
  with a warning (an English-only Whisper model takes `-l en` silently).
- **Long recordings** are cut into chunks of up to 30 s, in pauses, and each chunk goes to the model whole. The
  pipeline's own `chunk_length_s`, whose merging of overlapping chunks is experimental for sequence-to-sequence models,
  isn't used.
- **Precision and devices** are as for the other transformers backends: `--compute-type` and `--device`, including
  `cuda:N` and `mps`.
- **Dedicated backends do more** for some models: `-b parakeet` gives Parakeet TDT's word timestamps, which the
  pipeline doesn't, and `-b canary` and `-b voxtral` know their models' prompts and token limits.
- **Not supported yet:** models that need `trust_remote_code`; MMS's language adapters (it loads its default,
  English); and models whose feature extractor expects another sampling rate than 16 kHz need `torchaudio` installed,
  which the pipeline uses to resample.

**When requests are checked.** Before a model loads, the backend's capabilities are what *some* models can do, so
`-t translate` and `--word-timestamps` pass the checks made then (the model name's shape, the device and the options
are checked as for every backend). Once the model has loaded, the backend's capabilities are that model's:

- A translation the model can't do stops with `UnsupportedOptionError` before any audio is decoded, as with any
  backend that can't translate.
- `--word-timestamps` for a model without them prints a warning and is skipped, as with any backend without them. So
  `--diarize` and `--max-line-width` don't ask such a model for word timings.
- The server rejects both requests with a 400 error.
- Called directly (`get_backend("transformers")`), the backend raises `UnsupportedOptionError` for a task, word
  timestamps or a language the model can't take, before it reads the audio.

## Adding a backend

1. **Subclass** `TranscriptionBackend` in `speech_toolkit/backends/` and implement `available_models()`, `load_model()`
   and `transcribe()`.
2. **Import heavy libraries lazily,** inside `load_model()`, so `--help` and `--list-backends` stay instant.
3. **Register** the backend in `speech_toolkit/backends/__init__.py`.
4. **Package** it: add its packages as an extra in `pyproject.toml`.
5. **List its packages** in the class's `requires`: their top-level import names, such as `("faster_whisper",)`.
   `--list-backends` looks them up without importing them, to say whether the backend is installed.
6. **Declare its capabilities** in the class's `capabilities` set:
   - `"translate"`: `task="translate"` translates to English. Without it, `Transcriber` raises `UnsupportedOptionError`
     for a translation, before any audio is decoded.
   - `"language_detection"`: with no language given, the backend detects it and reports it in `result.language`.
   - `"prompt"`, `"vad"` and `"word_timestamps"`: optional features, accepted as keyword-only arguments of
     `transcribe()`. `Transcriber` passes only the declared ones and warns about the rest.
   - For `"vad"`, [`speech_toolkit.vad`](reference/vad.md) has the shared detector: `detect_speech()` finds the
     speech, `split_audio(..., regions=...)` cuts chunks from it, `SpeechMap` joins it and moves the times back, and
     `VAD_OPTIONS` declares the tuning options. With `"waveform"` too, `Transcriber` decodes the audio for it.
7. **Use Whisper's format for word timings:** `segment["words"] = [{"word": " Hi", "start": 0.0, "end": 0.4}]`.
8. **Declare what it loads** (each is optional; without it, `load_model()` gets the model name and device as given):
   - `MODEL_SOURCES`: the custom models it takes besides `available_models()`, a dict from `"hub"` (a repo id),
     `"directory"` or `"file"` to the words messages use for them (`""` for the usual ones). `resolve_model()` checks
     a name against it and gives what to load: a model name's id (override `_model_id()`), or a custom one unchanged.
   - `DEVICES`: the device kinds it runs on, of `"cpu"`, `"cuda"` (also `cuda:N`) and `"mps"`. In `load_model()`,
     `self._parse_device(device)` gives a `Device` (`kind`, `index`, `auto`); `None` and `"auto"` mean "pick one".
   - `OPTIONS`: its settings, a dict from names to `BackendOption(type, default, help, choices=(), minimum=None)`,
     from `speech_toolkit.backends`. A `compute_type` option is what `--compute-type` sets. `load_model()` then takes
     them keyword-only, `load_model(self, model_name, device=None, *, options=None)`: `self._resolve_options(options)`
     checks the ones given and adds the defaults, and the backend keeps the result in `self._options`, which
     `self.options` shows.

   `Transcriber` checks the model name, device and options before anything loads, and passes `options` only when the
   user gave some.
9. **Backends on transformers** can subclass `_TransformersBackend` (`speech_toolkit/backends/_transformers.py`),
   which loads `MODEL_CLASS` and its `AutoProcessor` from a repo id or folder, with the device and precision chosen
   by `torch_device()` and `torch_dtype()`.

To use a backend without changing the package, register it at runtime with `speech_toolkit.register_backend()`; see
[Python API](python-api.md#custom-backends).

## Benchmarks

The benchmark harness in
[`benchmarks/`](https://github.com/MysterionRise/speech-transcription-toolkit/tree/main/benchmarks) measures each
backend and model on a fixed evaluation set: 100 LibriSpeech test-clean utterances (English, 12.5 minutes) and 30
FLEURS recordings in German, Spanish and French (6.3 minutes). It reports the word error rate (WER, after Whisper's
text normalizers), the speed as RTFx (seconds of audio per second of transcription: higher is faster) and the peak
memory. It runs from a checkout:

```bash
pip install -e ".[whisper]" -r requirements-bench.txt     # plus the extras of the backends to measure
python -m benchmarks.run -b faster-whisper -m tiny -m base -b whisper -m tiny
```

Each `-m` is a model of the `-b` before it, and each `-O KEY=VALUE` a [backend option](#backend-options) for its models,
so one run compares settings:

```bash
python -m benchmarks.run -b faster-whisper -m tiny -b faster-whisper -m tiny -O beam_size=1
```

The table goes to stdout and the results to a JSON file, which `python -m benchmarks.compare BEFORE.json AFTER.json`
compares. The first run downloads the evaluation set (about 40 MB)
and checks every clip against its SHA-256.

**On a GPU,** pass `--device cuda`; the table then names the GPU and shows its peak memory. For example:

```bash
python -m benchmarks.run --device cuda -b faster-whisper -m large-v3 -m turbo -b whisper -m turbo -b parakeet
```

A workflow runs the tiny models on the CPU every week and fails when a set's WER rises above the committed baseline by
more than one percentage point (three for FLEURS); speed and memory vary too much on shared runners to gate on. The
[harness's README](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/benchmarks/README.md)
describes the evaluation set and its licenses (CC BY 4.0), how each number is measured, the JSON format, the
tolerances, and how to update the baseline.

### Measured so far

WER per set (lower is better) and RTFx over all clips (higher is faster), with speech_toolkit 0.4.0. Each model ran in
a process of its own, with the clips' audio decoded beforehand and one warm-up transcription; peak RSS is that
process's.

| Machine | Backend | Model | Device | WER LibriSpeech | WER FLEURS (de, es, fr) | RTFx | Peak RSS (MiB) |
|---|---|---|---|--:|--:|--:|--:|
| GitHub's `ubuntu-latest` runner: Intel Xeon Platinum 8370C, 4 CPUs, 16 GB | faster-whisper 1.2.1 | tiny | CPU, int8 | 0.072 | 0.284 (0.273, 0.153, 0.392) | 12.7 | 356 |
| GitHub's `ubuntu-latest` runner: Intel Xeon Platinum 8370C, 4 CPUs, 16 GB | whisper (openai-whisper 20250625, torch 2.14.1) | tiny | CPU, float32 | 0.078 | 0.324 (0.286, 0.191, 0.458) | 18.3 | 727 |
| GitHub's `ubuntu-latest` runner: AMD EPYC 7763, 4 CPUs, 16 GB | faster-whisper 1.2.1 | tiny | CPU, int8 | 0.071 | 0.302 (0.253, 0.191, 0.425) | 12.8 | 346 |
| GitHub's `ubuntu-latest` runner: AMD EPYC 7763, 4 CPUs, 16 GB | whisper (openai-whisper 20250625, torch 2.14.1) | tiny | CPU, float32 | 0.078 | 0.324 (0.286, 0.191, 0.458) | 11.4 | 726 |
| A cloud VM: Intel Xeon @ 2.8 GHz, 4 CPUs, 16 GB | faster-whisper 1.2.1 | tiny | CPU, int8 | 0.072 | 0.284 (0.273, 0.153, 0.392) | 6.6 | 332 |

The Intel runner's rows are the workflow's baseline (`benchmarks/baseline.json`). faster-whisper decodes with beam
search (5 beams) and openai-whisper greedily: openai-whisper's `tiny` is the faster of the two on the Intel runner, and
faster-whisper's the more accurate. openai-whisper gets the same WER on every machine. faster-whisper's int8 kernels
compute a little differently on Intel and AMD CPUs, which moves its WER on FLEURS's hard clips.

No GPU numbers yet: run the command above on one and add a row, naming the GPU.
