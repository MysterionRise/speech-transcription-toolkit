# Backends

| Backend | Models (default in bold) | Notes |
|---|---|---|
| `faster-whisper` (default) | tiny … large-v3, distil-\*, **turbo** | Whisper's models, several times faster; runs on CTranslate2 instead of PyTorch |
| `whisper` | tiny … large-v3, **turbo** | OpenAI Whisper, on PyTorch |
| `voxtral` | **voxtral-mini**, voxtral-small | Mistral Voxtral (3B and 24B); transcription only, timestamps per ~30 s |
| `parakeet` | **parakeet-tdt-0.6b-v3**, parakeet-tdt-0.6b-v2 | NVIDIA Parakeet: fast and accurate, word timestamps. v3: English + 24 European languages, detected automatically; v2: English only |
| `canary` | **canary-1b-v2** | NVIDIA Canary: the same 25 languages; translates to English. Can't detect the language: give it with `-l` (default English). Timestamps per ~30 s |

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
| `vad`: `--vad` | no | yes | no | no | no |
| `word_timestamps`: `--word-timestamps` | yes | yes | no | yes | no |

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
| `mps` | An Apple Silicon GPU. | whisper, voxtral, parakeet, canary; not faster-whisper (CTranslate2 has no Apple GPU support) |

- A device the backend can't use, such as `-b faster-whisper --device mps`, stops before anything loads.
- [Speaker labels](diarization.md) run on the same device.
- An NVIDIA GPU needs NVIDIA's CUDA libraries, and on Windows a CUDA build of PyTorch: see
  [GPU setup](install.md#gpu).

**Precision:** `--compute-type` (`compute_type=` in Python) sets it; the default, `auto`, picks one that suits the
device:

| Backend | `auto` on an NVIDIA GPU | `auto` on an Apple GPU | `auto` on the CPU | `--compute-type` also takes |
|---|---|---|---|---|
| faster-whisper | float16, or the next of int8_float16, int8, float32 the GPU supports | (no Apple GPU) | int8, or float32 | CTranslate2's types: `int8`, `int8_float32`, `int8_float16`, `int8_bfloat16`, `int16`, `float16`, `bfloat16`, `float32`, and `default` (the type the model was saved in) |
| voxtral, parakeet, canary | bfloat16, or float16 on a GPU without bfloat16 | float16 | float32 | `float32`, `float16`, `bfloat16` |
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
| voxtral, parakeet, canary | `compute_type` | `auto` | The precision, as `--compute-type` sets it. |
| voxtral, canary | `max_new_tokens` | `500` | Most tokens of text per ~30 s chunk; see [Text limit per chunk](#text-limit-per-chunk). |

```bash
transcribe talk.mp3 -O cpu_threads=8
transcribe talk.mp3 -b canary -l de -O max_new_tokens=1000 --device cuda:1
```

- Values are converted to the option's type: numbers, `true`/`false`, or one of the listed choices.
- An option the backend doesn't have stops before anything loads, with a message that lists the ones it has; so does
  a value that doesn't fit.

## Text limit per chunk

Canary and Voxtral transcribe ~30 s chunks and write at most 500 tokens for each, plenty for normal speech. A chunk
that reaches the limit, from very fast speech or the model repeating itself, may be cut off, so they warn once per file
with the chunk's time range. Check the transcript there, raise the limit with `-O max_new_tokens=1000`, or transcribe
the file with another backend.

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

Each `-m` is a model of the `-b` before it. The table goes to stdout and the results to a JSON file, which
`python -m benchmarks.compare BEFORE.json AFTER.json` compares. The first run downloads the evaluation set (about 40 MB)
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
