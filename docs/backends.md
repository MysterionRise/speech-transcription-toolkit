# Backends

| Backend | Models (default in bold) | Notes |
|---|---|---|
| `whisper` | tiny … large-v3, **turbo** | OpenAI Whisper |
| `faster-whisper` | tiny … large-v3, distil-\*, **turbo** | The same models, several times faster; runs on CTranslate2 instead of PyTorch |
| `voxtral` | **voxtral-mini**, voxtral-small | Mistral Voxtral (3B and 24B); transcription only, timestamps per ~30 s |
| `parakeet` | **parakeet-tdt-0.6b-v3**, parakeet-tdt-0.6b-v2 | NVIDIA Parakeet: fast and accurate, word timestamps. v3: English + 24 European languages, detected automatically; v2: English only |
| `canary` | **canary-1b-v2** | NVIDIA Canary: the same 25 languages; translates to English. Can't detect the language: give it with `-l` (default English). Timestamps per ~30 s |

`transcribe --list-backends` lists them with their models and capabilities, and says whether each one's packages are
installed. `transcribe --list-models -b <backend>` lists one backend's models. Each backend except `whisper` needs its
[extra](install.md#extras).

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

The GPU is used when available; force a device with `--device cpu` or `--device cuda`.

| Backend | GPU (CUDA) | CPU |
|---|---|---|
| whisper | fp16 | fp32 |
| faster-whisper | float16 | int8 |
| voxtral, parakeet, canary | bfloat16, or float16 where bfloat16 isn't supported | float32 |

Out of memory? Use a smaller model (`-m small`) or `-b faster-whisper`.

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

To use a backend without changing the package, register it at runtime with `speech_toolkit.register_backend()`; see
[Python API](python-api.md#custom-backends).
