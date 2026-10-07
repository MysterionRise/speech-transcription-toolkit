# Python API

```python
from speech_toolkit import Transcriber, transcribe

result = transcribe("talk.mp3", backend="faster-whisper", model="small")
print(result.text)        # result.segments has start, end and text for each segment
result.save("talk.srt")   # txt, srt, vtt or json, from the extension

transcriber = Transcriber(backend="faster-whisper", model="small")  # load once, reuse
for path in ["a.mp3", "b.mp3"]:
    transcriber.transcribe(path, language="en").save(f"{path}.vtt")
```

## Transcriber

`Transcriber(backend="whisper", model=None, *, device=None, diarize=False, hf_token=None)` loads the model once.
With `diarize=True` it also loads the pyannote pipeline, and does that first, so a missing package or token fails
before a large model download.

`transcriber.transcribe(audio, language=None, task="transcribe", *, ...)` transcribes one file. Keyword options:

| Option | Meaning |
|---|---|
| `prompt` | Names, terms or a sample sentence that guide spelling. |
| `vad` | Skip silence first. |
| `word_timestamps` | Add per-word timings. Defaults to on when diarizing, so the speaker can change mid-segment. |
| `num_speakers`, `min_speakers`, `max_speakers` | Speaker-count hints; need `diarize=True`. |
| `verbose` | Show the backend's progress on stderr. |

Options the backend doesn't support are skipped with a warning. `transcriber.supports("vad")` (or `"prompt"`,
`"word_timestamps"`) tells you in advance.

`transcribe(audio, backend=..., model=..., **options)` loads a model and transcribes one file in a single call. To
transcribe several files, create a `Transcriber` once instead.

The Hugging Face token, from `hf_token`, `HF_TOKEN` or `HUGGINGFACE_TOKEN`, is exported as `HF_TOKEN` for model downloads.

## Results

`TranscriptionResult` has:
- `text`: the full transcript.
- `segments`: dicts with `start`, `end` (seconds) and `text`.
  - With word timestamps on, each also has `words`: a list of `{"word", "start", "end"}` dicts.
  - Some backends add their own fields.
- `language`: the language code, detected or given.
- `speaker_segments`: with diarization, the segments with a `speaker` label each; otherwise `None`.
- `raw`: extra backend output, for example faster-whisper's `duration` and `language_probability`.
- `render(fmt="txt", max_line_width=None)`: the result as txt, srt, vtt or json text.
- `save(path, fmt=None, max_line_width=None)`: writes the rendered result, taking the format from the extension.

See [Output formats](formats.md) for what each format contains.

## Errors and warnings

The library raises exceptions and never exits. The errors below are all `SpeechToolkitError`s, and each is also the
built-in error the library raised before, so `except ValueError` and the like keep working:

| Error | Also a | Raised when |
|---|---|---|
| `BackendNotFoundError` | `ValueError` | No backend has that name. |
| `ModelNotFoundError` | `ValueError` | The backend has no model of that name. |
| `UnsupportedOptionError` | `ValueError` | The request can't be honoured: translating with a backend that only transcribes, or speaker hints without `diarize=True`. |
| `BackendUnavailableError` | `ImportError` | An optional package is missing, for the backend or for diarization; the message names the extra to install. |
| `ModelLoadError` | `RuntimeError` | The model can't be downloaded or loaded. |
| `AudioDecodeError` | `RuntimeError` | The audio can't be decoded. |
| `DiarizationError` | `RuntimeError` | The diarization model can't be downloaded: accept its terms and set a token, see [Speaker labels](diarization.md). |

A missing input file raises the built-in `FileNotFoundError`, and the `whisper` backend passes on openai-whisper's own
download and loading errors.

```python
from speech_toolkit import ModelNotFoundError, SpeechToolkitError, transcribe

try:
    result = transcribe("talk.mp3", model="huge")
except ModelNotFoundError as e:
    print(e)  # lists the backend's models
except SpeechToolkitError as e:
    print(f"transcription failed: {e}")
```

Warnings, such as an option the backend doesn't support, are `SpeechToolkitWarning`s, a kind of `UserWarning`. To hide
them:

```python
import warnings

from speech_toolkit import SpeechToolkitWarning

warnings.filterwarnings("ignore", category=SpeechToolkitWarning)
```

## Decoding audio

`load_audio()` decodes audio or video with ffmpeg into a mono float32 NumPy array at 16 kHz, the input speech models
take.

```python
from speech_toolkit.media import load_audio

audio = load_audio("talk.mp3")                   # a path
audio = load_audio(data, strict=True)            # bytes, from an untrusted upload say
with open("talk.ogg", "rb") as file:
    audio = load_audio(file, max_seconds=3600)   # a binary file object, streamed to ffmpeg
```

| Option | Meaning |
|---|---|
| `sr` | Sample rate (default 16000). |
| `max_seconds` | Decode only the first N seconds. |
| `timeout` | Stop ffmpeg after N seconds and raise `AudioDecodeError`. |
| `strict` | For untrusted input: accept only common audio and video formats (listed in `speech_toolkit.media.STRICT_FORMATS`), and refuse playlists (HLS, DASH), concat lists and anything else that would make ffmpeg open more files. |

- **Bytes and file objects** reach ffmpeg through a pipe; a file object is read in chunks from its current position,
  never whole. A pipe can't seek, so an MP4, M4A or MOV file with its index at the end, as phones often record them,
  decodes only from a path.
- **Safety:** ffmpeg opens local files only, never URLs, and runs without your environment's secrets, such as
  `HF_TOKEN`.
- **Errors:** failures raise `AudioDecodeError`, whose message ends with ffmpeg's reason, without file paths.
- **Memory:** decoding peaks at about 6 bytes per sample, some 1 GB for three hours of audio.

## Custom backends

Subclass `TranscriptionBackend`, implement `available_models()`, `load_model()` and `transcribe()`, then register it:

```python
from speech_toolkit import TranscriptionBackend, register_backend

class MyBackend(TranscriptionBackend):
    ...

register_backend("mine", MyBackend)
```

See [Adding a backend](backends.md#adding-a-backend) for the details.
