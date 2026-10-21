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

`transcriber.transcribe(audio, language=None, task="transcribe", *, ...)` transcribes one recording: a file, its
content, a file object or the samples, see [Audio input](#audio-input). Keyword options:

| Option | Meaning |
|---|---|
| `prompt` | Names, terms or a sample sentence that guide spelling. |
| `vad` | Skip silence first. |
| `word_timestamps` | Add per-word timings. Defaults to on when diarizing, so the speaker can change mid-segment. |
| `num_speakers`, `min_speakers`, `max_speakers` | Speaker-count hints; need `diarize=True`. |
| `verbose` | Show the backend's progress on stderr. |

Options the backend doesn't support are skipped with a warning. A translation the backend can't do raises
`UnsupportedOptionError` before any audio is decoded; the one-shot `transcribe()` raises it before loading the model.
`transcriber.supports("vad")` tells you in advance; it also takes `"prompt"`, `"word_timestamps"`, `"translate"`,
`"language_detection"` and `"waveform"`, see [Backends](backends.md#what-each-backend-supports) and
[Custom backends](#custom-backends).

`transcribe(audio, backend=..., model=..., **options)` loads a model and transcribes one file in a single call. To
transcribe several files, create a `Transcriber` once instead.

The Hugging Face token, from `hf_token`, `HF_TOKEN` or `HUGGINGFACE_TOKEN`, is exported as `HF_TOKEN` for model downloads.

### Audio input

`audio` can be any of these:

| Input | For example |
|---|---|
| A path to anything ffmpeg decodes, as a `str` or `pathlib.Path` | `"talk.mp3"` |
| The file's content: `bytes`, `bytearray` or `memoryview` | an upload's body |
| A binary file object, read from its current position to the end and left open | `open("talk.ogg", "rb")`, `sys.stdin.buffer`, `io.BytesIO(data)` |
| The samples: a 1-D `float32` NumPy array of mono audio at 16 kHz | what `load_audio()` returns |

```python
import numpy as np
from speech_toolkit.media import load_audio

transcriber.transcribe(data)                                   # bytes
with open("talk.ogg", "rb") as file:
    transcriber.transcribe(file)
transcriber.transcribe(load_audio("talk.mp3"))                 # samples
transcriber.transcribe(int16_samples.astype(np.float32) / 32768)
```

- **Arrays** must be 1-D `float32` samples at 16 kHz. Other arrays raise `ValueError` instead of being converted,
  because their scale and sample rate can't be told from the data: convert `int16` samples as above, average stereo
  channels, resample other rates, or pass the file instead. Other types raise `TypeError`.
- **Decoded once:** in-memory input is decoded with ffmpeg once, and the backend and speaker labelling share the
  samples. A path is decoded once too: by the backend itself or, with `diarize=True` and a backend that takes samples
  (all the built-in ones do), by the `Transcriber` for both.
- **MP4, M4A and MOV in memory:** ffmpeg reads in-memory input through a pipe, which can't seek, so it can't decode
  these when their index comes after the audio, as phones record them. `transcribe()` checks where the index is (in a
  file object, within its first 64 KiB): such a file is copied to a temporary file, decoded from there and deleted. A
  file with its index first streams through the pipe.
- **Untrusted input:** `transcribe()` decodes with `load_audio()`'s defaults. For uploads, decode with
  `load_audio(data, strict=True, max_seconds=..., timeout=...)` first and pass the samples; see
  [Decoding audio](#decoding-audio).

## Results

`TranscriptionResult` has:
- `text`: the full transcript.
- `segments`: dicts with `id` (counting from 0), `start` and `end` (seconds) and `text`.
  - With word timestamps on, each also has `words`: a list of `{"word", "start", "end"}` dicts.
  - Some backends add their own fields.
- `language`: the language code, detected or given.
- `duration`: the audio's length in seconds. The faster-whisper, Parakeet, Canary and Voxtral backends report it, and
  otherwise it comes from the decoded samples: for in-memory input, and with diarization. It is `None` only when a
  backend that doesn't report it, such as whisper, transcribes a path without diarization.
- `language_probability`: how likely the detected language is, from 0 to 1, if the backend reports it (faster-whisper
  does); otherwise `None`.
- `speaker_segments`: with diarization, the segments with a `speaker` label each, and a `segment_id`: the `id` of
  the segment in `segments` each one comes from. Otherwise `None`.
- `raw`: other backend output. faster-whisper also keeps `duration` and `language_probability` here, as before.
- `render(fmt="txt", max_line_width=None)`: the result as txt, srt, vtt or json text.
- `save(path, fmt=None, max_line_width=None)`: writes the rendered result, taking the format from the extension.
- `to_dict()`: the result as a dict, as the JSON output has it.

See [Output formats](formats.md) for what each format contains.

### Loading a saved result

A result saved as JSON, by `result.save("talk.json")` or `transcribe --json`, loads back, so you can render it in
another format without transcribing again:

```python
from speech_toolkit import TranscriptionResult

result = TranscriptionResult.load("talk.json")
result.save("talk.srt", max_line_width=42)
```

- `TranscriptionResult.from_dict(data)` does the same for a dict, such as `to_dict()`'s or `json.load()`'s:
  `from_dict(result.to_dict())` gives back an equal result, speaker segments included.
- Keys that aren't result fields go to `raw`, so backend-specific fields come back too. Keys of `raw` that repeat a
  result field aren't written, so they don't come back: whisper's `raw`, its whole output, comes back empty, and
  faster-whisper's `duration` and `language_probability` come back as the attributes only.
- JSON saved by earlier versions, without `schema_version`, loads too.
- Data that isn't a transcription result raises `ValueError`, with a message that says what is wrong and where. So
  does a newer `schema_version` than this version reads: upgrade to load the file. A missing file raises
  `FileNotFoundError`.

### Types

`Segment` and `Word` are `TypedDict`s that describe a segment and a word for type checkers; at run time they are plain
dicts. A backend can build its segments with them:

```python
from typing import List

from speech_toolkit import Segment, TranscriptionResult

segments: List[Segment] = [{"id": 0, "start": 0.0, "end": 1.5, "text": " Hello."}]
result = TranscriptionResult(text=" Hello.", segments=segments)
```

`result.segments` is typed `List[Dict[str, Any]]` for now, so code that reads backend-specific keys keeps
type-checking.

## Errors and warnings

The library raises exceptions and never exits. The errors below are all `SpeechToolkitError`s, and each is also the
built-in error the library raised before, so `except ValueError` and the like keep working:

| Error | Also a | Raised when |
|---|---|---|
| `BackendNotFoundError` | `ValueError` | No backend has that name. |
| `ModelNotFoundError` | `ValueError` | The backend has no model of that name. |
| `UnsupportedOptionError` | `ValueError` | The request can't be honoured: translating with a backend that only transcribes, a `task` other than `transcribe` or `translate`, or speaker hints without `diarize=True`. |
| `BackendUnavailableError` | `ImportError` | An optional package is missing, for the backend or for diarization; the message names the extra to install. |
| `ModelLoadError` | `RuntimeError` | The model can't be downloaded or loaded. |
| `AudioDecodeError` | `RuntimeError` | The audio can't be decoded. |
| `DiarizationError` | `RuntimeError` | The diarization model can't be downloaded: accept its terms and set a token, see [Speaker labels](diarization.md). |

A missing input file raises the built-in `FileNotFoundError`, and the `whisper` backend passes on openai-whisper's own
download and loading errors. Audio of a type `transcribe()` can't take raises the built-in `TypeError`, and an array
that isn't 1-D `float32` raises `ValueError`, see [Audio input](#audio-input). Loading a saved result that isn't one raises the built-in `ValueError`; see
[Loading a saved result](#loading-a-saved-result).

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
  decodes only from a path. `Transcriber.transcribe()` takes care of that itself, see [Audio input](#audio-input).
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

`Transcriber` passes the audio as the first positional argument of `transcribe()`: a `pathlib.Path` to the file, or,
for in-memory input, a temporary 16-bit WAV file that is deleted when `transcribe()` returns.

**Samples instead of a file:** declare the `waveform` capability, and `transcribe()` may get the samples instead of
the path: a 1-D `float32` NumPy array of mono audio at 16 kHz. `Transcriber` passes them whenever it has decoded the
audio, for in-memory input and for diarization, so nothing is decoded twice and no temporary file is written.

```python
from pathlib import Path

from speech_toolkit.media import load_audio

class MyBackend(TranscriptionBackend):
    capabilities = frozenset({"waveform"})

    def transcribe(self, audio_path, language=None, task="transcribe", verbose=True):
        samples = load_audio(audio_path) if isinstance(audio_path, Path) else audio_path
        ...
```

See [Adding a backend](backends.md#adding-a-backend) for the details.
