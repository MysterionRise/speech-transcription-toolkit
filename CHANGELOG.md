# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

Entries come from the fragments in [`changelog.d/`](https://github.com/MysterionRise/speech-transcription-toolkit/tree/main/changelog.d),
collected with towncrier when a release is made.

<!-- towncrier release notes start -->

## [0.4.0](https://github.com/MysterionRise/speech-transcription-toolkit/releases/tag/v0.4.0) - 2026-10-22

### Security

- ffmpeg, which decodes audio for the Parakeet, Canary and Voxtral backends and for speaker labels, now opens only local files, never URLs or other protocols, and runs without your environment's secrets such as `HF_TOKEN`. For untrusted input, `speech_toolkit.media.load_audio()` takes `strict=True`, which accepts only common audio and video formats and refuses playlists and concat lists, as well as a `timeout` and a `max_seconds` length cap. ([#18](https://github.com/MysterionRise/speech-transcription-toolkit/issues/18))
- `transcribe-server` now checks the API key and the upload size before it reads an upload, so unauthenticated or oversized uploads can no longer fill the disk, and it refuses playlists and manifests (HLS, DASH, ffconcat), which made ffmpeg open other files and URLs. Error responses no longer include temporary paths or ffmpeg's output: undecodable audio gets a 400 on every backend, and other failures a generic 500 whose details go to the server's log. ([#21](https://github.com/MysterionRise/speech-transcription-toolkit/issues/21))

### Removed

- The pydub and audioop-lts dependencies: `ogg2wav` now decodes with ffmpeg through `load_audio()` and writes WAV files with Python's `wave` module, so importing it no longer warns when ffmpeg is missing or on Python 3.12+. ([#22](https://github.com/MysterionRise/speech-transcription-toolkit/issues/22))

### Added

- Exceptions that tell failures apart: `BackendNotFoundError`, `BackendUnavailableError`, `ModelNotFoundError`, `ModelLoadError`, `AudioDecodeError`, `UnsupportedOptionError` and `DiarizationError`, all subclasses of `SpeechToolkitError` and of the built-in error raised before, so existing `except ValueError` (or `ImportError`, `RuntimeError`) code keeps working; library warnings are now `SpeechToolkitWarning`s. In batch mode, `transcribe` reports a file that fails with an unexpected error and carries on with the others instead of stopping. ([#16](https://github.com/MysterionRise/speech-transcription-toolkit/issues/16))
- Translating with a backend that can't (Voxtral, Parakeet) now fails at once with a clear error, before the model loads, and the server's `/v1/audio/translations` answers 400 without running the model. `transcribe --list-backends` also shows what each backend supports and whether its packages are installed; custom backends declare translation with the new `translate` capability. ([#17](https://github.com/MysterionRise/speech-transcription-toolkit/issues/17))
- `speech_toolkit.media.load_audio()` also decodes `bytes` and binary file objects, such as an upload; a file object is streamed to ffmpeg in chunks rather than read into memory whole. ([#18](https://github.com/MysterionRise/speech-transcription-toolkit/issues/18))
- `speech_toolkit.media.load_audio()` takes `channels`: the default, 1, mixes down to mono as before, and `channels=2` keeps stereo, as a `(samples, 2)` array. ([#22](https://github.com/MysterionRise/speech-transcription-toolkit/issues/22))
- `Transcriber.transcribe()` and `transcribe()` also take audio in memory: the file's content as `bytes`, a binary file object such as an upload or `sys.stdin.buffer`, or the samples as a 1-D float32 NumPy array at 16 kHz, M4A files recorded by phones included. With speaker labels, each recording is now decoded once instead of twice, and `result.duration` is set whenever the audio was decoded and by the Parakeet, Canary and Voxtral backends; custom backends can declare the new `waveform` capability to get the decoded samples instead of a file. ([#23](https://github.com/MysterionRise/speech-transcription-toolkit/issues/23))
- Typed results: `Segment` and `Word` TypedDicts, exported from `speech_toolkit`; `TranscriptionResult.duration` and `language_probability` (reported by faster-whisper); and `TranscriptionResult.load(path)` and `from_dict()`, which read a saved JSON result back so it can be rendered in another format. JSON output now starts with a `schema_version` and always has `duration` and `language_probability` (`null` when the backend doesn't report them). ([#24](https://github.com/MysterionRise/speech-transcription-toolkit/issues/24))

### Changed

- The reference documentation moved from the README into topic pages under `docs/`. These pages also cover every `transcribe` option and the server's endpoints and form fields. ([#15](https://github.com/MysterionRise/speech-transcription-toolkit/issues/15))
- **Breaking:** with an API key set (`--api-key` or `TRANSCRIBE_API_KEY`), `transcribe-server` no longer serves `/docs`, `/redoc` and `/openapi.json`, and every path except `/health` needs the key. ([#21](https://github.com/MysterionRise/speech-transcription-toolkit/issues/21))
- **Breaking:** faster-whisper is now the default backend and comes with every install, so `pip install speech-transcription-toolkit` no longer installs PyTorch (several GB). OpenAI's Whisper moved to the `whisper` extra: to keep using it, `pip install "speech-transcription-toolkit[whisper]"` and pass `-b whisper` (or `backend="whisper"`). The model names are the same and the default is still `turbo`. The `voxtral` and `nvidia` extras now install PyTorch themselves, and the `faster-whisper` extra still works but adds nothing. ([#22](https://github.com/MysterionRise/speech-transcription-toolkit/issues/22))
- faster-whisper segment ids now start at 0, like every other backend's; they started at 1. ([#24](https://github.com/MysterionRise/speech-transcription-toolkit/issues/24))

### Fixed

- `ogg2wav` no longer crashes on Windows when its output is redirected: the lines it prints to stdout are now plain ASCII. ([#14](https://github.com/MysterionRise/speech-transcription-toolkit/issues/14))
- Parakeet no longer reports the language you pass with `-l` as the detected one. It detects the language itself, so it now ignores `-l` with a warning and leaves the result's language empty. ([#17](https://github.com/MysterionRise/speech-transcription-toolkit/issues/17))
- With the Parakeet, Canary and Voxtral backends and with speaker labels, a file whose name ffmpeg took for a URL, such as `10:30.mp3`, now decodes, and a file ffmpeg can't get any audio from fails with an error instead of giving an empty transcript. Decoding also needs less memory: about 1 GB for three hours of audio instead of 1.7 GB. ([#18](https://github.com/MysterionRise/speech-transcription-toolkit/issues/18))
- Subtitles are valid in more cases: WebVTT escapes `&`, `<` and `>` in the text and in speaker names (so `-->` in the text no longer breaks a cue), and srt and vtt cues no longer overlap or have zero length. `--max-line-width` now counts the srt speaker label, counts Chinese, Japanese and Korean characters as two columns, and wraps Chinese and Japanese text without adding spaces. Diarized txt output no longer crashes on a segment without a speaker, and an unknown output extension such as `.tsv` now warns before writing plain text. ([#19](https://github.com/MysterionRise/speech-transcription-toolkit/issues/19))
- `ogg2wav` no longer overwrites one output with another when two inputs differ only in their extension, such as `talk.ogg` and `talk.opus`: they become `talk.ogg.wav` and `talk.opus.wav`, as `transcribe` names its outputs. It prints only the path of each WAV it writes to stdout, and skipped files go to stderr with the errors. faster-whisper now installs on Python 3.10 too: the PyAV version it was pinned to needs Python 3.11. ([#22](https://github.com/MysterionRise/speech-transcription-toolkit/issues/22))
- Speaker labels are cleaner: one word given to the wrong speaker no longer splits a line in three, a word in a pause goes to the nearer speaker, and merging speakers into a long transcript is much faster. In `speaker_segments`, each segment now has a unique `id` and a `segment_id`, the `id` of the transcript segment it comes from (use it to match the two); pieces of a split segment drop `tokens` and Whisper's other decoding details. Speaker hints that contradict each other, such as `--min-speakers 3 --max-speakers 2`, now fail before any model loads, and audio shorter than half a second isn't diarized. ([#26](https://github.com/MysterionRise/speech-transcription-toolkit/issues/26))
- Parakeet no longer glues two words into one ("Helloworld"), and no longer ends a segment at an abbreviation, initials or a decimal point ("Dr. Smith", "J. R. R. Tolkien", "3.5"). Canary and Voxtral now warn, naming the time range, when a ~30 s chunk reaches their limit of 500 tokens and its text may be cut off. ([#27](https://github.com/MysterionRise/speech-transcription-toolkit/issues/27))
- On a computer with an NVIDIA GPU but without NVIDIA's CUDA 12 libraries (cuBLAS, plus cuDNN for CTranslate2 older than 4.6.3), faster-whisper no longer fails, or on Windows crashes, on the first transcription. It now warns, naming the missing libraries and how to install them, and runs on the CPU. With `--device cuda` it stops with that message instead. The [installation guide](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/docs/install.md#gpu) has a new GPU section for Windows and Linux. ([#56](https://github.com/MysterionRise/speech-transcription-toolkit/issues/56))


## Before 0.4.0

Versions 0.1.0 to 0.3.0 were never published to PyPI. They added:

- **0.3.0:** `transcribe-server`, a local OpenAI-compatible speech-to-text API
  ([#11](https://github.com/MysterionRise/speech-transcription-toolkit/pull/11)).
- **0.2.0** ([#10](https://github.com/MysterionRise/speech-transcription-toolkit/pull/10)):
  - the `--prompt`, `--vad`, `--word-timestamps` and `--max-line-width` options;
  - word-level speaker labels;
  - the NVIDIA Parakeet and Canary backends.
- **0.1.0:** an installable package with a Python API (`Transcriber`, `transcribe`) and the `transcribe` and `ogg2wav`
  commands ([#8](https://github.com/MysterionRise/speech-transcription-toolkit/pull/8)).
- **Earlier work:**
  - pluggable backends, with Voxtral and faster-whisper;
  - SRT, WebVTT and JSON output;
  - batch mode;
  - speaker diarization with pyannote.audio
    ([#3](https://github.com/MysterionRise/speech-transcription-toolkit/pull/3)).
