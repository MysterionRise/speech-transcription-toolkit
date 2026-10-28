# API reference

Every class and function here is described by its docstring. The
[documentation site](https://mysterionrise.github.io/speech-transcription-toolkit/reference/) builds these pages from
the docstrings; on GitHub, each page links to the source file that holds them. For a guided tour, with examples, read
[Python API](../python-api.md) first.

| Page | Covers | Import from |
|---|---|---|
| [Transcriber](transcriber.md) | `Transcriber`, `transcribe()` | `speech_toolkit` |
| [Results and types](results.md) | `TranscriptionResult`, `Segment`, `Word` | `speech_toolkit` |
| [Errors and warnings](errors.md) | `SpeechToolkitError` and its subclasses, `SpeechToolkitWarning` | `speech_toolkit` |
| [Backends](backends.md) | `list_backends()`, `register_backend()`, `TranscriptionBackend` | `speech_toolkit` |
| | `get_backend()` | `speech_toolkit.backends` |
| [Decoding audio](media.md) | `load_audio()` | `speech_toolkit.media` |
| [Server](server.md) | `create_app()` | `speech_toolkit.server` |
