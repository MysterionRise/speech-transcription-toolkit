# Results and types

Every transcription returns a `TranscriptionResult`
([speech_toolkit/backends/base.py](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/speech_toolkit/backends/base.py)).
Its segments are dicts shaped like `Segment`, and a segment's words like `Word`
([speech_toolkit/types.py](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/speech_toolkit/types.py)).
[Output formats](../formats.md#json) shows the JSON a result is saved as; `speech_toolkit.types.SCHEMA_VERSION` is the
version of its layout, its `schema_version`.

::: speech_toolkit.TranscriptionResult

::: speech_toolkit.Segment
    options:
      inherited_members: true
      show_bases: false
      show_if_no_docstring: true

::: speech_toolkit.Word
    options:
      inherited_members: true
      show_bases: false
      show_if_no_docstring: true
