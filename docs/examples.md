# Examples

Short scripts that use the [Python API](python-api.md), in
[examples/](https://github.com/MysterionRise/speech-transcription-toolkit/tree/main/examples). Each one runs as it is,
and its docstring says how. They aren't installed with the package: after
`pip install speech-transcription-toolkit`, download the one you want, or run it from a clone of the repository.
The tests run every one of them, with stand-in models.

On the documentation site each script is shown below; on GitHub, follow the links.

| Script | What it does |
|---|---|
| [folder_to_srt.py](#a-folder-of-recordings-to-subtitles) | Subtitles for every recording in a folder, with the model loaded once. |
| [meeting_notes.py](#meeting-notes-with-speakers) | Meeting notes in Markdown: who said what, and when. |
| [openai_client.py](#openais-sdk-against-transcribe-server) | OpenAI's Python SDK, pointed at `transcribe-server`. |
| [custom_backend.py](#a-custom-backend) | A backend of your own, registered with `register_backend()`. |
| [rerender_json.py](#a-saved-transcript-as-subtitles) | A transcript saved as JSON, rendered again as srt and vtt. |

## A folder of recordings to subtitles

[examples/folder_to_srt.py](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/examples/folder_to_srt.py)
writes an `.srt` file for every audio or video file in a folder. It skips recordings that have their subtitles
already, and carries on past one that fails, like `transcribe recordings/ -f srt --skip-existing`
([Command line](cli.md#several-files-and-formats)).

```python
--8<-- "examples/folder_to_srt.py"
```

## Meeting notes with speakers

[examples/meeting_notes.py](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/examples/meeting_notes.py)
labels the speakers ([Speaker labels](diarization.md) explains the setup) and writes a Markdown paragraph for each
turn, with its time and the speaker's name.

```python
--8<-- "examples/meeting_notes.py"
```

## OpenAI's SDK against transcribe-server

[examples/openai_client.py](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/examples/openai_client.py)
sends a file to [`transcribe-server`](server.md) with OpenAI's Python SDK: code written for OpenAI's API changes only
its `base_url`.

```python
--8<-- "examples/openai_client.py"
```

## A custom backend

[examples/custom_backend.py](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/examples/custom_backend.py)
is a complete backend that needs no model: it marks where someone speaks, from the loudness. Replace its
`load_model()` and `transcribe()` to plug in your own model; [Custom backends](python-api.md#custom-backends) has the
rest of the contract.

```python
--8<-- "examples/custom_backend.py"
```

## A saved transcript as subtitles

[examples/rerender_json.py](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/examples/rerender_json.py)
loads a result saved as JSON with `TranscriptionResult.load()` and writes it as srt and vtt, without transcribing
again ([Loading a saved result](python-api.md#loading-a-saved-result)).

```python
--8<-- "examples/rerender_json.py"
```
