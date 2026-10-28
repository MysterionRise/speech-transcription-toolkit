# Server

`create_app()` builds the FastAPI app behind `transcribe-server`, so you can serve a `Transcriber` from your own ASGI
server or test it with FastAPI's `TestClient`. It is in
[speech_toolkit/server.py](https://github.com/MysterionRise/speech-transcription-toolkit/blob/main/speech_toolkit/server.py)
and needs the `server` extra. [OpenAI-compatible server](../server.md) describes the endpoints.

::: speech_toolkit.server.create_app
