"""Tests for the OpenAI-compatible server (speech_toolkit.server) with fake backends."""

from __future__ import annotations

import io
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from speech_toolkit import Transcriber
from speech_toolkit.server import INSTALL_HINT, _is_loopback, create_app, main, parse_args
from tests.conftest import FakeWordsBackend

URL = "/v1/audio/transcriptions"


def upload(name: str = "talk.wav", data: bytes = b"RIFF fake audio") -> dict:
    return {"file": (name, io.BytesIO(data), "audio/wav")}


@pytest.fixture
def temp_dir(tmp_path, monkeypatch):
    """Send the server's temporary uploads to a folder the test can inspect."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


@pytest.fixture
def client(fake_backend, temp_dir):
    return TestClient(create_app(Transcriber("fake-words")))


class TestResponseFormats:
    def test_json_is_the_default(self, client, temp_dir):
        response = client.post(URL, files=upload(), data={"model": "whisper-1"})

        assert response.status_code == 200
        assert list(response.json()) == ["text"]
        assert response.json()["text"].startswith("Hello from ")
        assert list(temp_dir.iterdir()) == []  # the upload is deleted afterwards

    @pytest.mark.parametrize("fmt, start", [("text", "Hello from "), ("srt", "1\n00:00:00,000 --> 00:00:01,500\n")])
    def test_plain_text_formats(self, client, fmt, start):
        response = client.post(URL, files=upload(), data={"response_format": fmt})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert response.text.startswith(start)

    def test_vtt(self, client):
        assert client.post(URL, files=upload(), data={"response_format": "vtt"}).text.startswith("WEBVTT\n")

    def test_verbose_json_has_openai_segment_fields(self, client):
        body = client.post(URL, files=upload(), data={"response_format": "verbose_json"}).json()

        assert (body["task"], body["language"], body["duration"]) == ("transcribe", "en", 1.5)
        assert set(body["segments"][0]) == {
            "id",
            "seek",
            "start",
            "end",
            "text",
            "tokens",
            "temperature",
            "avg_logprob",
            "compression_ratio",
            "no_speech_prob",
        }
        assert "words" not in body

    def test_word_timestamps(self, client):
        data = {"response_format": "verbose_json", "timestamp_granularities[]": ["word", "segment"]}
        body = client.post(URL, files=upload(), data=data).json()

        assert FakeWordsBackend.calls == [{"word_timestamps": True}]
        assert [w["word"] for w in body["words"]][:2] == ["Hello", "from"]
        assert len(body["words"]) == 3
        assert body["words"][0] == {"word": "Hello", "start": 0.0, "end": 0.5}

    def test_prompt_and_language_reach_the_backend(self, client):
        client.post(URL, files=upload(), data={"prompt": "Grafana", "language": "de"})
        assert FakeWordsBackend.calls == [{"prompt": "Grafana"}]

    def test_diarized_verbose_json_has_speakers(self, fake_backend, temp_dir, monkeypatch):
        monkeypatch.setattr("speech_toolkit.api.load_diarization_pipeline", lambda device: "pipeline")
        monkeypatch.setattr(
            "speech_toolkit.api.diarize_audio",
            lambda path, pipeline, **hints: [(0.0, 0.9, "SPEAKER_00"), (0.9, 1.5, "SPEAKER_01")],
        )
        client = TestClient(create_app(Transcriber("fake-words", diarize=True)))

        body = client.post(URL, files=upload(), data={"response_format": "verbose_json"}).json()

        assert [(s["id"], s["speaker"], s["text"]) for s in body["segments"]][0] == (0, "SPEAKER_00", " Hello from")
        assert body["segments"][1]["speaker"] == "SPEAKER_01"

    def test_translations(self, client):
        response = client.post("/v1/audio/translations", files=upload(), data={"model": "whisper-1"})

        assert response.status_code == 200
        assert FakeWordsBackend.tasks == ["translate"]


class TestErrors:
    @pytest.mark.parametrize(
        "data, param",
        [
            ({"stream": "true"}, "stream"),
            ({"response_format": "docx"}, "response_format"),
            ({"timestamp_granularities[]": "char"}, "timestamp_granularities[]"),
            ({"timestamp_granularities[]": "word"}, "timestamp_granularities[]"),  # needs verbose_json
        ],
    )
    def test_bad_fields_are_400s(self, client, data, param):
        response = client.post(URL, files=upload(), data=data)

        assert response.status_code == 400
        assert response.json()["error"]["param"] == param
        assert response.json()["error"]["type"] == "invalid_request_error"

    def test_missing_file(self, client):
        response = client.post(URL, data={"model": "whisper-1"})

        assert response.status_code == 400
        assert response.json()["error"] == {
            "message": "Field required: file",
            "type": "invalid_request_error",
            "param": "file",
            "code": None,
        }

    def test_word_timestamps_need_a_capable_backend(self, fake_backend, temp_dir):
        client = TestClient(create_app(Transcriber("fake")))
        data = {"response_format": "verbose_json", "timestamp_granularities[]": "word"}

        response = client.post(URL, files=upload(), data=data)

        assert response.status_code == 400
        assert "doesn't give word timestamps" in response.json()["error"]["message"]

    def test_backend_failure_is_a_500(self, fake_backend, temp_dir, caplog):
        client = TestClient(create_app(Transcriber("fake")))
        transcriber_error = RuntimeError("CUDA out of memory")

        with patch.object(Transcriber, "transcribe", side_effect=transcriber_error):
            response = client.post(URL, files=upload())

        assert response.status_code == 500
        assert response.json()["error"]["type"] == "server_error"
        assert "CUDA" not in response.json()["error"]["message"]  # the details are in the server's log
        assert "CUDA out of memory" in caplog.text
        assert list(temp_dir.iterdir()) == []

    def test_value_error_is_a_400(self, client):
        with patch.object(Transcriber, "transcribe", side_effect=ValueError("Unsupported language: xx")):
            response = client.post(URL, files=upload(), data={"language": "xx"})

        assert response.status_code == 400
        assert response.json()["error"]["message"] == "Unsupported language: xx"

    def test_upload_too_large(self, fake_backend, temp_dir):
        client = TestClient(create_app(Transcriber("fake-words"), max_upload_mb=0.0001))  # about 100 bytes

        response = client.post(URL, files=upload(data=b"x" * 1000))

        assert response.status_code == 413
        assert response.json()["error"]["param"] == "file"
        assert FakeWordsBackend.calls == []
        assert list(temp_dir.iterdir()) == []

    def test_unknown_route(self, client):
        response = client.get("/v1/nope")
        assert response.status_code == 404
        assert response.json()["error"]["message"] == "Not Found"


class TestAuthAndInfo:
    @pytest.fixture
    def secured(self, fake_backend, temp_dir):
        return TestClient(create_app(Transcriber("fake-words"), api_key="s3cret"))

    @pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic s3cret"}])
    def test_missing_or_wrong_key(self, secured, headers):
        response = secured.post(URL, files=upload(), headers=headers)

        assert response.status_code == 401
        assert response.json()["error"]["code"] == "invalid_api_key"
        assert secured.get("/v1/models", headers=headers).status_code == 401

    def test_right_key(self, secured):
        response = secured.post(URL, files=upload(), headers={"Authorization": "Bearer s3cret"})
        assert response.status_code == 200

    def test_health_needs_no_key(self, secured):
        assert secured.get("/health").json() == {"status": "ok", "backend": "fake-words", "model": "small"}

    def test_models_lists_the_loaded_model(self, client):
        data = client.get("/v1/models").json()["data"]
        assert [(m["id"], m["object"]) for m in data] == [("small", "model")]


class TestMain:
    @pytest.fixture
    def uvicorn(self):
        module = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": module}):
            yield module

    def test_serves_the_loaded_model(self, fake_backend, uvicorn, capsys):
        main(["-b", "fake", "-m", "small", "--port", "9000", "--api-key", "k"])

        (app,), kwargs = uvicorn.run.call_args
        assert kwargs == {"host": "127.0.0.1", "port": 9000}
        assert TestClient(app).get("/health").json()["backend"] == "fake"
        assert "Loading fake model 'small'" in capsys.readouterr().err

    def test_warns_when_open_to_the_network_without_a_key(self, fake_backend, uvicorn, capsys):
        main(["-b", "fake", "--host", "0.0.0.0"])
        assert "without --api-key" in capsys.readouterr().err

        main(["-b", "fake", "--host", "::1"])
        assert "without --api-key" not in capsys.readouterr().err

    @pytest.mark.parametrize(
        "host, loopback",
        [("127.0.0.1", True), ("localhost", True), ("::1", True), ("0.0.0.0", False), ("myhost", False)],
    )
    def test_loopback_detection(self, host, loopback):
        assert _is_loopback(host) is loopback

    def test_api_key_from_environment(self, monkeypatch):
        monkeypatch.setenv("TRANSCRIBE_API_KEY", "from-env")
        assert parse_args([]).api_key == "from-env"

    def test_missing_server_extra(self, uvicorn):
        with patch.dict(sys.modules, {"fastapi": None}):
            with pytest.raises(SystemExit, match=r"speech-transcription-toolkit\[server\]"):
                main([])
        assert "server extra" in INSTALL_HINT

    def test_model_load_error_exits(self, fake_backend, uvicorn):
        with pytest.raises(SystemExit, match="Unknown fake model"):
            main(["-b", "fake", "-m", "huge"])
        uvicorn.run.assert_not_called()
