import json
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from video_edit_agent.editing.ollama_client import (
    CHAT_TIMEOUT_SECONDS,
    ModelClientError,
    OllamaClient,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_chat_json_builds_structured_request_and_uses_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434/")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "10m")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse({"message": {"content": '{"title": "A story"}'}})

    monkeypatch.setattr("video_edit_agent.editing.ollama_client.urlopen", fake_urlopen)
    result = OllamaClient().chat_json(
        "story-model",
        [{"role": "user", "content": "Make a story"}],
        {"type": "object"},
    )

    assert result == {"title": "A story"}
    assert captured["url"] == "http://ollama:11434/api/chat"
    assert captured["payload"]["keep_alive"] == "10m"
    assert captured["payload"]["format"] == {"type": "object"}
    assert captured["payload"]["stream"] is False
    assert captured["timeout"] == CHAT_TIMEOUT_SECONDS


def test_client_retries_transient_http_error_once(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise HTTPError(
                request.full_url,
                503,
                "temporarily unavailable",
                hdrs=None,
                fp=BytesIO(b"try again"),
            )
        return FakeResponse({"embeddings": [[0.1, 0.2]]})

    monkeypatch.setattr("video_edit_agent.editing.ollama_client.urlopen", fake_urlopen)
    monkeypatch.setattr("video_edit_agent.editing.ollama_client.time.sleep", lambda *_: None)

    assert OllamaClient().embed("embed-model", ["one"]) == [[0.1, 0.2]]
    assert len(calls) == 2


def test_client_does_not_retry_permanent_http_error(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        raise HTTPError(
            request.full_url,
            404,
            "model missing",
            hdrs=None,
            fp=BytesIO(b"not found"),
        )

    monkeypatch.setattr("video_edit_agent.editing.ollama_client.urlopen", fake_urlopen)
    with pytest.raises(ModelClientError, match="HTTP 404") as error:
        OllamaClient().embed("missing-model", ["one"])
    assert error.value.status_code == 404
    assert len(calls) == 1


def test_client_retries_connection_errors_and_then_raises(monkeypatch):
    calls = []

    def fake_urlopen(*_args, **_kwargs):
        calls.append(1)
        raise URLError("connection refused")

    monkeypatch.setattr("video_edit_agent.editing.ollama_client.urlopen", fake_urlopen)
    monkeypatch.setattr("video_edit_agent.editing.ollama_client.time.sleep", lambda *_: None)
    with pytest.raises(ModelClientError, match="Could not reach Ollama"):
        OllamaClient().embed("embed-model", ["one"])
    assert len(calls) == 2


def test_embed_validates_vector_count_and_dimensions(monkeypatch):
    monkeypatch.setattr(
        "video_edit_agent.editing.ollama_client.urlopen",
        lambda *_args, **_kwargs: FakeResponse({"embeddings": [[0.1], [0.2, 0.3]]}),
    )
    with pytest.raises(ModelClientError, match="inconsistent dimensions"):
        OllamaClient().embed("embed-model", ["one", "two"])


def test_generate_supports_best_effort_model_unload(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse({"response": ""})

    monkeypatch.setattr("video_edit_agent.editing.ollama_client.urlopen", fake_urlopen)
    OllamaClient().generate("story-model", keep_alive=0, timeout_seconds=30)
    assert captured["payload"]["model"] == "story-model"
    assert captured["payload"]["keep_alive"] == 0
    assert captured["timeout"] == 30
