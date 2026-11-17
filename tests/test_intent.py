import json
from urllib.request import Request

import pytest
from pydantic import ValidationError

from travel_video_agent.editing.intent import IntentParsingError, OllamaIntentParser
from travel_video_agent.editing.intent_schemas import EditingPlan


@pytest.fixture(autouse=True)
def disable_environment_dependent_memory_preflight(monkeypatch):
    monkeypatch.setattr(
        "travel_video_agent.editing.intent.ensure_memory_available",
        lambda required_gb, task: None,
    )


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_editing_plan_defaults_are_safe_and_reviewable() -> None:
    plan = EditingPlan(source_prompt="Create a travel video")
    assert plan.target_duration_seconds == 60
    assert plan.style == "cinematic"
    assert plan.transitions.default == "cut"
    assert plan.transitions.duration_seconds == 0
    assert plan.approval_required is True
    assert plan.rendering_performed is False


def test_editing_plan_rejects_unsupported_fields_and_invalid_values() -> None:
    with pytest.raises(ValidationError):
        EditingPlan(source_prompt="Make a video", shell_command="ffmpeg ...")
    with pytest.raises(ValidationError):
        EditingPlan(source_prompt="Make a video", target_duration_seconds=5000)
    with pytest.raises(ValidationError):
        EditingPlan(
            source_prompt="Make a video",
            transitions={"default": "cut", "duration_seconds": 1},
        )


def test_ollama_intent_parser_validates_structured_response(monkeypatch) -> None:
    content = {
        "target_duration_seconds": 60,
        "style": "cinematic",
        "aspect_ratio": "16:9",
        "resolution": "1080p",
        "pacing": "moderate",
        "priorities": ["scenic_landscapes", "memorable_moments"],
        "transitions": {"default": "crossfade", "duration_seconds": 0.5},
        "subtitles": {"enabled": True, "position": "bottom"},
        "audio": {"preserve_original": True, "background_music": False},
        "remove_silence": True,
        "assumptions": [],
    }
    captured = {}

    def fake_urlopen(request: Request, timeout: int):
        captured["request"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse({"message": {"content": json.dumps(content)}})

    monkeypatch.setattr("travel_video_agent.editing.intent.urlopen", fake_urlopen)
    plan = OllamaIntentParser(model="qwen2.5vl:3b").parse(
        "  Create a 60-second cinematic travel video  "
    )

    assert plan.source_prompt == "Create a 60-second cinematic travel video"
    assert plan.target_duration_seconds == 60
    assert plan.transitions.default == "crossfade"
    assert plan.approval_required is True
    assert plan.rendering_performed is False
    assert captured["request"]["keep_alive"] == "5m"
    assert captured["request"]["options"]["temperature"] == 0
    assert captured["request"]["format"]["properties"].get("source_prompt") is None


def test_ollama_intent_parser_rejects_invalid_model_output(monkeypatch) -> None:
    content = {"style": "make_shell_command", "target_duration_seconds": 60}
    monkeypatch.setattr(
        "travel_video_agent.editing.intent.urlopen",
        lambda request, timeout: FakeResponse({"message": {"content": json.dumps(content)}}),
    )
    with pytest.raises(IntentParsingError, match="did not satisfy"):
        OllamaIntentParser().parse("Make a video")


def test_ollama_intent_parser_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        OllamaIntentParser().parse("   ")



def test_intent_parser_unloads_model(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request: Request, timeout: int):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"response": ""})

    monkeypatch.setattr("travel_video_agent.editing.intent.urlopen", fake_urlopen)
    parser = OllamaIntentParser(model="qwen2.5vl:3b")
    parser._model_used = True
    assert parser.unload_model() is True
    assert captured["url"].endswith("/api/generate")
    assert captured["payload"] == {"model": "qwen2.5vl:3b", "keep_alive": 0}
