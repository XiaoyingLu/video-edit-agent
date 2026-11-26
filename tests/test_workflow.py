from pathlib import Path
from types import SimpleNamespace

import pytest

from video_edit_agent.editing import workflow
from video_edit_agent.runtime_capabilities import RuntimeCapabilityError


def test_create_video_draft_parses_prompt_before_running_expensive_stages(
    tmp_path: Path, monkeypatch
) -> None:
    def reject_prompt(prompt: str, model: str | None = None):
        raise ValueError("invalid prompt")

    def unexpected(*args, **kwargs):
        raise AssertionError("analysis must not start when intent parsing fails")

    monkeypatch.setattr(
        workflow,
        "probe_runtime_capabilities",
        lambda _root: SimpleNamespace(require=lambda *_names: None),
    )
    monkeypatch.setattr(workflow, "parse_editing_intent", reject_prompt)
    monkeypatch.setattr(workflow, "understand_folder", unexpected)
    monkeypatch.setattr(workflow, "transcribe_folder", unexpected)
    monkeypatch.setattr(workflow, "find_highlights_folder", unexpected)

    with pytest.raises(ValueError, match="invalid prompt"):
        workflow.create_video_draft(tmp_path, "   ")


def test_create_video_draft_writes_review_artifacts_and_stops_before_render(
    tmp_path: Path, monkeypatch
) -> None:
    plan = SimpleNamespace(model_dump=lambda **kwargs: {"style": "cinematic"})
    understanding = SimpleNamespace(model_dump=lambda **kwargs: {"scenes": []})
    transcription = SimpleNamespace(model_dump=lambda **kwargs: {"videos": []})
    highlights = SimpleNamespace(
        candidates=[object()],
        candidate_count=1,
        model_dump=lambda **kwargs: {"candidates": [{}]},
    )
    selection = SimpleNamespace(
        selected_clip_count=1,
        selected_duration_seconds=5.0,
        model_dump=lambda **kwargs: {"clips": [{}]},
    )
    timeline = SimpleNamespace(clips=[object()])
    calls = []
    selected_models = {}
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5vl:3b")
    monkeypatch.setattr(
        workflow,
        "probe_runtime_capabilities",
        lambda _root: SimpleNamespace(require=lambda *_names: None),
    )

    def fake_parse(prompt, model=None):
        selected_models["intent"] = model
        return plan

    def fake_understand(*args, **kwargs):
        selected_models["understanding"] = kwargs.get("model")
        calls.append("understand")
        return understanding

    monkeypatch.setattr(workflow, "parse_editing_intent", fake_parse)
    monkeypatch.setattr(workflow, "understand_folder", fake_understand)
    monkeypatch.setattr(
        workflow,
        "transcribe_folder",
        lambda *args, **kwargs: calls.append("transcribe") or transcription,
    )
    monkeypatch.setattr(
        workflow,
        "find_highlights_folder",
        lambda *args, **kwargs: calls.append("highlights") or highlights,
    )
    monkeypatch.setattr(workflow, "build_draft_selection", lambda *args, **kwargs: selection)
    monkeypatch.setattr(workflow, "build_draft_timeline", lambda *args, **kwargs: timeline)

    def save_timeline(value, destination):
        calls.append("timeline")
        destination.write_text("review-only", encoding="utf-8")

    monkeypatch.setattr(workflow, "save_draft_timeline", save_timeline)

    result = workflow.create_video_draft(tmp_path, "Create a short travel video")

    assert result[0] is plan
    assert result[1] is highlights
    assert result[2] is selection
    assert result[3] is timeline
    assert result[4].is_file()
    assert calls == ["understand", "transcribe", "highlights", "timeline"]
    assert selected_models == {"intent": "qwen2.5vl:3b", "understanding": "qwen2.5vl:3b"}
    for artifact in (
        "editing-plan.json",
        "understanding.json",
        "transcription.json",
        "highlights.json",
        "draft-selection.json",
        "optimized-timeline.json",
    ):
        assert (tmp_path / ".video-agent" / artifact).is_file()


def test_create_video_draft_stops_when_no_highlights_are_found(tmp_path: Path, monkeypatch) -> None:
    plan = SimpleNamespace(model_dump=lambda **kwargs: {"style": "cinematic"})
    empty_highlights = SimpleNamespace(
        candidates=[],
        candidate_count=0,
        model_dump=lambda **kwargs: {"candidates": []},
    )
    monkeypatch.setattr(
        workflow,
        "probe_runtime_capabilities",
        lambda _root: SimpleNamespace(require=lambda *_names: None),
    )
    monkeypatch.setattr(workflow, "parse_editing_intent", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        workflow,
        "understand_folder",
        lambda *args, **kwargs: SimpleNamespace(model_dump=lambda **kwargs: {}),
    )
    monkeypatch.setattr(
        workflow,
        "transcribe_folder",
        lambda *args, **kwargs: SimpleNamespace(model_dump=lambda **kwargs: {}),
    )
    monkeypatch.setattr(
        workflow, "find_highlights_folder", lambda *args, **kwargs: empty_highlights
    )

    with pytest.raises(ValueError, match="No highlight candidates"):
        workflow.create_video_draft(tmp_path, "Create a travel video")



def test_create_video_draft_checks_capabilities_before_parsing_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []

    class MissingCapabilities:
        def require(self, *_names):
            calls.append("capability_check")
            raise RuntimeCapabilityError("transcription is unavailable")

    monkeypatch.setattr(
        workflow, "probe_runtime_capabilities", lambda _root: MissingCapabilities()
    )
    monkeypatch.setattr(
        workflow,
        "parse_editing_intent",
        lambda *_args, **_kwargs: calls.append("parse") or None,
    )

    with pytest.raises(RuntimeCapabilityError, match="transcription is unavailable"):
        workflow.create_video_draft(tmp_path, "Create a travel video")

    assert calls == ["capability_check"]
