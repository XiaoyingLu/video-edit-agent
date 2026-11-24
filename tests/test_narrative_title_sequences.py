import json
from pathlib import Path

import pytest

from video_edit_agent.editing.intent_schemas import EditingPlan
from video_edit_agent.editing.planning_schemas import (
    CreativeBrief,
    StoryTreatment,
    VideoProjectPlan,
)
from video_edit_agent.editing.timeline_schemas import (
    DraftTimelineClip,
    ProjectDraftTimeline,
)
from video_edit_agent.rendering import title_overlays


def _write_inputs(root: Path, *, approved=True, include_story=True):
    artifact_dir = root / ".video-agent"
    artifact_dir.mkdir(parents=True)
    timeline = ProjectDraftTimeline(
        project_path=str(root),
        style="cinematic",
        pacing="moderate",
        target_duration_seconds=19,
        proposed_duration_seconds=19,
        approved_duration_seconds=19 if approved else 0,
        proposed_clip_count=2,
        timeline_approved=approved,
        clips=[
            DraftTimelineClip(
                clip_id="clip-1",
                sequence_index=1,
                video_path="one.mp4",
                source_start_seconds=0,
                source_end_seconds=10,
                duration_seconds=10,
                combined_score=90,
                pacing_note="Opening establishing shot.",
                decision="approved" if approved else "pending",
            ),
            DraftTimelineClip(
                clip_id="clip-2",
                sequence_index=2,
                video_path="two.mp4",
                source_start_seconds=0,
                source_end_seconds=10,
                duration_seconds=10,
                combined_score=90,
                pacing_note="Closing scenic shot.",
                transition_in="crossfade",
                transition_duration_seconds=1,
                decision="approved" if approved else "pending",
            ),
        ],
    )
    timeline_path = artifact_dir / "optimized-timeline.json"
    timeline_path.write_text(timeline.model_dump_json(), encoding="utf-8")
    treatment = (
        StoryTreatment(
            working_title="A Weekend in the Rockies",
            logline="A quiet journey through mountain roads and alpine lakes.",
            narrative_arc=["Leaving the city", "Finding stillness in the mountains"],
            emotional_journey="From anticipation to calm.",
        )
        if include_story
        else None
    )
    plan = VideoProjectPlan(
        project_path=str(root),
        brief=CreativeBrief(
            source_prompt="Create a cinematic mountain travel film.",
            target_duration_seconds=19,
            style="cinematic",
            aspect_ratio="16:9",
            resolution="1080p",
            pacing="moderate",
        ),
        editing_preferences=EditingPlan(source_prompt="Create a cinematic mountain travel film."),
        story_treatment=treatment,
        planning_stage="story_ready" if include_story else "brief_ready",
    )
    plan_path = artifact_dir / "video-project-plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    return timeline_path, plan_path


def test_suggest_narrative_title_sequences_uses_story_and_approved_timeline(
    tmp_path: Path, monkeypatch
):
    timeline_path, plan_path = _write_inputs(tmp_path)
    registered = []
    monkeypatch.setattr(title_overlays, "validate_artifact", lambda *_: None)
    monkeypatch.setattr(
        title_overlays,
        "register_artifact",
        lambda root, artifact, dependencies: registered.append((artifact, dependencies)),
    )

    document, destination = title_overlays.suggest_narrative_title_sequences(tmp_path)

    assert destination == tmp_path / ".video-agent" / "title-overlays.json"
    assert [item.overlay_id for item in document.overlays] == [
        "opening-title",
        "opening-logline",
        "closing-card",
    ]
    assert document.overlays[0].text == "A Weekend in the Rockies"
    assert document.source_plan == str(plan_path.relative_to(tmp_path))
    assert document.template_id == "cinematic"
    assert document.template_version == 1
    assert document.schema_version == "1.1"
    assert document.overlays[2].text == "Finding stillness in the mountains"
    assert len(document.overlays[0].text) <= 40
    assert len(document.overlays[1].text) <= 48
    assert len(document.overlays[2].text) <= 40
    assert document.overlays[0].sequence_role == "opening"
    assert document.overlays[2].sequence_role == "closing"
    assert all(item.decision == "pending" for item in document.overlays)
    assert document.overlays[0].start_seconds == 0
    assert document.overlays[2].end_seconds == pytest.approx(19)
    assert registered[0][1][:2] == [timeline_path, plan_path]
    assert Path(registered[0][1][2]).name == "cinematic.yaml"
    title_overlays.save_title_overlays(tmp_path, document, destination)
    assert registered[1][1][:2] == [timeline_path, plan_path]
    assert Path(registered[1][1][2]).name == "cinematic.yaml"
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["overlays"][0]["decision"] == "pending"


def test_narrative_title_suggestions_require_approved_timeline(tmp_path: Path, monkeypatch):
    _write_inputs(tmp_path, approved=False)
    monkeypatch.setattr(title_overlays, "validate_artifact", lambda *_: None)
    with pytest.raises(ValueError, match="must be approved"):
        title_overlays.suggest_narrative_title_sequences(tmp_path)


def test_narrative_title_suggestions_require_story_treatment(tmp_path: Path, monkeypatch):
    _write_inputs(tmp_path, include_story=False)
    monkeypatch.setattr(title_overlays, "validate_artifact", lambda *_: None)
    with pytest.raises(ValueError, match="story treatment"):
        title_overlays.suggest_narrative_title_sequences(tmp_path)


def test_narrative_title_suggestions_do_not_overwrite_existing_review(tmp_path: Path, monkeypatch):
    _write_inputs(tmp_path)
    monkeypatch.setattr(title_overlays, "validate_artifact", lambda *_: None)
    destination = tmp_path / ".video-agent" / "title-overlays.json"
    destination.write_text("keep this review", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        title_overlays.suggest_narrative_title_sequences(tmp_path)
    assert destination.read_text(encoding="utf-8") == "keep this review"


def test_template_changes_sequence_content_and_styles(tmp_path: Path, monkeypatch):
    _write_inputs(tmp_path)
    monkeypatch.setattr(title_overlays, "validate_artifact", lambda *_: None)
    document, _ = title_overlays.suggest_narrative_title_sequences(
        tmp_path, template_id="travel-montage"
    )
    assert document.template_id == "travel-montage"
    assert [item.overlay_id for item in document.overlays] == [
        "opening-1",
        "opening-2",
        "closing-1",
    ]
    assert document.overlays[0].style == "bold"
    assert document.overlays[1].style == "minimal"
    assert document.overlays[2].animation == "slide_up"
    assert all(item.decision == "pending" for item in document.overlays)


def test_template_title_sequences_can_be_overridden_per_project(tmp_path: Path, monkeypatch):
    _write_inputs(tmp_path)
    template_dir = tmp_path / ".video-agent" / "templates"
    template_dir.mkdir(parents=True)
    (template_dir / "custom.yaml").write_text(
        """id: custom
version: 2
extends: cinematic
sequences:
  opening:
    - slot: "{title}"
      style: minimal
  closing:
    - slot: "{closing_line}"
      style: closing
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(title_overlays, "validate_artifact", lambda *_: None)
    document, _ = title_overlays.suggest_narrative_title_sequences(tmp_path, template_id="custom")
    assert document.template_id == "custom"
    assert document.template_version == 2
    assert document.overlays[0].style == "minimal"


def test_cinematic_override_keeps_unique_ids_for_repeated_slots(tmp_path: Path, monkeypatch):
    _write_inputs(tmp_path)
    template_dir = tmp_path / ".video-agent" / "templates"
    template_dir.mkdir(parents=True)
    (template_dir / "cinematic.yaml").write_text(
        """id: cinematic
version: 2
title_styles:
  minimal:
    placement: bottom
    size: 0.72
    box: 0.0
    animation: fade
sequences:
  opening:
    - slot: "{title}"
      style: minimal
    - slot: "{title}"
      style: minimal
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(title_overlays, "validate_artifact", lambda *_: None)
    document, _ = title_overlays.suggest_narrative_title_sequences(tmp_path)
    assert [item.overlay_id for item in document.overlays] == ["opening-title", "opening-2"]
