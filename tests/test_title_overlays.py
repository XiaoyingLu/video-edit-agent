from pathlib import Path

import pytest
from pydantic import ValidationError

from travel_video_agent.rendering.shot_renderer import _build_shot_render_command
from travel_video_agent.rendering.title_overlays import (
    TitleOverlay,
    TitleOverlayDocument,
    review_title_overlays,
)


def _overlay(**overrides):
    values = {
        "overlay_id": "opening",
        "text": "A New: Journey",
        "start_seconds": 0,
        "end_seconds": 3,
        "placement": "bottom",
        "style": "cinematic",
        "animation": "fade",
        "decision": "approved",
    }
    values.update(overrides)
    return TitleOverlay(**values)


def test_title_overlay_validates_interval_and_supported_presets():
    with pytest.raises(ValidationError, match="at least 0.5 seconds"):
        _overlay(end_seconds=0.2)
    with pytest.raises(ValidationError):
        _overlay(animation="arbitrary_filter")
    with pytest.raises(ValidationError):
        _overlay(text="x" * 121)


def test_title_overlay_review_requires_known_explicit_decisions():
    document = TitleOverlayDocument(
        project_path="/tmp/project",
        source_timeline=".video-agent/optimized-timeline.json",
        overlays=[_overlay(decision="pending")],
    )
    reviewed = review_title_overlays(document, {"opening"}, set())
    assert reviewed.overlays[0].decision == "approved"
    assert document.overlays[0].decision == "pending"
    rejected = review_title_overlays(reviewed, set(), {"opening"})
    assert rejected.overlays[0].decision == "rejected"
    with pytest.raises(ValueError, match="unknown title overlay"):
        review_title_overlays(document, {"missing"}, set())
    with pytest.raises(ValueError, match="both approved and rejected"):
        review_title_overlays(document, {"opening"}, {"opening"})


def test_title_overlay_document_rejects_duplicate_ids():
    with pytest.raises(ValidationError, match="IDs must be unique"):
        TitleOverlayDocument(
            project_path="/tmp/project",
            source_timeline=".video-agent/optimized-timeline.json",
            overlays=[_overlay(), _overlay()],
        )


@pytest.mark.parametrize("width,height", [(1920, 1080), (1080, 1920), (1080, 1080)])
def test_renderer_builds_safe_animated_title_filter_for_all_aspect_ratios(
    tmp_path: Path, width: int, height: int
):
    approved = _overlay(text="100%: let's go, [now];")
    pending = _overlay(overlay_id="pending", decision="pending")
    command = _build_shot_render_command(
        "ffmpeg",
        [(tmp_path / "clip.mp4", 0, 4, False, "cut", 0)],
        tmp_path / "output.mp4",
        width,
        height,
        title_overlays=[approved, pending],
    )
    graph = command[command.index("-filter_complex") + 1]
    assert "drawtext=" in graph
    assert "expansion=none" in graph
    assert "enable='between(t,0.000000,3.000000)'" in graph
    assert "A New" not in graph
    assert "100%" in graph
    assert "vtitle0" in graph
    assert "vtitle1" not in graph


def test_renderer_supports_slide_up_and_minimal_title_preset(tmp_path: Path):
    overlay = _overlay(
        style="minimal",
        animation="slide_up",
        placement="top",
        text="Hello",
    )
    command = _build_shot_render_command(
        "ffmpeg",
        [(tmp_path / "clip.mp4", 0, 4, False, "cut", 0)],
        tmp_path / "output.mp4",
        1920,
        1080,
        title_overlays=[overlay],
    )
    graph = command[command.index("-filter_complex") + 1]
    assert "box=0" in graph
    assert "if(lt(t,0.450000)" in graph
    assert "y='if(" in graph



def test_lower_third_schema_supports_bounded_secondary_context():
    overlay = _overlay(
        overlay_kind="lower_third",
        text="Lake Louise",
        secondary_text="Banff National Park · Alberta",
        placement="bottom",
    )
    assert overlay.overlay_kind == "lower_third"
    with pytest.raises(ValidationError, match="only supported for lower-third"):
        _overlay(secondary_text="Context without lower-third kind")


@pytest.mark.parametrize("width,height", [(1920, 1080), (1080, 1920), (1080, 1080)])
def test_renderer_builds_safe_lower_third_for_all_aspect_ratios(
    tmp_path: Path, width: int, height: int
):
    overlay = _overlay(
        overlay_kind="lower_third",
        text="Lake Louise: Alberta",
        secondary_text="Canadian Rockies",
        animation="fade",
    )
    command = _build_shot_render_command(
        "ffmpeg",
        [(tmp_path / "clip.mp4", 0, 4, False, "cut", 0)],
        tmp_path / "output.mp4",
        width,
        height,
        title_overlays=[overlay],
    )
    graph = command[command.index("-filter_complex") + 1]
    assert "drawbox=x=w*0.06" in graph
    assert ("w*0.88" if height > width else "w*0.62") in graph
    assert "Lake Louise" in graph
    assert "Canadian Rockies" in graph
    assert "expansion=none" in graph
    assert "vtitle0secondary" in graph
