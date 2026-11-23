from pathlib import Path

import pytest
from pydantic import ValidationError

from video_edit_agent.rendering.overlay_filters import _animation_expressions
from video_edit_agent.rendering.shot_renderer import _build_shot_render_command
from video_edit_agent.rendering.title_overlays import (
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


def test_legacy_linear_fade_and_slide_up_filter_expressions_are_unchanged():
    fade = _overlay(start_seconds=0, end_seconds=3, animation="fade")
    alpha, y_position = _animation_expressions(fade, "h*0.12")
    assert alpha == "if(lt(t,0.450000),(t-0.000000)/0.450000,if(gt(t,2.550000),(3.000000-t)/0.450000,1))"
    assert y_position == "h*0.12"

    slide = _overlay(start_seconds=0, end_seconds=3, animation="slide_up")
    alpha, y_position = _animation_expressions(slide, "h*0.12")
    assert alpha == "1"
    assert y_position == "if(lt(t,0.450000),(h*0.12)+80*(0.450000-t)/0.45,(h*0.12))"


def test_lower_third_slide_up_moves_card_and_both_text_lines(tmp_path: Path):
    overlay = _overlay(
        overlay_kind="lower_third",
        text="Lake Louise",
        secondary_text="Alberta",
        animation="slide_up",
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
    assert "drawbox=x=w*0.06:y='if(lt(t,0.450000)" in graph
    assert "y='if(lt(t,0.450000),(h*0.785)+80*(0.450000-t)/0.45" in graph
    assert "y='if(lt(t,0.450000),(h*0.835)+80*(0.450000-t)/0.45" in graph


def test_title_overlay_supports_easing_and_exit_animation():
    overlay = _overlay(
        animation="slide_up",
        easing="ease_in_out",
        exit_animation="slide_down",
    )
    alpha, y_position = _animation_expressions(overlay, "h*0.12")
    assert alpha == "1"
    assert "pow(" in y_position
    assert "gt(t,2.550000)" in y_position
    assert "80*(" in y_position


def test_title_font_uses_configured_family_and_dejavu_for_non_ascii(monkeypatch):
    from video_edit_agent.rendering.overlay_filters import _font_option

    monkeypatch.setenv("TITLE_FONT", "DejaVu Sans")
    assert _font_option("A title") == "font='DejaVu Sans'"
    monkeypatch.setenv("TITLE_FONT", "Lato")
    assert _font_option("A title").startswith("fontfile=")
    assert _font_option("Été à Québec") == "font='DejaVu Sans'"


def test_renderer_keeps_legacy_linear_fade_and_slide_filter_strings(tmp_path: Path):
    fade_command = _build_shot_render_command(
        "ffmpeg",
        [(tmp_path / "fade.mp4", 0, 4, False, "cut", 0)],
        tmp_path / "fade-out.mp4",
        1920,
        1080,
        title_overlays=[_overlay(animation="fade")],
    )
    fade_graph = fade_command[fade_command.index("-filter_complex") + 1]
    assert (
        "alpha='if(lt(t,0.450000),(t-0.000000)/0.450000,"
        "if(gt(t,2.550000),(3.000000-t)/0.450000,1))'"
    ) in fade_graph

    slide_command = _build_shot_render_command(
        "ffmpeg",
        [(tmp_path / "slide.mp4", 0, 4, False, "cut", 0)],
        tmp_path / "slide-out.mp4",
        1920,
        1080,
        title_overlays=[_overlay(animation="slide_up")],
    )
    slide_graph = slide_command[slide_command.index("-filter_complex") + 1]
    assert "y='if(lt(t,0.450000),(h-text_h-h*0.12)+80*(0.450000-t)/0.45,(h-text_h-h*0.12))'" in slide_graph


def test_explicit_exit_animation_replaces_legacy_fade_out():
    overlay = _overlay(
        animation="fade",
        exit_animation="slide_down",
        easing="ease_out",
    )
    alpha, y_position = _animation_expressions(overlay, "h*0.12")
    assert alpha.startswith("if(lt(t,0.450000),")
    assert "gt(t,2.550000)" not in alpha
    assert "gt(t,2.550000)" in y_position
