from pathlib import Path

import pytest
from pydantic import ValidationError

from video_edit_agent.rendering.overlay_filters import _animation_expressions
from video_edit_agent.rendering.shot_renderer import _build_shot_render_command
from video_edit_agent.rendering.title_overlays import (
    TitleOverlay,
    TitleOverlayDocument,
    resolve_asset_overlays,
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
@pytest.mark.parametrize("accent_bar", [False, True])
def test_renderer_builds_safe_lower_third_for_all_aspect_ratios(
    tmp_path: Path, width: int, height: int, accent_bar: bool
):
    overlay = _overlay(
        overlay_kind="lower_third",
        text="Lake Louise: Alberta",
        secondary_text="Canadian Rockies",
        animation="fade",
        accent_bar=accent_bar,
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
    assert "color=c=black:s=" in graph
    assert "colorchannelmixer=aa=0.62" in graph
    expected_card_width = "iw*0.88" if height > width else "iw*0.62"
    assert f"crop=w={expected_card_width}:h=ih*0.16" in graph
    assert "overlay=x='main_w*0.06':y=" in graph
    assert "drawbox=" not in graph
    assert ("w*0.88" if height > width else "w*0.62") in graph
    assert "Lake Louise" in graph
    assert "Canadian Rockies" in graph
    assert "expansion=none" in graph
    assert "vtitle0secondary" in graph
    if accent_bar:
        assert "crop=w=iw*0.22:h='max(3,ih*0.012)':x=0:y=0" in graph
        assert "[vtitle0bg][vtitle0accentsrc]overlay=" in graph
        assert "[vtitle0accent]drawtext=" in graph
    else:
        assert "[vtitle0bg]drawtext=" in graph


def test_legacy_linear_fade_and_slide_up_filter_expressions_are_unchanged():
    fade = _overlay(start_seconds=0, end_seconds=3, animation="fade")
    alpha, y_position = _animation_expressions(fade, "h*0.12")
    expected_alpha = "".join(
        [
            "if(lt(t,0.450000),(t-0.000000)/0.450000,",
            "if(gt(t,2.550000),(3.000000-t)/0.450000,1))",
        ]
    )
    assert alpha == expected_alpha
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
    assert "overlay=x='main_w*0.06':y='if(lt(t,0.450000)" in graph
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
    expected_fade_alpha = "".join(
        [
            "alpha='if(lt(t,0.450000),(t-0.000000)/0.450000,",
            "if(gt(t,2.550000),(3.000000-t)/0.450000,1))'",
        ]
    )
    assert expected_fade_alpha in fade_graph

    slide_command = _build_shot_render_command(
        "ffmpeg",
        [(tmp_path / "slide.mp4", 0, 4, False, "cut", 0)],
        tmp_path / "slide-out.mp4",
        1920,
        1080,
        title_overlays=[_overlay(animation="slide_up")],
    )
    slide_graph = slide_command[slide_command.index("-filter_complex") + 1]
    expected_slide_y = "".join(
        [
            "y='if(lt(t,0.450000),(h-text_h-h*0.12)+80*(0.450000-t)/0.45,",
            "(h-text_h-h*0.12))'",
        ]
    )
    assert expected_slide_y in slide_graph


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


def test_asset_overlay_schema_requires_template_path_and_format():
    asset = _overlay(
        overlay_kind="asset",
        text=None,
        asset_path="templates/overlays/logo.webm",
        asset_format="webm",
    )
    assert asset.overlay_kind == "asset"
    with pytest.raises(ValidationError, match="templates/overlays"):
        _overlay(
            overlay_kind="asset",
            text=None,
            asset_path="../logo.webm",
            asset_format="webm",
        )
    with pytest.raises(ValidationError, match="numbered"):
        _overlay(
            overlay_kind="asset",
            text=None,
            asset_path="templates/overlays/sticker.png",
            asset_format="png_sequence",
        )


def test_resolve_asset_overlay_enforces_template_folder_and_png_pattern(tmp_path: Path):
    asset_dir = tmp_path / "templates" / "overlays"
    asset_dir.mkdir(parents=True)
    webm = asset_dir / "logo.webm"
    webm.write_bytes(b"webm")
    sequence = asset_dir / "spark-%04d.png"
    (asset_dir / "spark-0001.png").write_bytes(b"png")
    webm_overlay = _overlay(
        overlay_kind="asset",
        text=None,
        asset_path="templates/overlays/logo.webm",
        asset_format="webm",
    )
    png_overlay = _overlay(
        overlay_id="spark",
        overlay_kind="asset",
        text=None,
        asset_path="templates/overlays/spark-%04d.png",
        asset_format="png_sequence",
    )
    resolved = resolve_asset_overlays(tmp_path, [webm_overlay, png_overlay])
    assert [path for _, path in resolved] == [webm.resolve(), sequence.resolve()]
    outside = _overlay(
        overlay_id="outside",
        overlay_kind="asset",
        text=None,
        asset_path="templates/overlays/../../outside.webm",
        asset_format="webm",
    )
    with pytest.raises(ValueError, match="inside templates/overlays"):
        resolve_asset_overlays(tmp_path, [outside])


def test_renderer_builds_webm_and_png_asset_inputs_and_overlay_filters(tmp_path: Path):
    webm = _overlay(
        overlay_kind="asset",
        text=None,
        asset_path="templates/overlays/logo.webm",
        asset_format="webm",
        start_seconds=1,
        end_seconds=3,
        asset_x=0.1,
        asset_y=0.2,
        asset_scale=0.25,
    )
    png = _overlay(
        overlay_id="spark",
        overlay_kind="asset",
        text=None,
        asset_path="templates/overlays/spark-%04d.png",
        asset_format="png_sequence",
        start_seconds=0,
        end_seconds=2,
        asset_fps=24,
    )
    command = _build_shot_render_command(
        "ffmpeg",
        [(tmp_path / "clip.mp4", 0, 4, False, "cut", 0)],
        tmp_path / "output.mp4",
        1920,
        1080,
        asset_overlays=[
            (webm, tmp_path / "logo.webm"),
            (png, tmp_path / "spark-%04d.png"),
        ],
    )
    graph = command[command.index("-filter_complex") + 1]
    assert command.count("-stream_loop") == 2
    assert "-framerate" in command
    assert "scale=480:-1" in graph
    assert "between(t,1.000000,3.000000)" in graph
    assert "between(t,0.000000,2.000000)" in graph
    assert "overlay=x='(main_w-overlay_w)*0.100000'" in graph


def test_resolve_asset_overlay_rejects_symlinked_template_directory(tmp_path: Path):
    outside = tmp_path / "external-assets"
    outside.mkdir()
    (outside / "logo.webm").write_bytes(b"webm")
    project = tmp_path / "project"
    project.mkdir()
    (project / "templates").mkdir()
    (project / "templates" / "overlays").symlink_to(outside, target_is_directory=True)
    overlay = _overlay(
        overlay_kind="asset",
        text=None,
        asset_path="templates/overlays/logo.webm",
        asset_format="webm",
    )
    with pytest.raises(ValueError, match="template folder must be inside"):
        resolve_asset_overlays(project, [overlay])


def test_slide_left_template_animation_is_preserved_and_rendered(tmp_path: Path):
    from video_edit_agent.rendering.overlay_filters import _horizontal_animation_expression
    from video_edit_agent.templates import TitleStyleSpec

    style = TitleStyleSpec(animation="slide_left")
    assert style.animation == "slide_left"
    overlay = _overlay(animation=style.animation, text="A title")
    expression = _horizontal_animation_expression(overlay, "(w-text_w)/2")
    assert expression.startswith("if(lt(t,0.450000),")
    assert "80*(0.450000-t)/0.45" in expression

    command = _build_shot_render_command(
        "ffmpeg",
        [(tmp_path / "clip.mp4", 0, 4, False, "cut", 0)],
        tmp_path / "output.mp4",
        1920,
        1080,
        title_overlays=[overlay],
    )
    graph = command[command.index("-filter_complex") + 1]
    assert "x='if(lt(t,0.450000)" in graph


def test_lower_third_ffmpeg_smoke_renders_and_moves_card_on_lavfi_source(tmp_path: Path):
    """Exercise real FFmpeg filter parsing and verify the card moves between frames."""
    import shutil
    import subprocess

    import numpy as np
    import pytest

    from video_edit_agent.rendering.overlay_filters import _append_title_overlay_filters

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg is required for the lower-third rendering smoke test")

    overlay = _overlay(
        overlay_kind="lower_third",
        text="Smoke test",
        secondary_text="Static lavfi background",
        animation="slide_up",
        accent_bar=True,
        start_seconds=0,
        end_seconds=2,
    )
    filters: list[str] = []
    output_label = _append_title_overlay_filters(filters, [overlay], 320, 180, "0:v")
    rendered = tmp_path / "lower-third-smoke.mkv"
    render = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=320x180:r=25:d=2",
            "-filter_complex",
            ";".join(filters),
            "-map",
            f"[{output_label}]",
            "-t",
            "2",
            "-c:v",
            "ffv1",
            str(rendered),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert render.returncode == 0, render.stderr
    assert rendered.is_file() and rendered.stat().st_size > 0

    def dark_card_rows(timestamp: float) -> np.ndarray:
        frame = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(timestamp),
                "-i",
                str(rendered),
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "pipe:1",
            ],
            capture_output=True,
            check=False,
        )
        assert frame.returncode == 0, frame.stderr.decode(errors="replace")
        pixels = np.frombuffer(frame.stdout, dtype=np.uint8).reshape(180, 320)
        # Sample inside the card but to the left of the white title glyphs.
        dark_pixels_per_row = (pixels[:, 20:27] < 100).sum(axis=1)
        return np.flatnonzero(dark_pixels_per_row > 5)

    early_rows = dark_card_rows(0.25)
    settled_rows = dark_card_rows(0.8)
    assert early_rows.size >= 5
    assert settled_rows.size >= 20
    assert float(np.median(early_rows)) > float(np.median(settled_rows)) + 20

    color_frame = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "0.8",
            "-i",
            str(rendered),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    assert color_frame.returncode == 0, color_frame.stderr.decode(errors="replace")
    rgb = np.frombuffer(color_frame.stdout, dtype=np.uint8).reshape(180, 320, 3)
    accent_pixels = (rgb[:, :, 0] > 180) & (rgb[:, :, 1] > 100) & (rgb[:, :, 2] < 130)
    assert int(accent_pixels.sum()) >= 10


def test_asset_overlay_ffmpeg_smoke_composites_png_sequence_only_during_approved_window(
    tmp_path: Path,
):
    """Exercise real FFmpeg alpha compositing and verify the asset's timed appearance."""
    import shutil
    import subprocess

    import numpy as np

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg is required for the asset-overlay rendering smoke test")

    source = tmp_path / "source.mp4"
    asset_dir = tmp_path / "templates" / "overlays"
    asset_dir.mkdir(parents=True)
    asset_pattern = asset_dir / "marker-%04d.png"
    rendered = tmp_path / "asset-overlay-smoke.mp4"

    def run(command: list[str]) -> None:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
        assert result.returncode == 0, result.stderr

    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=320x180:r=25:d=2",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(source),
        ]
    )
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=40x40:d=1",
            "-vf",
            "format=rgba,colorchannelmixer=aa=0,drawbox=x=0:y=0:w=40:h=40:color=red@1:t=fill",
            "-frames:v",
            "1",
            "-y",
            str(asset_dir / "marker-0001.png"),
        ]
    )

    overlay = _overlay(
        overlay_id="marker",
        overlay_kind="asset",
        text=None,
        asset_path="templates/overlays/marker-%04d.png",
        asset_format="png_sequence",
        start_seconds=0.5,
        end_seconds=1.5,
        asset_fps=1,
        asset_x=0.5,
        asset_y=0.5,
        asset_scale=0.25,
    )
    command = _build_shot_render_command(
        ffmpeg,
        [(source, 0, 2, False, "cut", 0)],
        rendered,
        320,
        180,
        asset_overlays=[(overlay, asset_pattern)],
    )
    run(command)
    assert rendered.is_file() and rendered.stat().st_size > 0

    def red_pixel_count(timestamp: float) -> int:
        frame = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(timestamp),
                "-i",
                str(rendered),
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert frame.returncode == 0, frame.stderr.decode(errors="replace")
        pixels = np.frombuffer(frame.stdout, dtype=np.uint8).reshape(180, 320, 3)
        red_pixels = (pixels[:, :, 0] > 150) & (pixels[:, :, 1] < 100) & (pixels[:, :, 2] < 100)
        return int(red_pixels.sum())

    assert red_pixel_count(0.25) == 0
    assert red_pixel_count(1.0) > 100
    assert red_pixel_count(1.75) == 0
