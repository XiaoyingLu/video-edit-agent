from pathlib import Path

import pytest

from video_edit_agent.rendering.renderer import (
    _build_ffmpeg_command,
    _subtitle_force_style,
)


@pytest.mark.parametrize(
    ("position", "alignment", "margin"),
    [
        ("bottom", 2, 48),
        ("center", 5, 0),
        ("top", 8, 48),
    ],
)
def test_classic_subtitle_style_preserves_existing_defaults(
    position: str, alignment: int, margin: int
):
    style = _subtitle_force_style("classic", 28, position, 1920, 1080)

    assert style == (
        "FontName=DejaVu Sans,FontSize=28,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        f"BorderStyle=1,Outline=2,Shadow=1,Alignment={alignment},MarginV={margin}"
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("cinematic", "Outline=3,Shadow=1,Bold=1"),
        ("boxed", "BorderStyle=3,Outline=12,Shadow=0"),
        ("minimal", "Outline=1,Shadow=1,Bold=0"),
    ],
)
def test_subtitle_presets_use_fixed_appearance(name: str, expected: str):
    style = _subtitle_force_style(name, 28, "bottom", 1920, 1080)

    assert expected in style
    assert "FontName=DejaVu Sans" in style
    assert "Alignment=2" in style


def test_nonclassic_subtitle_font_and_safe_margin_scale_with_portrait_height():
    landscape = _subtitle_force_style("cinematic", 28, "bottom", 1920, 1080)
    portrait = _subtitle_force_style("cinematic", 28, "bottom", 1080, 1920)

    assert "FontSize=28" in landscape
    assert "MarginV=65" in landscape
    assert "FontSize=50" in portrait
    assert "MarginV=115" in portrait


@pytest.mark.parametrize("position,alignment", [("bottom", 2), ("center", 5), ("top", 8)])
def test_subtitle_position_maps_to_fixed_ass_alignment(position: str, alignment: int):
    style = _subtitle_force_style("boxed", 28, position, 1080, 1920)

    assert f"Alignment={alignment}" in style


@pytest.mark.parametrize(
    ("style_name", "font_size", "position", "width", "height"),
    [
        ("unknown", 28, "bottom", 1920, 1080),
        ("classic", 11, "bottom", 1920, 1080),
        ("classic", 28, "left", 1920, 1080),
        ("classic", 28, "bottom", 0, 1080),
    ],
)
def test_subtitle_style_rejects_unsupported_values(
    style_name: str, font_size: int, position: str, width: int, height: int
):
    with pytest.raises(ValueError):
        _subtitle_force_style(style_name, font_size, position, width, height)


def test_ffmpeg_command_uses_requested_subtitle_preset(tmp_path: Path):
    command = _build_ffmpeg_command(
        "ffmpeg",
        [(tmp_path / "clip.mp4", {}, 4.0)],
        [(0, 0.0, 4.0, False)],
        tmp_path / "output.mp4",
        1080,
        1920,
        30.0,
        subtitle_path=tmp_path / "captions.srt",
        subtitle_style="boxed",
    )

    graph = command[command.index("-filter_complex") + 1]
    assert "BorderStyle=3" in graph
    assert "BackColour=&H80000000" in graph
