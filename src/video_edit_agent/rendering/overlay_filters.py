"""FFmpeg filter construction for reviewed title and lower-third overlays."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from ..config import DEFAULT_TITLE_FONT, get_setting
from .renderer import _seconds
from .title_overlays import TitleOverlay

_BUNDLED_FONT = Path(__file__).parent / "fonts" / "Lato-Regular.ttf"


def _escape_drawtext(value: str) -> str:
    """Escape user text for a quoted FFmpeg filter value with expansion disabled."""
    escaped = value.replace("\\", "\\\\")
    for character in (":", "'", ",", ";", "[", "]"):
        escaped = escaped.replace(character, "\\" + character)
    return escaped


def _bundled_font_path() -> Path | None:
    """Copy the bundled OFL-licensed font into a temporary cache for FFmpeg."""
    if not _BUNDLED_FONT.is_file():
        return None
    destination = Path(tempfile.gettempdir()) / "video-edit-agent" / "fonts" / "Lato-Regular.ttf"
    if destination.is_file():
        return destination
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
        shutil.copyfile(_BUNDLED_FONT, temporary)
        temporary.replace(destination)
    except OSError:
        return None
    return destination if destination.is_file() else None


def _font_option(text: str) -> str:
    """Resolve configured title font, preferring the bundled font with DejaVu fallback."""
    configured = (get_setting("TITLE_FONT", DEFAULT_TITLE_FONT) or DEFAULT_TITLE_FONT).strip()
    if configured.casefold() in {"lato", "bundled"}:
        # Use DejaVu for non-ASCII text to keep unsupported glyphs readable.
        if text.isascii():
            font_path = _bundled_font_path()
            if font_path is not None:
                return f"fontfile='{_escape_drawtext(str(font_path))}'"
        return "font='DejaVu Sans'"
    path = Path(configured).expanduser()
    if path.is_file():
        return f"fontfile='{_escape_drawtext(str(path.resolve()))}'"
    return f"font='{_escape_drawtext(configured or 'DejaVu Sans')}'"


def _ease_progress(start: float, duration: float, easing: str, *, reverse: bool = False) -> str:
    """Return a clamped 0..1 FFmpeg progress expression for the selected easing curve."""
    if easing == "linear":
        numerator = f"({_seconds(start + duration)}-t)" if reverse else f"(t-{_seconds(start)})"
        return f"{numerator}/{_seconds(duration)}"
    progress = (
        f"max(0,min(1,({_seconds(start + duration)}-t)/{_seconds(duration)}))"
        if reverse
        else f"max(0,min(1,(t-{_seconds(start)})/{_seconds(duration)}))"
    )
    if easing == "ease_in":
        return f"pow({progress},2)"
    if easing == "ease_out":
        return f"(1-pow(1-({progress}),2))"
    if easing == "ease_in_out":
        return f"if(lt({progress},0.5),2*pow({progress},2),1-pow(-2*({progress})+2,2)/2)"
    return progress


def _animation_expressions(
    overlay: TitleOverlay,
    base_y: str,
    *,
    lower_third: bool = False,
) -> tuple[str, str]:
    """Build shared alpha and vertical-position expressions for entry and exit motion."""
    alpha = "1"
    y_position = base_y
    duration = overlay.end_seconds - overlay.start_seconds
    fade_seconds = min(0.45, duration / 2)
    fade_in_end = overlay.start_seconds + fade_seconds
    fade_out_start = overlay.end_seconds - fade_seconds

    legacy_fade_exit = overlay.animation == "fade" and overlay.exit_animation == "auto"
    if overlay.animation == "fade":
        fade_in = _ease_progress(overlay.start_seconds, fade_seconds, overlay.easing)
        alpha = f"if(lt(t,{_seconds(fade_in_end)}),{fade_in},1)"
        if legacy_fade_exit:
            fade_out = _ease_progress(fade_out_start, fade_seconds, overlay.easing, reverse=True)
            alpha = (
                f"if(lt(t,{_seconds(fade_in_end)}),{fade_in},"
                f"if(gt(t,{_seconds(fade_out_start)}),{fade_out},1))"
            )
    elif overlay.animation == "slide_up":
        slide_end = overlay.start_seconds + 0.45
        progress = _ease_progress(overlay.start_seconds, 0.45, overlay.easing)
        if lower_third:
            alpha = f"if(lt(t,{_seconds(slide_end)}),{progress},1)"
        if overlay.easing == "linear":
            y_position = (
                f"if(lt(t,{_seconds(slide_end)}),"
                f"({base_y})+80*({_seconds(slide_end)}-t)/0.45,"
                f"({base_y}))"
            )
        else:
            slide_condition = f"if(lt(t,{_seconds(slide_end)}),"
            slide_motion = f"({base_y})+80*(1-({progress})),({base_y}))"
            y_position = slide_condition + slide_motion

    exit_animation = overlay.exit_animation
    if exit_animation == "auto":
        exit_animation = "fade" if overlay.animation == "fade" else "none"
    if exit_animation == "fade" and not legacy_fade_exit:
        fade_out = _ease_progress(fade_out_start, fade_seconds, overlay.easing, reverse=True)
        alpha = f"if(gt(t,{_seconds(fade_out_start)}),{fade_out},{alpha})"
    elif exit_animation in {"slide_down", "slide_up"}:
        slide_start = overlay.end_seconds - 0.45
        progress = _ease_progress(slide_start, 0.45, overlay.easing)
        direction = 80 if exit_animation == "slide_down" else -80
        if overlay.easing == "linear":
            movement = f"{direction}*(t-{_seconds(slide_start)})/0.45"
        else:
            movement = f"{direction}*({progress})"
        y_position = f"if(gt(t,{_seconds(slide_start)}),({base_y})+{movement},({y_position}))"
    return alpha, y_position


def _append_asset_overlay_filters(
    filters: list[str],
    asset_overlays: list[tuple[TitleOverlay, Path]],
    width: int,
    height: int,
    current_video: str,
    first_input_index: int,
) -> str:
    """Composite reviewed transparent assets over the rendered timeline."""
    video_label = current_video
    for index, (overlay, _asset_path) in enumerate(asset_overlays):
        scaled_label = f"asset{index}scaled"
        next_label = f"vasset{index}"
        filters.append(
            f"[{first_input_index + index}:v:0]format=rgba,"
            f"setpts=PTS-STARTPTS+{_seconds(overlay.start_seconds)}/TB,"
            f"scale={max(2, int(width * overlay.asset_scale))}:-1[{scaled_label}]"
        )
        filters.append(
            f"[{video_label}][{scaled_label}]overlay="
            f"x='(main_w-overlay_w)*{overlay.asset_x:.6f}':"
            f"y='(main_h-overlay_h)*{overlay.asset_y:.6f}':"
            f"eof_action=pass:shortest=0:"
            f"enable='between(t,{_seconds(overlay.start_seconds)},"
            f"{_seconds(overlay.end_seconds)})'"
            f"[{next_label}]"
        )
        video_label = next_label
    return video_label


def _append_title_overlay_filters(
    filters: list[str],
    title_overlays: list[TitleOverlay] | None,
    width: int,
    height: int,
    current_video: str,
) -> str:
    """Append approved title and lower-third filters, returning the final video label."""
    video_label = current_video
    for index, overlay in enumerate(title_overlays or []):
        if overlay.decision != "approved" or overlay.overlay_kind == "asset":
            continue
        escaped_text = _escape_drawtext(overlay.text or "")
        font_option = _font_option(overlay.text or "")
        next_label = f"vtitle{index}"
        if overlay.overlay_kind == "lower_third":
            # Percentage-based geometry keeps the card inside safe margins for
            # landscape, portrait, and square output profiles.
            card_width = "w*0.88" if height > width else "w*0.62"
            card_height = "h*0.16"
            _, card_y = _animation_expressions(overlay, "h*0.76", lower_third=True)
            filters.append(
                f"[{video_label}]drawbox=x=w*0.06:y='{card_y}':w={card_width}:"
                f"h={card_height}:color=black@0.62:t=fill:"
                f"enable='between(t,{_seconds(overlay.start_seconds)},"
                f"{_seconds(overlay.end_seconds)})'"
                f"[{next_label}bg]"
            )
            primary_size = max(26, min(64, int(width * 0.034)))
            secondary_size = max(20, min(42, int(width * 0.022)))
            alpha, primary_y = _animation_expressions(overlay, "h*0.785", lower_third=True)
            primary_label = f"vtitle{index}primary"
            filters.append(
                f"[{next_label}bg]drawtext=text='{escaped_text}':expansion=none:"
                f"{font_option}:fontsize={primary_size}:fontcolor=white:"
                f"x=w*0.09:y='{primary_y}':alpha='{alpha}':"
                f"enable='between(t,{_seconds(overlay.start_seconds)},"
                f"{_seconds(overlay.end_seconds)})'"
                f"[{primary_label}]"
            )
            if overlay.secondary_text:
                secondary_label = f"vtitle{index}secondary"
                secondary_alpha, secondary_y = _animation_expressions(
                    overlay, "h*0.835", lower_third=True
                )
                filters.append(
                    f"[{primary_label}]drawtext=text='{_escape_drawtext(overlay.secondary_text)}':"
                    f"expansion=none:{_font_option(overlay.secondary_text)}:fontsize={secondary_size}:"
                    f"fontcolor=white@0.88:x=w*0.09:y='{secondary_y}':alpha='{secondary_alpha}':"
                    f"enable='between(t,{_seconds(overlay.start_seconds)},"
                    f"{_seconds(overlay.end_seconds)})'"
                    f"[{secondary_label}]"
                )
                video_label = secondary_label
            else:
                video_label = primary_label
            continue

        font_size = max(32, min(96, int(width * 0.045)))
        if overlay.style == "minimal":
            font_size = max(28, int(width * 0.035))
            box_options = "box=0"
        elif overlay.style == "bold":
            font_size = max(40, min(112, int(width * 0.055)))
            box_options = "box=1:boxcolor=black@0.62:boxborderw=22"
        else:
            box_options = "box=1:boxcolor=black@0.42:boxborderw=18"
        base_y = {"top": "h*0.12", "center": "(h-text_h)/2", "bottom": "h-text_h-h*0.12"}[
            overlay.placement
        ]
        alpha, y_position = _animation_expressions(overlay, base_y)
        filters.append(
            f"[{video_label}]drawtext=text='{escaped_text}':expansion=none:"
            f"{font_option}:fontsize={font_size}:fontcolor=white:"
            f"x=(w-text_w)/2:y='{y_position}':alpha='{alpha}':{box_options}:"
            f"enable='between(t,{_seconds(overlay.start_seconds)},"
            f"{_seconds(overlay.end_seconds)})'"
            f"[{next_label}]"
        )
        video_label = next_label
    return video_label
