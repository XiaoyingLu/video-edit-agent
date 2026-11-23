"""Named render profiles for common personal-video editing styles."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RenderStylePreset:
    """A conservative set of render defaults; explicit CLI options take precedence."""

    name: str
    description: str
    fit_mode: str
    music_mode: str
    music_volume: float
    music_fade_in: float
    music_fade_out: float
    music_ducking: bool
    beat_punch_in: bool = False


STYLE_PRESETS: dict[str, RenderStylePreset] = {
    "cinematic": RenderStylePreset(
        name="cinematic",
        description="Measured pacing, gentle soundtrack, and preserved source ambience.",
        fit_mode="crop",
        music_mode="blend",
        music_volume=0.16,
        music_fade_in=1.8,
        music_fade_out=2.5,
        music_ducking=True,
    ),
    "vlog": RenderStylePreset(
        name="vlog",
        description="Natural framing with clear original audio and restrained background music.",
        fit_mode="pad",
        music_mode="blend",
        music_volume=0.18,
        music_fade_in=0.5,
        music_fade_out=1.0,
        music_ducking=True,
    ),
    "travel-montage": RenderStylePreset(
        name="travel-montage",
        description="Full-frame crop with a more prominent, smoothly faded music bed.",
        fit_mode="crop",
        music_mode="blend",
        music_volume=0.25,
        music_fade_in=1.5,
        music_fade_out=2.5,
        music_ducking=False,
        beat_punch_in=True,
    ),
    "social": RenderStylePreset(
        name="social",
        description="Tighter framing and short audio fades for concise social edits.",
        fit_mode="crop",
        music_mode="blend",
        music_volume=0.20,
        music_fade_in=0.25,
        music_fade_out=0.8,
        music_ducking=True,
        beat_punch_in=True,
    ),
}


def resolve_style_preset(name: str | None) -> RenderStylePreset | None:
    """Return a named preset, preserving legacy defaults when no preset is selected."""
    if name is None:
        return None
    try:
        return STYLE_PRESETS[name]
    except KeyError as exc:
        available = ", ".join(STYLE_PRESETS)
        raise ValueError(f"unknown style preset {name!r}; choose from: {available}") from exc
