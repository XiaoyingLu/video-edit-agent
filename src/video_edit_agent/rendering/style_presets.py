"""Legacy style-preset API backed by the built-in reusable templates."""

from dataclasses import dataclass

from ..templates import load_builtin_template


@dataclass(frozen=True)
class RenderStylePreset:
    """Compatibility view of a reusable template's render and music settings."""

    name: str
    description: str
    fit_mode: str
    music_mode: str
    music_volume: float
    music_fade_in: float
    music_fade_out: float
    music_ducking: bool
    beat_punch_in: bool = False


# Descriptions remain CLI-facing copy; all executable settings live in YAML.
_DESCRIPTIONS = {
    "cinematic": "Measured pacing, gentle soundtrack, and preserved source ambience.",
    "vlog": "Natural framing with clear original audio and restrained background music.",
    "travel-montage": "Full-frame crop with a more prominent, smoothly faded music bed.",
    "social": "Tighter framing and short audio fades for concise social edits.",
}


def _preset_from_template(name: str) -> RenderStylePreset:
    spec, _ = load_builtin_template(name)
    return RenderStylePreset(
        name=name,
        description=_DESCRIPTIONS[name],
        fit_mode=spec.render.fit_mode,
        music_mode=spec.music.mode,
        music_volume=spec.music.volume,
        music_fade_in=spec.music.fade_in,
        music_fade_out=spec.music.fade_out,
        music_ducking=spec.music.ducking,
        beat_punch_in=spec.render.beat_punch_in,
    )


STYLE_PRESETS: dict[str, RenderStylePreset] = {
    name: _preset_from_template(name) for name in _DESCRIPTIONS
}


def resolve_style_preset(name: str | None) -> RenderStylePreset | None:
    """Return a template-backed preset, preserving the legacy CLI/API alias."""
    if name is None:
        return None
    try:
        return STYLE_PRESETS[name]
    except KeyError as exc:
        available = ", ".join(STYLE_PRESETS)
        raise ValueError(f"unknown style preset {name!r}; choose from: {available}") from exc
