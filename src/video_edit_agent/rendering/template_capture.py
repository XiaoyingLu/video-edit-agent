"""Capture reusable, data-only templates from reviewed project artifacts."""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Literal

import yaml

from ..editing.timeline_schemas import ProjectDraftTimeline
from ..templates import MusicSpec, TemplateSpec, TitleStyleSpec
from .title_overlays import TitleOverlayDocument

_TRANSITIONS = {
    "cut": "cut",
    "crossfade": "crossfade",
    "dissolve": "crossfade",
    "fadeblack": "fadeblack",
    "wipeleft": "wipeleft",
    "slideleft": "slideleft",
    "circleopen": "circleopen",
}
_STYLE_PROFILES = {
    "cinematic": "cinematic",
    "vlog": "vlog",
    "documentary": "cinematic",
    "highlight_reel": "travel-montage",
    "social_media": "social",
}


def _majority(values: list[str], default: str) -> str:
    return Counter(values).most_common(1)[0][0] if values else default


def _style_from_overlay(overlay) -> TitleStyleSpec:
    size = {"minimal": 0.8, "cinematic": 1.0, "bold": 1.2}[overlay.style]
    box = {"minimal": 0.0, "cinematic": 0.35, "bold": 0.55}[overlay.style]
    return TitleStyleSpec(
        placement=overlay.placement,
        size=size,
        box=box,
        animation=overlay.animation,
    )


def _capture_title_sequences(document: TitleOverlayDocument | None):
    styles: dict[str, TitleStyleSpec] = {}
    sequences: dict[str, list[dict[str, str]]] = {"opening": [], "closing": []}
    notes: list[str] = []
    if document is None:
        return styles, {}, ["No title-overlay document was found; title styling was not captured."]

    used_slots: set[str] = set()
    counters: Counter[str] = Counter()
    for overlay in document.overlays:
        if overlay.decision != "approved" or overlay.overlay_kind == "asset":
            continue
        role = overlay.sequence_role
        if role not in {"opening", "closing"}:
            continue
        slot = (
            "{title}"
            if role == "opening" and not sequences["opening"]
            else "{logline}"
            if role == "opening"
            else "{closing_line}"
        )
        if slot in used_slots:
            notes.append(f"Skipped duplicate {role} title overlay {overlay.overlay_id}.")
            continue
        used_slots.add(slot)
        counters[role] += 1
        style_name = f"{role}-{counters[role]}"
        styles[style_name] = _style_from_overlay(overlay)
        sequences[role].append({"slot": slot, "style": style_name})

    captured = {name: items for name, items in sequences.items() if items}
    if not captured:
        notes.append("No approved opening/closing text overlays were available to capture.")
    return styles, captured, notes


def save_template_from_project(
    folder: Path,
    template_id: str,
    *,
    output_path: Path | None = None,
    music_mode: Literal["original", "music_only", "blend"] | None = None,
    music_volume: float | None = None,
    music_fade_in: float | None = None,
    music_fade_out: float | None = None,
    music_ducking: bool | None = None,
) -> tuple[TemplateSpec, Path, list[str]]:
    """Capture reviewed title structure and timeline transitions into a reusable template."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project folder does not exist: {root}")
    timeline_path = root / ".video-agent" / "optimized-timeline.json"
    if not timeline_path.is_file():
        raise FileNotFoundError(f"Missing optimized timeline: {timeline_path}")
    timeline = ProjectDraftTimeline.model_validate_json(timeline_path.read_text(encoding="utf-8"))
    if Path(timeline.project_path).resolve() != root:
        raise ValueError("optimized timeline belongs to a different project folder")
    if not timeline.timeline_approved:
        raise ValueError("the optimized timeline must be approved before saving a template")

    profile_id = _STYLE_PROFILES[timeline.style]
    from .style_presets import resolve_style_preset

    preset = resolve_style_preset(profile_id)
    assert preset is not None
    music = MusicSpec(
        mode=music_mode or preset.music_mode,
        volume=music_volume if music_volume is not None else preset.music_volume,
        fade_in=music_fade_in if music_fade_in is not None else preset.music_fade_in,
        fade_out=music_fade_out if music_fade_out is not None else preset.music_fade_out,
        ducking=music_ducking if music_ducking is not None else preset.music_ducking,
    )

    overlay_path = root / ".video-agent" / "title-overlays.json"
    overlay_document = None
    if overlay_path.is_file():
        overlay_document = TitleOverlayDocument.model_validate_json(
            overlay_path.read_text(encoding="utf-8")
        )
        if Path(overlay_document.project_path).resolve() != root:
            raise ValueError("title overlay document belongs to a different project folder")
        source_timeline = Path(overlay_document.source_timeline)
        if not source_timeline.is_absolute():
            source_timeline = root / source_timeline
        if source_timeline.resolve() != timeline_path.resolve():
            raise ValueError("title overlays reference a different optimized timeline")

    title_styles, sequences, notes = _capture_title_sequences(overlay_document)
    if not title_styles:
        title_styles = {
            "opening": TitleStyleSpec(),
            "closing": TitleStyleSpec(placement="bottom"),
        }
        sequences = {
            "opening": [{"slot": "{title}", "style": "opening"}],
            "closing": [{"slot": "{closing_line}", "style": "closing"}],
        }
        notes.append("Used safe built-in title defaults because no approved title styles were found.")

    transitions = [
        _TRANSITIONS[clip.transition_in]
        for clip in timeline.clips
        if clip.decision == "approved" and clip.transition_in != "cut"
    ]
    default_transition = _majority(transitions, "crossfade")
    template = TemplateSpec(
        id=template_id,
        version=1,
        title_styles=title_styles,
        sequences=sequences,
        transitions={
            "default": default_transition,
            "on_scene_change": default_transition,
            "on_beat": "cut",
        },
        music=music,
        render={"fit_mode": preset.fit_mode, "beat_punch_in": preset.beat_punch_in},
        tags={
            "source_style": [timeline.style],
            "best_for": [_STYLE_PROFILES[timeline.style]],
        },
    )

    destination = (
        (output_path or Path.home() / ".video-agent" / "templates" / f"{template_id}.yaml")
        .expanduser()
        .resolve()
    )
    if destination.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("template output must use a .yaml or .yml extension")
    if destination.exists():
        raise FileExistsError(f"Template already exists: {destination}; choose another output path.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(
        template.model_dump(mode="json", exclude_defaults=True),
        sort_keys=False,
        allow_unicode=True,
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-linking publishes atomically and fails rather than replacing an existing file.
        os.link(temporary_name, destination)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return template, destination, notes
