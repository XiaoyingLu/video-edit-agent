"""Validated, data-only reusable editing templates."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator

_TEMPLATE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_ALLOWED_SLOTS = {"title", "logline", "closing_line", "date", "location"}


class TypographySpec(BaseModel):
    """Typography tokens; font values are names, never executable instructions."""

    model_config = ConfigDict(extra="forbid")

    font: str = "Lato"
    color: str = "#FFFFFF"
    accent: str = "#F2B544"

    @model_validator(mode="after")
    def validate_colors(self) -> TypographySpec:
        for name in ("color", "accent"):
            if not _COLOR.fullmatch(getattr(self, name)):
                raise ValueError(f"typography.{name} must be a six-digit hex color")
        if not self.font.strip() or any(ord(char) < 32 for char in self.font):
            raise ValueError("typography.font must be a non-empty font family or filename")
        return self


class TitleStyleSpec(BaseModel):
    """A constrained title style expressed only through built-in renderer options."""

    model_config = ConfigDict(extra="forbid")

    placement: Literal["top", "center", "bottom"] = "center"
    size: float = Field(default=1.0, ge=0.25, le=3.0)
    box: float = Field(default=0.0, ge=0.0, le=1.0)
    animation: Literal["none", "fade", "slide_up", "slide_left"] = "fade"
    accent_bar: bool = False


class SequenceItemSpec(BaseModel):
    """A content slot and the named style to apply to it."""

    model_config = ConfigDict(extra="forbid")

    slot: str
    style: str

    @model_validator(mode="after")
    def validate_slot(self) -> SequenceItemSpec:
        if self.slot not in {f"{{{slot}}}" for slot in _ALLOWED_SLOTS}:
            raise ValueError(f"unsupported template slot {self.slot!r}")
        return self


class TransitionSpec(BaseModel):
    """Transition names mapped later to the renderer's built-in transition allowlist."""

    model_config = ConfigDict(extra="forbid")

    default: Literal["cut", "crossfade", "fadeblack", "wipeleft", "slideleft", "circleopen"] = (
        "crossfade"
    )
    on_scene_change: Literal[
        "cut", "crossfade", "fadeblack", "wipeleft", "slideleft", "circleopen"
    ] = "fadeblack"
    on_beat: Literal["cut", "crossfade", "fadeblack", "wipeleft", "slideleft", "circleopen"] = "cut"


class MusicSpec(BaseModel):
    """Safe soundtrack mix settings; no filter expressions are accepted."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["original", "music_only", "blend"] = "blend"
    volume: float = Field(default=0.18, ge=0.01, le=2.0)
    fade_in: float = Field(default=1.5, ge=0, le=30)
    fade_out: float = Field(default=2.5, ge=0, le=30)
    ducking: bool = True


class RenderSpec(BaseModel):
    """Non-title rendering options migrated from the existing named presets."""

    model_config = ConfigDict(extra="forbid")

    fit_mode: Literal["crop", "pad"] = "crop"
    beat_punch_in: bool = False


class TemplateSpec(BaseModel):
    """A validated, versioned visual template; never contains raw FFmpeg syntax."""

    model_config = ConfigDict(extra="forbid")

    id: str
    version: int = Field(ge=1)
    extends: str | None = None
    typography: TypographySpec = Field(default_factory=TypographySpec)
    title_styles: dict[str, TitleStyleSpec] = Field(default_factory=dict)
    sequences: dict[str, list[SequenceItemSpec]] = Field(default_factory=dict)
    transitions: TransitionSpec = Field(default_factory=TransitionSpec)
    music: MusicSpec = Field(default_factory=MusicSpec)
    render: RenderSpec = Field(default_factory=RenderSpec)
    tags: dict[str, list[str]] = Field(default_factory=dict)
    assets: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_template(self) -> TemplateSpec:
        if not _TEMPLATE_ID.fullmatch(self.id):
            raise ValueError("template id must contain lowercase letters, digits, and hyphens")
        if self.extends is not None and not _TEMPLATE_ID.fullmatch(self.extends):
            raise ValueError("extends must be a valid template id")
        for sequence_name, items in self.sequences.items():
            if not sequence_name.strip():
                raise ValueError("sequence names cannot be empty")
            for item in items:
                if item.style not in self.title_styles:
                    raise ValueError(
                        f"sequence {sequence_name!r} references unknown title style {item.style!r}"
                    )
        for category, values in self.tags.items():
            if not category.strip() or any(not value.strip() for value in values):
                raise ValueError("template tag categories and values must be non-empty")
        return self


def _merge(base: dict, override: dict) -> dict:
    """Deep-merge template data while replacing lists and scalar values."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _template_directories(
    project_root: Path | None,
    home_dir: Path | None,
    library_dir: Path | None,
) -> list[Path]:
    directories: list[Path] = []
    if project_root is not None:
        directories.append(project_root.expanduser().resolve() / ".video-agent" / "templates")
    home = (home_dir or Path.home()).expanduser().resolve()
    directories.append(home / ".video-agent" / "templates")
    directories.append(
        (library_dir or Path(__file__).with_name("template_library")).expanduser().resolve()
    )
    # Project templates override home templates, which override shipped templates.
    return directories


def _read_template(template_id: str, directories: list[Path]) -> tuple[dict, Path]:
    if not _TEMPLATE_ID.fullmatch(template_id):
        raise ValueError("template id must contain lowercase letters, digits, and hyphens")
    for directory in directories:
        for suffix in (".yaml", ".yml"):
            path = directory / f"{template_id}{suffix}"
            if path.is_file():
                try:
                    data = yaml.safe_load(path.read_text(encoding="utf-8"))
                except (OSError, yaml.YAMLError) as exc:
                    raise ValueError(f"could not read template {path}: {exc}") from exc
                if not isinstance(data, dict):
                    raise ValueError(f"template {path} must contain a YAML mapping")
                return data, path.resolve()
    raise FileNotFoundError(f"unknown template {template_id!r}")


def _resolve_template(
    template_id: str,
    directories: list[Path],
    stack: tuple[str, ...],
) -> tuple[TemplateSpec, Path]:
    if template_id in stack:
        chain = " -> ".join((*stack, template_id))
        raise ValueError(f"template inheritance cycle: {chain}")
    raw, source = _read_template(template_id, directories)
    declared_id = raw.get("id")
    if declared_id != template_id:
        raise ValueError(f"template filename {source.name!r} does not match id {declared_id!r}")
    assets = raw.get("assets", [])
    if not isinstance(assets, list) or not all(isinstance(asset, str) for asset in assets):
        raise ValueError("template assets must be a list of relative paths")
    _validate_template_assets(assets, source.parent)
    parent_id = raw.get("extends")
    if parent_id is None:
        effective = raw
    else:
        parent, _ = _resolve_template(parent_id, directories, (*stack, template_id))
        effective = _merge(parent.model_dump(mode="python"), raw)
        if "assets" not in raw:
            effective["assets"] = []
    spec = TemplateSpec.model_validate(effective)
    return spec, source


def _validate_template_assets(assets: list[str], template_dir: Path) -> None:
    root = template_dir.resolve()
    for asset in assets:
        relative = Path(asset)
        if relative.is_absolute() or "\\" in asset:
            raise ValueError("template asset paths must be relative POSIX paths")
        resolved = (root / relative).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"template asset escapes its template folder: {asset}") from exc
        if not resolved.is_file():
            raise FileNotFoundError(f"template asset does not exist: {resolved}")


def load_template(
    template_id: str,
    project_root: Path | None = None,
    *,
    home_dir: Path | None = None,
    library_dir: Path | None = None,
) -> tuple[TemplateSpec, Path]:
    """Load a project, home, or built-in template and resolve its inheritance."""
    directories = _template_directories(project_root, home_dir, library_dir)
    return _resolve_template(template_id, directories, ())
