"""Reviewable beat-alignment suggestions for an approved or proposed shot timeline."""

from bisect import bisect_left
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..artifact_tracking import validate_artifact
from ..atomic_io import write_json_atomic
from ..editing.timeline_schemas import ProjectDraftTimeline
from .schemas import MusicAnalysis


class MusicSyncBoundary(BaseModel):
    """One proposed transition boundary compared with nearby musical anchors."""

    model_config = ConfigDict(extra="forbid")

    previous_clip_id: str
    next_clip_id: str
    boundary_seconds: float = Field(ge=0)
    transition: Literal["cut", "crossfade", "dissolve"]
    transition_duration_seconds: float = Field(ge=0)
    nearest_beat_seconds: float | None = Field(default=None, ge=0)
    beat_delta_seconds: float | None = None
    nearest_downbeat_seconds: float | None = Field(default=None, ge=0)
    downbeat_delta_seconds: float | None = None
    recommended_anchor: Literal["none", "beat", "downbeat"] = "none"
    recommended_boundary_seconds: float | None = Field(default=None, ge=0)
    adjustment_seconds: float | None = None


class MusicSyncSuggestions(BaseModel):
    """Non-destructive comparison of timeline boundaries with soundtrack beats."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    project_path: str
    timeline_path: str
    music_analysis_path: str
    music_path: str
    bpm: float | None = None
    beat_confidence: float = Field(ge=0, le=1)
    tolerance_seconds: float = Field(gt=0, le=1)
    boundary_count: int = Field(ge=0)
    aligned_boundary_count: int = Field(ge=0)
    boundaries: list[MusicSyncBoundary] = Field(default_factory=list)
    review_required: Literal[True] = True
    timeline_modified: Literal[False] = False
    notes: list[str] = Field(default_factory=list)


class MusicSyncError(ValueError):
    """Raised when music sync suggestions cannot be generated safely."""


def _nearest(values: list[float], target: float) -> float | None:
    if not values:
        return None
    index = bisect_left(values, target)
    candidates = []
    if index < len(values):
        candidates.append(values[index])
    if index > 0:
        candidates.append(values[index - 1])
    return min(candidates, key=lambda value: abs(value - target))


def suggest_music_sync(
    folder: Path,
    *,
    timeline_path: Path | None = None,
    music_analysis_path: Path | None = None,
    tolerance_seconds: float = 0.18,
) -> MusicSyncSuggestions:
    """Suggest nearby beat/downbeat anchors without changing timeline durations."""
    if not 0 < tolerance_seconds <= 1:
        raise MusicSyncError("tolerance_seconds must be greater than 0 and at most 1 second")
    root = folder.expanduser().resolve()
    timeline_file = (
        (timeline_path or root / ".video-agent" / "optimized-timeline.json").expanduser().resolve()
    )
    music_file = (
        (music_analysis_path or root / ".video-agent" / "music-analysis.json")
        .expanduser()
        .resolve()
    )
    for path, label in (
        (timeline_file, "timeline"),
        (music_file, "music analysis"),
    ):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise MusicSyncError(f"{label} artifact must be inside the project folder") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} artifact: {path}")
        validate_artifact(root, path)

    timeline = ProjectDraftTimeline.model_validate_json(timeline_file.read_text(encoding="utf-8"))
    music = MusicAnalysis.model_validate_json(music_file.read_text(encoding="utf-8"))
    if Path(timeline.project_path).resolve() != root:
        raise MusicSyncError("optimized timeline belongs to a different project folder")
    if Path(music.project_path).resolve() != root:
        raise MusicSyncError("music analysis belongs to a different project folder")
    soundtrack = Path(music.music_path).expanduser().resolve()
    if not soundtrack.is_file():
        raise FileNotFoundError(f"Analyzed soundtrack no longer exists: {soundtrack}")

    boundaries: list[MusicSyncBoundary] = []
    elapsed = 0.0
    aligned_count = 0
    beats = sorted(music.beat_times_seconds)
    downbeats = sorted(music.downbeat_times_seconds)
    for index, clip in enumerate(timeline.clips):
        if index:
            transition_duration = clip.transition_duration_seconds
            boundary = elapsed - transition_duration / 2
            beat = _nearest(beats, boundary)
            downbeat = _nearest(downbeats, boundary)
            beat_delta = beat - boundary if beat is not None else None
            downbeat_delta = downbeat - boundary if downbeat is not None else None

            candidates = []
            if beat_delta is not None and abs(beat_delta) <= tolerance_seconds:
                candidates.append((abs(beat_delta), "beat", beat, beat_delta))
            if downbeat_delta is not None and abs(downbeat_delta) <= tolerance_seconds:
                candidates.append((abs(downbeat_delta), "downbeat", downbeat, downbeat_delta))
            recommendation = min(
                candidates,
                default=None,
                key=lambda item: (item[0], 0 if item[1] == "downbeat" else 1),
            )
            if recommendation is not None and music.beat_confidence >= 0.1:
                _, anchor, recommended_time, adjustment = recommendation
                aligned_count += 1
            else:
                anchor, recommended_time, adjustment = "none", None, None

            boundaries.append(
                MusicSyncBoundary(
                    previous_clip_id=timeline.clips[index - 1].clip_id,
                    next_clip_id=clip.clip_id,
                    boundary_seconds=round(max(0.0, boundary), 3),
                    transition=clip.transition_in,
                    transition_duration_seconds=clip.transition_duration_seconds,
                    nearest_beat_seconds=beat,
                    beat_delta_seconds=round(beat_delta, 3) if beat_delta is not None else None,
                    nearest_downbeat_seconds=downbeat,
                    downbeat_delta_seconds=(
                        round(downbeat_delta, 3) if downbeat_delta is not None else None
                    ),
                    recommended_anchor=anchor,
                    recommended_boundary_seconds=(
                        round(recommended_time, 3) if recommended_time is not None else None
                    ),
                    adjustment_seconds=round(adjustment, 3) if adjustment is not None else None,
                )
            )
            elapsed += clip.duration_seconds - transition_duration
        else:
            elapsed += clip.duration_seconds

    notes = [
        "Suggestions compare each transition midpoint (or hard-cut boundary) "
        "with the nearest detected beat.",
        "No clip duration, source interval, transition, or render artifact is modified.",
        "A recommendation is shown only when an anchor is within the configured tolerance "
        "and beat confidence is at least 0.10.",
        "Review the soundtrack and footage before applying any timing change; "
        "signal-based beat estimates can be imperfect.",
    ]
    if not beats:
        notes.append(
            "No stable beat grid is available; no beat-alignment recommendations were made."
        )
    return MusicSyncSuggestions(
        project_path=str(root),
        timeline_path=timeline_file.relative_to(root).as_posix(),
        music_analysis_path=music_file.relative_to(root).as_posix(),
        music_path=str(soundtrack),
        bpm=music.bpm,
        beat_confidence=music.beat_confidence,
        tolerance_seconds=tolerance_seconds,
        boundary_count=len(boundaries),
        aligned_boundary_count=aligned_count,
        boundaries=boundaries,
        notes=notes,
    )


def save_music_sync_suggestions(suggestions: MusicSyncSuggestions, destination: Path) -> None:
    """Write the report as a standalone JSON artifact."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(destination, suggestions)
