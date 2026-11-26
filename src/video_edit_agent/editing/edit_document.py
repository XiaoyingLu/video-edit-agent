"""Canonical, renderer-independent edit document from a reviewed shot timeline."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .timeline_schemas import ProjectDraftTimeline


class EditTrackClip(BaseModel):
    """One media or title item with source and output time ranges."""

    model_config = ConfigDict(extra="forbid")

    clip_id: str
    kind: Literal["video", "audio", "title"]
    source_path: str | None = None
    source_start_seconds: float | None = Field(default=None, ge=0)
    source_end_seconds: float | None = Field(default=None, gt=0)
    timeline_start_seconds: float = Field(ge=0)
    timeline_end_seconds: float = Field(gt=0)
    transition_in: str | None = None
    transition_duration_seconds: float = Field(default=0, ge=0, le=2)
    text: str | None = None
    gain_db: float | None = Field(default=None, ge=-60, le=24)
    decision: Literal["pending", "approved", "rejected"] = "pending"

    @model_validator(mode="after")
    def validate_ranges(self) -> "EditTrackClip":
        if self.timeline_end_seconds <= self.timeline_start_seconds:
            raise ValueError("timeline end must be greater than timeline start")
        if self.kind in {"video", "audio"}:
            media_fields = (
                self.source_path,
                self.source_start_seconds,
                self.source_end_seconds,
            )
            if any(value is None for value in media_fields):
                raise ValueError(
                    "media clips require a source path and complete source interval"
                )
            if self.source_end_seconds <= self.source_start_seconds:
                raise ValueError("source end must be greater than source start")
        if self.kind == "title" and not self.text:
            raise ValueError("title clips require text")
        return self


class EditTrack(BaseModel):
    """An ordered lane in the edit."""

    model_config = ConfigDict(extra="forbid")

    track_id: str
    kind: Literal["video", "audio", "title"]
    name: str
    clips: list[EditTrackClip] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_clip_kinds(self) -> "EditTrack":
        if any(clip.kind != self.kind for clip in self.clips):
            raise ValueError("track clips must match the track kind")
        return self


class StructuredEditDocument(BaseModel):
    """Versioned project edit representation with explicit tracks and output timing."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    project_path: str
    source_timeline: str
    aspect_ratio: Literal["16:9", "9:16", "1:1"]
    style: str
    pacing: str
    duration_seconds: float = Field(ge=0)
    approved: bool = False
    tracks: list[EditTrack] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_tracks(self) -> "StructuredEditDocument":
        ids = [track.track_id for track in self.tracks]
        if len(ids) != len(set(ids)):
            raise ValueError("track IDs must be unique")
        video_tracks = [track for track in self.tracks if track.kind == "video"]
        if len(video_tracks) != 1:
            raise ValueError("an edit document must contain exactly one primary video track")
        for track in self.tracks:
            previous_start = -1.0
            for clip in track.clips:
                if clip.timeline_start_seconds < previous_start:
                    raise ValueError("track clips must be ordered by timeline start")
                previous_start = clip.timeline_start_seconds
                if clip.timeline_end_seconds > self.duration_seconds + 0.02:
                    raise ValueError("track clip exceeds document duration")
        return self


def build_edit_document(
    timeline: ProjectDraftTimeline, *, source_timeline: str
) -> StructuredEditDocument:
    """Map a reviewed shot list into explicit video, audio and title tracks."""
    video_clips: list[EditTrackClip] = []
    audio_clips: list[EditTrackClip] = []
    title_clips: list[EditTrackClip] = []
    cursor = 0.0
    for clip in timeline.clips:
        overlap = clip.transition_duration_seconds if clip.transition_in != "cut" else 0
        start = max(0.0, cursor - overlap)
        end = start + clip.duration_seconds
        common = {
            "clip_id": clip.clip_id,
            "source_path": clip.video_path,
            "source_start_seconds": clip.source_start_seconds,
            "source_end_seconds": clip.source_end_seconds,
            "timeline_start_seconds": round(start, 3),
            "timeline_end_seconds": round(end, 3),
            "transition_in": clip.transition_in,
            "transition_duration_seconds": clip.transition_duration_seconds,
            "decision": clip.decision,
        }
        video_clips.append(EditTrackClip(kind="video", **common))
        audio_clips.append(EditTrackClip(kind="audio", **common, gain_db=clip.gain_db))
        if clip.caption:
            title_clips.append(
                EditTrackClip(
                    clip_id=f"{clip.clip_id}:caption",
                    kind="title",
                    timeline_start_seconds=round(start, 3),
                    timeline_end_seconds=round(end, 3),
                    text=clip.caption,
                    decision=clip.decision,
                )
            )
        cursor = end
    tracks = [
        EditTrack(track_id="video-main", kind="video", name="Main video", clips=video_clips),
        EditTrack(track_id="audio-source", kind="audio", name="Source audio", clips=audio_clips),
    ]
    if title_clips:
        tracks.append(
            EditTrack(
                track_id="titles",
                kind="title",
                name="Titles & captions",
                clips=title_clips,
            )
        )
    return StructuredEditDocument(
        project_path=timeline.project_path,
        source_timeline=source_timeline,
        aspect_ratio=timeline.aspect_ratio,
        style=timeline.style,
        pacing=timeline.pacing,
        duration_seconds=round(timeline.proposed_duration_seconds, 3),
        approved=timeline.timeline_approved,
        tracks=tracks,
    )
