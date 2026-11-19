"""Deterministic compression of project analysis into a compact story ledger."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

from ..highlights.schemas import ProjectHighlights
from ..models import ProjectAnalysis
from ..transcription.schemas import ProjectTranscription
from ..understanding.schemas import VideoScene, VideoUnderstanding


@dataclass(frozen=True)
class StoryLedgerEntry:
    """One compact, grounded line describing a source scene."""

    scene_id: str
    video_path: str
    day_label: str
    timestamp_label: str
    setting: str
    subject: str
    shot_scale: str
    mood: str
    editorial_role: str
    motion: str
    highlight_score: float | None
    speech: tuple[str, ...]
    start_seconds: float
    end_seconds: float

    def render(self) -> str:
        """Render a concise, stable, human- and model-readable scene line."""
        setting_subject = ", ".join(value for value in (self.setting, self.subject) if value)
        scale = self.shot_scale or "unknown"
        mood = self.mood or "unknown"
        role = self.editorial_role or "unknown"
        motion = self.motion or "unknown"
        highlight = (
            f"{self.highlight_score:.2f}" if self.highlight_score is not None else "n/a"
        )
        speech = " ".join(self.speech)
        speech = speech.replace('"', "'").replace("\n", " ").strip()
        if len(speech) > 180:
            speech = speech[:177].rstrip() + "..."
        speech_text = f' | speech: "{speech}"' if speech else ""
        return (
            f"{self.scene_id} | {self.day_label} {self.timestamp_label} | "
            f"{setting_subject or 'setting unknown'} | {scale} | {mood} | "
            f"role: {role} | motion: {motion} | highlight {highlight}{speech_text}"
        )


@dataclass
class StoryLedgerCluster:
    """A deterministic group of related scenes, usually from one day/location."""

    cluster_id: str
    day_label: str
    label: str
    entries: list[StoryLedgerEntry] = field(default_factory=list)

    @property
    def scene_ids(self) -> list[str]:
        return [entry.scene_id for entry in self.entries]

    def render(self) -> str:
        lines = [f"[{self.cluster_id} | {self.day_label} | {self.label}]"]
        lines.extend(f"  {entry.render()}" for entry in self.entries)
        return "\n".join(lines)


@dataclass
class StoryLedger:
    """Compact story context with the original scene IDs preserved."""

    entries: list[StoryLedgerEntry]
    clusters: list[StoryLedgerCluster]
    timestamp_note: str = (
        "Day/time labels use source-file modification timestamps plus scene offsets; "
        "they are an ordering hint, not guaranteed camera capture metadata."
    )

    @property
    def scene_ids(self) -> set[str]:
        return {entry.scene_id for entry in self.entries}

    def render(self) -> str:
        """Return the grouped ledger text intended for downstream story prompting."""
        sections = [
            "STORY LEDGER",
            self.timestamp_note,
            f"{len(self.entries)} scenes grouped into {len(self.clusters)} clusters.",
        ]
        sections.extend(cluster.render() for cluster in self.clusters)
        return "\n\n".join(sections)


def _video_key(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]


def _clean(value: str | None) -> str:
    return " ".join((value or "").strip().split())


def _first_known(values: Iterable[str]) -> str:
    for value in values:
        normalized = _clean(value)
        if normalized and normalized.lower() != "unknown":
            return normalized
    return ""


def _scene_attributes(scene: VideoScene) -> tuple[str, str, str, str]:
    observations = scene.observations
    setting = _first_known(item.setting for item in observations)
    subject = _first_known(item.primary_subject for item in observations)
    scales = list(
        dict.fromkeys(
            item.shot_scale for item in observations if item.shot_scale != "unknown"
        )
    )
    scale = "→".join(scales) if scales else "unknown"
    motion_parts: list[str] = []
    if scene.dynamics is not None:
        for value in (
            scene.dynamics.subject_motion,
            scene.dynamics.camera_motion,
        ):
            if value != "unknown" and value not in motion_parts:
                motion_parts.append(value)
    motion = "/".join(motion_parts) if motion_parts else "unknown"
    return setting, subject, scale, motion


def _overlap(start: float, end: float, other_start: float, other_end: float) -> bool:
    return start < other_end and other_start < end


def _build_entries(
    project_root: Path,
    understanding: VideoUnderstanding,
    transcription: ProjectTranscription | None,
    highlights: ProjectHighlights | None,
    analysis: ProjectAnalysis,
) -> list[StoryLedgerEntry]:
    media_by_key = {_video_key(item.path): item for item in analysis.videos}
    media_by_path = {item.path: item for item in analysis.videos}
    transcript_by_path = (
        {item.path: item for item in transcription.videos} if transcription else {}
    )
    candidates_by_path: dict[str, list] = defaultdict(list)
    if highlights is not None:
        for candidate in highlights.candidates:
            candidates_by_path[candidate.video_path].append(candidate)

    timed_scenes: list[tuple[datetime, str, VideoScene, str]] = []
    for scene in understanding.scenes:
        video_key = scene.scene_id.split("-scene-", maxsplit=1)[0]
        media = media_by_key.get(video_key)
        if media is None:
            continue
        source_path = project_root / media.path
        try:
            modified = datetime.fromtimestamp(source_path.stat().st_mtime).astimezone()
        except OSError:
            modified = datetime.fromtimestamp(0, tz=ZoneInfo("UTC"))
        scene_time = modified.timestamp() + scene.start_seconds
        timestamp = datetime.fromtimestamp(scene_time).astimezone()
        timed_scenes.append((timestamp, media.path, scene, timestamp.strftime("%H:%M")))

    timed_scenes.sort(key=lambda item: (item[0], item[1], item[2].start_seconds))
    day_numbers: dict[str, int] = {}
    entries: list[StoryLedgerEntry] = []
    for timestamp, video_path, scene, time_label in timed_scenes:
        date_key = timestamp.date().isoformat()
        if date_key not in day_numbers:
            day_numbers[date_key] = len(day_numbers) + 1
        day_label = f"day{day_numbers[date_key]}"

        setting, subject, scale, motion = _scene_attributes(scene)
        transcript = transcript_by_path.get(video_path)
        speech: list[str] = []
        if transcript is not None:
            speech = [
                segment.text
                for segment in transcript.segments
                if segment.text
                and _overlap(
                    scene.start_seconds,
                    scene.end_seconds,
                    segment.start_seconds,
                    segment.end_seconds,
                )
            ]

        scores = [
            candidate.score
            for candidate in candidates_by_path.get(video_path, [])
            if _overlap(
                scene.start_seconds,
                scene.end_seconds,
                candidate.start_seconds,
                candidate.end_seconds,
            )
        ]
        entries.append(
            StoryLedgerEntry(
                scene_id=scene.scene_id,
                video_path=video_path,
                day_label=day_label,
                timestamp_label=time_label,
                setting=setting,
                subject=subject,
                shot_scale=scale,
                mood=scene.dominant_mood,
                editorial_role=scene.story_function,
                motion=motion,
                highlight_score=max(scores) / 100 if scores else None,
                speech=tuple(speech),
                start_seconds=scene.start_seconds,
                end_seconds=scene.end_seconds,
            )
        )
    return entries


def _cluster_entries(entries: list[StoryLedgerEntry]) -> list[StoryLedgerCluster]:
    """Group same-day scenes by matching setting/subject, preserving chronology."""
    clusters: list[StoryLedgerCluster] = []
    by_day: dict[str, list[StoryLedgerEntry]] = defaultdict(list)
    for entry in entries:
        by_day[entry.day_label].append(entry)

    for day_label, day_entries in by_day.items():
        day_clusters: list[StoryLedgerCluster] = []
        for entry in day_entries:
            setting = entry.setting.casefold()
            subject = entry.subject.casefold()
            matching = next(
                (
                    cluster
                    for cluster in day_clusters
                    if (setting and setting in cluster.label.casefold())
                    or (subject and subject in cluster.label.casefold())
                    or any(
                        (setting and setting == item.setting.casefold())
                        or (subject and subject == item.subject.casefold())
                        for item in cluster.entries
                    )
                ),
                None,
            )
            if matching is None:
                label = entry.setting or entry.subject or Path(entry.video_path).stem
                matching = StoryLedgerCluster(
                    cluster_id=f"{day_label}-block-{len(day_clusters) + 1:02d}",
                    day_label=day_label,
                    label=label[:80],
                )
                day_clusters.append(matching)
            matching.entries.append(entry)
        clusters.extend(day_clusters)
    return clusters


def build_story_ledger(
    project_root: Path,
    understanding: VideoUnderstanding,
    analysis: ProjectAnalysis,
    *,
    transcription: ProjectTranscription | None = None,
    highlights: ProjectHighlights | None = None,
) -> StoryLedger:
    """Build a compact deterministic ledger from validated project artifacts.

    Scene IDs and scene intervals always come from the understanding artifact.
    Optional speech and highlight context is joined by source path and time overlap.
    """
    root = project_root.expanduser().resolve()
    analysis_paths = {item.path for item in analysis.videos}
    unknown_scene_keys = {
        scene.scene_id.split("-scene-", maxsplit=1)[0] for scene in understanding.scenes
    } - {_video_key(path) for path in analysis_paths}
    if unknown_scene_keys:
        raise ValueError(
            "Understanding contains scenes whose source videos are absent from analysis: "
            + ", ".join(sorted(unknown_scene_keys))
        )
    entries = _build_entries(root, understanding, transcription, highlights, analysis)
    return StoryLedger(entries=entries, clusters=_cluster_entries(entries))
