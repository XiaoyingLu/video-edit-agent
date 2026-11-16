"""Build deterministic edit proposals from existing analysis artifacts."""

import hashlib
from pathlib import Path

from ..silence.schemas import ProjectSilenceAnalysis
from ..understanding.schemas import VideoUnderstanding
from .schemas import EditReviewItem, ProjectEditProposal, VideoEditProposal


def _scene_context(
    start_seconds: float,
    end_seconds: float,
    scenes: list,
) -> tuple[str | None, str | None]:
    """Return the scene with the greatest temporal overlap, if any."""
    overlapping = [
        scene
        for scene in scenes
        if scene.start_seconds < end_seconds and scene.end_seconds > start_seconds
    ]
    if not overlapping:
        return None, None
    scene = max(
        overlapping,
        key=lambda item: max(
            0.0,
            min(end_seconds, item.end_seconds) - max(start_seconds, item.start_seconds),
        ),
    )
    return scene.scene_id, scene.summary


def build_edit_proposal(folder: Path) -> ProjectEditProposal:
    """Create a review-only proposal; never modify or render source media.

    Requires .video-agent/silence-analysis.json. Scene descriptions are optional
    and are used only as context for human review, never to auto-approve a cut.
    """
    root = folder.expanduser().resolve()
    artifact_dir = root / ".video-agent"
    silence_path = artifact_dir / "silence-analysis.json"
    if not silence_path.is_file():
        raise FileNotFoundError(
            f"Missing {silence_path}; run 'video-agent detect-silence' first."
        )

    silence = ProjectSilenceAnalysis.model_validate_json(
        silence_path.read_text(encoding="utf-8")
    )
    understanding_path = artifact_dir / "understanding.json"
    understanding = (
        VideoUnderstanding.model_validate_json(
            understanding_path.read_text(encoding="utf-8")
        )
        if understanding_path.is_file()
        else None
    )

    scenes_by_video: dict[str, list] = {}
    if understanding is not None:
        for video in silence.videos:
            video_key = hashlib.sha256(video.path.encode("utf-8")).hexdigest()[:12]
            scenes_by_video[video.path] = [
                scene
                for scene in understanding.scenes
                if scene.scene_id.startswith(f"{video_key}-")
            ]

    videos: list[VideoEditProposal] = []
    item_count = 0
    for video in silence.videos:
        scenes = scenes_by_video.get(video.path, [])
        items: list[EditReviewItem] = []
        for index, candidate in enumerate(video.candidates, start=1):
            scene_id, scene_summary = _scene_context(
                candidate.start_seconds, candidate.end_seconds, scenes
            )
            items.append(
                EditReviewItem(
                    item_id=f"{Path(video.path).stem}-review-{index:03d}",
                    start_seconds=candidate.start_seconds,
                    end_seconds=candidate.end_seconds,
                    duration_seconds=candidate.duration_seconds,
                    review_reason=(
                        "Quiet audio interval with no overlapping transcript speech. "
                        "Inspect the picture and listen for ambience, music, wind, or "
                        "intentional scenic pauses before deciding."
                    ),
                    scene_id=scene_id,
                    scene_summary=scene_summary,
                )
            )
        item_count += len(items)
        videos.append(
            VideoEditProposal(
                path=video.path,
                duration_seconds=video.duration_seconds,
                review_items=items,
            )
        )

    sources = [".video-agent/silence-analysis.json"]
    if understanding is not None:
        sources.append(".video-agent/understanding.json")
    return ProjectEditProposal(
        project_path=str(root),
        source_artifacts=sources,
        video_count=len(videos),
        review_item_count=item_count,
        videos=videos,
    )
