"""Build deterministic edit proposals and validated retained timelines."""

import hashlib
from pathlib import Path

from ..artifact_tracking import validate_artifact
from ..atomic_io import write_json_atomic
from ..silence.schemas import ProjectSilenceAnalysis
from ..understanding.schemas import VideoScene, VideoUnderstanding
from .schemas import (
    EditReviewItem,
    ProjectEditProposal,
    ProjectTimeline,
    TimelineSegment,
    VideoEditProposal,
    VideoTimeline,
)


def _scene_context(
    start_seconds: float,
    end_seconds: float,
    scenes: list[VideoScene],
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
        raise FileNotFoundError(f"Missing {silence_path}; run 'video-agent detect-silence' first.")

    validate_artifact(root, silence_path)
    silence = ProjectSilenceAnalysis.model_validate_json(silence_path.read_text(encoding="utf-8"))
    understanding_path = artifact_dir / "understanding.json"
    if understanding_path.is_file():
        validate_artifact(root, understanding_path)
        understanding = VideoUnderstanding.model_validate_json(
            understanding_path.read_text(encoding="utf-8")
        )
    else:
        understanding = None

    scenes_by_video: dict[str, list[VideoScene]] = {}
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


def load_edit_proposal(
    folder: Path, proposal_path: Path | None = None
) -> tuple[ProjectEditProposal, Path]:
    """Load the saved review artifact and return its resolved path."""
    root = folder.expanduser().resolve()
    destination = (
        (proposal_path or root / ".video-agent" / "edit-proposal.json").expanduser().resolve()
    )
    if not destination.is_file():
        raise FileNotFoundError(f"Missing {destination}; run 'video-agent propose-edits' first.")
    validate_artifact(root, destination)
    proposal = ProjectEditProposal.model_validate_json(destination.read_text(encoding="utf-8"))
    if Path(proposal.project_path).resolve() != root:
        raise ValueError("edit proposal belongs to a different project folder")
    return proposal, destination


def save_edit_proposal(proposal: ProjectEditProposal, destination: Path) -> None:
    """Persist review decisions without touching source media."""
    write_json_atomic(destination, proposal)


def apply_review_decisions(
    proposal: ProjectEditProposal,
    approve_ids: set[str],
    reject_ids: set[str],
) -> ProjectEditProposal:
    """Apply explicit decisions, rejecting unknown or conflicting item IDs."""
    if approve_ids & reject_ids:
        raise ValueError("an item cannot be both approved and rejected")
    all_items = [item for video in proposal.videos for item in video.review_items]
    item_ids = [item.item_id for item in all_items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("review item IDs must be unique across the project")
    items = {item.item_id: item for item in all_items}
    requested = approve_ids | reject_ids
    unknown = requested - items.keys()
    if unknown:
        raise ValueError(f"unknown review item ID(s): {', '.join(sorted(unknown))}")
    for video in proposal.videos:
        for item in video.review_items:
            if item.item_id in approve_ids:
                item.decision = "approved"
            elif item.item_id in reject_ids:
                item.decision = "rejected"
    return proposal


def build_project_timeline(proposal: ProjectEditProposal) -> ProjectTimeline:
    """Build retained source intervals after every candidate has an explicit decision."""
    pending = [
        item.item_id
        for video in proposal.videos
        for item in video.review_items
        if item.decision == "pending"
    ]
    if pending:
        raise ValueError(
            "all review items must be approved or rejected before building a timeline; "
            f"pending: {', '.join(pending)}"
        )

    timelines: list[VideoTimeline] = []
    approved_count = 0
    project_retained = 0.0
    all_item_ids = [item.item_id for video in proposal.videos for item in video.review_items]
    if len(all_item_ids) != len(set(all_item_ids)):
        raise ValueError("review item IDs must be unique across the project")

    for video in proposal.videos:
        approved = sorted(
            (item for item in video.review_items if item.decision == "approved"),
            key=lambda item: (item.start_seconds, item.end_seconds),
        )
        previous_end = 0.0
        retained: list[TimelineSegment] = []
        for item in approved:
            # Revalidate because Pydantic assignment validation is intentionally not required
            # for the editable review artifact.
            EditReviewItem.model_validate(item.model_dump())
            if item.end_seconds > video.duration_seconds + 0.01:
                raise ValueError(f"approved cut {item.item_id} exceeds source duration")
            if item.start_seconds < previous_end - 0.01:
                raise ValueError(f"approved cuts overlap in {video.path}")
            if item.start_seconds > previous_end:
                retained.append(
                    TimelineSegment(start_seconds=previous_end, end_seconds=item.start_seconds)
                )
            previous_end = item.end_seconds
            approved_count += 1
        if previous_end < video.duration_seconds:
            retained.append(
                TimelineSegment(start_seconds=previous_end, end_seconds=video.duration_seconds)
            )
        retained_duration = sum(segment.end_seconds - segment.start_seconds for segment in retained)
        project_retained += retained_duration
        timelines.append(
            VideoTimeline(
                path=video.path,
                source_duration_seconds=video.duration_seconds,
                retained_duration_seconds=retained_duration,
                segments=retained,
            )
        )

    return ProjectTimeline(
        project_path=proposal.project_path,
        approved_cut_count=approved_count,
        retained_duration_seconds=project_retained,
        videos=timelines,
    )
