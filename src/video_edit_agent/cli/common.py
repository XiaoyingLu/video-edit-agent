"""Shared CLI application and command dependencies."""

# ruff: noqa: F401

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ..artifact_tracking import register_artifact, validate_all_artifacts
from ..config import get_setting
from ..editing import (
    IntentParsingError,
    apply_review_decisions,
    apply_timeline_review,
    build_draft_selection,
    build_draft_timeline,
    build_edit_proposal,
    build_project_timeline,
    create_video_project_plan,
    load_draft_timeline,
    load_edit_proposal,
    parse_editing_intent,
    save_draft_timeline,
    save_edit_proposal,
)
from ..editing.embeddings import EmbeddingModelError
from ..editing.footage_mapping import suggest_storyboard_footage
from ..editing.footage_review import review_storyboard_footage
from ..editing.footage_schemas import ProjectFootageSuggestions
from ..editing.planning_generation import VideoPlanningError
from ..editing.review_workspace import generate_review_workspace
from ..editing.revision_cycle import (
    RevisionCycleError,
    build_revision_cycle_report,
    save_revision_cycle_report,
)
from ..editing.revision_preview import render_timeline_revision_preview
from ..editing.revision_review import (
    RevisionSuggestionReviewError,
    review_draft_revision_suggestions,
)
from ..editing.revision_suggestions import (
    RevisionSuggestionError,
    build_draft_revision_suggestions,
    save_draft_revision_suggestions,
)
from ..editing.revision_visual_review import (
    TimelineRevisionVisualReviewError,
    build_timeline_revision_visual_review,
    save_timeline_revision_visual_review,
)
from ..editing.revisions import (
    TimelineIntervalEdit,
    TimelineRevisionError,
    compare_timeline_files,
    create_timeline_revision,
)
from ..editing.storyboard_production import (
    approve_storyboard,
    build_storyboard_timeline,
    save_storyboard_timeline,
)
from ..editing.storyboard_review import generate_storyboard_review
from ..editing.timeline_schemas import ProjectDraftTimeline
from ..editing.timeline_visual_editor import (
    apply_timeline_visual_edits,
    load_timeline_visual_edits,
)
from ..editing.timeline_visual_review import (
    TimelineVisualReviewError,
    build_timeline_visual_review,
    load_timeline_visual_review_decisions,
    save_timeline_visual_review,
)
from ..editing.workflow import create_video_draft
from ..highlights import find_highlights_folder
from ..media import MediaToolError, analyze_folder
from ..music import analyze_music
from ..music.pipeline import MusicAnalysisError
from ..music.refinement import apply_music_sync_review
from ..music.sync import MusicSyncSuggestions, save_music_sync_suggestions, suggest_music_sync
from ..music.timeline_preview import (
    build_music_timeline_preview,
    save_music_timeline_preview,
)
from ..rendering import (
    DraftComparisonError,
    DraftCritiqueError,
    RenderError,
    compare_rendered_drafts,
    critique_rendered_draft,
    render_approved_shot_timeline,
    render_comparison_html,
    render_project,
)
from ..rendering.title_overlays import (
    TitleOverlayDocument,
    load_title_overlays,
    review_title_overlays,
    save_title_overlays,
    suggest_narrative_title_sequences,
)
from ..silence import analyze_silence_folder
from ..silence.ffmpeg_detector import SilenceDetectionError
from ..soundtrack import create_soundtrack_brief, generate_soundtrack_from_brief
from ..soundtrack.analysis import analyze_selected_soundtrack
from ..soundtrack.generated import prepare_generated_soundtrack_candidates
from ..soundtrack.review import (
    build_soundtrack_candidate_review,
    save_soundtrack_candidate_review,
    search_soundstripe_catalog,
    select_soundtrack_candidate,
)
from ..storage_cleanup import cleanup_generated_storage
from ..story_proposal import (
    StoryProposalError,
    propose_story,
    select_story_angle,
)
from ..subtitles import SubtitleError, generate_subtitles
from ..transcription import transcribe_folder
from ..transcription.whisper_backend import TranscriptionError
from ..understanding import understand_folder
from ..understanding.visual_analyzer import VisionModelError

app = typer.Typer(
    name="video-agent",
    help="Analyze and edit local personal video footage with an AI-assisted workflow.",
    no_args_is_help=True,
)
console = Console()
DEFAULT_VISION_PROVIDER = get_setting("VISION_PROVIDER")
DEFAULT_FOOTAGE_EMBEDDING_MODEL = get_setting("FOOTAGE_EMBEDDING_MODEL")
_MUSIC_ANALYSIS_OPTION = typer.Option(
    None, "--music-analysis", help="Optional soundtrack energy analysis JSON path."
)


def _track_artifact(
    folder: Path,
    artifact: Path,
    dependencies: tuple[str, ...] = (),
    config: dict | None = None,
) -> None:
    """Register an artifact and whichever optional upstream files are present."""
    root = folder.expanduser().resolve()
    if not artifact.is_file():
        return
    existing = [root / item for item in dependencies if (root / item).is_file()]
    register_artifact(root, artifact, dependencies=existing, config=config)


@app.callback()
def main() -> None:
    """Analyze and edit local personal video footage."""
