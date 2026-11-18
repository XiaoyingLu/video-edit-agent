"""Command-line interface."""

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .artifact_tracking import register_artifact, validate_all_artifacts
from .editing import (
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
from .editing.embeddings import EmbeddingModelError
from .editing.footage_mapping import suggest_storyboard_footage
from .editing.footage_review import review_storyboard_footage
from .editing.planning_generation import VideoPlanningError
from .editing.revision_preview import render_timeline_revision_preview
from .editing.revision_review import (
    RevisionSuggestionReviewError,
    review_draft_revision_suggestions,
)
from .editing.revision_suggestions import (
    RevisionSuggestionError,
    build_draft_revision_suggestions,
    save_draft_revision_suggestions,
)
from .editing.revision_visual_review import (
    TimelineRevisionVisualReviewError,
    build_timeline_revision_visual_review,
    save_timeline_revision_visual_review,
)
from .editing.revisions import (
    TimelineIntervalEdit,
    TimelineRevisionError,
    compare_timeline_files,
    create_timeline_revision,
)
from .editing.storyboard_production import (
    approve_storyboard,
    build_storyboard_timeline,
    save_storyboard_timeline,
)
from .editing.storyboard_review import generate_storyboard_review
from .editing.timeline_schemas import ProjectDraftTimeline
from .editing.workflow import create_video_draft
from .highlights import find_highlights_folder
from .media import MediaToolError, analyze_folder
from .music import analyze_music
from .music.pipeline import MusicAnalysisError
from .music.refinement import apply_music_sync_review
from .music.sync import MusicSyncSuggestions, save_music_sync_suggestions, suggest_music_sync
from .rendering import (
    DraftComparisonError,
    DraftCritiqueError,
    RenderError,
    compare_rendered_drafts,
    critique_rendered_draft,
    render_approved_shot_timeline,
    render_comparison_html,
    render_project,
)
from .rendering.title_overlays import (
    TitleOverlayDocument,
    load_title_overlays,
    review_title_overlays,
    save_title_overlays,
    suggest_narrative_title_sequences,
)
from .silence import analyze_silence_folder
from .silence.ffmpeg_detector import SilenceDetectionError
from .storage_cleanup import cleanup_generated_storage
from .subtitles import SubtitleError, generate_subtitles
from .transcription import transcribe_folder
from .transcription.whisper_backend import TranscriptionError
from .understanding import understand_folder
from .understanding.visual_analyzer import VisionModelError

app = typer.Typer(
    name="video-agent",
    help="Analyze and edit local travel footage with an AI-assisted workflow.",
    no_args_is_help=True,
)
console = Console()
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
    """Analyze and edit local travel footage."""


@app.command()
def analyze(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Folder containing travel videos.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="JSON output path (default: <folder>/.video-agent/analysis.json).",
        ),
    ] = None,
) -> None:
    """Discover videos and extract metadata using ffprobe."""
    try:
        analysis = analyze_folder(folder)
    except (FileNotFoundError, NotADirectoryError, MediaToolError) as exc:
        console.print(f"[red]Analysis failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = output or folder.resolve() / ".video-agent" / "analysis.json"
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(analysis.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    _track_artifact(folder, destination)

    table = Table(title="Travel Video Analysis")
    table.add_column("Video", style="cyan")
    table.add_column("Duration", justify="right")
    table.add_column("Resolution", justify="right")
    table.add_column("Video codec")
    table.add_column("Audio")
    for video in analysis.videos:
        resolution = f"{video.width}x{video.height}" if video.width and video.height else "—"
        table.add_row(
            video.path,
            f"{video.duration_seconds:.2f}s",
            resolution,
            video.video_codec or "—",
            video.audio_codec or ("No" if not video.has_audio else "Yes"),
        )
    console.print(table)
    console.print(
        f"Found [bold]{analysis.video_count}[/bold] video(s), "
        f"{analysis.total_duration_seconds:.2f}s total."
    )
    console.print(f"Analysis saved to [green]{destination}[/green]")


@app.command()
def understand(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Folder containing travel videos.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="JSON output path (default: <folder>/.video-agent/understanding.json).",
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", help="Ollama vision model (default: qwen2.5vl:7b)."),
    ] = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b"),
    samples_per_scene: Annotated[
        int,
        typer.Option(
            "--samples-per-scene", min=1, max=10, help="Representative frames to inspect per scene."
        ),
    ] = 3,
) -> None:
    """Detect scenes, sample frames, and describe footage with a local vision model."""
    try:
        understanding = understand_folder(folder, model=model, samples_per_scene=samples_per_scene)
    except (FileNotFoundError, NotADirectoryError, MediaToolError, VisionModelError) as exc:
        console.print(f"[red]Video understanding failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = output or folder.resolve() / ".video-agent" / "understanding.json"
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(understanding.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    _track_artifact(
        folder, destination, config={"model": model, "samples_per_scene": samples_per_scene}
    )

    table = Table(title="Video Understanding")
    table.add_column("Scene", style="cyan")
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")
    table.add_column("Frames", justify="right")
    table.add_column("Subject motion")
    table.add_column("Camera motion")
    table.add_column("Temporal summary")
    table.add_column("Summary")
    for scene in understanding.scenes:
        dynamics = scene.dynamics
        table.add_row(
            scene.scene_id,
            f"{scene.start_seconds:.2f}s",
            f"{scene.end_seconds:.2f}s",
            str(len(scene.observations)),
            dynamics.subject_motion if dynamics else "—",
            dynamics.camera_motion if dynamics else "—",
            dynamics.temporal_summary if dynamics else "—",
            scene.summary,
        )
    console.print(table)
    console.print(
        f"Analyzed [bold]{understanding.video_count}[/bold] video(s), "
        f"found [bold]{understanding.scene_count}[/bold] scene(s)."
    )
    console.print(f"Understanding saved to [green]{destination}[/green]")
    console.print(
        "Extracted frames are saved under "
        f"[green]{folder.resolve() / '.video-agent' / 'frames'}[/green]"
    )


@app.command()
def transcribe(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Folder containing travel videos.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="JSON output path (default: <folder>/.video-agent/transcription.json).",
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", help="faster-whisper model size (default: small)."),
    ] = "small",
    device: Annotated[
        str,
        typer.Option("--device", help="Inference device: cpu or cuda."),
    ] = "cpu",
    compute_type: Annotated[
        str,
        typer.Option("--compute-type", help="Model precision (default: int8 on CPU)."),
    ] = "int8",
    min_pause_duration: Annotated[
        float,
        typer.Option(
            "--min-pause-duration",
            min=0.1,
            help="Minimum speech gap to report, in seconds.",
        ),
    ] = 1.0,
) -> None:
    """Transcribe speech locally and save timestamped segments as JSON."""
    try:
        transcription = transcribe_folder(
            folder,
            model=model,
            device=device,
            compute_type=compute_type,
            min_pause_duration=min_pause_duration,
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        MediaToolError,
        TranscriptionError,
        ValueError,
    ) as exc:
        console.print(f"[red]Transcription failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = output or folder.resolve() / ".video-agent" / "transcription.json"
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(transcription.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    _track_artifact(
        folder,
        destination,
        config={"model": model, "device": device, "compute_type": compute_type},
    )

    table = Table(title="Speech Transcription")
    table.add_column("Video", style="cyan")
    table.add_column("Language")
    table.add_column("Segments", justify="right")
    table.add_column("Long gaps", justify="right")
    table.add_column("Transcript preview")
    for video in transcription.videos:
        preview = " ".join(segment.text for segment in video.segments)
        table.add_row(
            video.path,
            video.language or ("No audio" if not video.has_audio else "—"),
            str(len(video.segments)),
            str(len(video.pauses)),
            preview[:100] or "—",
        )
    console.print(table)
    console.print(
        f"Transcribed [bold]{transcription.transcribed_video_count}[/bold] video(s), "
        f"recognized [bold]{transcription.segment_count}[/bold] speech segment(s)."
    )
    console.print(f"Transcription saved to [green]{destination}[/green]")
    console.print(
        "[yellow]Long gaps are not cut recommendations; "
        "preserve quiet scenic footage by default.[/yellow]"
    )


@app.command("detect-silence")
def detect_silence(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Folder containing travel videos.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="JSON output path (default: <folder>/.video-agent/silence-analysis.json).",
        ),
    ] = None,
    noise_threshold_db: Annotated[
        float,
        typer.Option("--noise-db", min=-90.0, max=-1.0, help="Audio silence threshold in dB."),
    ] = -35.0,
    min_silence_duration: Annotated[
        float,
        typer.Option("--min-duration", min=0.1, help="Minimum quiet interval in seconds."),
    ] = 1.5,
    speech_padding_seconds: Annotated[
        float,
        typer.Option("--speech-padding", min=0.0, help="Protection around transcript speech."),
    ] = 0.25,
) -> None:
    """Detect quiet audio and exclude intervals overlapping timestamped speech."""
    try:
        report = analyze_silence_folder(
            folder,
            noise_threshold_db=noise_threshold_db,
            min_silence_duration=min_silence_duration,
            speech_padding_seconds=speech_padding_seconds,
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        MediaToolError,
        SilenceDetectionError,
        ValueError,
    ) as exc:
        console.print(f"[red]Silence analysis failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = output or folder.resolve() / ".video-agent" / "silence-analysis.json"
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    _track_artifact(
        folder,
        destination,
        dependencies=(".video-agent/transcription.json",),
        config={
            "noise_threshold_db": noise_threshold_db,
            "min_silence_duration": min_silence_duration,
            "speech_padding_seconds": speech_padding_seconds,
        },
    )

    table = Table(title="Speech-aware Silence Analysis")
    table.add_column("Video", style="cyan")
    table.add_column("Quiet intervals", justify="right")
    table.add_column("Review candidates", justify="right")
    table.add_column("Transcript")
    for video in report.videos:
        table.add_row(
            video.path,
            str(len(video.audio_silences)),
            str(len(video.candidates)),
            "Available" if video.transcript_available else "Missing",
        )
    console.print(table)
    console.print(f"Silence report saved to [green]{destination}[/green]")
    console.print(
        "[yellow]Candidates are review-only. Check scenic ambience, quiet footage, and music; "
        "source media is never modified.[/yellow]"
    )


@app.command("propose-edits")
def propose_edits(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Folder containing travel videos.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="JSON output path (default: <folder>/.video-agent/edit-proposal.json).",
        ),
    ] = None,
) -> None:
    """Create a human-reviewable edit proposal from existing analysis artifacts."""
    try:
        proposal = build_edit_proposal(folder)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Edit proposal failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = output or folder.resolve() / ".video-agent" / "edit-proposal.json"
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(proposal.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    _track_artifact(
        folder,
        destination,
        dependencies=(".video-agent/silence-analysis.json", ".video-agent/understanding.json"),
    )

    table = Table(title="Draft Edit Proposal — Human Review Required")
    table.add_column("Video", style="cyan")
    table.add_column("Review items", justify="right")
    table.add_column("Candidate duration", justify="right")
    for video in proposal.videos:
        total = sum(item.duration_seconds for item in video.review_items)
        table.add_row(video.path, str(len(video.review_items)), f"{total:.2f}s")
    console.print(table)
    console.print(f"Proposal saved to [green]{destination}[/green]")
    console.print(
        "[yellow]No edits were applied and no video was rendered. "
        "Every item is pending human approval; review visual context and audio "
        "before deciding.[/yellow]"
    )


@app.command("review-edits")
def review_edits(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Folder containing the saved edit proposal.",
        ),
    ],
    approve: Annotated[
        list[str] | None,
        typer.Option("--approve", help="Review item ID to approve; repeat for multiple IDs."),
    ] = None,
    reject: Annotated[
        list[str] | None,
        typer.Option("--reject", help="Review item ID to reject; repeat for multiple IDs."),
    ] = None,
    proposal_path: Annotated[
        Path | None,
        typer.Option("--proposal", help="Path to an edit proposal JSON file."),
    ] = None,
) -> None:
    """Approve or reject proposed cuts; without IDs, review pending items interactively."""
    try:
        proposal, destination = load_edit_proposal(folder, proposal_path)
        approve_ids = set(approve or [])
        reject_ids = set(reject or [])
        if approve is None and reject is None:
            for video in proposal.videos:
                for item in video.review_items:
                    if item.decision != "pending":
                        continue
                    console.print(
                        f"\n[bold cyan]{video.path}[/bold cyan] — {item.item_id} "
                        f"({item.start_seconds:.2f}s–{item.end_seconds:.2f}s, "
                        f"{item.duration_seconds:.2f}s)"
                    )
                    if item.scene_summary:
                        console.print(f"Scene: {item.scene_summary}")
                    console.print(item.review_reason)
                    while True:
                        choice = (
                            console.input("Decision [a]pprove / [r]eject / [s]kip: ")
                            .strip()
                            .lower()
                        )
                        if choice in {"a", "approve"}:
                            approve_ids.add(item.item_id)
                            break
                        if choice in {"r", "reject"}:
                            reject_ids.add(item.item_id)
                            break
                        if choice in {"s", "skip", ""}:
                            break
                        console.print("[yellow]Enter a, r, or s.[/yellow]")
        apply_review_decisions(proposal, approve_ids, reject_ids)
        save_edit_proposal(proposal, destination)
        _track_artifact(
            folder,
            destination,
            dependencies=(".video-agent/silence-analysis.json", ".video-agent/understanding.json"),
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Review failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    counts = {
        decision: sum(
            item.decision == decision for video in proposal.videos for item in video.review_items
        )
        for decision in ("approved", "rejected", "pending")
    }
    console.print(
        f"Review saved to [green]{destination}[/green] — "
        f"{counts['approved']} approved, {counts['rejected']} rejected, "
        f"{counts['pending']} pending."
    )
    console.print("[yellow]No source media was modified or rendered.[/yellow]")


@app.command("build-timeline")
def build_timeline(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Folder containing the reviewed edit proposal.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="JSON output path (default: <folder>/.video-agent/timeline.json).",
        ),
    ] = None,
    proposal_path: Annotated[
        Path | None,
        typer.Option("--proposal", help="Path to an edit proposal JSON file."),
    ] = None,
) -> None:
    """Build a validated retained-segment timeline after all cuts are reviewed."""
    try:
        proposal, _ = load_edit_proposal(folder, proposal_path)
        timeline = build_project_timeline(proposal)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Timeline generation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = output or folder.resolve() / ".video-agent" / "timeline.json"
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(timeline.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    _track_artifact(folder, destination, dependencies=(".video-agent/edit-proposal.json",))
    table = Table(title="Validated Edit Timeline — No Rendering")
    table.add_column("Video", style="cyan")
    table.add_column("Approved cuts", justify="right")
    table.add_column("Retained", justify="right")
    for source, video in zip(proposal.videos, timeline.videos, strict=True):
        approved_count = sum(item.decision == "approved" for item in source.review_items)
        table.add_row(video.path, str(approved_count), f"{video.retained_duration_seconds:.2f}s")
    console.print(table)
    console.print(f"Timeline saved to [green]{destination}[/green]")
    console.print(
        "[yellow]This is a validated edit-decision artifact only. "
        "No FFmpeg commands were run and no video was rendered.[/yellow]"
    )


@app.command("subtitles")
def subtitles_command(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder containing the reviewed timeline and transcription.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="SRT output path (default: <folder>/.video-agent/subtitles.srt).",
        ),
    ] = None,
    timeline: Annotated[
        Path | None,
        typer.Option("--timeline", help="Path to a validated timeline JSON file."),
    ] = None,
    transcription: Annotated[
        Path | None,
        typer.Option("--transcription", help="Path to a Whisper transcription JSON file."),
    ] = None,
) -> None:
    """Generate an SRT subtitle file aligned to the reviewed final timeline."""
    try:
        destination, cue_count = generate_subtitles(
            folder,
            timeline_path=timeline,
            transcription_path=transcription,
            output_path=output,
        )
    except (FileNotFoundError, ValueError, SubtitleError) as exc:
        console.print(f"[red]Subtitle generation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _track_artifact(
        folder,
        destination,
        dependencies=(".video-agent/timeline.json", ".video-agent/transcription.json"),
    )
    console.print(f"[bold green]Subtitles generated.[/bold green] {cue_count} cue(s).")
    console.print(f"SRT: [green]{destination}[/green]", soft_wrap=True)
    console.print(
        "[yellow]Subtitle timestamps follow the concatenated, reviewed timeline.[/yellow]"
    )


@app.command()
def render(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder containing the reviewed timeline and source videos.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="MP4 output path (default: <folder>/.video-agent/rendered.mp4).",
        ),
    ] = None,
    timeline: Annotated[
        Path | None,
        typer.Option("--timeline", help="Path to a validated timeline JSON file."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing output MP4."),
    ] = False,
    timeout_seconds: Annotated[
        float,
        typer.Option("--timeout", min=1, help="Maximum render time in seconds."),
    ] = 3600,
    burn_subtitles: Annotated[
        bool,
        typer.Option("--burn-subtitles", help="Burn .video-agent/subtitles.srt into the video."),
    ] = False,
    subtitle_file: Annotated[
        Path | None,
        typer.Option("--subtitle-file", help="Custom SRT file to burn into the output."),
    ] = None,
    subtitle_font_size: Annotated[
        int,
        typer.Option("--subtitle-font-size", min=12, max=72, help="Burned-in subtitle font size."),
    ] = 28,
    subtitle_position: Annotated[
        str,
        typer.Option("--subtitle-position", help="Subtitle position: bottom, center, or top."),
    ] = "bottom",
    subtitle_style: Annotated[
        str,
        typer.Option(
            "--subtitle-style",
            help="Subtitle appearance: classic, cinematic, boxed, or minimal.",
        ),
    ] = "classic",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Probe inputs and validate the FFmpeg filter graph without writing an MP4.",
        ),
    ] = False,
) -> None:
    """Render the human-approved timeline to a verified H.264/AAC MP4."""
    try:
        report = render_project(
            folder,
            timeline_path=timeline,
            output_path=output,
            overwrite=overwrite,
            timeout_seconds=timeout_seconds,
            subtitle_path=(
                subtitle_file or folder.resolve() / ".video-agent" / "subtitles.srt"
                if burn_subtitles or subtitle_file
                else None
            ),
            subtitle_font_size=subtitle_font_size,
            subtitle_position=subtitle_position,
            subtitle_style=subtitle_style,
            dry_run=dry_run,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RenderError) as exc:
        console.print(f"[red]Render failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if dry_run:
        console.print("[bold green]FFmpeg filter-graph preflight passed.[/bold green]")
        console.print("Inputs and stream metadata were probed; no MP4 was written.")
        return

    console.print("[bold green]Render completed and verified.[/bold green]")
    console.print(f"Output: [green]{report.output_path}[/green]")
    console.print(
        f"Duration: {report.actual_duration_seconds:.2f}s "
        f"({report.segment_count} retained segments from "
        f"{report.source_video_count} source video(s))"
    )
    console.print(
        f"Format: {report.width}x{report.height}, "
        f"{report.video_codec.upper()} video + {report.audio_codec.upper()} audio"
    )
    if burn_subtitles or subtitle_file:
        subtitle_display_path = subtitle_file or folder.resolve() / ".video-agent" / "subtitles.srt"
        console.print(f"Burned-in subtitles: [green]{subtitle_display_path}[/green]")
    console.print("[yellow]Original source videos were not modified.[/yellow]")


@app.command("init-title-overlays")
def init_title_overlays(
    folder: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    timeline: Annotated[
        Path | None,
        typer.Option("--timeline", help="Approved/reviewed optimized timeline JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", help="Output JSON path (default: .video-agent/title-overlays.json)."
        ),
    ] = None,
) -> None:
    """Create an editable title overlay review template for the optimized timeline."""
    root = folder.expanduser().resolve()
    timeline_file = (
        (timeline or root / ".video-agent" / "optimized-timeline.json").expanduser().resolve()
    )
    destination = (output or root / ".video-agent" / "title-overlays.json").expanduser().resolve()
    try:
        _, timeline_file = load_draft_timeline(root, timeline)
        timeline_relative = timeline_file.relative_to(root)
        if destination.exists():
            raise FileExistsError(f"Title overlay file already exists: {destination}")
        document = TitleOverlayDocument(
            project_path=str(root),
            source_timeline=str(timeline_relative),
            overlays=[],
        )
        save_title_overlays(root, document, destination)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        console.print(f"[red]Could not initialize title overlays:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Title overlay template created at [green]{destination}[/green]")
    console.print("Add overlay entries to the JSON, then review them with review-title-overlays.")


@app.command("suggest-title-sequences")
def suggest_title_sequences(
    folder: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    timeline: Annotated[
        Path | None,
        typer.Option("--timeline", help="Approved optimized timeline JSON."),
    ] = None,
    plan: Annotated[
        Path | None,
        typer.Option("--plan", help="Creative video project plan JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output title overlay JSON path."),
    ] = None,
) -> None:
    """Suggest story-driven opening and closing titles for human review."""
    try:
        document, destination = suggest_narrative_title_sequences(
            folder,
            timeline_path=timeline,
            plan_path=plan,
            output_path=output,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        console.print(f"[red]Title sequence suggestion failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Suggested Narrative Title Sequences")
    table.add_column("ID", style="cyan")
    table.add_column("Role")
    table.add_column("Text")
    table.add_column("Time", justify="right")
    table.add_column("Animation")
    table.add_column("Decision")
    for item in document.overlays:
        table.add_row(
            item.overlay_id,
            item.sequence_role,
            item.text,
            f"{item.start_seconds:.2f}–{item.end_seconds:.2f}s",
            item.animation,
            item.decision,
        )
    console.print(table)
    console.print(f"Review suggestions in [green]{destination}[/green]")
    console.print("Nothing is approved automatically; use review-title-overlays before rendering.")


@app.command("review-title-overlays")
def review_titles(
    folder: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    overlay_file: Annotated[
        Path | None,
        typer.Option(
            "--overlay-file",
            help="Title overlay JSON (default: .video-agent/title-overlays.json).",
        ),
    ] = None,
    approve: Annotated[
        list[str] | None,
        typer.Option("--approve", help="Title overlay ID to approve; repeat for multiple IDs."),
    ] = None,
    reject: Annotated[
        list[str] | None,
        typer.Option("--reject", help="Title overlay ID to reject; repeat for multiple IDs."),
    ] = None,
) -> None:
    """Review title suggestions; only approved overlays are burned into the render."""
    try:
        document, destination = load_title_overlays(
            folder, overlay_file, validate_overlay_artifact=False
        )
        reviewed = review_title_overlays(document, set(approve or []), set(reject or []))
        save_title_overlays(folder, reviewed, destination)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Title review failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Title Overlay Review")
    table.add_column("ID", style="cyan")
    table.add_column("Text")
    table.add_column("Time", justify="right")
    table.add_column("Role")
    table.add_column("Placement")
    table.add_column("Animation")
    table.add_column("Decision")
    for item in reviewed.overlays:
        table.add_row(
            item.overlay_id,
            item.text,
            f"{item.start_seconds:.2f}–{item.end_seconds:.2f}s",
            item.sequence_role,
            item.placement,
            item.animation,
            item.decision,
        )
    console.print(table)
    console.print(f"Title overlay review saved to [green]{destination}[/green]")


@app.command("render-revision-preview")
def render_revision_preview(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder containing an approved timeline revision and source videos.",
        ),
    ],
    timeline: Annotated[
        Path,
        typer.Option(
            "--timeline",
            help="Approved timeline revision under .video-agent/timeline-revisions/.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Separate MP4 output path under .video-agent/revision-previews/ by default.",
        ),
    ] = None,
    timeout_seconds: Annotated[
        float,
        typer.Option("--timeout", min=1, help="Maximum render time in seconds."),
    ] = 3600,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate inputs and FFmpeg graph without writing a preview.",
        ),
    ] = False,
    fit_mode: Annotated[
        str,
        typer.Option(
            "--fit-mode",
            help="Source framing: crop to fill or pad to preserve the full frame.",
        ),
    ] = "crop",
    title_overlays: Annotated[
        Path | None,
        typer.Option("--title-overlays", help="Optional reviewed title overlay JSON artifact."),
    ] = None,
) -> None:
    """Render an approved timeline revision to a separate, traceable preview MP4."""
    try:
        report, manifest = render_timeline_revision_preview(
            folder,
            timeline_path=timeline,
            output_path=output,
            timeout_seconds=timeout_seconds,
            dry_run=dry_run,
            fit_mode=fit_mode,
            title_overlay_path=title_overlays,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        OSError,
        ValueError,
        RenderError,
    ) as exc:
        console.print(f"[red]Revision preview failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if dry_run:
        console.print("[bold green]Revision preview preflight passed.[/bold green]")
        console.print("Inputs and FFmpeg graph were validated; no MP4 was written.")
        return
    console.print("[bold green]Approved revision preview rendered and verified.[/bold green]")
    console.print(f"Preview: [green]{report.output_path}[/green]")
    console.print(f"Duration: {report.actual_duration_seconds:.2f}s")
    console.print(
        f"Format: {report.width}x{report.height}, "
        f"{report.video_codec.upper()} video + {report.audio_codec.upper()} audio"
    )
    console.print(f"Preview manifest: [green]{manifest}[/green]")
    console.print("[yellow]Original timeline and source media were not modified.[/yellow]")


@app.command("render-draft")
def render_draft(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder containing optimized-timeline.json and source videos.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="MP4 output path (default: <folder>/.video-agent/rendered-draft.mp4).",
        ),
    ] = None,
    timeline: Annotated[
        Path | None,
        typer.Option("--timeline", help="Path to the reviewed optimized timeline JSON."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing output MP4."),
    ] = False,
    timeout_seconds: Annotated[
        float,
        typer.Option("--timeout", min=1, help="Maximum render time in seconds."),
    ] = 3600,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Probe inputs and validate the FFmpeg transition graph without writing an MP4.",
        ),
    ] = False,
    title_overlays: Annotated[
        Path | None,
        typer.Option("--title-overlays", help="Optional reviewed title overlay JSON artifact."),
    ] = None,
    fit_mode: Annotated[
        str,
        typer.Option(
            "--fit-mode",
            help="Source framing: crop to fill or pad to preserve the full frame.",
        ),
    ] = "crop",
) -> None:
    """Render the approved timeline using the creative plan output profile."""
    try:
        report = render_approved_shot_timeline(
            folder,
            timeline_path=timeline,
            output_path=output,
            overwrite=overwrite,
            timeout_seconds=timeout_seconds,
            dry_run=dry_run,
            fit_mode=fit_mode,
            title_overlay_path=title_overlays,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RenderError) as exc:
        console.print(f"[red]Draft render failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if dry_run:
        console.print("[bold green]FFmpeg transition-graph preflight passed.[/bold green]")
        console.print("Inputs and stream metadata were probed; no MP4 was written.")
        return

    console.print("[bold green]Approved shot timeline rendered and verified.[/bold green]")
    console.print(f"Output: [green]{report.output_path}[/green]")
    console.print(
        f"Duration: {report.actual_duration_seconds:.2f}s "
        f"({report.segment_count} approved shots from {report.source_video_count} source video(s))"
    )
    console.print(
        f"Format: {report.width}x{report.height}, "
        f"{report.video_codec.upper()} video + {report.audio_codec.upper()} audio"
    )
    console.print("[yellow]Original source videos were not modified.[/yellow]")


@app.command("critique-draft")
def critique_draft(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder containing the rendered draft and creative artifacts.",
        ),
    ],
    video: Annotated[
        Path | None,
        typer.Option("--video", help="Rendered MP4 path (default: rendered-draft.mp4)."),
    ] = None,
    timeline: Annotated[
        Path | None, typer.Option("--timeline", help="Approved timeline JSON path.")
    ] = None,
    plan: Annotated[
        Path | None, typer.Option("--plan", help="Creative project plan JSON path.")
    ] = None,
    samples: Annotated[
        int,
        typer.Option("--samples", min=1, max=24, help="Number of evenly spaced frames to inspect."),
    ] = 8,
    semantic: Annotated[
        bool,
        typer.Option(
            "--semantic",
            is_flag=True,
            help="Use the local vision model to compare rendered shots with storyboard intent.",
        ),
    ] = False,
    semantic_samples: Annotated[
        int,
        typer.Option(
            "--semantic-samples",
            min=1,
            max=12,
            help="Maximum storyboard shots sampled for semantic critique.",
        ),
    ] = 8,
    vision_model: Annotated[
        str | None,
        typer.Option("--vision-model", help="Ollama vision model (default: OLLAMA_MODEL)."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="JSON report path (default: .video-agent/draft-critique.json)."
        ),
    ] = None,
) -> None:
    """Inspect a rendered draft against the creative brief and approved timeline."""
    try:
        report = critique_rendered_draft(
            folder,
            video_path=video,
            timeline_path=timeline,
            plan_path=plan,
            sample_count=samples,
            semantic=semantic,
            semantic_samples=semantic_samples,
            vision_model=vision_model,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError, DraftCritiqueError) as exc:
        console.print(f"[red]Draft critique failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = (
        (output or folder.resolve() / ".video-agent" / "draft-critique.json").expanduser().resolve()
    )
    root = folder.expanduser().resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        console.print("[red]Critique report output must be inside the project folder.[/red]")
        raise typer.Exit(code=1) from exc
    if destination in {
        root / report.rendered_video,
        root / report.source_timeline,
        root / report.source_plan,
    }:
        console.print("[red]Critique report cannot overwrite an input artifact.[/red]")
        raise typer.Exit(code=1)
    if destination.exists():
        console.print(
            "[red]Critique report already exists: "
            f"{destination}; choose another --output path.[/red]"
        )
        raise typer.Exit(code=1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    register_artifact(
        root,
        destination,
        dependencies=[
            root / report.rendered_video,
            root / report.source_timeline,
            root / report.source_plan,
        ],
        config={
            "sample_count": samples,
            "semantic": semantic,
            "semantic_samples": semantic_samples,
            "vision_model": vision_model,
        },
    )

    table = Table(title="Rendered Draft Critique — Review Required")
    table.add_column("Severity")
    table.add_column("Category")
    table.add_column("Observation")
    for finding in report.findings:
        table.add_row(finding.severity, finding.category, finding.message)
    console.print(table)
    console.print(
        f"Inspected {report.duration_seconds:.2f}s at {report.width}x{report.height}; "
        f"{len(report.frames)} frames sampled."
    )
    if report.semantic_model:
        console.print(
            f"Semantic review: {len(report.semantic_reviews)} shots via {report.semantic_model}."
        )
    console.print(f"Report saved to [green]{destination}[/green]")
    console.print(
        "[yellow]Findings are evidence-based review prompts, not aesthetic judgments. "
        "No timeline, rendered video, or source media was modified.[/yellow]"
    )


@app.command("suggest-revisions")
def suggest_revisions(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder containing a critique report and approved timeline.",
        ),
    ],
    critique: Annotated[
        Path | None,
        typer.Option("--critique", help="Critique JSON path (default: draft-critique.json)."),
    ] = None,
    timeline: Annotated[
        Path | None,
        typer.Option("--timeline", help="Approved timeline JSON path."),
    ] = None,
    minimum_confidence: Annotated[
        float,
        typer.Option(
            "--minimum-confidence",
            min=0.0,
            max=1.0,
            help="Minimum semantic-review confidence for story suggestions.",
        ),
    ] = 0.55,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Suggestion JSON path (default: draft-revision-suggestions.json).",
        ),
    ] = None,
) -> None:
    """Turn critique evidence into shot-level suggestions for human review."""
    try:
        report = build_draft_revision_suggestions(
            folder,
            critique_path=critique,
            timeline_path=timeline,
            minimum_semantic_confidence=minimum_confidence,
        )
        destination = save_draft_revision_suggestions(folder, report, output)
    except (
        FileNotFoundError,
        NotADirectoryError,
        FileExistsError,
        ValueError,
        OSError,
        RevisionSuggestionError,
    ) as exc:
        console.print(f"[red]Revision suggestions failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Shot-level Revision Suggestions — Human Review Required")
    table.add_column("Priority")
    table.add_column("Shot")
    table.add_column("Category")
    table.add_column("Action")
    table.add_column("Recommendation")
    for item in report.suggestions:
        table.add_row(
            item.priority,
            item.clip_id,
            item.category,
            item.action,
            item.recommendation,
        )
    if not report.suggestions:
        console.print("No shot-level revision suggestions met the configured evidence thresholds.")
    else:
        console.print(table)
    console.print(f"Suggestions saved to [green]{destination}[/green]")
    console.print(
        "[yellow]Suggestions are advisory. No source intervals, footage assignments, "
        "timeline approvals, or rendered videos were changed. Review each recommendation "
        "and use the existing revision workflow for any explicit trim.[/yellow]"
    )


@app.command("review-revision-suggestions")
def review_revision_suggestions(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder containing draft-revision-suggestions.json.",
        ),
    ],
    suggestions: Annotated[
        Path | None,
        typer.Option("--suggestions", help="Suggestion JSON path."),
    ] = None,
    accept: Annotated[
        list[str] | None,
        typer.Option("--accept", help="Suggestion ID to accept; repeat for multiple IDs."),
    ] = None,
    reject: Annotated[
        list[str] | None,
        typer.Option("--reject", help="Suggestion ID to reject; repeat for multiple IDs."),
    ] = None,
    set_interval: Annotated[
        list[str] | None,
        typer.Option(
            "--set-interval",
            help=(
                "Explicit trim as clip_id:start:end; only for accepted overlong timing suggestions."
            ),
        ),
    ] = None,
) -> None:
    """Review critique suggestions and optionally create a separate timeline revision."""
    root = folder.expanduser().resolve()
    suggestion_file = (
        (suggestions or root / ".video-agent" / "draft-revision-suggestions.json")
        .expanduser()
        .resolve()
    )
    try:
        payload = json.loads(suggestion_file.read_text(encoding="utf-8"))
        items = payload.get("suggestions", [])
        known_ids = {str(item["suggestion_id"]) for item in items}
        accepted = set(accept or [])
        rejected = set(reject or [])
        interactive_edits = []
        if not accepted and not rejected:
            for item in items:
                console.print(
                    f"\n[bold]{item['suggestion_id']}[/bold] — {item['summary']} "
                    f"({item['clip_id']})"
                )
                console.print(f"Evidence: {item['evidence']}")
                console.print(f"Recommendation: {item['recommendation']}")
                while True:
                    choice = (
                        console.input("Decision [a]ccept / [r]eject / [s]kip: ").strip().lower()
                    )
                    if choice in {"a", "accept"}:
                        accepted.add(item["suggestion_id"])
                        break
                    if choice in {"r", "reject"}:
                        rejected.add(item["suggestion_id"])
                        break
                    if choice in {"s", "skip"}:
                        break
                    console.print("[yellow]Enter a, r, or s.[/yellow]")
                if (
                    item["suggestion_id"] in accepted
                    and item["action"] == "review_timing"
                    and "should be shorter" in item.get("recommendation", "")
                ):
                    console.print(
                        f"Current source interval: "
                        f"{item['current_source_start_seconds']:.3f}–"
                        f"{item['current_source_end_seconds']:.3f}s"
                    )
                    raw_interval = console.input(
                        "New source interval start:end (Enter to skip the trim): "
                    ).strip()
                    if raw_interval:
                        bounds = raw_interval.split(":")
                        if len(bounds) != 2:
                            raise RevisionSuggestionReviewError(
                                "Enter the trim as start:end, using seconds."
                            )
                        interactive_edits.append(
                            TimelineIntervalEdit(
                                clip_id=item["clip_id"],
                                source_start_seconds=float(bounds[0]),
                                source_end_seconds=float(bounds[1]),
                            )
                        )
        if (accepted | rejected) - known_ids:
            raise RevisionSuggestionReviewError(
                f"Unknown suggestion ID: {sorted((accepted | rejected) - known_ids)[0]}"
            )

        edits = []
        edits.extend(interactive_edits)
        for raw in set_interval or []:
            parts = raw.split(":")
            if len(parts) != 3:
                raise RevisionSuggestionReviewError("--set-interval must use clip_id:start:end")
            edits.append(
                TimelineIntervalEdit(
                    clip_id=parts[0],
                    source_start_seconds=float(parts[1]),
                    source_end_seconds=float(parts[2]),
                )
            )
        result, review_file, revision = review_draft_revision_suggestions(
            root,
            suggestions_path=suggestion_file,
            accept_ids=accepted,
            reject_ids=rejected,
            interval_edits=edits,
        )
    except (
        FileNotFoundError,
        ValueError,
        OSError,
        RevisionSuggestionReviewError,
    ) as exc:
        console.print(f"[red]Revision suggestion review failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Revision Suggestion Decisions")
    table.add_column("Suggestion")
    table.add_column("Shot")
    table.add_column("Decision")
    for item in result.decisions:
        table.add_row(item.suggestion_id, item.clip_id, item.status)
    console.print(table)
    console.print(f"Review saved to [green]{review_file}[/green]")
    if revision:
        console.print(f"Separate timeline revision: [green]{revision.revision_path}[/green]")
        console.print(
            "[yellow]The new revision is unapproved. Review it before rendering; "
            "the original approved timeline remains unchanged.[/yellow]"
        )
    else:
        console.print(
            "No source intervals were changed. Accepted footage-review suggestions remain "
            "follow-up tasks for the separate storyboard footage workflow."
        )


@app.command("compare-drafts")
def compare_drafts(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder containing both rendered MP4 drafts.",
        ),
    ],
    before: Annotated[
        Path, typer.Option("--before", help="Earlier rendered MP4, relative to the project folder.")
    ],
    after: Annotated[
        Path, typer.Option("--after", help="Revised rendered MP4, relative to the project folder.")
    ],
    samples: Annotated[
        int,
        typer.Option("--samples", min=1, max=16, help="Number of paired frame samples."),
    ] = 8,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output base path; creates JSON and HTML reports.",
        ),
    ] = None,
) -> None:
    """Create a read-only side-by-side comparison of two rendered drafts."""
    try:
        report = compare_rendered_drafts(
            folder,
            before_video=before,
            after_video=after,
            sample_count=samples,
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
        OSError,
        DraftComparisonError,
    ) as exc:
        console.print(f"[red]Draft comparison failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    root = folder.expanduser().resolve()
    base = (output or root / ".video-agent" / "draft-comparison").expanduser()
    if not base.is_absolute():
        base = root / base
    base = base.resolve()
    try:
        base.relative_to(root)
    except ValueError as exc:
        console.print("[red]Comparison output must be inside the project folder.[/red]")
        raise typer.Exit(code=1) from exc

    json_path = base.with_suffix(".json")
    html_path = base.with_suffix(".html")
    input_paths = {root / report.before.path, root / report.after.path}
    if json_path in input_paths or html_path in input_paths:
        console.print("[red]Comparison output cannot overwrite either input video.[/red]")
        raise typer.Exit(code=1)
    if json_path.exists() or html_path.exists():
        console.print("[red]Comparison output already exists; choose another --output path.[/red]")
        raise typer.Exit(code=1)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    html_path.write_text(render_comparison_html(report), encoding="utf-8")
    dependencies = [root / report.before.path, root / report.after.path]
    register_artifact(root, json_path, dependencies=dependencies, config={"samples": samples})
    register_artifact(root, html_path, dependencies=dependencies, config={"samples": samples})

    table = Table(title="Rendered Draft Comparison — Human Review")
    table.add_column("Metric")
    table.add_column("Before")
    table.add_column("After")
    table.add_row(
        "Duration",
        f"{report.before.duration_seconds:.2f}s",
        f"{report.after.duration_seconds:.2f}s",
    )
    table.add_row(
        "Resolution",
        f"{report.before.width}×{report.before.height}",
        f"{report.after.width}×{report.after.height}",
    )
    table.add_row(
        "Audio mean",
        f"{report.before.audio.mean_volume_db:.2f} dB"
        if report.before.audio.mean_volume_db is not None
        else "unavailable",
        f"{report.after.audio.mean_volume_db:.2f} dB"
        if report.after.audio.mean_volume_db is not None
        else "unavailable",
    )
    console.print(table)
    console.print(f"Side-by-side review: [green]{html_path}[/green]")
    console.print(f"Metrics report: [green]{json_path}[/green]")
    console.print(
        "[yellow]Samples use matched normalized playback positions. Metrics are evidence, "
        "not aesthetic scores; inspect and listen to both complete drafts before deciding. "
        "No source videos or timelines were modified.[/yellow]"
    )


@app.command("find-highlights")
def find_highlights(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Folder containing travel videos.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="JSON output path (default: <folder>/.video-agent/highlights.json).",
        ),
    ] = None,
    samples_per_scene: Annotated[
        int, typer.Option("--samples-per-scene", min=1, max=10, help="Frames sampled per scene.")
    ] = 3,
    understanding: Annotated[
        Path | None, typer.Option("--understanding", help="Optional path to understanding.json.")
    ] = None,
) -> None:
    """Find and rank visual highlight candidates; every candidate remains review-only."""
    try:
        report = find_highlights_folder(
            folder, samples_per_scene=samples_per_scene, understanding_path=understanding
        )
    except (FileNotFoundError, NotADirectoryError, MediaToolError, ValueError) as exc:
        console.print(f"[red]Highlight analysis failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = output or folder.resolve() / ".video-agent" / "highlights.json"
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    _track_artifact(
        folder,
        destination,
        dependencies=(".video-agent/understanding.json",),
        config={"samples_per_scene": samples_per_scene},
    )
    table = Table(title="Visual Highlight Candidates — Human Review Required")
    table.add_column("Rank", justify="right")
    table.add_column("Video", style="cyan")
    table.add_column("Time")
    table.add_column("Score", justify="right")
    table.add_column("Scene")
    for rank, candidate in enumerate(report.candidates, start=1):
        table.add_row(
            str(rank),
            candidate.video_path,
            f"{candidate.start_seconds:.1f}–{candidate.end_seconds:.1f}s",
            f"{candidate.score:.1f}",
            (candidate.scene_summary or "Visual signals only")[:55],
        )
    console.print(table)
    console.print(
        f"Found {report.candidate_count} candidate(s). Saved to [green]{destination}[/green]"
    )
    console.print(
        "[yellow]Scores are explainable visual heuristics, not aesthetic judgments. "
        "Review every candidate; quiet scenic footage is not penalized for low motion. "
        "No edits were applied and source media was not modified.[/yellow]"
    )


@app.command("create-video")
def create_video(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Folder containing travel videos.",
        ),
    ],
    prompt: Annotated[str, typer.Option("--prompt", "-p", help="Describe the video you want.")],
    model: Annotated[str | None, typer.Option("--model", help="Ollama model.")] = None,
    whisper_model: Annotated[
        str, typer.Option("--whisper-model", help="Whisper model size.")
    ] = "small",
    samples_per_scene: Annotated[int, typer.Option("--samples-per-scene", min=1, max=10)] = 2,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Rendered MP4 path.")
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace existing output.")
    ] = False,
) -> None:
    """Create a video from natural language with mandatory shot-by-shot approval."""
    try:
        plan, highlights, selection, timeline, timeline_path = create_video_draft(
            folder,
            prompt,
            vision_model=model,
            whisper_model=whisper_model,
            samples_per_scene=samples_per_scene,
        )
        console.print("[bold green]Analysis and draft planning complete.[/bold green]")
        console.print(
            f"Intent: {plan.style}, {plan.target_duration_seconds:.0f}s, "
            f"{plan.aspect_ratio}, {plan.resolution}"
        )
        console.print(
            f"Found {highlights.candidate_count} highlights; selected "
            f"{selection.selected_clip_count} shots ({selection.selected_duration_seconds:.1f}s)."
        )
        console.print(f"Review artifact: [green]{timeline_path}[/green]")
        console.print(
            "[yellow]Output resolution/aspect ratio and subtitle/audio preferences are "
            "recorded in the plan; this renderer currently preserves the first source shot "
            "dimensions.[/yellow]"
        )
        console.print("[bold]Human review required — no video has been rendered.[/bold]")
        approve_ids: set[str] = set()
        reject_ids: set[str] = set()
        for clip in timeline.clips:
            console.print(
                f"\n[bold cyan]{clip.sequence_index}. {clip.video_path}[/bold cyan] "
                f"— {clip.clip_id} ({clip.source_start_seconds:.2f}–"
                f"{clip.source_end_seconds:.2f}s)"
            )
            if clip.scene_summary:
                console.print(f"Scene: {clip.scene_summary}")
            console.print(clip.pacing_note)
            if clip.transition_in != "cut":
                console.print(
                    f"Incoming transition: {clip.transition_in} "
                    f"({clip.transition_duration_seconds:.2f}s)"
                )
            while True:
                choice = console.input("Decision [a]pprove / [r]eject: ").strip().lower()
                if choice in {"a", "approve"}:
                    approve_ids.add(clip.clip_id)
                    break
                if choice in {"r", "reject"}:
                    reject_ids.add(clip.clip_id)
                    break
                console.print(
                    "[yellow]Enter a or r. Every shot needs an explicit decision.[/yellow]"
                )

        timeline = apply_timeline_review(timeline, approve_ids, reject_ids)
        save_draft_timeline(timeline, timeline_path)
        if not timeline.timeline_approved:
            console.print("[yellow]No shots were approved. Rendering was skipped.[/yellow]")
            raise typer.Exit(code=2)

        report = render_approved_shot_timeline(
            folder,
            timeline_path=timeline_path,
            output_path=output,
            overwrite=overwrite,
        )
    except typer.Exit:
        raise
    except (
        FileNotFoundError,
        ValueError,
        IntentParsingError,
        VisionModelError,
        TranscriptionError,
        MediaToolError,
        RenderError,
    ) as exc:
        console.print(f"[red]Video creation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("[bold green]Approved video rendered and verified.[/bold green]")
    console.print(f"Output: [green]{report.output_path}[/green]")
    console.print(
        f"Duration: {report.actual_duration_seconds:.2f}s — {report.segment_count} approved shots."
    )
    console.print(
        f"Format: {report.width}x{report.height}, "
        f"{report.video_codec.upper()} video + {report.audio_codec.upper()} audio"
    )
    console.print("[yellow]Original source videos were not modified.[/yellow]")


@app.command("plan")
def plan_edit(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Folder containing travel videos.",
        ),
    ],
    prompt: Annotated[
        str,
        typer.Option("--prompt", "-p", help="Describe the video you want to create."),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="JSON output path (default: <folder>/.video-agent/editing-plan.json).",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Ollama model (default: OLLAMA_MODEL or qwen2.5vl:3b)."),
    ] = None,
) -> None:
    """Convert a natural-language request into a validated, reviewable editing plan."""
    try:
        editing_plan = parse_editing_intent(prompt, model=model)
    except (IntentParsingError, ValueError) as exc:
        console.print(f"[red]Editing-plan generation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = output or folder.resolve() / ".video-agent" / "editing-plan.json"
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(editing_plan.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    _track_artifact(folder, destination, config={"prompt": prompt, "model": model})
    console.print("[bold]Validated Natural-language Editing Plan[/bold]")
    console.print(f"Target duration: {editing_plan.target_duration_seconds:.1f}s")
    console.print(f"Style: {editing_plan.style} | Pacing: {editing_plan.pacing}")
    console.print(f"Priorities: {', '.join(editing_plan.priorities)}")
    console.print(f"Plan saved to [green]{destination}[/green]")
    console.print(
        "[yellow]This is an intent artifact only. No clips were selected, "
        "no timeline was changed, and no video was rendered.[/yellow]"
    )


@app.command("create-plan")
def create_plan(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder for the video plan.",
        ),
    ],
    prompt: Annotated[
        str,
        typer.Option("--prompt", "-p", help="Describe the video you want to plan."),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="JSON output path (default: <folder>/.video-agent/video-project-plan.json).",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model", help="Planning model (default: VIDEO_PLANNER_MODEL or qwen2.5:1.5b)."
        ),
    ] = None,
    music_analysis: Annotated[
        Path | None,
        typer.Option(
            "--music-analysis",
            help="Optional music-analysis JSON from analyze-music.",
        ),
    ] = None,
) -> None:
    """Generate a creative brief, story treatment, and storyboard."""
    try:
        project_plan, destination = create_video_project_plan(
            folder,
            prompt,
            model=model,
            output_path=output,
            music_analysis_path=music_analysis,
        )
    except (FileNotFoundError, IntentParsingError, VideoPlanningError, ValueError) as exc:
        console.print(f"[red]Video planning failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    brief = project_plan.brief
    console.print("[bold green]Video story and storyboard plan created.[/bold green]")
    console.print(f"Target: {brief.target_duration_seconds:.1f}s | {brief.style}")
    console.print(f"Format: {brief.aspect_ratio} | {brief.resolution} | {brief.pacing} pacing")
    console.print(f"Creative priorities: {', '.join(brief.creative_priorities)}")
    console.print(f"Working title: [bold]{project_plan.story_treatment.working_title}[/bold]")
    shot_count = sum(len(scene.shots) for scene in project_plan.scenes)
    console.print(f"Scenes: {len(project_plan.scenes)} | Storyboard shots: {shot_count}")
    if project_plan.source_music_analysis:
        energy_summary = ", ".join(
            f"{scene.title}: {scene.music_energy}" for scene in project_plan.scenes
        )
        console.print(f"Soundtrack energy: {energy_summary}")
    console.print(f"Planning stage: [cyan]{project_plan.planning_stage}[/cyan]")
    console.print(f"Plan saved to [green]{destination}[/green]")
    console.print(
        "[yellow]Storyboard shots are creative proposals only; music energy is advisory, "
        "no source footage was assigned, and no video was rendered.[/yellow]"
    )


@app.command("suggest-footage")
def suggest_footage(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder containing a storyboard and analyzed source footage.",
        ),
    ],
    plan: Annotated[
        Path | None,
        typer.Option("--plan", help="Video project plan JSON path."),
    ] = None,
    understanding: Annotated[
        Path | None,
        typer.Option("--understanding", help="Video understanding JSON path."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Footage suggestions JSON output path."),
    ] = None,
    max_candidates: Annotated[
        int,
        typer.Option(
            "--max-candidates",
            min=1,
            max=10,
            help="Candidates to show per storyboard shot.",
        ),
    ] = 3,
    embedding_model: Annotated[
        str | None,
        typer.Option(
            "--embedding-model",
            help=(
                "Optional Ollama embedding model for hybrid semantic ranking "
                "(e.g. nomic-embed-text)."
            ),
        ),
    ] = os.getenv("FOOTAGE_EMBEDDING_MODEL"),
) -> None:
    """Rank analyzed source scenes for storyboard shots; never assign footage automatically."""
    try:
        report = suggest_storyboard_footage(
            folder,
            plan_path=plan,
            understanding_path=understanding,
            max_candidates_per_shot=max_candidates,
            embedding_model=embedding_model,
        )
    except (FileNotFoundError, ValueError, EmbeddingModelError) as exc:
        console.print(f"[red]Footage suggestion failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = output or folder.resolve() / ".video-agent" / "footage-suggestions.json"
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    _track_artifact(
        folder,
        destination,
        dependencies=(report.source_plan, report.source_understanding),
        config={
            "max_candidates_per_shot": max_candidates,
            "embedding_model": embedding_model,
        },
    )

    table = Table(title="Storyboard Footage Suggestions — Human Review Required")
    table.add_column("Shot", style="cyan")
    table.add_column("Candidate footage")
    table.add_column("Source interval")
    table.add_column("Score", justify="right")
    table.add_column("Semantic", justify="right")
    table.add_column("Lexical", justify="right")
    for item in report.suggestions:
        if not item.candidates:
            table.add_row(item.shot_id, "No analyzed scene candidates", "—", "—", "—", "—")
            continue
        for index, candidate in enumerate(item.candidates, start=1):
            table.add_row(
                item.shot_id if index == 1 else "",
                candidate.video_path,
                f"{candidate.source_start_seconds:.2f}–{candidate.source_end_seconds:.2f}s",
                f"{candidate.score:.1f}",
                (
                    f"{candidate.semantic_similarity:.2f}"
                    if candidate.semantic_similarity is not None
                    else "—"
                ),
                f"{candidate.lexical_score:.1f}",
            )
    console.print(table)
    console.print(f"Matching method: {report.matching_method}")
    if report.embedding_model:
        console.print(f"Embedding model: {report.embedding_model}")
    console.print(
        f"Suggested {report.candidate_count} candidate(s) for "
        f"{report.matched_shot_count}/{len(report.suggestions)} storyboard shot(s)."
    )
    console.print(f"Suggestions saved to [green]{destination}[/green]")
    console.print(
        "[yellow]Candidates are ranked suggestions only. No footage was assigned, "
        "approved, trimmed, or rendered.[/yellow]"
    )


@app.command("approve-storyboard")
def approve_storyboard_command(
    folder: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    plan: Annotated[Path | None, typer.Option("--plan", help="Project plan JSON path.")] = None,
) -> None:
    """Explicitly approve a fully footage-mapped creative storyboard."""
    try:
        project_plan, destination = approve_storyboard(folder, plan_path=plan)
    except (FileNotFoundError, ValueError, OSError) as exc:
        console.print(f"[red]Storyboard approval failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    shot_count = sum(len(scene.shots) for scene in project_plan.scenes)
    console.print(
        f"[bold green]Storyboard approved[/bold green] — {shot_count} shot(s), "
        f"planning stage: {project_plan.planning_stage}."
    )
    console.print(f"Updated plan: [green]{destination}[/green]")
    console.print("[yellow]No timeline was built and no video was rendered.[/yellow]")


@app.command("build-storyboard-timeline")
def build_storyboard_timeline_command(
    folder: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    plan: Annotated[
        Path | None, typer.Option("--plan", help="Approved project plan JSON path.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Timeline output path.")
    ] = None,
) -> None:
    """Compile an approved storyboard into a reviewable FFmpeg shot timeline."""
    try:
        timeline = build_storyboard_timeline(folder, plan_path=plan)
    except (FileNotFoundError, ValueError, OSError) as exc:
        console.print(f"[red]Storyboard timeline build failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = output or folder.resolve() / ".video-agent" / "optimized-timeline.json"
    destination = destination.expanduser().resolve()
    try:
        destination.relative_to(folder.expanduser().resolve())
    except ValueError as exc:
        console.print("[red]Timeline output must be inside the project folder.[/red]")
        raise typer.Exit(code=1) from exc
    save_storyboard_timeline(timeline, destination)

    table = Table(title="Storyboard Production Timeline — Human Review Required")
    table.add_column("#", justify="right")
    table.add_column("Shot", style="cyan")
    table.add_column("Source interval")
    table.add_column("Duration", justify="right")
    table.add_column("Transition")
    for clip in timeline.clips:
        transition = (
            f"{clip.transition_in} ({clip.transition_duration_seconds:.2f}s)"
            if clip.transition_in != "cut"
            else "cut"
        )
        table.add_row(
            str(clip.sequence_index),
            clip.clip_id,
            f"{clip.source_start_seconds:.2f}–{clip.source_end_seconds:.2f}s",
            f"{clip.duration_seconds:.2f}s",
            transition,
        )
    console.print(table)
    console.print(
        f"Compiled {timeline.proposed_clip_count} storyboard shot(s): "
        f"{timeline.proposed_duration_seconds:.1f}s proposed, "
        f"{timeline.target_duration_seconds:.1f}s target."
    )
    console.print(f"Timeline saved to [green]{destination}[/green]")
    console.print(
        "[yellow]All timeline clips remain pending. Use review-timeline before render-draft; "
        "no source media was modified and no video was rendered.[/yellow]"
    )


@app.command("review-storyboard")
def review_storyboard(
    folder: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    plan: Annotated[Path | None, typer.Option("--plan", help="Project plan JSON path.")] = None,
    understanding: Annotated[
        Path | None, typer.Option("--understanding", help="Video understanding JSON path.")
    ] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="HTML output path.")] = None,
) -> None:
    """Generate a local, read-only HTML storyboard with representative footage frames."""
    try:
        destination, shot_count, image_count = generate_storyboard_review(
            folder,
            plan_path=plan,
            understanding_path=understanding,
            output_path=output,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        console.print(f"[red]Storyboard review failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"Storyboard review saved to [green]{destination}[/green] — "
        f"{shot_count} shot(s), {image_count} representative frame(s)."
    )
    console.print(
        "[yellow]Open the HTML file locally to inspect the sequence. "
        "This is read-only review; no approval or rendering was performed.[/yellow]"
    )


@app.command("review-footage")
def review_footage(
    folder: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    select: Annotated[
        list[str] | None,
        typer.Option(
            "--select", help="Approve a candidate as SHOT_ID=CANDIDATE_NUMBER; repeatable."
        ),
    ] = None,
    reject: Annotated[
        list[str] | None,
        typer.Option("--reject", help="Leave a shot unassigned; repeatable."),
    ] = None,
    plan: Annotated[Path | None, typer.Option("--plan", help="Project plan JSON path.")] = None,
    suggestions: Annotated[
        Path | None, typer.Option("--suggestions", help="Footage suggestions JSON path.")
    ] = None,
) -> None:
    """Review ranked footage candidates and explicitly approve shot assignments."""
    selections = {}
    try:
        for item in select or []:
            shot_id, separator, number = item.partition("=")
            if not separator or not shot_id.strip():
                raise ValueError("each --select value must use SHOT_ID=CANDIDATE_NUMBER")
            if shot_id.strip() in selections:
                raise ValueError(f"shot '{shot_id.strip()}' was selected more than once")
            selections[shot_id.strip()] = int(number)
        plan_result, destination = review_storyboard_footage(
            folder,
            selections=selections,
            rejected_shot_ids=set(reject or []),
            plan_path=plan,
            suggestions_path=suggestions,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Footage review failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    assigned = sum(
        shot.assignment_status != "unassigned"
        for scene in plan_result.scenes
        for shot in scene.shots
    )
    total = sum(len(scene.shots) for scene in plan_result.scenes)
    console.print(
        f"Saved [green]{destination}[/green] — {assigned}/{total} shot(s) assigned; "
        f"stage: [cyan]{plan_result.planning_stage}[/cyan]."
    )
    if plan_result.planning_stage == "footage_mapped":
        console.print(
            "[bold green]Every storyboard shot has an explicitly approved footage match."
            "[/bold green]"
        )
    else:
        console.print(
            "[yellow]Unassigned shots remain in storyboard_ready; "
            "review more candidates to complete mapping.[/yellow]"
        )
    console.print(
        "[yellow]This does not approve the overall creative plan or render video.[/yellow]"
    )


@app.command("draft-edit")
def draft_edit(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Folder containing analyzed travel videos.",
        ),
    ],
    plan: Annotated[
        Path | None,
        typer.Option(
            "--plan",
            help="Editing plan JSON (default: <folder>/.video-agent/editing-plan.json).",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Draft selection JSON output path."),
    ] = None,
) -> None:
    """Select and rank footage using the editing plan and existing analysis artifacts."""
    try:
        draft = build_draft_selection(folder, plan_path=plan)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Draft selection failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = output or folder.resolve() / ".video-agent" / "draft-selection.json"
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(draft.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    _track_artifact(
        folder,
        destination,
        dependencies=(
            ".video-agent/editing-plan.json",
            ".video-agent/highlights.json",
            ".video-agent/understanding.json",
            ".video-agent/transcription.json",
        ),
    )
    table = Table(title="Proposed Footage Sequence — Human Review Required")
    table.add_column("Selected", justify="center")
    table.add_column("Video", style="cyan")
    table.add_column("Source interval")
    table.add_column("Duration", justify="right")
    table.add_column("Score", justify="right")
    for clip in draft.clips:
        if clip.selected:
            table.add_row(
                "Yes",
                clip.video_path,
                f"{clip.source_start_seconds:.2f}–{clip.source_end_seconds:.2f}s",
                f"{clip.duration_seconds:.2f}s",
                f"{clip.combined_score:.1f}",
            )
    console.print(table)
    console.print(
        f"Selected {draft.selected_duration_seconds:.1f}s of "
        f"{draft.target_duration_seconds:.1f}s requested footage "
        f"from {draft.candidates_considered} candidate(s)."
    )
    console.print(f"Draft selection saved to [green]{destination}[/green]")
    console.print(
        "[yellow]All selections require human review. No existing timeline was changed "
        "and no video was rendered.[/yellow]"
    )


@app.command("optimize-timeline")
def optimize_timeline(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Folder containing the draft selection and editing plan.",
        ),
    ],
    selection: Annotated[
        Path | None,
        typer.Option("--selection", help="Draft selection JSON path."),
    ] = None,
    plan: Annotated[
        Path | None,
        typer.Option("--plan", help="Editing plan JSON path."),
    ] = None,
    music_analysis: Path | None = _MUSIC_ANALYSIS_OPTION,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output path (default: <folder>/.video-agent/optimized-timeline.json).",
        ),
    ] = None,
) -> None:
    """Order selected shots and propose pacing and transitions for human review."""
    try:
        timeline = build_draft_timeline(
            folder, selection_path=selection, plan_path=plan, music_analysis_path=music_analysis
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Timeline optimization failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = output or folder.resolve() / ".video-agent" / "optimized-timeline.json"
    destination = destination.expanduser().resolve()
    if timeline.source_music_analysis is not None:
        music_file = (folder.expanduser().resolve() / timeline.source_music_analysis).resolve()
        if destination == music_file:
            console.print(
                "[red]Timeline output cannot overwrite the music analysis artifact.[/red]"
            )
            raise typer.Exit(code=1)
    save_draft_timeline(timeline, destination)

    table = Table(title="Optimized Shot Timeline — Human Review Required")
    table.add_column("#", justify="right")
    table.add_column("Video", style="cyan")
    table.add_column("Source interval")
    table.add_column("Duration", justify="right")
    table.add_column("Transition")
    table.add_column("Role")
    table.add_column("Scale")
    table.add_column("Motion")
    table.add_column("Composition")
    table.add_column("Pacing target", justify="right")
    table.add_column("Music energy")
    for clip in timeline.clips:
        transition = (
            f"{clip.transition_in} ({clip.transition_duration_seconds:.2f}s)"
            if clip.transition_in != "cut"
            else "cut"
        )
        table.add_row(
            str(clip.sequence_index),
            clip.video_path,
            f"{clip.source_start_seconds:.2f}–{clip.source_end_seconds:.2f}s",
            f"{clip.duration_seconds:.2f}s",
            transition,
            clip.visual_role.replace("_", " "),
            clip.shot_scale.replace("_", " "),
            (f"{clip.subject_motion.replace('_', ' ')} / {clip.camera_motion.replace('_', ' ')}"),
            clip.composition.replace("_", " "),
            f"{clip.pacing_target_seconds:.1f}s ({clip.pacing_fit})",
            clip.music_energy,
        )
    console.print(table)
    console.print(
        f"Proposed {timeline.proposed_duration_seconds:.1f}s across "
        f"{timeline.proposed_clip_count} clip(s); target {timeline.target_duration_seconds:.1f}s."
    )
    if timeline.source_music_analysis:
        console.print(f"Music-aware pacing source: {timeline.source_music_analysis}")
    console.print(f"Optimized timeline saved to [green]{destination}[/green]")
    console.print(
        "[yellow]Review every clip with review-timeline. This is not the renderer's "
        "timeline.json; no source media was changed and no video was rendered.[/yellow]"
    )


@app.command("review-timeline")
def review_timeline(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Folder containing optimized-timeline.json.",
        ),
    ],
    approve: Annotated[
        list[str] | None,
        typer.Option("--approve", help="Clip ID to approve; repeat for multiple IDs."),
    ] = None,
    reject: Annotated[
        list[str] | None,
        typer.Option("--reject", help="Clip ID to reject; repeat for multiple IDs."),
    ] = None,
    timeline_path: Annotated[
        Path | None,
        typer.Option("--timeline", help="Path to an optimized timeline JSON file."),
    ] = None,
) -> None:
    """Approve or reject proposed shots; skipped clips remain pending."""
    try:
        timeline, destination = load_draft_timeline(folder, timeline_path)
        approve_ids = set(approve or [])
        reject_ids = set(reject or [])
        if approve is None and reject is None:
            for clip in timeline.clips:
                if clip.decision != "pending":
                    continue
                console.print(
                    f"\n[bold cyan]{clip.sequence_index}. {clip.video_path}[/bold cyan] "
                    f"— {clip.clip_id} ({clip.source_start_seconds:.2f}s–"
                    f"{clip.source_end_seconds:.2f}s)"
                )
                if clip.scene_summary:
                    console.print(f"Scene: {clip.scene_summary}")
                console.print(f"Sequence rationale: {clip.sequence_reason}")
                if clip.temporal_summary:
                    console.print(f"Temporal action: {clip.temporal_summary}")
                if clip.action_progression:
                    console.print("Action progression: " + " → ".join(clip.action_progression))
                if clip.subject_motion != "unknown" or clip.camera_motion != "unknown":
                    console.print(
                        f"Motion: subject {clip.subject_motion.replace('_', ' ')} | "
                        f"camera {clip.camera_motion.replace('_', ' ')}"
                    )
                if clip.composition != "unknown":
                    console.print(f"Composition: {clip.composition.replace('_', ' ')}")
                console.print(f"Continuity: {clip.continuity_note}")
                if clip.visual_role != "unknown" or clip.shot_scale != "unknown":
                    console.print(
                        f"Editorial role: {clip.visual_role.replace('_', ' ')} | "
                        f"Shot scale: {clip.shot_scale.replace('_', ' ')}"
                    )
                console.print(clip.pacing_note)
                if clip.music_energy != "unknown":
                    console.print(f"Music: {clip.music_pacing_note}")
                if clip.transition_in != "cut":
                    console.print(
                        f"Incoming transition: {clip.transition_in} "
                        f"({clip.transition_duration_seconds:.2f}s)"
                    )
                while True:
                    choice = (
                        console.input("Decision [a]pprove / [r]eject / [s]kip: ").strip().lower()
                    )
                    if choice in {"a", "approve"}:
                        approve_ids.add(clip.clip_id)
                        break
                    if choice in {"r", "reject"}:
                        reject_ids.add(clip.clip_id)
                        break
                    if choice in {"s", "skip", ""}:
                        break
                    console.print("[yellow]Enter a, r, or s.[/yellow]")
        timeline = apply_timeline_review(timeline, approve_ids, reject_ids)
        save_draft_timeline(timeline, destination)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Timeline review failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    counts = {
        decision: sum(clip.decision == decision for clip in timeline.clips)
        for decision in ("approved", "rejected", "pending")
    }
    console.print(
        f"Review saved to [green]{destination}[/green] — "
        f"{counts['approved']} approved, {counts['rejected']} rejected, "
        f"{counts['pending']} pending."
    )
    if timeline.timeline_approved:
        console.print("[bold green]Shot sequence approved.[/bold green]")
    else:
        console.print("[yellow]Sequence is not approved yet; resolve all pending clips.[/yellow]")
    console.print("[yellow]No renderer timeline was created and no video was rendered.[/yellow]")


@app.command("revise-timeline")
def revise_timeline(
    folder: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    set_interval: Annotated[
        list[str] | None,
        typer.Option(
            "--set-interval",
            help="Trim a shot using CLIP_ID:SOURCE_START:SOURCE_END; repeat for multiple shots.",
        ),
    ] = None,
    timeline: Annotated[
        Path | None, typer.Option("--timeline", help="Approved source timeline JSON path.")
    ] = None,
    minimum_clip_duration: Annotated[
        float,
        typer.Option("--minimum-clip-duration", min=0.1, help="Minimum resulting shot duration."),
    ] = 1.0,
) -> None:
    """Create a separate revision with explicitly requested shot trims."""
    edits: list[TimelineIntervalEdit] = []
    try:
        for value in set_interval or []:
            parts = value.rsplit(":", maxsplit=2)
            if len(parts) != 3 or not parts[0]:
                raise ValueError(f"invalid --set-interval '{value}'; expected CLIP_ID:START:END")
            edits.append(
                TimelineIntervalEdit(
                    clip_id=parts[0],
                    source_start_seconds=float(parts[1]),
                    source_end_seconds=float(parts[2]),
                )
            )
        result = create_timeline_revision(
            folder,
            edits,
            timeline_path=timeline,
            minimum_clip_duration_seconds=minimum_clip_duration,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        console.print(f"[red]Timeline revision failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Timeline Revision — Human Review Required")
    table.add_column("Clip ID")
    table.add_column("Decision")
    table.add_column("New interval", justify="right")
    for clip in result.timeline.clips:
        if clip.clip_id in result.changed_clip_ids:
            table.add_row(
                clip.clip_id,
                clip.decision,
                f"{clip.source_start_seconds:.2f}–{clip.source_end_seconds:.2f}s",
            )
    console.print(table)
    console.print(
        f"Duration: {result.before_duration_seconds:.2f}s → {result.after_duration_seconds:.2f}s"
    )
    console.print(f"Revision saved to [green]{folder.resolve() / result.revision_path}[/green]")
    console.print(
        "[yellow]Only changed shots are pending. Review this revision with "
        "review-timeline before rendering; the original timeline and source media "
        "are unchanged.[/yellow]"
    )


@app.command("review-timeline-revision")
def review_timeline_revision(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder containing both timeline versions and source media.",
        ),
    ],
    before: Annotated[Path, typer.Option("--before", help="Original timeline JSON path.")],
    after: Annotated[Path, typer.Option("--after", help="Revised timeline JSON path.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output base path for the JSON and HTML reports."),
    ] = None,
) -> None:
    """Create a visual before/after source-frame review for a timeline revision."""
    try:
        report = build_timeline_revision_visual_review(
            folder,
            before_path=before,
            after_path=after,
        )
        json_file, html_file = save_timeline_revision_visual_review(
            folder,
            report,
            output_path=output,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        OSError,
        ValueError,
        TimelineRevisionVisualReviewError,
        TimelineRevisionError,
    ) as exc:
        console.print(f"[red]Timeline revision review failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Timeline Revision Visual Review")
    table.add_column("Shot")
    table.add_column("Before interval")
    table.add_column("After interval")
    for frame in report.frames:
        table.add_row(
            frame.clip_id,
            f"{frame.before_start_seconds:.2f}–{frame.before_end_seconds:.2f}s",
            f"{frame.after_start_seconds:.2f}–{frame.after_end_seconds:.2f}s",
        )
    console.print(table)
    console.print(f"JSON report: [green]{json_file}[/green]")
    console.print(f"Visual review: [green]{html_file}[/green]")
    console.print(
        "[yellow]This report shows representative source frames, not rendered playback. "
        "Review full shot context and transitions before approving the revised timeline. "
        "Neither timeline nor source media was modified.[/yellow]"
    )


@app.command("compare-timelines")
def compare_timelines(
    folder: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    before: Annotated[
        Path,
        typer.Option("--before", help="Earlier project-local timeline JSON."),
    ],
    after: Annotated[
        Path,
        typer.Option("--after", help="Revised project-local timeline JSON."),
    ],
) -> None:
    """Show a read-only before/after comparison of two timeline versions."""
    try:
        comparison = compare_timeline_files(folder, before, after)
    except (FileNotFoundError, ValueError, OSError) as exc:
        console.print(f"[red]Timeline comparison failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Timeline Comparison")
    table.add_column("Clip ID")
    table.add_column("Before interval", justify="right")
    table.add_column("After interval", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Transition")
    for change in comparison.changes:
        table.add_row(
            change.clip_id,
            f"{change.before_start_seconds:.2f}–{change.before_end_seconds:.2f}s",
            f"{change.after_start_seconds:.2f}–{change.after_end_seconds:.2f}s",
            f"{change.before_duration_seconds:.2f}s → {change.after_duration_seconds:.2f}s",
            f"{change.before_transition} → {change.after_transition}",
        )
    console.print(table)
    console.print(
        f"Total duration: {comparison.before_duration_seconds:.2f}s → "
        f"{comparison.after_duration_seconds:.2f}s "
        f"({comparison.duration_delta_seconds:+.2f}s)"
    )
    if not comparison.changes:
        console.print("[dim]No source-interval or transition changes were found.[/dim]")


@app.command("analyze-music")
def analyze_music_command(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder for the soundtrack analysis artifact.",
        ),
    ],
    music: Annotated[
        Path,
        typer.Option(
            "--music",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Explicit soundtrack file to analyze.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="JSON output path (default: <folder>/.video-agent/music-analysis.json).",
        ),
    ] = None,
) -> None:
    """Estimate soundtrack tempo, beats, downbeats, and broad energy sections."""
    try:
        report = analyze_music(folder, music)
    except (FileNotFoundError, ValueError, MusicAnalysisError) as exc:
        console.print(f"[red]Music analysis failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = output or folder.resolve() / ".video-agent" / "music-analysis.json"
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    register_artifact(folder, destination, dependencies=[music.expanduser().resolve()])

    table = Table(title="Soundtrack Analysis")
    table.add_column("Tempo", justify="right")
    table.add_column("Beat confidence", justify="right")
    table.add_column("Beats", justify="right")
    table.add_column("Downbeats", justify="right")
    table.add_column("Energy sections", justify="right")
    table.add_row(
        f"{report.bpm:.1f} BPM" if report.bpm else "Undetected",
        f"{report.beat_confidence:.2f}",
        str(len(report.beat_times_seconds)),
        str(len(report.downbeat_times_seconds)),
        str(len(report.sections)),
    )
    console.print(table)
    console.print(f"Music analysis saved to [green]{destination}[/green]")
    for note in report.notes:
        console.print(f"[dim]• {note}[/dim]")


@app.command("suggest-music-sync")
def suggest_music_sync_command(
    folder: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    timeline: Annotated[
        Path | None, typer.Option("--timeline", help="Shot timeline JSON path.")
    ] = None,
    music_analysis: Annotated[
        Path | None, typer.Option("--music-analysis", help="Soundtrack analysis JSON path.")
    ] = None,
    tolerance: Annotated[
        float,
        typer.Option(
            "--tolerance",
            min=0.001,
            max=1.0,
            help="Maximum beat distance for a recommendation, in seconds (default: 0.18).",
        ),
    ] = 0.18,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Suggestion report output path.")
    ] = None,
) -> None:
    """Compare timeline transitions with soundtrack beats without changing the timeline."""
    try:
        report = suggest_music_sync(
            folder,
            timeline_path=timeline,
            music_analysis_path=music_analysis,
            tolerance_seconds=tolerance,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        console.print(f"[red]Music sync analysis failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = (
        (output or folder.resolve() / ".video-agent" / "music-sync-suggestions.json")
        .expanduser()
        .resolve()
    )
    try:
        destination.relative_to(folder.expanduser().resolve())
    except ValueError as exc:
        console.print("[red]Music sync output must be inside the project folder.[/red]")
        raise typer.Exit(code=1) from exc
    timeline_file = folder.resolve() / report.timeline_path
    analysis_file = folder.resolve() / report.music_analysis_path
    if destination in {timeline_file, analysis_file, Path(report.music_path).resolve()}:
        console.print(
            "[red]Music sync output cannot overwrite an input artifact or soundtrack.[/red]"
        )
        raise typer.Exit(code=1)
    save_music_sync_suggestions(report, destination)
    register_artifact(
        folder,
        destination,
        dependencies=[timeline_file, analysis_file],
        config={"tolerance_seconds": tolerance},
    )

    table = Table(title="Music Sync Suggestions — Review Required")
    table.add_column("Transition")
    table.add_column("Current", justify="right")
    table.add_column("Nearest beat", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Recommendation")
    for boundary in report.boundaries:
        table.add_row(
            f"{boundary.previous_clip_id} → {boundary.next_clip_id}",
            f"{boundary.boundary_seconds:.2f}s",
            (
                f"{boundary.nearest_beat_seconds:.2f}s"
                if boundary.nearest_beat_seconds is not None
                else "—"
            ),
            (
                f"{boundary.beat_delta_seconds:+.2f}s"
                if boundary.beat_delta_seconds is not None
                else "—"
            ),
            (
                f"{boundary.recommended_anchor}: {boundary.recommended_boundary_seconds:.2f}s"
                if boundary.recommended_boundary_seconds is not None
                else "No change"
            ),
        )
    console.print(table)
    console.print(
        f"Suggested {report.aligned_boundary_count}/{report.boundary_count} "
        f"beat-aligned transition(s); report saved to [green]{destination}[/green]."
    )
    console.print(
        "[yellow]This is a review-only report. Timeline timing and source intervals "
        "were not changed.[/yellow]"
    )


@app.command("review-music-sync")
def review_music_sync(
    folder: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    accept: Annotated[
        list[str] | None,
        typer.Option("--accept", help="Accept a boundary ID; repeat for multiple IDs."),
    ] = None,
    reject: Annotated[
        list[str] | None,
        typer.Option("--reject", help="Reject a boundary ID; repeat for multiple IDs."),
    ] = None,
    timeline: Annotated[
        Path | None, typer.Option("--timeline", help="Shot timeline JSON path.")
    ] = None,
    suggestions: Annotated[
        Path | None, typer.Option("--suggestions", help="Music sync suggestions JSON path.")
    ] = None,
    minimum_clip_duration: Annotated[
        float,
        typer.Option(
            "--minimum-clip-duration",
            help="Minimum resulting source clip duration in seconds (default: 1.0).",
        ),
    ] = 1.0,
) -> None:
    """Review beat-alignment suggestions and save a separate, unapproved timeline revision."""
    root = folder.expanduser().resolve()
    timeline_file = (timeline or root / ".video-agent" / "optimized-timeline.json").resolve()
    suggestions_file = (
        suggestions or root / ".video-agent" / "music-sync-suggestions.json"
    ).resolve()
    accept_ids = set(accept or [])
    reject_ids = set(reject or [])
    if accept_ids & reject_ids:
        console.print("[red]A boundary cannot be both accepted and rejected.[/red]")
        raise typer.Exit(code=1)

    try:
        report = MusicSyncSuggestions.model_validate_json(
            suggestions_file.read_text(encoding="utf-8")
        )
        timeline_model = ProjectDraftTimeline.model_validate_json(
            timeline_file.read_text(encoding="utf-8")
        )
        clips = {clip.clip_id: clip for clip in timeline_model.clips}
        known_ids = {f"{item.previous_clip_id}->{item.next_clip_id}" for item in report.boundaries}
        unknown = (accept_ids | reject_ids) - known_ids
        if unknown:
            raise ValueError(f"unknown music sync boundary: {sorted(unknown)[0]}")

        if accept is None and reject is None:
            for item in report.boundaries:
                if item.adjustment_seconds is None:
                    continue
                boundary_id = f"{item.previous_clip_id}->{item.next_clip_id}"
                previous = clips.get(item.previous_clip_id)
                following = clips.get(item.next_clip_id)
                if previous is None or following is None:
                    raise ValueError(
                        f"music sync boundary '{boundary_id}' references a missing timeline clip"
                    )
                console.print(f"\n[bold cyan]{boundary_id}[/bold cyan] — {item.transition}")
                suggested = (
                    f"{item.recommended_boundary_seconds:.3f}s"
                    if item.recommended_boundary_seconds is not None
                    else "—"
                )
                console.print(
                    f"Current boundary: {item.boundary_seconds:.3f}s | "
                    f"Suggested: {suggested} | Adjustment: {item.adjustment_seconds:+.3f}s"
                )
                console.print(
                    f"Outgoing source: {previous.video_path} "
                    f"{previous.source_start_seconds:.2f}–{previous.source_end_seconds:.2f}s"
                )
                console.print(
                    f"Incoming source: {following.video_path} "
                    f"{following.source_start_seconds:.2f}–{following.source_end_seconds:.2f}s"
                )
                while True:
                    choice = console.input("Decision (a)ccept / (r)eject / (s)kip: ")
                    choice = choice.strip().lower()
                    if choice in {"a", "accept"}:
                        accept_ids.add(boundary_id)
                        break
                    if choice in {"r", "reject"}:
                        reject_ids.add(boundary_id)
                        break
                    if choice in {"s", "skip", ""}:
                        break
                    console.print("[yellow]Enter a, r, or s.[/yellow]")

        result = apply_music_sync_review(
            folder,
            suggestions_path=suggestions,
            timeline_path=timeline,
            accept_ids=accept_ids,
            reject_ids=reject_ids,
            minimum_clip_duration_seconds=minimum_clip_duration,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        console.print(f"[red]Music sync review failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Music Sync Review Decisions")
    table.add_column("Boundary")
    table.add_column("Decision")
    table.add_column("Reason")
    for decision in result.decisions:
        table.add_row(decision.boundary_id, decision.status, decision.reason or "—")
    console.print(table)
    revision = root / result.revision_path
    console.print(f"Revision saved to [green]{revision}[/green]")
    console.print(
        "[yellow]The revision is not approved. Run review-timeline with "
        "--timeline to approve it before rendering. No video was rendered and "
        "source media/original timeline were not modified.[/yellow]"
    )


@app.command("validate-artifacts")
def validate_artifacts(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder containing generated artifacts.",
        ),
    ],
) -> None:
    """Check tracked artifacts against source videos and upstream dependencies."""
    try:
        stale = validate_all_artifacts(folder)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Artifact validation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if stale:
        console.print("[red]Stale artifacts detected:[/red]")
        for item in stale:
            console.print(f"  • {item}")
        raise typer.Exit(code=1)
    console.print("[bold green]No stale tracked artifacts detected.[/bold green]")
    console.print(
        "[dim]Legacy artifacts without manifest entries are not checked; regenerate them "
        "with the current CLI to enable dependency tracking.[/dim]"
    )


@app.command()
def clean(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder whose generated frame caches should be cleaned.",
        ),
    ],
    max_age_days: Annotated[
        float,
        typer.Option(
            "--max-age-days", min=0, help="Prune cached frames older than this many days."
        ),
    ] = 7,
    max_cache_gb: Annotated[
        float,
        typer.Option("--max-cache-gb", min=0.001, help="Maximum combined frame-cache size in GB."),
    ] = 5,
    keep_artifacts: Annotated[
        bool,
        typer.Option(
            "--keep-artifacts",
            help="Preserve frames referenced by understanding.json as well as durable artifacts.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Report cleanup candidates without deleting files."),
    ] = False,
) -> None:
    """Prune old or oversized extracted-frame caches without deleting project outputs."""
    try:
        report = cleanup_generated_storage(
            folder,
            max_age_days=max_age_days,
            max_cache_gb=max_cache_gb,
            keep_artifacts=keep_artifacts,
            dry_run=dry_run,
        )
    except (NotADirectoryError, ValueError, OSError) as exc:
        console.print(f"[red]Cleanup failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    action = "Would remove" if dry_run else "Removed"
    console.print(
        f"{action} [bold]{report.candidate_files}[/bold] frame-cache file(s) "
        f"({report.reclaimed_bytes / 1024**2:.1f} MiB)."
    )
    console.print(
        f"Scanned {report.scanned_files} file(s); "
        f"estimated remaining cache: {report.remaining_bytes / 1024**2:.1f} MiB."
    )
    if dry_run:
        console.print("[yellow]Dry run only; no files were deleted.[/yellow]")
    else:
        console.print(
            "[yellow]JSON artifacts, manifests, rendered videos, model weights, "
            "and original source media were preserved.[/yellow]"
        )


if __name__ == "__main__":
    app()
