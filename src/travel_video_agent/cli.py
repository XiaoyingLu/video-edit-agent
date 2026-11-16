"""Command-line interface."""

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .editing import (
    apply_review_decisions,
    build_edit_proposal,
    build_project_timeline,
    load_edit_proposal,
    save_edit_proposal,
)
from .media import MediaToolError, analyze_folder
from .silence import analyze_silence_folder
from .silence.ffmpeg_detector import SilenceDetectionError
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

    table = Table(title="Video Understanding")
    table.add_column("Scene", style="cyan")
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")
    table.add_column("Frames", justify="right")
    table.add_column("Summary")
    for scene in understanding.scenes:
        table.add_row(
            scene.scene_id,
            f"{scene.start_seconds:.2f}s",
            f"{scene.end_seconds:.2f}s",
            str(len(scene.observations)),
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
                        choice = console.input("Decision [a]pprove / [r]eject / [s]kip: ").strip().lower()
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
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Review failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    counts = {
        decision: sum(
            item.decision == decision
            for video in proposal.videos
            for item in video.review_items
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


if __name__ == "__main__":
    app()
