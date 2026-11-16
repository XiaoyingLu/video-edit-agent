"""Command-line interface."""

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .audio import transcribe_folder
from .audio.transcriber import TranscriptionError
from .media import MediaToolError, analyze_folder
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
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="JSON output path."),
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
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="JSON output path."),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", help="Ollama vision model."),
    ] = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b"),
    samples_per_scene: Annotated[
        int,
        typer.Option("--samples-per-scene", min=1, max=10),
    ] = 3,
) -> None:
    """Detect scenes, sample frames, and describe footage with a local vision model."""
    try:
        understanding = understand_folder(
            folder, model=model, samples_per_scene=samples_per_scene
        )
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


@app.command()
def transcribe(
    folder: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="JSON output path."),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="faster-whisper model size (tiny, base, small, medium, large-v3).",
        ),
    ] = "tiny",
    language: Annotated[
        str | None,
        typer.Option("--language", help="Optional language code, e.g. en or zh."),
    ] = None,
    min_pause_duration: Annotated[
        float,
        typer.Option(
            "--min-pause-duration",
            min=0.1,
            help="Minimum gap length to report, in seconds.",
        ),
    ] = 1.0,
) -> None:
    """Transcribe speech locally and report gaps without marking them as cuts."""
    try:
        result = transcribe_folder(
            folder,
            model=model,
            language=language,
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
        json.dumps(result.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )

    table = Table(title="Speech Transcription")
    table.add_column("Video", style="cyan")
    table.add_column("Language")
    table.add_column("Speech segments", justify="right")
    table.add_column("Long gaps", justify="right")
    for transcript in result.transcripts:
        table.add_row(
            transcript.path,
            transcript.language or ("No audio" if not transcript.has_audio else "Unknown"),
            str(len(transcript.segments)),
            str(len(transcript.pauses)),
        )
    console.print(table)
    console.print(
        f"Transcribed [bold]{result.transcribed_count}[/bold] of "
        f"[bold]{result.video_count}[/bold] video(s)."
    )
    console.print(f"Transcript saved to [green]{destination}[/green]")
    console.print(
        "[yellow]Reported gaps are not cut recommendations; quiet footage is preserved.[/yellow]"
    )


if __name__ == "__main__":
    app()
