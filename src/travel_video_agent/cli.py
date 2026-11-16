"""Command-line interface."""

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .media import MediaToolError, analyze_folder
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
    except (FileNotFoundError, NotADirectoryError, MediaToolError, TranscriptionError, ValueError) as exc:
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
        "[yellow]Long gaps are not cut recommendations; preserve quiet scenic footage by default.[/yellow]"
    )


if __name__ == "__main__":
    app()
