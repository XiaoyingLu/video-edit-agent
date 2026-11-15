"""Command-line interface."""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .media import MediaToolError, analyze_folder

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
        json.dumps(analysis.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
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


if __name__ == "__main__":
    app()
