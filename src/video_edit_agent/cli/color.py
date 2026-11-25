"""Review-only color consistency analysis command."""

# ruff: noqa: F403,F405,I001
from .common import *  # noqa: F403
from ..color_consistency import (
    ColorAnalysisError,
    analyze_project_color,
    save_project_color_analysis,
)


@app.command("analyze-color")
def analyze_color(
    folder: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    samples: Annotated[
        int,
        typer.Option("--samples", min=1, max=12, help="Representative frames per source video."),
    ] = 5,
    analysis: Annotated[
        Path | None,
        typer.Option("--analysis", help="Source media analysis JSON path."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Report path (default: .video-agent/color-analysis.json).",
        ),
    ] = None,
) -> None:
    """Measure shot-to-shot luma/chroma variation and suggest bounded corrections."""
    try:
        report = analyze_project_color(folder, sample_count=samples, analysis_path=analysis)
        destination = save_project_color_analysis(folder, report, output)
    except (FileNotFoundError, FileExistsError, OSError, ValueError, ColorAnalysisError) as exc:
        console.print(f"[red]Color analysis failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Color Consistency — Review Suggestions")
    table.add_column("Video", style="cyan")
    table.add_column("Samples", justify="right")
    table.add_column("Baseline Y", justify="right")
    table.add_column("Non-neutral", justify="right")
    for video in report.videos:
        non_neutral = sum(
            any(abs(value) >= 2 for value in (
                item.luma_offset, item.chroma_u_offset, item.chroma_v_offset
            ))
            for item in video.suggestions
        )
        table.add_row(
            video.video_path,
            str(len(video.samples)),
            f"{video.baseline_luma:.1f}",
            str(non_neutral),
        )
    console.print(table)
    console.print(f"Report saved to [green]{destination}[/green]")
    console.print(
        "[yellow]Suggestions are relative measurements, not automatic grading. "
        "No source video or render was modified.[/yellow]"
    )
