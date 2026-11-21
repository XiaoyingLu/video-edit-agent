"""Maintenance and reporting commands."""

# ruff: noqa: F403,F405,I001
from .common import *  # noqa: F403


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


@app.command("report-revision-cycle")
def report_revision_cycle(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder containing the revision-cycle artifacts.",
        ),
    ],
    critique: Annotated[Path, typer.Option("--critique", help="Rendered-draft critique JSON.")],
    suggestions: Annotated[
        Path, typer.Option("--suggestions", help="Shot-level suggestions JSON.")
    ],
    original_timeline: Annotated[
        Path, typer.Option("--original-timeline", help="Original approved timeline JSON.")
    ],
    revised_timeline: Annotated[
        Path, typer.Option("--revised-timeline", help="Revised timeline JSON.")
    ],
    preview_manifest: Annotated[
        Path, typer.Option("--preview-manifest", help="Approved revision preview manifest JSON.")
    ],
    comparison: Annotated[
        Path, typer.Option("--comparison", help="Rendered before/after comparison JSON.")
    ],
    timeline_review: Annotated[
        Path | None, typer.Option("--timeline-review", help="Optional source-frame review JSON.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output base path for JSON and HTML.")
    ] = None,
) -> None:
    """Create a traceable index of one critique-to-render revision iteration."""
    try:
        report = build_revision_cycle_report(
            folder,
            critique_path=critique,
            suggestions_path=suggestions,
            original_timeline_path=original_timeline,
            revised_timeline_path=revised_timeline,
            preview_manifest_path=preview_manifest,
            comparison_path=comparison,
            timeline_review_path=timeline_review,
        )
        json_file, html_file = save_revision_cycle_report(folder, report, output_path=output)
    except (
        FileNotFoundError,
        NotADirectoryError,
        FileExistsError,
        OSError,
        ValueError,
        RevisionCycleError,
    ) as exc:
        console.print(f"[red]Revision-cycle report failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Revision Iteration — Artifact Trace")
    table.add_column("Stage")
    table.add_column("Artifact")
    table.add_column("Summary")
    for item in report.artifacts:
        table.add_row(item.role, item.path, item.summary)
    console.print(table)
    console.print(f"HTML report: [green]{html_file}[/green]")
    console.print(f"JSON report: [green]{json_file}[/green]")
    console.print(
        "[yellow]This report links evidence; it does not approve edits or modify timelines, "
        "source media, or rendered videos.[/yellow]"
    )

