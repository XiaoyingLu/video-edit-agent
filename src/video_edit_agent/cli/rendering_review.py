"""Rendering and visual review command workflows."""

# ruff: noqa: F403,F405,I001
from ..atomic_io import write_json_atomic
from .common import *  # noqa: F403


@render_app.command("critique-draft")
@app.command("critique-draft", hidden=True)
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
    write_json_atomic(destination, report)
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


@render_app.command("suggest-revisions")
@app.command("suggest-revisions", hidden=True)
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


@render_app.command("review-revision-suggestions")
@app.command("review-revision-suggestions", hidden=True)
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


@render_app.command("compare-drafts")
@app.command("compare-drafts", hidden=True)
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
    write_json_atomic(json_path, report)
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


@render_app.command("review-workspace")
@app.command("review-workspace", hidden=True)
def review_workspace(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder containing generated review artifacts.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Workspace HTML output path."),
    ] = None,
) -> None:
    """Create a local landing page for the project's available review tools."""
    try:
        destination = generate_review_workspace(folder, output_path=output)
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
        console.print(f"[red]Review workspace failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Project review workspace: [green]{destination}[/green]")
    console.print(
        "[yellow]Open this HTML file in a browser. It links to existing review pages and "
        "shows which project artifacts are available. Decisions and rendering still require "
        "their explicit CLI workflows.[/yellow]"
    )
