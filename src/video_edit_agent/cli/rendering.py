"""Rendering and visual review commands."""

# ruff: noqa: F403,F405,I001
from .common import *  # noqa: F403



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

