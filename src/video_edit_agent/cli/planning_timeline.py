"""Story planning and timeline command workflows."""

# ruff: noqa: F403,F405,I001
import os
import webbrowser
from urllib.parse import quote

from . import common
from .common import *  # noqa: F403
from ..editing.timeline_review_server import start_timeline_review_server
from ..editing.timeline_visual_editor import (
    apply_timeline_visual_edits,
    load_timeline_visual_edits,
)
from ..editing.timeline_visual_review import render_timeline_visual_review_html


@story_app.command("optimize-timeline")
@app.command("optimize-timeline", hidden=True)
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
    music_analysis: Path | None = common._MUSIC_ANALYSIS_OPTION,
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


@story_app.command("review-timeline")
@app.command("review-timeline", hidden=True)
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
    decisions_file: Annotated[
        Path | None,
        typer.Option(
            "--decisions-file",
            help="JSON decisions exported from review-timeline-visual.",
        ),
    ] = None,
) -> None:
    """Approve or reject proposed shots; skipped clips remain pending."""
    try:
        timeline, destination = load_draft_timeline(folder, timeline_path)
        approve_ids = set(approve or [])
        reject_ids = set(reject or [])
        if decisions_file is not None:
            if approve is not None or reject is not None:
                raise ValueError("--decisions-file cannot be combined with --approve or --reject")
            exported = load_timeline_visual_review_decisions(decisions_file)
            expected_timeline_path = destination.relative_to(folder.expanduser().resolve())
            if Path(exported.timeline_path) != expected_timeline_path:
                raise ValueError(
                    "Decision file targets a different timeline; regenerate the visual review"
                )
            current_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
            if exported.timeline_sha256 != current_hash:
                raise ValueError("Timeline changed after visual review; regenerate the review page")
            known_ids = {clip.clip_id for clip in timeline.clips}
            if set(exported.decisions) != known_ids:
                raise ValueError(
                    "Decision file must contain exactly one decision for every timeline clip"
                )
            approve_ids = {
                clip_id
                for clip_id, decision in exported.decisions.items()
                if decision == "approved"
            }
            reject_ids = {
                clip_id
                for clip_id, decision in exported.decisions.items()
                if decision == "rejected"
            }
            for clip in timeline.clips:
                if exported.decisions[clip.clip_id] == "pending":
                    clip.decision = "pending"
        elif approve is None and reject is None:
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


@story_app.command("review-timeline-visual")
@app.command("review-timeline-visual", hidden=True)
def review_timeline_visual(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder containing an optimized shot timeline and source videos.",
        ),
    ],
    timeline: Annotated[
        Path | None,
        typer.Option("--timeline", help="Timeline JSON path; defaults to optimized-timeline.json."),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output base path for JSON and HTML.")
    ] = None,
    live: Annotated[
        bool,
        typer.Option(
            "--live",
            help="Open a local browser session and submit visual edits directly.",
        ),
    ] = False,
) -> Path | None:
    """Create a browser review page; --live applies submitted edits as an unapproved revision."""
    try:
        report = build_timeline_visual_review(folder, timeline_path=timeline)
        if live:
            root = folder.expanduser().resolve()
            review_file = root / ".video-agent" / "timeline-live-review.html"
            edits_file = root / ".video-agent" / "timeline-live-edits.json"
            review_file.parent.mkdir(parents=True, exist_ok=True)
            server, submitted, review_url, submission_url = start_timeline_review_server(
                root, edits_file
            )
            media_sources = {
                clip.clip_id: quote(
                    os.path.relpath(
                        (root / clip.video_path).resolve()
                        if not Path(clip.video_path).is_absolute()
                        else Path(clip.video_path).resolve(),
                        review_file.parent,
                    ).replace(os.sep, "/"),
                    safe="/:@-._~",
                )
                for clip in report.clips
            }
            try:
                review_file.write_text(
                    render_timeline_visual_review_html(
                        report,
                        media_sources=media_sources,
                        submission_url=submission_url,
                    ),
                    encoding="utf-8",
                )
                console.print(f"Opening visual timeline editor: [green]{review_url}[/green]")
                webbrowser.open(review_url)
                console.print(
                    "Reorder shots and adjust trims in the browser, then choose "
                    "'Submit visual edits'. Waiting up to 30 minutes."
                )
                if not submitted.wait(timeout=1800):
                    raise ValueError("Timeline review timed out after 30 minutes.")
            finally:
                server.shutdown()
                server.server_close()
            edits = load_timeline_visual_edits(edits_file)
            revision = apply_timeline_visual_edits(root, edits)
            console.print(f"Unapproved timeline revision: [green]{revision}[/green]")
            console.print(
                "[yellow]Edits were saved as a separate revision. Review and approve "
                "that revision before rendering; the original timeline is unchanged.[/yellow]"
            )
            return revision
        json_file, html_file = save_timeline_visual_review(folder, report, output_path=output)
    except (
        FileNotFoundError,
        NotADirectoryError,
        FileExistsError,
        OSError,
        ValueError,
        TimelineVisualReviewError,
    ) as exc:
        console.print(f"[red]Visual timeline review failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"Review page: [green]{html_file}[/green]")
    console.print(f"Review metadata: [green]{json_file}[/green]")
    console.print(
        "[yellow]Open the HTML file in a browser, export decisions, then apply them with "
        "review-timeline --decisions-file. No decisions are applied by this command.[/yellow]"
    )
    return html_file


@story_app.command("apply-timeline-visual-edits")
@app.command("apply-timeline-visual-edits", hidden=True)
def apply_timeline_visual_edits_command(
    folder: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    edits_file: Annotated[
        Path,
        typer.Option(
            "--edits-file", help="JSON proposal exported from the visual timeline editor."
        ),
    ],
    timeline: Annotated[
        Path | None,
        typer.Option("--timeline", help="Source timeline path; defaults to the export's timeline."),
    ] = None,
) -> None:
    """Create a separate unapproved revision from browser-authored order and trim edits."""
    try:
        edits = load_timeline_visual_edits(edits_file)
        destination = apply_timeline_visual_edits(folder, edits, timeline_path=timeline)
    except (OSError, ValueError) as exc:
        console.print(f"[red]Visual timeline edits rejected:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"Unapproved timeline revision: [green]{destination}[/green]")
    console.print(
        "[yellow]Review the new revision with review-timeline before rendering. "
        "The original timeline and source media are unchanged.[/yellow]"
    )


@story_app.command("revise-timeline")
@app.command("revise-timeline", hidden=True)
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


@story_app.command("review-timeline-revision")
@app.command("review-timeline-revision", hidden=True)
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


@story_app.command("compare-timelines")
@app.command("compare-timelines", hidden=True)
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
