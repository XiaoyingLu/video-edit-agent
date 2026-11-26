"""Story planning and timeline command workflows."""

# ruff: noqa: F403,F405,I001
from . import common
import webbrowser
from ..atomic_io import write_json_atomic
from ..editing.planning_schemas import VideoProjectPlan
from ..editing.footage_visual_review import (
    generate_footage_visual_review,
    load_footage_review_selections,
)
from ..editing.footage_review_server import start_footage_review_server
from .common import *  # noqa: F403


@story_app.command("find-highlights")
@app.command("find-highlights", hidden=True)
def find_highlights(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Folder containing video files.",
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
    write_json_atomic(destination, report)
    common._track_artifact(
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


def _create_storyboard_video_draft(
    folder: Path,
    prompt: str,
    model: str | None,
    samples_per_scene: int,
):
    """Build, explicitly confirm, and approve a storyboard-driven first draft."""
    understanding = understand_folder(
        folder,
        model=model or get_setting("OLLAMA_MODEL"),
        samples_per_scene=samples_per_scene,
    )
    understanding_path = folder.resolve() / ".video-agent" / "understanding.json"
    understanding_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(understanding_path, understanding)
    common._track_artifact(
        folder,
        understanding_path,
        config={
            "model": getattr(understanding, "model", model),
            "samples_per_scene": samples_per_scene,
        },
    )
    project_plan, plan_path = create_video_project_plan(folder, prompt)
    suggestions = suggest_storyboard_footage(
        folder,
        plan_path=plan_path,
        embedding_model=DEFAULT_FOOTAGE_EMBEDDING_MODEL,
    )
    suggestions_path = folder.resolve() / ".video-agent" / "footage-suggestions.json"
    write_json_atomic(suggestions_path, suggestions)
    common._track_artifact(
        folder,
        suggestions_path,
        dependencies=(
            ".video-agent/video-project-plan.json",
            ".video-agent/understanding.json",
        ),
        config={"matching_method": suggestions.matching_method},
    )
    shots = [shot for scene in project_plan.scenes for shot in scene.shots]
    suggestions_by_id = {item.shot_id: item for item in suggestions.suggestions}
    missing = [
        shot.shot_id
        for shot in shots
        if not suggestions_by_id.get(shot.shot_id) or not suggestions_by_id[shot.shot_id].candidates
    ]
    if missing:
        raise ValueError(
            "Footage matching found no candidates for "
            f"{len(missing)} storyboard shot(s): {', '.join(missing[:8])}. "
            "No footage was assigned; inspect footage-suggestions.json."
        )
    selection_path = folder.resolve() / ".video-agent" / "footage-review-selections.json"
    server, submitted, review_url, submission_url = start_footage_review_server(
        folder, selection_path
    )
    try:
        review_page, _, candidate_count = generate_footage_visual_review(
            folder,
            plan_path=plan_path,
            suggestions_path=suggestions_path,
            understanding_path=understanding_path,
            submission_url=submission_url,
        )
        console.print(
            f"Storyboard: {project_plan.story_treatment.working_title} — "
            f"{len(project_plan.scenes)} scene(s), {len(shots)} shot(s), "
            f"{candidate_count} footage candidate(s)."
        )
        console.print(f"Visual footage review: [green]{review_page}[/green]")
        console.print(f"Opening review in your browser: [link]{review_url}[/link]")
        webbrowser.open(review_url)
        console.print(
            "[yellow]Choose a footage option for each shot and click "
            "Submit selections. The choices will be applied automatically.[/yellow]"
        )
        if not submitted.wait(timeout=1800):
            raise ValueError("Footage review timed out after 30 minutes; no selections were applied.")
    finally:
        server.shutdown()
        server.server_close()
    selections, rejected = load_footage_review_selections(
        selection_path,
        expected_shot_ids={shot.shot_id for shot in shots},
        suggestions_path=suggestions_path,
    )
    project_plan, plan_path = review_storyboard_footage(
        folder,
        selections=selections,
        rejected_shot_ids=rejected,
        plan_path=plan_path,
        suggestions_path=suggestions_path,
    )
    storyboard_path, _, image_count = generate_storyboard_review(folder, plan_path=plan_path)
    console.print(
        f"Storyboard preview: [green]{storyboard_path}[/green] "
        f"({image_count} representative frame(s))."
    )
    console.print(
        "[yellow]Open storyboard.html to inspect the proposed footage assignments. "
        "The matching scores are recommendations, not guarantees.[/yellow]"
    )
    decision = console.input("Continue to build and render this storyboard? [y/N]: ")
    if decision.strip().lower() not in {"y", "yes"}:
        console.print(
            "[yellow]Rendering cancelled. The storyboard and footage assignments "
            "were preserved for review; nothing was approved or rendered.[/yellow]"
        )
        raise typer.Exit(code=2)
    project_plan, plan_path = approve_storyboard(folder, plan_path=plan_path)
    timeline = build_storyboard_timeline(folder, plan_path=plan_path)
    timeline_path = folder.resolve() / ".video-agent" / "optimized-timeline.json"
    save_storyboard_timeline(timeline, timeline_path)
    timeline = apply_timeline_review(timeline, {clip.clip_id for clip in timeline.clips}, set())
    save_draft_timeline(timeline, timeline_path)
    plan = project_plan.brief
    console.print(
        f"Approved storyboard: {len(shots)} footage assignments; "
        f"timeline saved to [green]{timeline_path}[/green]."
    )
    return plan, timeline_path


def _create_highlight_video_draft(
    folder: Path,
    prompt: str,
    model: str | None,
    whisper_model: str,
    samples_per_scene: int,
    review: bool,
):
    """Build and optionally confirm a relevance-ranked highlight draft."""
    plan, highlights, selection, timeline, timeline_path = create_video_draft(
        folder,
        prompt,
        vision_model=model,
        whisper_model=whisper_model,
        samples_per_scene=samples_per_scene,
    )
    console.print("[bold green]Analysis and automatic draft planning complete.[/bold green]")
    console.print(
        f"Intent: {plan.style}, {plan.target_duration_seconds:.0f}s, "
        f"{plan.aspect_ratio}, {plan.resolution}"
    )
    console.print(
        f"Found {highlights.candidate_count} highlights; selected "
        f"{selection.selected_clip_count} shots "
        f"({selection.selected_duration_seconds:.1f}s)."
    )
    console.print(f"Draft timeline: [green]{timeline_path}[/green]")
    if review:
        decision = console.input("Render this automatically assembled first draft? [y/N]: ")
        if decision.strip().lower() not in {"y", "yes"}:
            console.print("[yellow]Rendering cancelled. The draft timeline was preserved.[/yellow]")
            raise typer.Exit(code=2)
    approve_ids = {clip.clip_id for clip in timeline.clips}
    timeline = apply_timeline_review(timeline, approve_ids, set())
    save_draft_timeline(timeline, timeline_path)
    return plan, timeline_path


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
            help="Folder containing video files.",
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
    review: Annotated[
        bool,
        typer.Option(
            "--review",
            help=(
                "For --highlights mode, ask once before rendering; storyboard mode "
                "always asks after showing its visual preview."
            ),
        ),
    ] = False,
    storyboard: Annotated[
        bool,
        typer.Option(
            "--storyboard/--highlights",
            help=(
                "Use AI storyboard planning (default), or opt into the legacy "
                "highlight-ranked workflow with --highlights."
            ),
        ),
    ] = True,
) -> None:
    """Create and render an automatically assembled first draft.

    Use advanced commands for shot-by-shot review.
    """
    try:
        if storyboard:
            plan, timeline_path = _create_storyboard_video_draft(
                folder, prompt, model, samples_per_scene
            )
        else:
            plan, timeline_path = _create_highlight_video_draft(
                folder, prompt, model, whisper_model, samples_per_scene, review
            )

        report = render_approved_shot_timeline(
            folder,
            timeline_path=timeline_path,
            output_path=output,
            overwrite=overwrite,
            output_aspect_ratio=plan.aspect_ratio,
            output_resolution=plan.resolution,
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
        VideoPlanningError,
        EmbeddingModelError,
    ) as exc:
        console.print(f"[red]Video creation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("[bold green]First draft rendered and verified.[/bold green]")
    console.print(f"Output: [green]{report.output_path}[/green]")
    console.print(
        f"Duration: {report.actual_duration_seconds:.2f}s — {report.segment_count} shots."
    )
    console.print(
        f"Format: {report.width}x{report.height}, "
        f"{report.video_codec.upper()} video + {report.audio_codec.upper()} audio"
    )
    console.print(
        "[yellow]The selected project soundtrack is mixed automatically when a local "
        "eligible asset is available. Original source videos were not modified.[/yellow]"
    )


@story_app.command("plan")
@app.command("plan", hidden=True)
def plan_edit(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Folder containing video files.",
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
    write_json_atomic(destination, editing_plan)
    common._track_artifact(folder, destination, config={"prompt": prompt, "model": model})
    console.print("[bold]Validated Natural-language Editing Plan[/bold]")
    console.print(f"Target duration: {editing_plan.target_duration_seconds:.1f}s")
    console.print(f"Style: {editing_plan.style} | Pacing: {editing_plan.pacing}")
    console.print(f"Priorities: {', '.join(editing_plan.priorities)}")
    console.print(f"Plan saved to [green]{destination}[/green]")
    console.print(
        "[yellow]This is an intent artifact only. No clips were selected, "
        "no timeline was changed, and no video was rendered.[/yellow]"
    )


@story_app.command("create-plan")
@app.command("create-plan", hidden=True)
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
    story: Annotated[
        Path | None,
        typer.Option(
            "--story",
            help="Selected story-proposal JSON (select an angle first with select-story).",
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
            story_path=story,
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


@story_app.command("propose-story")
@app.command("propose-story", hidden=True)
def propose_story_command(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder with analyzed footage (run analyze and understand first).",
        ),
    ],
    prompt: Annotated[
        str | None,
        typer.Option(
            "--prompt",
            "-p",
            help="Optional tone or focus hint; an explicit duration here sets the target.",
        ),
    ] = None,
    target_seconds: Annotated[
        float | None,
        typer.Option("--target-seconds", help="Target video length (default: 90)."),
    ] = None,
    angles: Annotated[
        int,
        typer.Option("--angles", min=1, max=4, help="Number of alternative story angles."),
    ] = 2,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Story model (default: VIDEO_PLANNER_MODEL or qwen2.5:1.5b)."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="JSON output path (default: <folder>/.video-agent/story-proposal.json).",
        ),
    ] = None,
) -> None:
    """Propose story angles from the footage you actually shot; nothing is assigned or rendered."""
    try:
        proposal, destination = propose_story(
            folder,
            prompt=prompt,
            target_seconds=target_seconds,
            angle_count=angles,
            model=model,
            output_path=output,
        )
    except (FileNotFoundError, StoryProposalError, ValueError) as exc:
        console.print(f"[red]Story proposal failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("[bold green]Story angles proposed from your footage.[/bold green]")
    console.print(
        f"{proposal.ledger_scene_count} scene(s) | target {proposal.target_duration_seconds:g}s"
    )
    for angle in proposal.angles:
        console.print(f"\n[bold]{angle.angle_id}[/bold]: {angle.title} ({angle.story_angle})")
        console.print(f"  {angle.logline}")
        for chapter in angle.chapters:
            console.print(
                f"  - {chapter.name}: {chapter.target_seconds:.1f}s, "
                f"{len(chapter.candidate_scene_ids)} candidate scene(s)"
            )
        for gap in angle.gaps:
            console.print(f"  [yellow]gap:[/yellow] {gap}")
        for warning in angle.warnings:
            console.print(f"  [yellow]warning:[/yellow] {warning}")
    console.print(f"\nProposal saved to [green]{destination}[/green]")
    console.print(
        "[yellow]Proposals are advisory: pick one with select-story; no footage was assigned "
        "and no video was rendered.[/yellow]"
    )


@story_app.command("select-story")
@app.command("select-story", hidden=True)
def select_story_command(
    folder: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    angle_id: Annotated[
        str, typer.Option("--angle", help="Angle ID to select, for example angle-1.")
    ],
) -> None:
    """Record which proposed story angle you want to develop."""
    try:
        proposal, destination = select_story_angle(folder, angle_id)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Story selection failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    chosen = next(angle for angle in proposal.angles if angle.angle_id == angle_id)
    console.print(f"Selected [bold]{chosen.angle_id}[/bold]: {chosen.title}")
    console.print(f"Saved to [green]{destination}[/green]")


@story_app.command("suggest-footage")
@app.command("suggest-footage", hidden=True)
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
    ] = DEFAULT_FOOTAGE_EMBEDDING_MODEL,
) -> None:
    """Rank analyzed source scenes for storyboard shots; never assign footage automatically."""
    if (folder.resolve() / ".video-agent" / "footage-feedback.json").is_file():
        console.print("[dim]Applying preferences from previous footage reviews.[/dim]")
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
    write_json_atomic(destination, report)
    common._track_artifact(
        folder,
        destination,
        dependencies=(report.source_plan, report.source_understanding)
        + ((report.source_feedback,) if report.source_feedback else ()),
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


@story_app.command("approve-storyboard")
@app.command("approve-storyboard", hidden=True)
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


@story_app.command("build-storyboard-timeline")
@app.command("build-storyboard-timeline", hidden=True)
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


@story_app.command("review-storyboard")
@app.command("review-storyboard", hidden=True)
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


@story_app.command("review-footage")
@app.command("review-footage", hidden=True)
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
    selections_file: Annotated[
        Path | None,
        typer.Option("--selections-file", help="JSON exported from the visual footage review."),
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
        if selections_file is not None:
            if select or reject:
                raise ValueError("--selections-file cannot be combined with --select or --reject")
            current_plan = VideoProjectPlan.model_validate_json(
                (plan or folder.resolve() / ".video-agent" / "video-project-plan.json")
                .expanduser()
                .read_text(encoding="utf-8")
            )
            shot_ids = {shot.shot_id for scene in current_plan.scenes for shot in scene.shots}
            selections, rejected = load_footage_review_selections(
                selections_file,
                expected_shot_ids=shot_ids,
                suggestions_path=suggestions
                or folder.resolve() / ".video-agent" / "footage-suggestions.json",
            )
        else:
            for item in select or []:
                shot_id, separator, number = item.partition("=")
                if not separator or not shot_id.strip():
                    raise ValueError("each --select value must use SHOT_ID=CANDIDATE_NUMBER")
                if shot_id.strip() in selections:
                    raise ValueError(f"shot '{shot_id.strip()}' was selected more than once")
                selections[shot_id.strip()] = int(number)
            rejected = set(reject or [])
        plan_result, destination = review_storyboard_footage(
            folder,
            selections=selections,
            rejected_shot_ids=rejected,
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


@story_app.command("draft-edit")
@app.command("draft-edit", hidden=True)
def draft_edit(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Folder containing analyzed video files.",
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
    write_json_atomic(destination, draft)
    common._track_artifact(
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
