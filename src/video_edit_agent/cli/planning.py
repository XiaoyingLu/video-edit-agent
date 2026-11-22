"""Story and editing workflow commands."""

# ruff: noqa: F403,F405,I001
from . import common
from .common import *  # noqa: F403


@app.command("find-highlights")
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
    destination.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
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
            help="Ask once before rendering the automatically assembled draft.",
        ),
    ] = False,
    storyboard: Annotated[
        bool,
        typer.Option(
            "--storyboard",
            help="Use AI storyboard planning and semantic footage matching instead of highlight ranking.",
        ),
    ] = False,
) -> None:
    """Create and render a first draft; use advanced commands for shot-by-shot review."""
    try:
        if storyboard:
            understanding = understand_folder(
                folder,
                model=model or os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b"),
                samples_per_scene=samples_per_scene,
            )
            understanding_path = folder.resolve() / ".video-agent" / "understanding.json"
            understanding_path.parent.mkdir(parents=True, exist_ok=True)
            understanding_path.write_text(
                json.dumps(understanding.model_dump(mode="json"), indent=2) + "\n",
                encoding="utf-8",
            )
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
            suggestions_path.write_text(
                json.dumps(suggestions.model_dump(mode="json"), indent=2) + "\n",
                encoding="utf-8",
            )
            common._track_artifact(
                folder,
                suggestions_path,
                dependencies=(
                    ".video-agent/video-project-plan.json",
                    ".video-agent/understanding.json",
                ),
                config={"matching_method": suggestions.matching_method},
            )
            shots = [
                shot
                for scene in project_plan.scenes
                for shot in scene.shots
            ]
            suggestions_by_id = {item.shot_id: item for item in suggestions.suggestions}
            missing = [
                shot.shot_id
                for shot in shots
                if not suggestions_by_id.get(shot.shot_id)
                or not suggestions_by_id[shot.shot_id].candidates
            ]
            if missing:
                raise ValueError(
                    "Semantic footage matching found no candidates for "
                    f"{len(missing)} storyboard shot(s): {', '.join(missing[:8])}. "
                    "No footage was assigned; inspect footage-suggestions.json."
                )
            selections = {shot.shot_id: 1 for shot in shots}
            scores = [
                suggestions_by_id[shot.shot_id].candidates[0].score for shot in shots
            ]
            console.print(
                f"Storyboard: {project_plan.story_treatment.working_title} — "
                f"{len(project_plan.scenes)} scene(s), {len(shots)} shot(s)."
            )
            console.print(
                f"Semantic matching produced a top-ranked candidate for every shot "
                f"(score range {min(scores):.1f}–{max(scores):.1f})."
            )
            console.print(
                "[yellow]Review the saved footage-suggestions.json. The next confirmation "
                "explicitly approves the top-ranked match for every storyboard shot; "
                "matching scores are recommendations, not guarantees.[/yellow]"
            )
            decision = console.input(
                "Approve all top-ranked footage matches and render this storyboard? [y/N]: "
            )
            if decision.strip().lower() not in {"y", "yes"}:
                console.print(
                    "[yellow]Rendering cancelled. Storyboard and footage suggestions were preserved.[/yellow]"
                )
                raise typer.Exit(code=2)
            project_plan, plan_path = review_storyboard_footage(
                folder,
                selections=selections,
                plan_path=plan_path,
                suggestions_path=suggestions_path,
            )
            project_plan, plan_path = approve_storyboard(folder, plan_path=plan_path)
            timeline = build_storyboard_timeline(folder, plan_path=plan_path)
            timeline_path = folder.resolve() / ".video-agent" / "optimized-timeline.json"
            save_storyboard_timeline(timeline, timeline_path)
            timeline = apply_timeline_review(
                timeline, {clip.clip_id for clip in timeline.clips}, set()
            )
            save_draft_timeline(timeline, timeline_path)
            plan = project_plan.brief
            console.print(
                f"Approved storyboard: {len(shots)} footage assignments; "
                f"timeline saved to [green]{timeline_path}[/green]."
            )
        else:
            plan, highlights, selection, timeline, timeline_path = create_video_draft(
                folder,
                prompt,
                vision_model=model,
                whisper_model=whisper_model,
                samples_per_scene=samples_per_scene,
            )
            console.print(
                "[bold green]Analysis and automatic draft planning complete.[/bold green]"
            )
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
                decision = console.input(
                    "Render this automatically assembled first draft? [y/N]: "
                )
                if decision.strip().lower() not in {"y", "yes"}:
                    console.print(
                        "[yellow]Rendering cancelled. The draft timeline was preserved.[/yellow]"
                    )
                    raise typer.Exit(code=2)
            approve_ids = {clip.clip_id for clip in timeline.clips}
            timeline = apply_timeline_review(timeline, approve_ids, set())
            save_draft_timeline(timeline, timeline_path)

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


@app.command("plan")
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
    destination.write_text(
        json.dumps(editing_plan.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
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


@app.command("create-plan")
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


@app.command("propose-story")
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


@app.command("select-story")
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


@app.command("suggest-footage")
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
    destination.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    common._track_artifact(
        folder,
        destination,
        dependencies=(report.source_plan, report.source_understanding),
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


@app.command("approve-storyboard")
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


@app.command("build-storyboard-timeline")
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


@app.command("review-storyboard")
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


@app.command("review-footage")
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
        for item in select or []:
            shot_id, separator, number = item.partition("=")
            if not separator or not shot_id.strip():
                raise ValueError("each --select value must use SHOT_ID=CANDIDATE_NUMBER")
            if shot_id.strip() in selections:
                raise ValueError(f"shot '{shot_id.strip()}' was selected more than once")
            selections[shot_id.strip()] = int(number)
        plan_result, destination = review_storyboard_footage(
            folder,
            selections=selections,
            rejected_shot_ids=set(reject or []),
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


@app.command("draft-edit")
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
    destination.write_text(
        json.dumps(draft.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
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


@app.command("optimize-timeline")
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


@app.command("review-timeline")
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


@app.command("review-timeline-visual")
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
) -> None:
    """Create an offline browser review page for proposed timeline shots."""
    try:
        report = build_timeline_visual_review(folder, timeline_path=timeline)
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


@app.command("apply-timeline-visual-edits")
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


@app.command("revise-timeline")
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


@app.command("review-timeline-revision")
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


@app.command("compare-timelines")
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
