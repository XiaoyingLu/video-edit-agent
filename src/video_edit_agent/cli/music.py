"""Soundtrack and music commands."""

# ruff: noqa: F403,F405,I001
from .common import *  # noqa: F403


@app.command("suggest-soundtrack")
def suggest_soundtrack_command(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder containing video-project-plan.json.",
        ),
    ],
    plan: Annotated[
        Path | None, typer.Option("--plan", help="Video project plan JSON path.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Soundtrack brief output JSON path.")
    ] = None,
) -> None:
    """Derive a soundtrack brief from the project's narrative and storyboard."""
    try:
        brief, destination = create_soundtrack_brief(
            folder,
            plan_path=plan,
            output_path=output,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        console.print(f"[red]Soundtrack brief failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("[bold green]Soundtrack brief created.[/bold green]")
    console.print(f"Style: {brief.video_style} | Duration: {brief.target_duration_seconds:.1f}s")
    console.print(f"Mood: {', '.join(brief.mood)}")
    console.print(f"Genres: {', '.join(brief.genre_directions)}")
    console.print(f"Tempo: {brief.tempo_min_bpm}-{brief.tempo_max_bpm} BPM")
    console.print(f"Energy sections: {len(brief.energy_arc)}")
    console.print(f"Search terms: {', '.join(brief.search_terms)}")
    console.print(f"Brief saved to [green]{destination}[/green]")
    console.print(
        "[yellow]This is a music direction only. No track was downloaded or generated, "
        "and no timeline or video was changed.[/yellow]"
    )


@app.command("search-soundtrack")
def search_soundtrack_command(
    folder: Annotated[
        Path, typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True)
    ],
    license_profile: Annotated[
        Path | None,
        typer.Option(
            "--license-profile",
            help="Project-local JSON rights profile confirmed against your Soundstripe agreement.",
        ),
    ] = None,
    limit: Annotated[
        int, typer.Option("--limit", min=1, max=100, help="Maximum candidates to retrieve.")
    ] = 20,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Candidate response JSON output path.")
    ] = None,
) -> None:
    """Search Soundstripe using the project's soundtrack brief."""
    try:
        response, destination = asyncio.run(
            search_soundstripe_catalog(
                folder,
                license_profile_path=license_profile,
                output_path=output,
                limit=limit,
            )
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        console.print(f"[red]Soundtrack search failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"Found [bold]{len(response.candidates)}[/bold] Soundstripe candidate(s).")
    console.print(f"Candidates saved to [green]{destination}[/green]")
    if license_profile is None:
        console.print(
            "[yellow]No account-specific rights profile was supplied. Candidates remain "
            "ineligible until commercial, monetized-video, adaptation, and synchronization "
            "permissions are explicitly documented.[/yellow]"
        )
    console.print("[dim]No audio was downloaded and no timeline or video was changed.[/dim]")


@app.command("generate-soundtrack")
def generate_soundtrack_command(
    folder: Annotated[
        Path, typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True)
    ],
    output_format: Annotated[
        str, typer.Option("--format", help="Generated audio format: mp3 or wav.")
    ] = "mp3",
    seed: Annotated[
        int | None,
        typer.Option("--seed", min=0, max=4_294_967_294, help="Optional reproducibility seed."),
    ] = None,
    timeout: Annotated[
        float,
        typer.Option("--timeout", min=1, max=3600, help="Maximum generation wait in seconds."),
    ] = 900,
    poll_interval: Annotated[
        float,
        typer.Option("--poll-interval", min=1, max=60, help="Seconds between status checks."),
    ] = 10,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Project-local generated audio path.")
    ] = None,
) -> None:
    """Generate original music from the saved soundtrack brief with Stable Audio 3.0."""
    try:
        artifact, metadata_path = asyncio.run(
            generate_soundtrack_from_brief(
                folder,
                output_path=output,
                output_format=output_format,
                seed=seed,
                timeout_seconds=timeout,
                poll_interval_seconds=poll_interval,
            )
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError, TimeoutError) as exc:
        console.print(f"[red]Soundtrack generation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("[bold green]Original soundtrack generated.[/bold green]")
    console.print(f"Audio: [green]{folder / artifact.audio_path}[/green]")
    console.print(f"Metadata: [green]{metadata_path}[/green]")
    console.print(f"Provider generation ID: {artifact.provider_result.generation_id}")
    console.print(f"SHA-256: {artifact.provider_result.sha256}")
    console.print(
        "[yellow]Generation is not selection or approval. Review the output before use; "
        "no timeline was changed and no video was rendered.[/yellow]"
    )


@app.command("prepare-soundtrack-candidates")
def prepare_soundtrack_candidates_command(
    folder: Annotated[
        Path, typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True)
    ],
    catalog_candidates: Annotated[
        Path | None,
        typer.Option(
            "--catalog-candidates",
            help=(
                "Optional normalized catalog response JSON "
                "(defaults to the saved Soundstripe results)."
            ),
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Combined candidate collection JSON output path."),
    ] = None,
) -> None:
    """Analyze generated tracks and combine them with catalog candidates for review."""
    try:
        collection, destination = prepare_generated_soundtrack_candidates(
            folder,
            catalog_candidates_path=catalog_candidates,
            output_path=output,
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        console.print(f"[red]Soundtrack candidate preparation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    generated_count = sum(item.origin == "generated" for item in collection.candidates)
    catalog_count = len(collection.candidates) - generated_count
    console.print(
        f"Prepared [bold]{len(collection.candidates)}[/bold] candidate(s): "
        f"{catalog_count} catalog, {generated_count} generated."
    )
    console.print(f"Combined candidates saved to [green]{destination}[/green]")
    console.print(
        "[yellow]Generated tracks were analyzed for duration and estimated tempo. "
        "No candidate was selected, no timeline was changed, and no video was rendered.[/yellow]"
    )


@app.command("review-soundtrack")
def review_soundtrack_command(
    folder: Annotated[
        Path, typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True)
    ],
    candidates: Annotated[
        Path, typer.Option("--candidates", help="Normalized soundtrack search-response JSON.")
    ],
    brief: Annotated[
        Path | None, typer.Option("--brief", help="Soundtrack brief JSON path.")
    ] = None,
    video: Annotated[
        Path | None,
        typer.Option("--video", help="Optional project-local video to audition against."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output base path for review JSON and HTML."),
    ] = None,
) -> None:
    """Rank catalog and generated candidates in a read-only video/audio audition page."""
    try:
        report = build_soundtrack_candidate_review(
            folder,
            candidates_path=candidates,
            brief_path=brief,
            video_path=video,
        )
        json_file, html_file = save_soundtrack_candidate_review(folder, report, output_path=output)
    except (FileNotFoundError, ValueError, OSError) as exc:
        console.print(f"[red]Soundtrack review failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Soundtrack Candidates — Human Review Required")
    table.add_column("Candidate", style="cyan")
    table.add_column("Origin")
    table.add_column("Creator")
    table.add_column("Fit", justify="right")
    table.add_column("Rights")
    table.add_column("Duration", justify="right")
    for item in report.items:
        table.add_row(
            item.candidate.title,
            item.candidate.origin,
            item.candidate.creator or "—",
            f"{item.ranking.creative_score:.0%}",
            item.ranking.rights.status,
            f"{item.candidate.duration_seconds:.1f}s",
        )
    console.print(table)
    console.print(f"HTML audition: [green]{html_file}[/green]")
    console.print(f"Review data: [green]{json_file}[/green]")
    console.print(
        "[yellow]Review only. No candidate was selected or downloaded, and no timeline or "
        "source video was changed. Only rights-eligible candidates can be selected.[/yellow]"
    )


@app.command("select-soundtrack")
def select_soundtrack_command(
    folder: Annotated[
        Path, typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True)
    ],
    candidate_id: Annotated[
        str, typer.Option("--candidate-id", help="Exact candidate ID shown in the review JSON.")
    ],
    review: Annotated[
        Path | None, typer.Option("--review", help="Soundtrack review JSON path.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Selection JSON output path.")
    ] = None,
) -> None:
    """Explicitly select one rights-eligible candidate without downloading or rendering."""
    try:
        selection, destination = select_soundtrack_candidate(
            folder,
            review_path=review,
            candidate_id=candidate_id,
            output_path=output,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        console.print(f"[red]Soundtrack selection failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("[bold green]Soundtrack candidate selected.[/bold green]")
    console.print(f"Track: {selection.candidate.title}")
    console.print(f"Provider: {selection.candidate.provider_id}")
    console.print(f"Selection saved to [green]{destination}[/green]")
    console.print(
        "[yellow]The selected track has not been downloaded or added to a timeline. "
        "No source media was changed and no video was rendered.[/yellow]"
    )


@app.command("analyze-selected-soundtrack")
def analyze_selected_soundtrack_command(
    folder: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    music: Annotated[
        Path | None,
        typer.Option(
            "--music",
            help=(
                "Project-local licensed audio file for a catalog selection. "
                "Generated selections use their recorded local asset."
            ),
        ),
    ] = None,
    selection: Annotated[
        Path | None, typer.Option("--selection", help="Explicit soundtrack selection JSON path.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Music analysis JSON output path.")
    ] = None,
    provenance_output: Annotated[
        Path | None,
        typer.Option("--provenance-output", help="Selected-track provenance JSON output path."),
    ] = None,
) -> None:
    """Analyze the explicitly selected, rights-eligible local soundtrack."""
    try:
        report, provenance, analysis_file, provenance_file = analyze_selected_soundtrack(
            folder,
            selection_path=selection,
            music_path=music,
            analysis_output_path=output,
            record_output_path=provenance_output,
        )
    except (FileNotFoundError, ValueError, OSError, MusicAnalysisError) as exc:
        console.print(f"[red]Selected soundtrack analysis failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Selected Soundtrack Analysis")
    table.add_column("Candidate")
    table.add_column("Provider")
    table.add_column("Duration", justify="right")
    table.add_column("Tempo", justify="right")
    table.add_column("Beat confidence", justify="right")
    table.add_row(
        provenance.candidate_id,
        provenance.provider_id,
        f"{report.duration_seconds:.1f}s",
        f"{report.bpm:.1f} BPM" if report.bpm is not None else "Undetected",
        f"{report.beat_confidence:.2f}",
    )
    console.print(table)
    console.print(f"Music analysis saved to [green]{analysis_file}[/green]")
    console.print(f"Selection provenance saved to [green]{provenance_file}[/green]")
    console.print(
        "[yellow]Analysis only. The selected track was not added to the timeline; "
        "no timeline approval or rendering occurred.[/yellow]"
    )


@app.command("analyze-music")
def analyze_music_command(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder for the soundtrack analysis artifact.",
        ),
    ],
    music: Annotated[
        Path,
        typer.Option(
            "--music",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Explicit soundtrack file to analyze.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="JSON output path (default: <folder>/.video-agent/music-analysis.json).",
        ),
    ] = None,
) -> None:
    """Estimate soundtrack tempo, beats, downbeats, and broad energy sections."""
    try:
        report = analyze_music(folder, music)
    except (FileNotFoundError, ValueError, MusicAnalysisError) as exc:
        console.print(f"[red]Music analysis failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = output or folder.resolve() / ".video-agent" / "music-analysis.json"
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    register_artifact(folder, destination, dependencies=[music.expanduser().resolve()])

    table = Table(title="Soundtrack Analysis")
    table.add_column("Tempo", justify="right")
    table.add_column("Beat confidence", justify="right")
    table.add_column("Beats", justify="right")
    table.add_column("Downbeats", justify="right")
    table.add_column("Energy sections", justify="right")
    table.add_row(
        f"{report.bpm:.1f} BPM" if report.bpm else "Undetected",
        f"{report.beat_confidence:.2f}",
        str(len(report.beat_times_seconds)),
        str(len(report.downbeat_times_seconds)),
        str(len(report.sections)),
    )
    console.print(table)
    console.print(f"Music analysis saved to [green]{destination}[/green]")
    for note in report.notes:
        console.print(f"[dim]• {note}[/dim]")


@app.command("suggest-music-sync")
def suggest_music_sync_command(
    folder: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    timeline: Annotated[
        Path | None, typer.Option("--timeline", help="Shot timeline JSON path.")
    ] = None,
    music_analysis: Annotated[
        Path | None, typer.Option("--music-analysis", help="Soundtrack analysis JSON path.")
    ] = None,
    tolerance: Annotated[
        float,
        typer.Option(
            "--tolerance",
            min=0.001,
            max=1.0,
            help="Maximum beat distance for a recommendation, in seconds (default: 0.18).",
        ),
    ] = 0.18,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Suggestion report output path.")
    ] = None,
) -> None:
    """Compare timeline transitions with soundtrack beats without changing the timeline."""
    try:
        report = suggest_music_sync(
            folder,
            timeline_path=timeline,
            music_analysis_path=music_analysis,
            tolerance_seconds=tolerance,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        console.print(f"[red]Music sync analysis failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = (
        (output or folder.resolve() / ".video-agent" / "music-sync-suggestions.json")
        .expanduser()
        .resolve()
    )
    try:
        destination.relative_to(folder.expanduser().resolve())
    except ValueError as exc:
        console.print("[red]Music sync output must be inside the project folder.[/red]")
        raise typer.Exit(code=1) from exc
    timeline_file = folder.resolve() / report.timeline_path
    analysis_file = folder.resolve() / report.music_analysis_path
    if destination in {timeline_file, analysis_file, Path(report.music_path).resolve()}:
        console.print(
            "[red]Music sync output cannot overwrite an input artifact or soundtrack.[/red]"
        )
        raise typer.Exit(code=1)
    save_music_sync_suggestions(report, destination)
    register_artifact(
        folder,
        destination,
        dependencies=[timeline_file, analysis_file],
        config={"tolerance_seconds": tolerance},
    )

    table = Table(title="Music Sync Suggestions — Review Required")
    table.add_column("Transition")
    table.add_column("Current", justify="right")
    table.add_column("Nearest beat", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Recommendation")
    for boundary in report.boundaries:
        table.add_row(
            f"{boundary.previous_clip_id} → {boundary.next_clip_id}",
            f"{boundary.boundary_seconds:.2f}s",
            (
                f"{boundary.nearest_beat_seconds:.2f}s"
                if boundary.nearest_beat_seconds is not None
                else "—"
            ),
            (
                f"{boundary.beat_delta_seconds:+.2f}s"
                if boundary.beat_delta_seconds is not None
                else "—"
            ),
            (
                f"{boundary.recommended_anchor}: {boundary.recommended_boundary_seconds:.2f}s"
                if boundary.recommended_boundary_seconds is not None
                else "No change"
            ),
        )
    console.print(table)
    console.print(
        f"Suggested {report.aligned_boundary_count}/{report.boundary_count} "
        f"beat-aligned transition(s); report saved to [green]{destination}[/green]."
    )
    console.print(
        "[yellow]This is a review-only report. Timeline timing and source intervals "
        "were not changed.[/yellow]"
    )


@app.command("review-music-timeline")
def review_music_timeline(
    folder: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    timeline: Annotated[
        Path | None, typer.Option("--timeline", help="Shot timeline JSON path.")
    ] = None,
    music_analysis: Annotated[
        Path | None, typer.Option("--music-analysis", help="Soundtrack analysis JSON path.")
    ] = None,
    suggestions: Annotated[
        Path | None, typer.Option("--suggestions", help="Music sync suggestions JSON path.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output base path for JSON and HTML.")
    ] = None,
) -> None:
    """Create a read-only browser view of soundtrack beats and shot boundaries."""
    try:
        report = build_music_timeline_preview(
            folder,
            timeline_path=timeline,
            music_analysis_path=music_analysis,
            suggestions_path=suggestions,
        )
        json_file, html_file = save_music_timeline_preview(folder, report, output_path=output)
    except (FileNotFoundError, ValueError, OSError) as exc:
        console.print(f"[red]Music timeline preview failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"HTML preview: [green]{html_file}[/green]")
    console.print(f"JSON report: [green]{json_file}[/green]")
    console.print(
        "[yellow]This is a read-only synchronization view. Beat estimates are approximate; "
        "no timeline timing, approval, or source media was changed.[/yellow]"
    )


@app.command("review-music-sync")
def review_music_sync(
    folder: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    accept: Annotated[
        list[str] | None,
        typer.Option("--accept", help="Accept a boundary ID; repeat for multiple IDs."),
    ] = None,
    reject: Annotated[
        list[str] | None,
        typer.Option("--reject", help="Reject a boundary ID; repeat for multiple IDs."),
    ] = None,
    timeline: Annotated[
        Path | None, typer.Option("--timeline", help="Shot timeline JSON path.")
    ] = None,
    suggestions: Annotated[
        Path | None, typer.Option("--suggestions", help="Music sync suggestions JSON path.")
    ] = None,
    minimum_clip_duration: Annotated[
        float,
        typer.Option(
            "--minimum-clip-duration",
            help="Minimum resulting source clip duration in seconds (default: 1.0).",
        ),
    ] = 1.0,
) -> None:
    """Review beat-alignment suggestions and save a separate, unapproved timeline revision."""
    root = folder.expanduser().resolve()
    timeline_file = (timeline or root / ".video-agent" / "optimized-timeline.json").resolve()
    suggestions_file = (
        suggestions or root / ".video-agent" / "music-sync-suggestions.json"
    ).resolve()
    accept_ids = set(accept or [])
    reject_ids = set(reject or [])
    if accept_ids & reject_ids:
        console.print("[red]A boundary cannot be both accepted and rejected.[/red]")
        raise typer.Exit(code=1)

    try:
        report = MusicSyncSuggestions.model_validate_json(
            suggestions_file.read_text(encoding="utf-8")
        )
        timeline_model = ProjectDraftTimeline.model_validate_json(
            timeline_file.read_text(encoding="utf-8")
        )
        clips = {clip.clip_id: clip for clip in timeline_model.clips}
        known_ids = {f"{item.previous_clip_id}->{item.next_clip_id}" for item in report.boundaries}
        unknown = (accept_ids | reject_ids) - known_ids
        if unknown:
            raise ValueError(f"unknown music sync boundary: {sorted(unknown)[0]}")

        if accept is None and reject is None:
            for item in report.boundaries:
                if item.adjustment_seconds is None:
                    continue
                boundary_id = f"{item.previous_clip_id}->{item.next_clip_id}"
                previous = clips.get(item.previous_clip_id)
                following = clips.get(item.next_clip_id)
                if previous is None or following is None:
                    raise ValueError(
                        f"music sync boundary '{boundary_id}' references a missing timeline clip"
                    )
                console.print(f"\n[bold cyan]{boundary_id}[/bold cyan] — {item.transition}")
                suggested = (
                    f"{item.recommended_boundary_seconds:.3f}s"
                    if item.recommended_boundary_seconds is not None
                    else "—"
                )
                console.print(
                    f"Current boundary: {item.boundary_seconds:.3f}s | "
                    f"Suggested: {suggested} | Adjustment: {item.adjustment_seconds:+.3f}s"
                )
                console.print(
                    f"Outgoing source: {previous.video_path} "
                    f"{previous.source_start_seconds:.2f}–{previous.source_end_seconds:.2f}s"
                )
                console.print(
                    f"Incoming source: {following.video_path} "
                    f"{following.source_start_seconds:.2f}–{following.source_end_seconds:.2f}s"
                )
                while True:
                    choice = console.input("Decision (a)ccept / (r)eject / (s)kip: ")
                    choice = choice.strip().lower()
                    if choice in {"a", "accept"}:
                        accept_ids.add(boundary_id)
                        break
                    if choice in {"r", "reject"}:
                        reject_ids.add(boundary_id)
                        break
                    if choice in {"s", "skip", ""}:
                        break
                    console.print("[yellow]Enter a, r, or s.[/yellow]")

        result = apply_music_sync_review(
            folder,
            suggestions_path=suggestions,
            timeline_path=timeline,
            accept_ids=accept_ids,
            reject_ids=reject_ids,
            minimum_clip_duration_seconds=minimum_clip_duration,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        console.print(f"[red]Music sync review failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Music Sync Review Decisions")
    table.add_column("Boundary")
    table.add_column("Decision")
    table.add_column("Reason")
    for decision in result.decisions:
        table.add_row(decision.boundary_id, decision.status, decision.reason or "—")
    console.print(table)
    revision = root / result.revision_path
    console.print(f"Revision saved to [green]{revision}[/green]")
    console.print(
        "[yellow]The revision is not approved. Run review-timeline with "
        "--timeline to approve it before rendering. No video was rendered and "
        "source media/original timeline were not modified.[/yellow]"
    )
