"""Analysis and ingestion commands."""

# ruff: noqa: F403,F405
from .common import *  # noqa: F403
from .common import _MUSIC_ANALYSIS_OPTION, _track_artifact, app, console



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
            help="Folder containing video files.",
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
    _track_artifact(folder, destination)

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
            help="Folder containing video files.",
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
        str | None,
        typer.Option("--model", help="Provider-specific vision model identifier."),
    ] = None,
    provider: Annotated[
        str,
        typer.Option("--provider", help="Vision backend: ollama, mlx (SmolVLM2), or qwen3-vl."),
    ] = os.getenv("VISION_PROVIDER", "ollama"),
    samples_per_scene: Annotated[
        int,
        typer.Option(
            "--samples-per-scene", min=1, max=10, help="Representative frames to inspect per scene."
        ),
    ] = 3,
    no_resume: Annotated[
        bool,
        typer.Option(
            "--no-resume", help="Ignore saved progress and start understanding from scratch."
        ),
    ] = False,
) -> None:
    """Detect scenes, sample frames, and describe footage with a local vision model."""
    project_root = folder.expanduser().resolve()
    selected_provider = provider.strip().lower()
    if selected_provider == "mlx":
        default_model = os.getenv("MLX_VISION_MODEL", "mlx-community/SmolVLM2-2.2B-Instruct-mlx")
    elif selected_provider == "qwen3-vl":
        default_model = os.getenv("QWEN3_VL_MODEL", "mlx-community/Qwen3-VL-4B-Instruct-4bit")
    else:
        default_model = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
    selected_model = model or default_model
    checkpoint = project_root / ".video-agent" / "understanding-checkpoint.json"
    if checkpoint.is_file() and not no_resume:
        console.print(f"[cyan]Resuming video understanding from {checkpoint}[/cyan]")
    console.print(f"[dim]Vision provider: {provider} · model: {selected_model}[/dim]")
    console.print(
        "[dim]Progress is saved after each analyzed frame; rerun this command to resume.[/dim]"
    )
    try:

        def report_progress(event: str, details: dict) -> None:
            if event == "scene_detection_started":
                console.print(f"[cyan]Detecting scene boundaries:[/cyan] {details['video_path']}")
            elif event == "scene_detection_completed":
                console.print(
                    f"[green]Scene detection complete:[/green] {details['scene_count']} scenes "
                    f"in {details['elapsed_seconds']:.1f}s"
                )
            elif event == "scene_started":
                console.print(
                    f"[bold cyan]Scene {details['scene_index']}/"
                    f"{details['scene_count']}[/bold cyan] "
                    f"({details['start_seconds']:.1f}s–{details['end_seconds']:.1f}s)"
                )
            elif event == "frame_extraction_completed":
                console.print(
                    f"  Prepared {details['frame_count']} representative frame(s) "
                    f"in {details['elapsed_seconds']:.1f}s"
                )
            elif event == "frame_started":
                console.print(
                    f"  Analyzing frame {details['frame_index']}/{details['frame_count']}..."
                )
            elif event == "frame_completed":
                console.print(
                    f"  [green]Frame analyzed[/green] in {details['elapsed_seconds']:.1f}s"
                )
            elif event == "frame_cached":
                console.print(
                    "  [dim]Reused cached frame "
                    f"{details['frame_index']}/{details['frame_count']}[/dim]"
                )
            elif event == "temporal_analysis_started":
                console.print("  Analyzing temporal action and camera movement...")
            elif event == "temporal_analysis_completed":
                console.print(
                    f"  [green]Temporal analysis complete[/green] "
                    f"in {details['elapsed_seconds']:.1f}s"
                )
            elif event == "temporal_analysis_cached":
                console.print("  [dim]Reused cached temporal analysis[/dim]")
            elif event == "scene_completed":
                console.print(
                    "[green]Completed scene "
                    f"{details['scene_index']}/{details['scene_count']}[/green]"
                )

        understanding = understand_folder(
            folder,
            model=selected_model,
            provider=provider,
            samples_per_scene=samples_per_scene,
            resume=not no_resume,
            progress_callback=report_progress,
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        MediaToolError,
        VisionModelError,
        ValueError,
    ) as exc:
        console.print(f"[red]Video understanding failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = output or project_root / ".video-agent" / "understanding.json"
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(understanding.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    _track_artifact(
        folder,
        destination,
        config={
            "provider": provider,
            "model": selected_model,
            "samples_per_scene": samples_per_scene,
        },
    )
    checkpoint.unlink(missing_ok=True)

    table = Table(title="Video Understanding")
    table.add_column("Scene", style="cyan")
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")
    table.add_column("Frames", justify="right")
    table.add_column("Subject motion")
    table.add_column("Camera motion")
    table.add_column("Temporal summary")
    table.add_column("Summary")
    for scene in understanding.scenes:
        dynamics = scene.dynamics
        table.add_row(
            scene.scene_id,
            f"{scene.start_seconds:.2f}s",
            f"{scene.end_seconds:.2f}s",
            str(len(scene.observations)),
            dynamics.subject_motion if dynamics else "—",
            dynamics.camera_motion if dynamics else "—",
            dynamics.temporal_summary if dynamics else "—",
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
            help="Folder containing video files.",
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
    except (
        FileNotFoundError,
        NotADirectoryError,
        MediaToolError,
        TranscriptionError,
        ValueError,
    ) as exc:
        console.print(f"[red]Transcription failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = output or folder.resolve() / ".video-agent" / "transcription.json"
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(transcription.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    _track_artifact(
        folder,
        destination,
        config={"model": model, "device": device, "compute_type": compute_type},
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
        "[yellow]Long gaps are not cut recommendations; "
        "preserve quiet scenic footage by default.[/yellow]"
    )


@app.command("detect-silence")
def detect_silence(
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
            help="JSON output path (default: <folder>/.video-agent/silence-analysis.json).",
        ),
    ] = None,
    noise_threshold_db: Annotated[
        float,
        typer.Option("--noise-db", min=-90.0, max=-1.0, help="Audio silence threshold in dB."),
    ] = -35.0,
    min_silence_duration: Annotated[
        float,
        typer.Option("--min-duration", min=0.1, help="Minimum quiet interval in seconds."),
    ] = 1.5,
    speech_padding_seconds: Annotated[
        float,
        typer.Option("--speech-padding", min=0.0, help="Protection around transcript speech."),
    ] = 0.25,
) -> None:
    """Detect quiet audio and exclude intervals overlapping timestamped speech."""
    try:
        report = analyze_silence_folder(
            folder,
            noise_threshold_db=noise_threshold_db,
            min_silence_duration=min_silence_duration,
            speech_padding_seconds=speech_padding_seconds,
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        MediaToolError,
        SilenceDetectionError,
        ValueError,
    ) as exc:
        console.print(f"[red]Silence analysis failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = output or folder.resolve() / ".video-agent" / "silence-analysis.json"
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    _track_artifact(
        folder,
        destination,
        dependencies=(".video-agent/transcription.json",),
        config={
            "noise_threshold_db": noise_threshold_db,
            "min_silence_duration": min_silence_duration,
            "speech_padding_seconds": speech_padding_seconds,
        },
    )

    table = Table(title="Speech-aware Silence Analysis")
    table.add_column("Video", style="cyan")
    table.add_column("Quiet intervals", justify="right")
    table.add_column("Review candidates", justify="right")
    table.add_column("Transcript")
    for video in report.videos:
        table.add_row(
            video.path,
            str(len(video.audio_silences)),
            str(len(video.candidates)),
            "Available" if video.transcript_available else "Missing",
        )
    console.print(table)
    console.print(f"Silence report saved to [green]{destination}[/green]")
    console.print(
        "[yellow]Candidates are review-only. Check scenic ambience, quiet footage, and music; "
        "source media is never modified.[/yellow]"
    )


@app.command("propose-edits")
def propose_edits(
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
            help="JSON output path (default: <folder>/.video-agent/edit-proposal.json).",
        ),
    ] = None,
) -> None:
    """Create a human-reviewable edit proposal from existing analysis artifacts."""
    try:
        proposal = build_edit_proposal(folder)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Edit proposal failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = output or folder.resolve() / ".video-agent" / "edit-proposal.json"
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(proposal.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    _track_artifact(
        folder,
        destination,
        dependencies=(".video-agent/silence-analysis.json", ".video-agent/understanding.json"),
    )

    table = Table(title="Draft Edit Proposal — Human Review Required")
    table.add_column("Video", style="cyan")
    table.add_column("Review items", justify="right")
    table.add_column("Candidate duration", justify="right")
    for video in proposal.videos:
        total = sum(item.duration_seconds for item in video.review_items)
        table.add_row(video.path, str(len(video.review_items)), f"{total:.2f}s")
    console.print(table)
    console.print(f"Proposal saved to [green]{destination}[/green]")
    console.print(
        "[yellow]No edits were applied and no video was rendered. "
        "Every item is pending human approval; review visual context and audio "
        "before deciding.[/yellow]"
    )


@app.command("review-edits")
def review_edits(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Folder containing the saved edit proposal.",
        ),
    ],
    approve: Annotated[
        list[str] | None,
        typer.Option("--approve", help="Review item ID to approve; repeat for multiple IDs."),
    ] = None,
    reject: Annotated[
        list[str] | None,
        typer.Option("--reject", help="Review item ID to reject; repeat for multiple IDs."),
    ] = None,
    proposal_path: Annotated[
        Path | None,
        typer.Option("--proposal", help="Path to an edit proposal JSON file."),
    ] = None,
) -> None:
    """Approve or reject proposed cuts; without IDs, review pending items interactively."""
    try:
        proposal, destination = load_edit_proposal(folder, proposal_path)
        approve_ids = set(approve or [])
        reject_ids = set(reject or [])
        if approve is None and reject is None:
            for video in proposal.videos:
                for item in video.review_items:
                    if item.decision != "pending":
                        continue
                    console.print(
                        f"\n[bold cyan]{video.path}[/bold cyan] — {item.item_id} "
                        f"({item.start_seconds:.2f}s–{item.end_seconds:.2f}s, "
                        f"{item.duration_seconds:.2f}s)"
                    )
                    if item.scene_summary:
                        console.print(f"Scene: {item.scene_summary}")
                    console.print(item.review_reason)
                    while True:
                        choice = (
                            console.input("Decision [a]pprove / [r]eject / [s]kip: ")
                            .strip()
                            .lower()
                        )
                        if choice in {"a", "approve"}:
                            approve_ids.add(item.item_id)
                            break
                        if choice in {"r", "reject"}:
                            reject_ids.add(item.item_id)
                            break
                        if choice in {"s", "skip", ""}:
                            break
                        console.print("[yellow]Enter a, r, or s.[/yellow]")
        apply_review_decisions(proposal, approve_ids, reject_ids)
        save_edit_proposal(proposal, destination)
        _track_artifact(
            folder,
            destination,
            dependencies=(".video-agent/silence-analysis.json", ".video-agent/understanding.json"),
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Review failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    counts = {
        decision: sum(
            item.decision == decision for video in proposal.videos for item in video.review_items
        )
        for decision in ("approved", "rejected", "pending")
    }
    console.print(
        f"Review saved to [green]{destination}[/green] — "
        f"{counts['approved']} approved, {counts['rejected']} rejected, "
        f"{counts['pending']} pending."
    )
    console.print("[yellow]No source media was modified or rendered.[/yellow]")


@app.command("build-timeline")
def build_timeline(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Folder containing the reviewed edit proposal.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="JSON output path (default: <folder>/.video-agent/timeline.json).",
        ),
    ] = None,
    proposal_path: Annotated[
        Path | None,
        typer.Option("--proposal", help="Path to an edit proposal JSON file."),
    ] = None,
) -> None:
    """Build a validated retained-segment timeline after all cuts are reviewed."""
    try:
        proposal, _ = load_edit_proposal(folder, proposal_path)
        timeline = build_project_timeline(proposal)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Timeline generation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    destination = output or folder.resolve() / ".video-agent" / "timeline.json"
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(timeline.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    _track_artifact(folder, destination, dependencies=(".video-agent/edit-proposal.json",))
    table = Table(title="Validated Edit Timeline — No Rendering")
    table.add_column("Video", style="cyan")
    table.add_column("Approved cuts", justify="right")
    table.add_column("Retained", justify="right")
    for source, video in zip(proposal.videos, timeline.videos, strict=True):
        approved_count = sum(item.decision == "approved" for item in source.review_items)
        table.add_row(video.path, str(approved_count), f"{video.retained_duration_seconds:.2f}s")
    console.print(table)
    console.print(f"Timeline saved to [green]{destination}[/green]")
    console.print(
        "[yellow]This is a validated edit-decision artifact only. "
        "No FFmpeg commands were run and no video was rendered.[/yellow]"
    )


@app.command("subtitles")
def subtitles_command(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder containing the reviewed timeline and transcription.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="SRT output path (default: <folder>/.video-agent/subtitles.srt).",
        ),
    ] = None,
    timeline: Annotated[
        Path | None,
        typer.Option("--timeline", help="Path to a validated timeline JSON file."),
    ] = None,
    transcription: Annotated[
        Path | None,
        typer.Option("--transcription", help="Path to a Whisper transcription JSON file."),
    ] = None,
) -> None:
    """Generate an SRT subtitle file aligned to the reviewed final timeline."""
    try:
        destination, cue_count = generate_subtitles(
            folder,
            timeline_path=timeline,
            transcription_path=transcription,
            output_path=output,
        )
    except (FileNotFoundError, ValueError, SubtitleError) as exc:
        console.print(f"[red]Subtitle generation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _track_artifact(
        folder,
        destination,
        dependencies=(".video-agent/timeline.json", ".video-agent/transcription.json"),
    )
    console.print(f"[bold green]Subtitles generated.[/bold green] {cue_count} cue(s).")
    console.print(f"SRT: [green]{destination}[/green]", soft_wrap=True)
    console.print(
        "[yellow]Subtitle timestamps follow the concatenated, reviewed timeline.[/yellow]"
    )

