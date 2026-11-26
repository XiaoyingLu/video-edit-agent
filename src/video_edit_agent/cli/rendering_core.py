"""Rendering and visual review command workflows."""

# ruff: noqa: F403,F405,I001
from .common import *  # noqa: F403
from ..rendering.style_presets import STYLE_PRESETS, resolve_style_preset
from ..rendering.social_crop_anchors import create_crop_anchor_template
from ..templates import load_template


@render_app.command("video")
@app.command("render-video", hidden=True)
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
        int | None,
        typer.Option(
            "--subtitle-font-size",
            min=12,
            max=72,
            help="Override template subtitle font size.",
        ),
    ] = None,
    subtitle_position: Annotated[
        str | None,
        typer.Option("--subtitle-position", help="Override template subtitle position."),
    ] = None,
    subtitle_style: Annotated[
        str | None,
        typer.Option(
            "--subtitle-style",
            help="Override template subtitle style: classic, cinematic, boxed, or minimal.",
        ),
    ] = None,
    template: Annotated[
        str | None,
        typer.Option("--template", help="Reusable template supplying default subtitle styling."),
    ] = None,
    music: Annotated[
        Path | None, typer.Option("--music", help="Project-local selected soundtrack file.")
    ] = None,
    music_mode: Annotated[
        str, typer.Option("--music-mode", help="Audio mode: original, music_only, or blend.")
    ] = "blend",
    music_volume: Annotated[
        float, typer.Option("--music-volume", min=0.01, max=2.0, help="Music gain (default 0.22).")
    ] = 0.22,
    music_fade_in: Annotated[
        float, typer.Option("--music-fade-in", min=0, help="Music fade-in seconds.")
    ] = 1.0,
    music_fade_out: Annotated[
        float, typer.Option("--music-fade-out", min=0, help="Music fade-out seconds.")
    ] = 1.5,
    music_ducking: Annotated[
        bool,
        typer.Option(
            "--music-ducking/--no-music-ducking",
            help="Lower music when source audio is active.",
        ),
    ] = False,
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
        subtitle_spec = load_template(template, folder)[0].subtitles if template else None
        resolved_subtitle_font_size = (
            subtitle_font_size
            if subtitle_font_size is not None
            else subtitle_spec.font_size
            if subtitle_spec
            else 28
        )
        resolved_subtitle_position = (
            subtitle_position
            if subtitle_position is not None
            else subtitle_spec.position
            if subtitle_spec
            else "bottom"
        )
        resolved_subtitle_style = (
            subtitle_style
            if subtitle_style is not None
            else subtitle_spec.style
            if subtitle_spec
            else "classic"
        )
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
            subtitle_font_size=resolved_subtitle_font_size,
            subtitle_position=resolved_subtitle_position,
            subtitle_style=resolved_subtitle_style,
            music_path=music,
            music_mode=music_mode,
            music_volume=music_volume,
            music_fade_in=music_fade_in,
            music_fade_out=music_fade_out,
            music_ducking=music_ducking,
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


@render_app.command("init-title-overlays")
@app.command("init-title-overlays", hidden=True)
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


@render_app.command("suggest-title-sequences")
@app.command("suggest-title-sequences", hidden=True)
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
    template: Annotated[
        str, typer.Option("--template", help="Reusable template ID (default: cinematic).")
    ] = "cinematic",
) -> None:
    """Suggest template-driven opening and closing titles for human review."""
    try:
        document, destination = suggest_narrative_title_sequences(
            folder,
            timeline_path=timeline,
            plan_path=plan,
            output_path=output,
            template_id=template,
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
    console.print(f"Template: [cyan]{document.template_id}[/cyan] v{document.template_version}")
    console.print(f"Review suggestions in [green]{destination}[/green]")
    console.print("Nothing is approved automatically; use review-title-overlays before rendering.")


@render_app.command("review-title-overlays")
@app.command("review-title-overlays", hidden=True)
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
            item.text or item.asset_path or "(asset)",
            f"{item.start_seconds:.2f}–{item.end_seconds:.2f}s",
            item.sequence_role,
            item.placement,
            item.animation,
            item.decision,
        )
    console.print(table)
    console.print(f"Title overlay review saved to [green]{destination}[/green]")


@render_app.command("render-revision-preview")
@app.command("render-revision-preview", hidden=True)
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


@render_app.command("style-presets")
@app.command("style-presets", hidden=True)
def list_style_presets() -> None:
    """List available render style presets."""
    table = Table(title="Render Style Presets")
    table.add_column("Preset", style="cyan")
    table.add_column("Description")
    table.add_column("Framing")
    table.add_column("Music", justify="right")
    table.add_column("Ducking")
    table.add_column("Beat zoom")
    for preset in STYLE_PRESETS.values():
        table.add_row(
            preset.name,
            preset.description,
            preset.fit_mode,
            f"{preset.music_volume:.2f}",
            "on" if preset.music_ducking else "off",
            "on" if preset.beat_punch_in else "off",
        )
    console.print(table)
    console.print(
        "Pass --style-preset to render-draft; explicit render options override preset values."
    )


@render_app.command("render-draft")
@app.command("render-draft", hidden=True)
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
    output_aspect_ratio: Annotated[
        str | None,
        typer.Option("--aspect-ratio", help="Output aspect ratio: 16:9, 9:16, or 1:1."),
    ] = None,
    output_resolution: Annotated[
        str | None,
        typer.Option("--resolution", help="Output resolution: 720p, 1080p, or 4k."),
    ] = None,
    style_preset: Annotated[
        str | None,
        typer.Option(
            "--style-preset",
            help="Optional render profile: cinematic, vlog, travel-montage, or social.",
        ),
    ] = None,
    fit_mode: Annotated[
        str | None,
        typer.Option(
            "--fit-mode",
            help="Source framing override: crop to fill or pad to preserve the full frame.",
        ),
    ] = None,
    music: Annotated[
        Path | None, typer.Option("--music", help="Project-local selected soundtrack file.")
    ] = None,
    music_mode: Annotated[
        str | None,
        typer.Option("--music-mode", help="Audio mode override: original, music_only, or blend."),
    ] = None,
    music_volume: Annotated[
        float | None,
        typer.Option("--music-volume", min=0.01, max=2.0, help="Music gain override."),
    ] = None,
    music_fade_in: Annotated[
        float | None,
        typer.Option("--music-fade-in", min=0, help="Music fade-in override in seconds."),
    ] = None,
    music_fade_out: Annotated[
        float | None,
        typer.Option("--music-fade-out", min=0, help="Music fade-out override in seconds."),
    ] = None,
    music_ducking: Annotated[
        bool | None,
        typer.Option(
            "--music-ducking/--no-music-ducking",
            help="Override whether music is lowered when source audio is active.",
        ),
    ] = None,
) -> None:
    """Render the approved shot timeline using an optional named style preset."""
    try:
        preset = resolve_style_preset(style_preset)
        if (output_aspect_ratio is None) != (output_resolution is None):
            raise ValueError("--aspect-ratio and --resolution must be supplied together")
        report = render_approved_shot_timeline(
            folder,
            timeline_path=timeline,
            output_path=output,
            overwrite=overwrite,
            timeout_seconds=timeout_seconds,
            dry_run=dry_run,
            fit_mode=fit_mode or (preset.fit_mode if preset else "crop"),
            output_aspect_ratio=output_aspect_ratio,
            output_resolution=output_resolution,
            title_overlay_path=title_overlays,
            music_path=music,
            music_mode=music_mode or (preset.music_mode if preset else "blend"),
            music_volume=(
                music_volume
                if music_volume is not None
                else (preset.music_volume if preset else 0.22)
            ),
            music_fade_in=(
                music_fade_in
                if music_fade_in is not None
                else (preset.music_fade_in if preset else 1.0)
            ),
            music_fade_out=(
                music_fade_out
                if music_fade_out is not None
                else (preset.music_fade_out if preset else 1.5)
            ),
            music_ducking=(
                music_ducking
                if music_ducking is not None
                else (preset.music_ducking if preset else False)
            ),
            music_loudness_normalization=preset.music_loudness_normalization if preset else False,
            music_target_lufs=preset.music_target_lufs if preset else -16.0,
            music_true_peak_db=preset.music_true_peak_db if preset else -1.5,
            music_loudness_range=preset.music_loudness_range if preset else 11.0,
            beat_punch_in=preset.beat_punch_in if preset else False,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RenderError) as exc:
        console.print(f"[red]Draft render failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if dry_run:
        console.print("[bold green]FFmpeg transition-graph preflight passed.[/bold green]")
        console.print("Inputs and stream metadata were probed; no MP4 was written.")
        return

    console.print("[bold green]Approved shot timeline rendered and verified.[/bold green]")
    if style_preset:
        console.print(f"Style preset: [cyan]{style_preset}[/cyan]")
    if output_aspect_ratio:
        console.print(f"Aspect ratio: {output_aspect_ratio} ({output_resolution})")
    console.print(f"Output: [green]{report.output_path}[/green]")
    console.print(
        f"Duration: {report.actual_duration_seconds:.2f}s "
        f"({report.segment_count} approved shots from {report.source_video_count} source video(s))"
    )
    console.print(
        f"Format: {report.width}x{report.height}, "
        f"{report.video_codec.upper()} video + {report.audio_codec.upper()} audio"
    )
    console.print(
        f"Frame audit: {report.audit_sample_count} timestamps decoded in shuffled order "
        "and checked for deterministic output."
    )
    for warning in report.audit_warnings:
        console.print(f"[yellow]Audit warning:[/yellow] {warning}")
    console.print("[yellow]Original source videos were not modified.[/yellow]")


@render_app.command("init-social-crop-anchors")
@app.command("init-social-crop-anchors", hidden=True)
def init_social_crop_anchors(
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
            "--output",
            help="Template path (default: .video-agent/social-crop-anchors.json).",
        ),
    ] = None,
) -> None:
    """Create an editable per-clip crop-anchor template for the optimized timeline."""
    try:
        destination, clip_ids = create_crop_anchor_template(
            folder, timeline_path=timeline, output_path=output
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        console.print(f"[red]Could not initialize crop anchors:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Crop-anchor template created at [green]{destination}[/green]")
    console.print("Add entries to the anchors array using normalized x/y coordinates (0.0–1.0).")
    console.print(f"Available clip IDs: {', '.join(clip_ids)}")


@render_app.command("render-social-exports")
@app.command("render-social-exports", hidden=True)
def render_social_exports(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder containing the approved optimized timeline and source videos.",
        ),
    ],
    timeline: Annotated[
        Path | None,
        typer.Option("--timeline", help="Path to the reviewed optimized timeline JSON."),
    ] = None,
    resolution: Annotated[
        str,
        typer.Option("--resolution", help="Output resolution: 720p, 1080p, or 4k."),
    ] = "1080p",
    title_overlays: Annotated[
        Path | None,
        typer.Option("--title-overlays", help="Optional reviewed title overlay JSON artifact."),
    ] = None,
    fit_mode: Annotated[
        str | None,
        typer.Option(
            "--fit-mode",
            help="Framing: center crop to fill or pad to preserve the full frame.",
        ),
    ] = None,
    style_preset: Annotated[
        str | None,
        typer.Option("--style-preset", help="Optional audio/framing profile."),
    ] = "social",
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace existing social export files."),
    ] = False,
    reframe: Annotated[
        str,
        typer.Option(
            "--reframe",
            help="Crop positioning: center (default) or face (local face detection per shot).",
        ),
    ] = "center",
    crop_anchors: Annotated[
        Path | None,
        typer.Option(
            "--crop-anchors",
            help="Optional manual per-clip crop-anchor JSON; overrides face detection.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Validate all three output formats without writing MP4 files."
        ),
    ] = False,
) -> None:
    """Render approved footage in landscape, vertical, and square social formats."""
    root = folder.expanduser().resolve()
    export_dir = root / ".video-agent" / "social-exports"
    profiles = (
        ("landscape", "16:9"),
        ("vertical", "9:16"),
        ("square", "1:1"),
    )
    try:
        preset = resolve_style_preset(style_preset)
        # Validate the resolution and framing once before starting a multi-output render.
        from ..rendering.shot_renderer import _requested_output_dimensions

        _requested_output_dimensions("16:9", resolution)
        resolved_fit_mode = fit_mode or (preset.fit_mode if preset else "crop")
        if resolved_fit_mode not in {"crop", "pad"}:
            raise ValueError("fit_mode must be crop or pad")
        if reframe not in {"center", "face"}:
            raise ValueError("--reframe must be center or face")
        if resolved_fit_mode != "crop":
            if reframe == "face":
                raise ValueError("--reframe face requires --fit-mode crop")
            if crop_anchors is not None:
                raise ValueError("--crop-anchors requires --fit-mode crop")
        outputs = []
        for label, aspect_ratio in profiles:
            destination = export_dir / f"{label}-{resolution}.mp4"
            report = render_approved_shot_timeline(
                root,
                timeline_path=timeline,
                output_path=destination,
                overwrite=overwrite,
                dry_run=dry_run,
                fit_mode=resolved_fit_mode,
                reframe_mode=reframe,
                crop_anchors_path=crop_anchors,
                output_aspect_ratio=aspect_ratio,
                output_resolution=resolution,
                title_overlay_path=title_overlays,
                music_mode=preset.music_mode if preset else "blend",
                music_volume=preset.music_volume if preset else 0.22,
                music_fade_in=preset.music_fade_in if preset else 1.0,
                music_fade_out=preset.music_fade_out if preset else 1.5,
                music_ducking=preset.music_ducking if preset else False,
                music_loudness_normalization=(
                    preset.music_loudness_normalization if preset else False
                ),
                music_target_lufs=preset.music_target_lufs if preset else -16.0,
                music_true_peak_db=preset.music_true_peak_db if preset else -1.5,
                music_loudness_range=preset.music_loudness_range if preset else 11.0,
                beat_punch_in=preset.beat_punch_in if preset else False,
            )
            outputs.append((label, aspect_ratio, report))
    except (FileNotFoundError, FileExistsError, ValueError, RenderError) as exc:
        console.print(f"[red]Social export failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if dry_run:
        console.print("[bold green]All three social export preflights passed.[/bold green]")
        console.print("No MP4 files were written.")
        return

    table = Table(title="Social Exports")
    table.add_column("Format", style="cyan")
    table.add_column("Aspect ratio")
    table.add_column("Resolution")
    table.add_column("Output")
    for label, aspect_ratio, report in outputs:
        table.add_row(
            label, aspect_ratio, f"{report.width}x{report.height}", str(report.output_path)
        )
    console.print(table)
    if reframe == "face":
        console.print(
            "[yellow]Face-aware framing uses one stable face-centered crop anchor per shot. "
            "Shots without a detected face fall back to center crop; this is not continuous "
            "face tracking.[/yellow]"
        )
    else:
        console.print(
            "[yellow]Framing uses a centered crop or letterbox; review each export for subject "
            "placement.[/yellow]"
        )
