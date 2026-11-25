"""Commands for discovering and validating reusable editing templates."""

from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.table import Table

from ..rendering.template_capture import save_template_from_project
from ..rendering.template_preview import TemplatePreviewError, render_template_previews
from ..rendering.title_overlays import suggest_narrative_title_sequences
from ..template_recommendations import recommend_templates
from ..templates import available_template_ids, load_template
from .common import console

template_app = typer.Typer(
    name="template", help="Discover and validate reusable editing templates."
)


@template_app.command("list")
def list_templates(
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder whose templates take precedence over home and built-ins.",
        ),
    ] = None,
) -> None:
    """List available project, home, and built-in templates."""
    table = Table(title="Available Templates")
    table.add_column("ID", style="bold")
    table.add_column("Version")
    table.add_column("Source")
    table.add_column("Status")
    found = False
    for template_id in available_template_ids(project):
        found = True
        try:
            spec, source = load_template(template_id, project)
            source_label = (
                "project"
                if project and source.is_relative_to(project.resolve())
                else "home"
                if source.is_relative_to(Path.home().resolve())
                else "built-in"
            )
            table.add_row(spec.id, str(spec.version), source_label, "valid")
        except (FileNotFoundError, OSError, ValueError) as exc:
            table.add_row(template_id, "—", "—", f"invalid: {exc}")
    if found:
        console.print(table)
    else:
        console.print("No templates found.")


@template_app.command("show")
def show_template(
    template_id: Annotated[str, typer.Argument(help="Template ID, such as cinematic.")],
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder used for template overrides.",
        ),
    ] = None,
) -> None:
    """Show the fully resolved template, including inherited values."""
    try:
        spec, source = load_template(template_id, project)
    except (FileNotFoundError, OSError, ValueError) as exc:
        console.print(f"[red]Template lookup failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[bold]{spec.id}[/bold] v{spec.version} — {source}")
    console.print_json(spec.model_dump_json(indent=2))


@template_app.command("validate")
def validate_template(
    template_id: Annotated[str, typer.Argument(help="Template ID to validate.")],
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder used for template overrides.",
        ),
    ] = None,
) -> None:
    """Validate a template and its inheritance chain without changing project files."""
    try:
        spec, source = load_template(template_id, project)
    except (FileNotFoundError, OSError, ValueError) as exc:
        console.print(f"[red]Template validation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[bold green]Valid template:[/bold green] {spec.id} v{spec.version} ({source})")
    if spec.extends:
        console.print(f"Inheritance: extends [cyan]{spec.extends}[/cyan]")
    console.print(
        f"{len(spec.title_styles)} title style(s), {len(spec.sequences)} sequence(s), "
        f"{len(spec.assets)} asset(s)."
    )

@template_app.command("recommend")
def recommend_template_command(
    mood: Annotated[
        list[str] | None, typer.Option("--mood", help="Mood tag to match; may be repeated.")
    ] = None,
    best_for: Annotated[
        list[str] | None,
        typer.Option("--best-for", help="Use-case tag to match; may be repeated."),
    ] = None,
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder whose templates take precedence over home and built-ins.",
        ),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=3)] = 3,
) -> None:
    """Recommend up to three templates from explicit mood and use-case tags."""
    try:
        results = recommend_templates(
            moods=mood or (), best_for=best_for or (), project_root=project, limit=limit
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        console.print(f"[red]Template recommendation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not results:
        console.print("No templates matched those tags. Try broader or different tags.")
        return

    table = Table(title="Template Recommendations")
    table.add_column("Template", style="bold cyan")
    table.add_column("Score", justify="right")
    table.add_column("Matched tags")
    table.add_column("Source")
    for item in results:
        table.add_row(
            item.template.id,
            str(item.score),
            ", ".join(item.matches),
            str(item.source),
        )
    console.print(table)
    console.print("Recommendations are suggestions only; no template was applied.")


@template_app.command("apply")
def apply_template(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder with an approved timeline and story treatment.",
        ),
    ],
    template_id: Annotated[str, typer.Argument(help="Template ID to apply.")],
    timeline: Annotated[
        Path | None, typer.Option("--timeline", help="Approved optimized timeline JSON.")
    ] = None,
    plan: Annotated[
        Path | None, typer.Option("--plan", help="Creative video project plan JSON.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", help="Output title overlay JSON path.")
    ] = None,
) -> None:
    """Apply a template by generating a review-only title overlay proposal."""
    try:
        document, destination = suggest_narrative_title_sequences(
            folder,
            timeline_path=timeline,
            plan_path=plan,
            output_path=output,
            template_id=template_id,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        console.print(f"[red]Template application failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Template Application Proposal")
    table.add_column("ID", style="cyan")
    table.add_column("Role")
    table.add_column("Text")
    table.add_column("Time", justify="right")
    table.add_column("Decision")
    for item in document.overlays:
        table.add_row(
            item.overlay_id,
            item.sequence_role,
            item.text or "",
            f"{item.start_seconds:.2f}–{item.end_seconds:.2f}s",
            item.decision,
        )
    console.print(table)
    console.print(
        f"Applied [cyan]{document.template_id}[/cyan] v{document.template_version}; "
        f"proposal saved to [green]{destination}[/green]."
    )
    console.print(
        "All generated overlays remain pending review; nothing is approved automatically."
    )


@template_app.command("preview")
def preview_templates(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder with an approved optimized timeline and story treatment.",
        ),
    ],
    template_ids: Annotated[
        list[str], typer.Argument(help="Two or three template IDs to compare.")
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Project-local preview output directory."),
    ] = None,
) -> None:
    """Render side-by-side opening-card previews without applying a template."""
    try:
        index_file, preview_files, timestamp = render_template_previews(
            folder, template_ids, output_dir
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError, TemplatePreviewError) as exc:
        console.print(f"[red]Template preview failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Template Opening-Card Previews")
    table.add_column("Template", style="cyan")
    table.add_column("Preview")
    for template_id, preview_file in zip(template_ids, preview_files, strict=True):
        table.add_row(template_id, str(preview_file))
    console.print(table)
    console.print(f"Sampled approved footage at {timestamp:.2f}s.")
    console.print(f"Comparison page: [green]{index_file}[/green]")
    console.print("Previews are visual comparisons only; no template or title was applied.")


@template_app.command("save")
def save_template(
    folder: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Project folder with an approved optimized timeline.",
        ),
    ],
    template_id: Annotated[
        str, typer.Argument(help="New template ID, using lowercase letters, digits, and hyphens.")
    ],
    from_project: Annotated[
        bool,
        typer.Option(
            "--from-project",
            help="Capture approved title styling and transitions from this project.",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="YAML destination (default: ~/.video-agent/templates/<id>.yaml).",
        ),
    ] = None,
    music_mode: Annotated[
        Literal["original", "music_only", "blend"] | None,
        typer.Option(
            "--music-mode", help="Override inferred profile: original, music_only, or blend."
        ),
    ] = None,
    music_volume: Annotated[float | None, typer.Option("--music-volume", min=0.01, max=2.0)] = None,
    music_fade_in: Annotated[float | None, typer.Option("--music-fade-in", min=0, max=30)] = None,
    music_fade_out: Annotated[float | None, typer.Option("--music-fade-out", min=0, max=30)] = None,
    music_ducking: Annotated[
        bool | None,
        typer.Option("--music-ducking/--no-music-ducking", help="Override inferred music ducking."),
    ] = None,
) -> None:
    """Save a reusable template from reviewed project settings."""
    if not from_project:
        console.print("[red]Specify --from-project to capture a project.[/red]")
        raise typer.Exit(code=1)
    try:
        template, destination, notes = save_template_from_project(
            folder,
            template_id,
            output_path=output,
            music_mode=music_mode,
            music_volume=music_volume,
            music_fade_in=music_fade_in,
            music_fade_out=music_fade_out,
            music_ducking=music_ducking,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        console.print(f"[red]Template save failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold green]Template saved:[/bold green] {template.id} v{template.version}")
    console.print(f"YAML: [green]{destination}[/green]")
    console.print(
        f"Captured {len(template.title_styles)} title style(s), "
        f"{len(template.sequences)} sequence(s), and transition '{template.transitions.default}'."
    )
    console.print(
        f"Music: {template.music.mode}, volume {template.music.volume:.2f}, "
        f"fades {template.music.fade_in:.1f}s/{template.music.fade_out:.1f}s, "
        f"ducking {'on' if template.music.ducking else 'off'}."
    )
    for note in notes:
        console.print(f"[yellow]Note:[/yellow] {note}")
    console.print("No footage, title text, or editorial decisions were copied.")
