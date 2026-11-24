"""Commands for discovering and validating reusable editing templates."""

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from ..rendering.title_overlays import suggest_narrative_title_sequences
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
