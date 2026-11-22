"""Commands for the live local review workspace."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from ..editing.review_server import create_review_server
from ..editing.review_workspace import generate_review_workspace
from .common import app

console = Console()


@app.command("review-serve")
def review_serve(
    folder: Annotated[
        Path,
        typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    host: Annotated[
        str,
        typer.Option("--host", help="Bind address: 127.0.0.1 (local) or 0.0.0.0 (container)."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", min=0, max=65535, help="Local port (0 chooses an available port)."),
    ] = 0,
) -> None:
    """Serve the project review workspace with direct saving of timeline draft edits."""
    try:
        workspace = generate_review_workspace(folder)
        server = create_review_server(folder, host=host, port=port)
    except (OSError, ValueError) as exc:
        console.print(f"[red]Review server failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    bind_address, selected_port = server.server_address
    address = "127.0.0.1" if bind_address == "0.0.0.0" else bind_address
    url = (
        f"http://{address}:{selected_port}/"
        f"{workspace.relative_to(folder.resolve()).as_posix()}"
        f"?token={server.session_token}"
    )
    console.print(f"Project review workspace: [green]{url}[/green]")
    console.print(
        "[yellow]Local-only server. Browser timeline edits save as separate unapproved "
        "revisions; review and approve them before rendering. Press Ctrl+C to stop.[/yellow]"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\nReview server stopped.")
    finally:
        server.server_close()
