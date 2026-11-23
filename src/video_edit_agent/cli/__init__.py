"""Video editing command-line interface."""

from . import maintenance, music, planning, rendering, review, understanding  # noqa: F401
from .common import app, music_app, render_app, story_app

app.add_typer(story_app, name="story")
app.add_typer(music_app, name="music")
app.add_typer(render_app, name="render")

__all__ = ["app"]
