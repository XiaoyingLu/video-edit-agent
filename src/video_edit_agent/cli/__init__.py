"""Video editing command-line interface."""

from . import color, maintenance, music, planning, rendering, review, templates, understanding  # noqa: F401
from .common import app, music_app, render_app, story_app

app.add_typer(story_app, name="story")
app.add_typer(music_app, name="music")
app.add_typer(render_app, name="render")
app.add_typer(templates.template_app, name="template")

__all__ = ["app"]
