"""Video editing command-line interface."""

from .common import app
from . import understanding, rendering, planning, music, maintenance  # noqa: F401

__all__ = ["app"]
