"""Video editing command-line interface."""

from .common import app
from . import maintenance, music, planning, rendering, understanding  # noqa: F401

__all__ = ["app"]
