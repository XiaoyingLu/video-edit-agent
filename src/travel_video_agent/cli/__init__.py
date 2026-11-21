"""Video editing command-line interface."""

from . import maintenance, music, planning, rendering, understanding  # noqa: F401
from .common import app

__all__ = ["app"]
