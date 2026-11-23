"""Compatibility facade for the split rendering CLI modules."""

from . import rendering_core as _rendering_core  # noqa: F401
from . import rendering_review as _rendering_review  # noqa: F401
from .rendering_core import *  # noqa: F403
from .rendering_review import *  # noqa: F403
