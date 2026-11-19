"""Content-aware soundtrack planning and discovery."""

from .brief import create_soundtrack_brief
from .schemas import SoundtrackBrief, SoundtrackEnergySegment

__all__ = ["SoundtrackBrief", "SoundtrackEnergySegment", "create_soundtrack_brief"]
