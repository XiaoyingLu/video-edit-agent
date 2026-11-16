"""Speech-aware audio silence analysis."""

from .pipeline import analyze_silence_folder
from .schemas import ProjectSilenceAnalysis, SilenceInterval, SilenceCandidate, VideoSilenceAnalysis

__all__ = [
    "ProjectSilenceAnalysis",
    "SilenceCandidate",
    "SilenceInterval",
    "VideoSilenceAnalysis",
    "analyze_silence_folder",
]
