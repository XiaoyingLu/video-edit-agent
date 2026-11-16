"""Speech transcription for local video projects."""

from .pipeline import transcribe_folder
from .schemas import ProjectTranscription, TranscriptSegment, VideoTranscript

__all__ = ["ProjectTranscription", "TranscriptSegment", "VideoTranscript", "transcribe_folder"]
