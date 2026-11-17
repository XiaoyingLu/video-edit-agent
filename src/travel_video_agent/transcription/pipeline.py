"""Project-level speech transcription orchestration."""

from pathlib import Path

from ..media import analyze_folder, discover_videos
from ..models import relative_media_path
from .schemas import (
    PauseInterval,
    ProjectTranscription,
    TranscriptSegment,
    VideoTranscript,
)
from .whisper_backend import FasterWhisperTranscriber


def detect_speech_pauses(
    segments: list[TranscriptSegment],
    duration_seconds: float,
    min_pause_duration: float = 1.0,
) -> list[PauseInterval]:
    """Find long gaps around speech; gaps are metadata, never automatic cut decisions."""
    if min_pause_duration < 0:
        raise ValueError("min_pause_duration must be non-negative")

    pauses: list[PauseInterval] = []
    cursor = 0.0
    for segment in sorted(segments, key=lambda item: item.start_seconds):
        start = min(duration_seconds, max(0.0, segment.start_seconds))
        end = min(duration_seconds, max(start, segment.end_seconds))
        gap = start - cursor
        if gap >= min_pause_duration and gap > 0:
            pauses.append(
                PauseInterval(
                    start_seconds=cursor,
                    end_seconds=start,
                    duration_seconds=gap,
                )
            )
        cursor = max(cursor, end)

    tail = max(0.0, duration_seconds - cursor)
    if tail >= min_pause_duration and tail > 0:
        pauses.append(
            PauseInterval(
                start_seconds=cursor,
                end_seconds=duration_seconds,
                duration_seconds=tail,
            )
        )
    return pauses


def transcribe_folder(
    folder: Path,
    model: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    min_pause_duration: float = 1.0,
) -> ProjectTranscription:
    """Transcribe every video with an audio stream; source media remains untouched."""
    if min_pause_duration < 0:
        raise ValueError("min_pause_duration must be non-negative")

    root = folder.expanduser().resolve()
    analysis = analyze_folder(root)
    media_with_audio = {item.path for item in analysis.videos if item.has_audio}
    backend = (
        FasterWhisperTranscriber(
            model=model,
            device=device,
            compute_type=compute_type,
            download_root=root / ".video-agent" / "models",
        )
        if media_with_audio
        else None
    )

    transcripts: list[VideoTranscript] = []
    try:
        for path in discover_videos(root):
            relative_path = relative_media_path(path, root)
            info = next(item for item in analysis.videos if item.path == relative_path)
            if not info.has_audio:
                transcripts.append(
                    VideoTranscript(
                        path=relative_path,
                        duration_seconds=info.duration_seconds,
                        has_audio=False,
                    )
                )
                continue

            assert backend is not None
            result = backend.transcribe(path)
            segments = [TranscriptSegment.model_validate(item) for item in result["segments"]]
            pauses = detect_speech_pauses(segments, info.duration_seconds, min_pause_duration)
            transcripts.append(
                VideoTranscript(
                    path=relative_path,
                    duration_seconds=info.duration_seconds,
                    has_audio=True,
                    language=result.get("language"),
                    language_probability=result.get("language_probability"),
                    segments=segments,
                    pauses=pauses,
                )
            )

        return ProjectTranscription(
            project_path=str(root),
            video_count=analysis.video_count,
            transcribed_video_count=sum(item.has_audio for item in transcripts),
            segment_count=sum(len(item.segments) for item in transcripts),
            model=model,
            videos=transcripts,
        )
    finally:
        if backend is not None:
            backend.close()
