"""Folder-level transcription and speech-bounded pause analysis."""

import hashlib
from pathlib import Path

from ..media import analyze_folder, discover_videos
from .schemas import AudioTranscript, PauseInterval, ProjectTranscription, SpeechSegment
from .transcriber import FasterWhisperTranscriber


def detect_speech_pauses(
    segments: list[SpeechSegment],
    duration_seconds: float,
    min_pause_duration: float = 1.0,
) -> list[PauseInterval]:
    """Find gaps between speech segments without declaring them safe to cut."""
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
    model: str = "tiny",
    language: str | None = None,
    min_pause_duration: float = 1.0,
) -> ProjectTranscription:
    """Transcribe all videos locally and write only derived JSON at the CLI layer."""
    root = folder.expanduser().resolve()
    analysis = analyze_folder(root)
    transcriber = FasterWhisperTranscriber(model=model, language=language)
    transcripts: list[AudioTranscript] = []
    transcribed_count = 0

    for video_path in discover_videos(root):
        relative_path = video_path.relative_to(root).as_posix()
        info = next(video for video in analysis.videos if video.path == relative_path)
        if not info.has_audio:
            transcripts.append(
                AudioTranscript(
                    path=relative_path,
                    duration_seconds=info.duration_seconds,
                    has_audio=False,
                )
            )
            continue

        key = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
        raw_segments, detected_language, language_probability = transcriber.transcribe(
            video_path, root / ".video-agent" / "audio-cache" / key
        )
        segments = [SpeechSegment(**item) for item in raw_segments]
        pauses = detect_speech_pauses(segments, info.duration_seconds, min_pause_duration)
        transcripts.append(
            AudioTranscript(
                path=relative_path,
                duration_seconds=info.duration_seconds,
                has_audio=True,
                language=detected_language,
                language_probability=language_probability,
                segments=segments,
                pauses=pauses,
            )
        )
        transcribed_count += 1

    return ProjectTranscription(
        project_path=str(root),
        video_count=analysis.video_count,
        transcribed_count=transcribed_count,
        model=model,
        transcripts=transcripts,
    )
