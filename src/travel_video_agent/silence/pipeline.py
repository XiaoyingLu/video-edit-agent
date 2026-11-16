"""Project-level speech-aware silence analysis."""

import json
from pathlib import Path

from ..media import analyze_folder, discover_videos
from ..models import relative_media_path
from ..transcription.schemas import ProjectTranscription, TranscriptSegment
from .ffmpeg_detector import FFmpegSilenceDetector
from .schemas import (
    ProjectSilenceAnalysis,
    SilenceCandidate,
    SilenceInterval,
    VideoSilenceAnalysis,
)


def _speech_free_candidates(
    silence_intervals: list[SilenceInterval],
    segments: list[TranscriptSegment],
    speech_padding_seconds: float,
) -> list[SilenceCandidate]:
    """Keep only quiet intervals that do not overlap padded transcript speech."""
    padded_speech = [
        (
            max(0.0, segment.start_seconds - speech_padding_seconds),
            segment.end_seconds + speech_padding_seconds,
        )
        for segment in segments
    ]
    candidates: list[SilenceCandidate] = []
    for silence in silence_intervals:
        overlaps_speech = any(
            silence.start_seconds < speech_end and silence.end_seconds > speech_start
            for speech_start, speech_end in padded_speech
        )
        if not overlaps_speech:
            candidates.append(
                SilenceCandidate(
                    start_seconds=silence.start_seconds,
                    end_seconds=silence.end_seconds,
                    duration_seconds=silence.duration_seconds,
                )
            )
    return candidates


def analyze_silence_folder(
    folder: Path,
    noise_threshold_db: float = -35.0,
    min_silence_duration: float = 1.5,
    speech_padding_seconds: float = 0.25,
) -> ProjectSilenceAnalysis:
    """Find quiet audio intervals and correlate them with an existing transcript.

    Without a transcript artifact, quiet intervals are reported as observations only;
    no speech-free candidates are emitted.
    """
    if not -90.0 <= noise_threshold_db <= -1.0:
        raise ValueError("noise_threshold_db must be between -90 and -1 dB")
    if min_silence_duration <= 0:
        raise ValueError("min_silence_duration must be positive")
    if speech_padding_seconds < 0:
        raise ValueError("speech_padding_seconds must be non-negative")

    root = folder.expanduser().resolve()
    analysis = analyze_folder(root)
    transcript_path = root / ".video-agent" / "transcription.json"
    transcripts_by_path: dict[str, list[TranscriptSegment]] = {}
    if transcript_path.is_file():
        transcription = ProjectTranscription.model_validate_json(
            transcript_path.read_text(encoding="utf-8")
        )
        transcripts_by_path = {
            video.path: video.segments for video in transcription.videos if video.has_audio
        }

    detector = FFmpegSilenceDetector()
    videos: list[VideoSilenceAnalysis] = []
    for path in discover_videos(root):
        relative_path = relative_media_path(path, root)
        info = next(item for item in analysis.videos if item.path == relative_path)
        if not info.has_audio:
            videos.append(
                VideoSilenceAnalysis(
                    path=relative_path,
                    duration_seconds=info.duration_seconds,
                    has_audio=False,
                )
            )
            continue

        intervals = detector.detect(
            path,
            info.duration_seconds,
            noise_threshold_db=noise_threshold_db,
            min_silence_duration=min_silence_duration,
        )
        has_transcript = relative_path in transcripts_by_path
        candidates = (
            _speech_free_candidates(
                intervals,
                transcripts_by_path[relative_path],
                speech_padding_seconds,
            )
            if has_transcript
            else []
        )
        videos.append(
            VideoSilenceAnalysis(
                path=relative_path,
                duration_seconds=info.duration_seconds,
                has_audio=True,
                transcript_available=has_transcript,
                audio_silences=intervals,
                candidates=candidates,
            )
        )

    return ProjectSilenceAnalysis(
        project_path=str(root),
        video_count=analysis.video_count,
        analyzed_video_count=sum(video.has_audio for video in videos),
        noise_threshold_db=noise_threshold_db,
        min_silence_duration=min_silence_duration,
        speech_padding_seconds=speech_padding_seconds,
        videos=videos,
    )
