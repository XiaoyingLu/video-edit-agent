"""Project-level speech-aware silence analysis."""

from pathlib import Path

from ..artifact_tracking import validate_artifact
from ..media import analyze_folder, discover_videos
from ..models import ProjectAnalysis, relative_media_path
from ..transcription.schemas import ProjectTranscription, TranscriptSegment
from .ffmpeg_detector import FFmpegSilenceDetector
from .schemas import (
    ProjectSilenceAnalysis,
    SilenceCandidate,
    SilenceInterval,
    VideoSilenceAnalysis,
)


def _load_transcripts_by_path(
    transcript_path: Path,
    root: Path,
    analysis: ProjectAnalysis,
) -> dict[str, list[TranscriptSegment]]:
    """Load only a transcript artifact that matches the current media inventory."""
    if not transcript_path.is_file():
        return {}

    transcription = ProjectTranscription.model_validate_json(
        transcript_path.read_text(encoding="utf-8")
    )
    if Path(transcription.project_path).expanduser().resolve() != root:
        raise ValueError(
            "transcription artifact belongs to a different project folder; "
            "rerun 'video-agent transcribe'"
        )

    media_by_path = {video.path: video for video in analysis.videos}
    if len(media_by_path) != len(analysis.videos):
        raise ValueError("media analysis contains duplicate video paths")

    transcripts_by_path = {}
    for video in transcription.videos:
        if video.path in transcripts_by_path:
            raise ValueError(f"transcription artifact contains duplicate video path: {video.path}")

        media = media_by_path.get(video.path)
        if media is None:
            raise ValueError(
                f"transcription artifact references unknown video {video.path!r}; "
                "rerun 'video-agent transcribe'"
            )
        if video.has_audio != media.has_audio:
            raise ValueError(
                f"audio-stream metadata changed for {video.path!r}; rerun 'video-agent transcribe'"
            )
        if abs(video.duration_seconds - media.duration_seconds) > 0.05:
            raise ValueError(
                f"video duration changed for {video.path!r}; rerun 'video-agent transcribe'"
            )
        if not video.has_audio and video.segments:
            raise ValueError(
                f"transcript contains speech for a video without audio: {video.path!r}"
            )

        for segment in video.segments:
            if segment.end_seconds <= segment.start_seconds:
                raise ValueError(
                    f"transcript contains an invalid speech interval for {video.path!r}"
                )
            if segment.end_seconds > media.duration_seconds + 0.01:
                raise ValueError(
                    f"transcript speech interval exceeds source duration for {video.path!r}; "
                    "rerun 'video-agent transcribe'"
                )

        transcripts_by_path[video.path] = video

    missing_paths = media_by_path.keys() - transcripts_by_path.keys()
    if missing_paths:
        missing = ", ".join(sorted(missing_paths))
        raise ValueError(
            f"transcription artifact is missing video(s): {missing}; rerun 'video-agent transcribe'"
        )

    if transcription.video_count != len(media_by_path):
        raise ValueError("transcription video count does not match media analysis")
    if transcription.transcribed_video_count != sum(
        video.has_audio for video in transcription.videos
    ):
        raise ValueError("transcription audio-video count is inconsistent")
    if transcription.segment_count != sum(len(video.segments) for video in transcription.videos):
        raise ValueError("transcription segment count is inconsistent")

    return {path: video.segments for path, video in transcripts_by_path.items() if video.has_audio}


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
    if transcript_path.is_file():
        validate_artifact(root, transcript_path)
    transcripts_by_path = _load_transcripts_by_path(transcript_path, root, analysis)

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
