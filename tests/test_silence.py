import json
from pathlib import Path

import pytest

from travel_video_agent.silence.ffmpeg_detector import FFmpegSilenceDetector
from travel_video_agent.silence.pipeline import _speech_free_candidates
from travel_video_agent.silence.schemas import SilenceInterval
from travel_video_agent.transcription.schemas import TranscriptSegment


def test_parse_ffmpeg_silencedetect_intervals_and_trailing_silence() -> None:
    log = """
[silencedetect] silence_start: 1.250
[silencedetect] silence_end: 3.750 | silence_duration: 2.500
[silencedetect] silence_start: 8.000
"""
    result = FFmpegSilenceDetector.parse_output(log, 10.0)
    assert [(item.start_seconds, item.end_seconds) for item in result] == [
        (1.25, 3.75),
        (8.0, 10.0),
    ]


def test_parse_ffmpeg_ignores_unmatched_end() -> None:
    result = FFmpegSilenceDetector.parse_output(
        "[silencedetect] silence_end: 2.0 | silence_duration: 1.0", 5.0
    )
    assert result == []


def test_speech_overlap_excludes_quiet_interval_and_padding() -> None:
    silences = [
        SilenceInterval(start_seconds=0, end_seconds=2, duration_seconds=2),
        SilenceInterval(start_seconds=3, end_seconds=5, duration_seconds=2),
        SilenceInterval(start_seconds=6, end_seconds=8, duration_seconds=2),
    ]
    speech = [TranscriptSegment(start_seconds=1, end_seconds=1.5, text="hello")]
    candidates = _speech_free_candidates(silences, speech, 0.25)
    assert [(item.start_seconds, item.end_seconds) for item in candidates] == [
        (3, 5),
        (6, 8),
    ]
    assert all(item.review_required for item in candidates)


def test_no_transcript_means_no_cut_candidates(tmp_path: Path, monkeypatch) -> None:
    from travel_video_agent.silence import pipeline
    from travel_video_agent.models import MediaInfo, ProjectAnalysis

    video = tmp_path / "clip.mp4"
    video.touch()
    analysis = ProjectAnalysis(
        project_path=str(tmp_path),
        video_count=1,
        total_duration_seconds=10,
        videos=[
            MediaInfo(
                path="clip.mp4",
                duration_seconds=10,
                has_video=True,
                has_audio=True,
            )
        ],
    )

    class FakeDetector:
        def detect(self, *args, **kwargs):
            return [SilenceInterval(start_seconds=2, end_seconds=4, duration_seconds=2)]

    monkeypatch.setattr(pipeline, "analyze_folder", lambda folder: analysis)
    monkeypatch.setattr(pipeline, "discover_videos", lambda folder: [video])
    monkeypatch.setattr(pipeline, "FFmpegSilenceDetector", FakeDetector)

    result = pipeline.analyze_silence_folder(tmp_path)
    assert result.videos[0].audio_silences
    assert result.videos[0].transcript_available is False
    assert result.videos[0].candidates == []


def test_transcript_speech_excludes_silence_candidate(tmp_path: Path, monkeypatch) -> None:
    from travel_video_agent.silence import pipeline
    from travel_video_agent.models import MediaInfo, ProjectAnalysis
    from travel_video_agent.transcription.schemas import ProjectTranscription, VideoTranscript

    video = tmp_path / "clip.mp4"
    video.touch()
    artifact_dir = tmp_path / ".video-agent"
    artifact_dir.mkdir()
    transcription = ProjectTranscription(
        project_path=str(tmp_path),
        video_count=1,
        transcribed_video_count=1,
        segment_count=1,
        model="tiny",
        videos=[
            VideoTranscript(
                path="clip.mp4",
                duration_seconds=10,
                has_audio=True,
                segments=[TranscriptSegment(start_seconds=2, end_seconds=3, text="speaking")],
            )
        ],
    )
    (artifact_dir / "transcription.json").write_text(
        transcription.model_dump_json(), encoding="utf-8"
    )
    analysis = ProjectAnalysis(
        project_path=str(tmp_path),
        video_count=1,
        total_duration_seconds=10,
        videos=[
            MediaInfo(
                path="clip.mp4",
                duration_seconds=10,
                has_video=True,
                has_audio=True,
            )
        ],
    )

    class FakeDetector:
        def detect(self, *args, **kwargs):
            return [
                SilenceInterval(start_seconds=1, end_seconds=4, duration_seconds=3),
                SilenceInterval(start_seconds=5, end_seconds=8, duration_seconds=3),
            ]

    monkeypatch.setattr(pipeline, "analyze_folder", lambda folder: analysis)
    monkeypatch.setattr(pipeline, "discover_videos", lambda folder: [video])
    monkeypatch.setattr(pipeline, "FFmpegSilenceDetector", FakeDetector)

    result = pipeline.analyze_silence_folder(tmp_path, speech_padding_seconds=0.2)
    assert len(result.videos[0].audio_silences) == 2
    assert [(item.start_seconds, item.end_seconds) for item in result.videos[0].candidates] == [
        (5, 8)
    ]
