from pathlib import Path

import pytest

from travel_video_agent.models import MediaInfo, ProjectAnalysis
from travel_video_agent.transcription.pipeline import transcribe_folder
from travel_video_agent.transcription.schemas import TranscriptSegment
from travel_video_agent.transcription.whisper_backend import FasterWhisperTranscriber


def test_transcript_segment_validates_timestamps() -> None:
    segment = TranscriptSegment(start_seconds=1.25, end_seconds=2.5, text="Hello")
    assert segment.start_seconds == 1.25
    assert segment.end_seconds == 2.5


def test_transcribe_folder_skips_video_without_audio(tmp_path: Path, monkeypatch) -> None:
    from travel_video_agent.transcription import pipeline

    silent_video = tmp_path / "silent.mp4"
    silent_video.touch()
    analysis = ProjectAnalysis(
        project_path=str(tmp_path),
        video_count=1,
        total_duration_seconds=4.0,
        videos=[
            MediaInfo(
                path="silent.mp4",
                duration_seconds=4.0,
                has_video=True,
                has_audio=False,
            )
        ],
    )
    monkeypatch.setattr(pipeline, "analyze_folder", lambda folder: analysis)
    monkeypatch.setattr(pipeline, "discover_videos", lambda folder: [silent_video])

    result = transcribe_folder(tmp_path)

    assert result.video_count == 1
    assert result.transcribed_video_count == 0
    assert result.segment_count == 0
    assert result.videos[0].has_audio is False


def test_transcribe_folder_builds_timestamped_transcript(tmp_path: Path, monkeypatch) -> None:
    from travel_video_agent.transcription import pipeline

    video = tmp_path / "clip.mp4"
    video.touch()
    analysis = ProjectAnalysis(
        project_path=str(tmp_path),
        video_count=1,
        total_duration_seconds=8.0,
        videos=[
            MediaInfo(
                path="clip.mp4",
                duration_seconds=8.0,
                has_video=True,
                has_audio=True,
            )
        ],
    )

    class FakeTranscriber:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def transcribe(self, path: Path) -> dict:
            return {
                "language": "en",
                "language_probability": 0.98,
                "segments": [
                    {
                        "start_seconds": 1.0,
                        "end_seconds": 2.2,
                        "text": " A scenic view. ",
                        "average_log_probability": -0.2,
                        "no_speech_probability": 0.01,
                    }
                ],
            }

    monkeypatch.setattr(pipeline, "analyze_folder", lambda folder: analysis)
    monkeypatch.setattr(pipeline, "discover_videos", lambda folder: [video])
    monkeypatch.setattr(pipeline, "FasterWhisperTranscriber", FakeTranscriber)

    result = transcribe_folder(tmp_path, model="tiny")

    assert result.segment_count == 1
    assert result.transcribed_video_count == 1
    assert result.videos[0].language == "en"
    assert result.videos[0].segments[0].text == "A scenic view."
    assert result.videos[0].segments[0].start_seconds == 1.0
    assert len(result.videos[0].pauses) == 2


def test_backend_reports_missing_optional_dependency(monkeypatch) -> None:
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    from travel_video_agent.transcription.whisper_backend import TranscriptionError

    with pytest.raises(TranscriptionError, match="optional dependency"):
        FasterWhisperTranscriber()


def test_detect_speech_pauses_reports_long_gaps_only() -> None:
    from travel_video_agent.transcription.pipeline import detect_speech_pauses

    segments = [
        TranscriptSegment(start_seconds=2.0, end_seconds=3.0, text="First"),
        TranscriptSegment(start_seconds=5.0, end_seconds=6.0, text="Second"),
    ]
    pauses = detect_speech_pauses(segments, 8.0, min_pause_duration=1.5)
    assert [(pause.start_seconds, pause.end_seconds) for pause in pauses] == [
        (0.0, 2.0),
        (3.0, 5.0),
        (6.0, 8.0),
    ]
