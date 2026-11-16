import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from travel_video_agent.audio.pipeline import detect_speech_pauses
from travel_video_agent.audio.schemas import SpeechSegment
from travel_video_agent.audio.transcriber import FasterWhisperTranscriber, TranscriptionError


def test_pause_detection_finds_gaps_before_between_and_after_speech() -> None:
    segments = [
        SpeechSegment(start_seconds=2, end_seconds=3, text="Hello"),
        SpeechSegment(start_seconds=5, end_seconds=6, text="World"),
    ]
    pauses = detect_speech_pauses(segments, 8, min_pause_duration=1)
    assert [(item.start_seconds, item.end_seconds) for item in pauses] == [
        (0, 2), (3, 5), (6, 8)
    ]


def test_pause_detection_ignores_short_gaps_and_clamps_to_duration() -> None:
    segments = [
        SpeechSegment(start_seconds=0.5, end_seconds=1, text="Hi"),
        SpeechSegment(start_seconds=1.4, end_seconds=5, text="There"),
    ]
    pauses = detect_speech_pauses(segments, 4, min_pause_duration=0.5)
    assert len(pauses) == 0


def test_pause_detection_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError):
        detect_speech_pauses([], 5, min_pause_duration=-1)


def test_transcriber_reports_missing_optional_dependency(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(TranscriptionError, match="audio"):
        FasterWhisperTranscriber()


def test_transcriber_cleans_up_temporary_audio(monkeypatch, tmp_path: Path) -> None:
    class FakeModel:
        def transcribe(self, audio_path, **kwargs):
            assert Path(audio_path).exists()
            segment = SimpleNamespace(start=0.5, end=1.5, text=" hello ", avg_logprob=-0.1)
            return iter([segment]), SimpleNamespace(language="en", language_probability=0.99)

    transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
    transcriber.model = FakeModel()
    transcriber.model_name = "tiny"
    transcriber.language = None
    monkeypatch.setattr("travel_video_agent.audio.transcriber.shutil.which", lambda _: "ffmpeg")
    monkeypatch.setattr(
        "travel_video_agent.audio.transcriber.subprocess.run", lambda *args, **kwargs: None
    )
    segments, language, probability = transcriber.transcribe(tmp_path / "clip.mp4", tmp_path)
    assert segments[0]["text"] == "hello"
    assert language == "en"
    assert probability == 0.99
    assert list(tmp_path.iterdir()) == []


def test_silent_project_does_not_load_whisper_model(monkeypatch, tmp_path: Path) -> None:
    from travel_video_agent.audio import pipeline

    monkeypatch.setattr(
        pipeline,
        "analyze_folder",
        lambda root: SimpleNamespace(video_count=0, videos=[]),
    )
    monkeypatch.setattr(
        pipeline,
        "FasterWhisperTranscriber",
        lambda **kwargs: pytest.fail("Whisper should not load for a project with no audio"),
    )
    result = pipeline.transcribe_folder(tmp_path)
    assert result.video_count == 0
    assert result.transcribed_count == 0
    assert result.transcripts == []
