"""Cross-stage context signal serialization and output attachment tests."""

from video_edit_agent.editing.intent_schemas import EditingPlan
from video_edit_agent.transcription.schemas import (
    PauseInterval,
    ProjectTranscription,
    TranscriptSegment,
    VideoTranscript,
)
from video_edit_agent.understanding.schemas import VideoScene, VideoUnderstanding


def test_understanding_output_attaches_compact_scene_context() -> None:
    result = VideoUnderstanding(
        project_path="/project",
        video_count=1,
        scene_count=1,
        model="test-model",
        scenes=[
            VideoScene(
                scene_id="scene-1",
                source_video_path="clip.mp4",
                start_seconds=1,
                end_seconds=4,
                summary="A person walks along a waterfront.",
                story_function="action",
                dominant_mood="serene",
            )
        ],
    )

    signal = result.context_signal
    assert signal is not None
    assert signal.scenes[0].scene_id == "scene-1"
    assert signal.scenes[0].visual_role == "action"
    assert signal.scenes[0].start_seconds == 1


def test_transcription_output_attaches_speech_and_silence_boundaries() -> None:
    result = ProjectTranscription(
        project_path="/project",
        video_count=1,
        transcribed_video_count=1,
        segment_count=1,
        model="small",
        videos=[
            VideoTranscript(
                path="clip.mp4",
                duration_seconds=8,
                has_audio=True,
                segments=[
                    TranscriptSegment(
                        start_seconds=2,
                        end_seconds=3,
                        text="Hello there",
                    )
                ],
                pauses=[
                    PauseInterval(
                        start_seconds=3,
                        end_seconds=5,
                        duration_seconds=2,
                    )
                ],
            )
        ],
    )

    signal = result.context_signal
    assert signal is not None
    assert signal.speech_boundaries[0].text == "Hello there"
    assert signal.silence_boundaries[0].start_seconds == 3
    assert signal.silence_boundaries[0].end_seconds == 5


def test_editing_plan_attaches_target_format_and_pacing() -> None:
    plan = EditingPlan(
        source_prompt="Make a concise travel film",
        aspect_ratio="9:16",
        pacing="fast",
    )

    signal = plan.context_signal
    assert signal is not None
    assert signal.target_aspect_ratio == "9:16"
    assert signal.pacing == "fast"
    assert signal.pacing_constraints[0].max_seconds == 3.5
