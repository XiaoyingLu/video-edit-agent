import pytest
from pydantic import ValidationError

from video_edit_agent.editing.edit_document import (
    StructuredEditDocument,
    build_edit_document,
)
from video_edit_agent.editing.timeline_schemas import (
    DraftTimelineClip,
    ProjectDraftTimeline,
)


def _timeline():
    clips = [
        DraftTimelineClip(
            clip_id="clip-1",
            sequence_index=1,
            video_path="a.mp4",
            source_start_seconds=2,
            source_end_seconds=7,
            duration_seconds=5,
            combined_score=90,
            pacing_note="Good pacing",
            caption="Opening",
        ),
        DraftTimelineClip(
            clip_id="clip-2",
            sequence_index=2,
            video_path="b.mp4",
            source_start_seconds=10,
            source_end_seconds=16,
            duration_seconds=6,
            combined_score=85,
            pacing_note="Good pacing",
            transition_in="crossfade",
            transition_duration_seconds=1,
        ),
    ]
    return ProjectDraftTimeline(
        project_path="/tmp/project",
        style="cinematic",
        pacing="moderate",
        target_duration_seconds=20,
        proposed_duration_seconds=10,
        proposed_clip_count=2,
        clips=clips,
    )


def test_build_edit_document_creates_typed_tracks_and_output_time_mapping():
    document = build_edit_document(
        _timeline(),
        source_timeline=".video-agent/optimized-timeline.json",
    )
    video = next(track for track in document.tracks if track.kind == "video")
    audio = next(track for track in document.tracks if track.kind == "audio")
    titles = next(track for track in document.tracks if track.kind == "title")

    assert video.clips[0].timeline_start_seconds == 0
    assert video.clips[1].timeline_start_seconds == 4
    assert video.clips[1].source_start_seconds == 10
    assert audio.clips[1].gain_db is None
    assert titles.clips[0].text == "Opening"
    assert document.duration_seconds == 10


def test_edit_document_rejects_duplicate_track_ids():
    document = build_edit_document(_timeline(), source_timeline="timeline.json")
    with pytest.raises(ValidationError, match="track IDs must be unique"):
        StructuredEditDocument.model_validate(
            {**document.model_dump(), "tracks": [document.tracks[0], document.tracks[0]]}
        )
