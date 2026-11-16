from pathlib import Path

from travel_video_agent.editing.schemas import (
    ProjectTimeline,
    TimelineSegment,
    VideoTimeline,
)
from travel_video_agent.subtitles import (
    SubtitleCue,
    build_subtitle_cues,
    generate_subtitles,
    render_srt,
)
from travel_video_agent.transcription.schemas import (
    ProjectTranscription,
    TranscriptSegment,
    VideoTranscript,
)


def _artifacts(root: Path):
    timeline = ProjectTimeline(
        project_path=str(root.resolve()),
        approved_cut_count=1,
        retained_duration_seconds=13,
        videos=[
            VideoTimeline(
                path="first.mp4",
                source_duration_seconds=10,
                retained_duration_seconds=8,
                segments=[
                    TimelineSegment(start_seconds=0, end_seconds=4),
                    TimelineSegment(start_seconds=6, end_seconds=10),
                ],
            ),
            VideoTimeline(
                path="second.mp4",
                source_duration_seconds=5,
                retained_duration_seconds=5,
                segments=[TimelineSegment(start_seconds=0, end_seconds=5)],
            ),
        ],
    )
    transcription = ProjectTranscription(
        project_path=str(root.resolve()),
        video_count=2,
        transcribed_video_count=2,
        segment_count=2,
        model="tiny",
        videos=[
            VideoTranscript(
                path="first.mp4",
                duration_seconds=10,
                has_audio=True,
                segments=[
                    TranscriptSegment(
                        start_seconds=2,
                        end_seconds=8,
                        text="one two three four",
                    )
                ],
            ),
            VideoTranscript(
                path="second.mp4",
                duration_seconds=5,
                has_audio=True,
                segments=[TranscriptSegment(start_seconds=1, end_seconds=2, text="next clip")],
            ),
        ],
    )
    return timeline, transcription


def test_build_subtitle_cues_remaps_cut_intervals_and_concatenated_videos(tmp_path):
    timeline, transcription = _artifacts(tmp_path)

    cues = build_subtitle_cues(timeline, transcription)

    assert cues == [
        SubtitleCue(2, 4, "one two"),
        SubtitleCue(4, 6, "three four"),
        SubtitleCue(9, 10, "next clip"),
    ]


def test_build_subtitle_cues_drops_speech_inside_removed_interval(tmp_path):
    timeline, transcription = _artifacts(tmp_path)
    transcription.videos[0].segments = [
        TranscriptSegment(start_seconds=4.5, end_seconds=5.5, text="removed speech")
    ]

    assert build_subtitle_cues(timeline, transcription) == []


def test_render_srt_uses_real_subrip_line_breaks():
    result = render_srt([SubtitleCue(1.25, 2.5, "Hello world")])

    assert result == "1\n00:00:01,250 --> 00:00:02,500\nHello world\n"
    assert "\\n" not in result


def test_generate_subtitles_writes_srt_from_project_artifacts(tmp_path):
    timeline, transcription = _artifacts(tmp_path)
    artifact_dir = tmp_path / ".video-agent"
    artifact_dir.mkdir()
    (artifact_dir / "timeline.json").write_text(timeline.model_dump_json(), encoding="utf-8")
    (artifact_dir / "transcription.json").write_text(
        transcription.model_dump_json(), encoding="utf-8"
    )

    destination, count = generate_subtitles(tmp_path)

    assert destination == artifact_dir / "subtitles.srt"
    assert count == 3
    content = destination.read_text(encoding="utf-8")
    assert "00:00:02,000 --> 00:00:04,000" in content
    assert "00:00:04,000 --> 00:00:06,000" in content
    assert "00:00:09,000 --> 00:00:10,000" in content
