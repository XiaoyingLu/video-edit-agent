"""Generate subtitle cues aligned to a validated, reviewed project timeline."""

from dataclasses import dataclass
from pathlib import Path

from ..editing.schemas import ProjectTimeline
from ..transcription.schemas import ProjectTranscription


class SubtitleError(RuntimeError):
    """Raised when subtitle inputs are missing, inconsistent, or unsafe."""


@dataclass(frozen=True)
class SubtitleCue:
    """One subtitle cue in the final rendered video's timebase."""

    start_seconds: float
    end_seconds: float
    text: str


def _load_timeline(root: Path, path: Path | None) -> tuple[ProjectTimeline, Path]:
    timeline_path = (path or root / ".video-agent" / "timeline.json").expanduser().resolve()
    if not timeline_path.is_file():
        raise FileNotFoundError(
            f"Missing {timeline_path}; run 'video-agent build-timeline' first."
        )
    timeline = ProjectTimeline.model_validate_json(timeline_path.read_text(encoding="utf-8"))
    if Path(timeline.project_path).resolve() != root:
        raise SubtitleError("timeline belongs to a different project folder")
    if timeline.rendering_performed:
        raise SubtitleError("timeline artifact must describe an edit decision, not a prior render")
    return timeline, timeline_path


def _split_text(text: str, durations: list[float]) -> list[str]:
    """Distribute words over fragments when a spoken cue crosses an edited cut."""
    words = text.split()
    if len(durations) <= 1 or len(words) <= 1:
        return [text] * len(durations)
    total = sum(durations)
    result: list[str] = []
    cursor = 0
    for index in range(len(durations)):
        if index == len(durations) - 1:
            end = len(words)
        else:
            end = max(cursor + 1, round(len(words) * (sum(durations[: index + 1]) / total)))
            end = min(end, len(words) - (len(durations) - index - 1))
        result.append(" ".join(words[cursor:end]))
        cursor = end
    return result


def build_subtitle_cues(
    timeline: ProjectTimeline, transcription: ProjectTranscription
) -> list[SubtitleCue]:
    """Map source-time transcript segments onto the concatenated retained timeline."""
    transcript_by_path = {video.path: video for video in transcription.videos}
    cues: list[SubtitleCue] = []
    output_cursor = 0.0

    for video in timeline.videos:
        transcript = transcript_by_path.get(video.path)
        if transcript is not None:
            if abs(transcript.duration_seconds - video.source_duration_seconds) > max(
                0.25, video.source_duration_seconds * 0.005
            ):
                raise SubtitleError(
                    f"transcription duration does not match timeline source: {video.path}"
                )
            for speech in transcript.segments:
                if speech.end_seconds <= speech.start_seconds:
                    continue
                if speech.end_seconds > video.source_duration_seconds + 0.01:
                    raise SubtitleError(
                        f"transcript segment exceeds source duration: {video.path}"
                    )
                fragments: list[tuple[float, float, float]] = []
                segment_cursor = output_cursor
                for retained in video.segments:
                    start = max(speech.start_seconds, retained.start_seconds)
                    end = min(speech.end_seconds, retained.end_seconds)
                    if end - start > 0.02:
                        mapped_start = segment_cursor + start - retained.start_seconds
                        mapped_end = segment_cursor + end - retained.start_seconds
                        fragments.append((mapped_start, mapped_end, end - start))
                    segment_cursor += retained.end_seconds - retained.start_seconds
                if fragments:
                    texts = _split_text(speech.text, [item[2] for item in fragments])
                    for (start, end, _), text in zip(fragments, texts, strict=True):
                        cleaned = " ".join(text.replace("\r", " ").replace("\n", " ").split())
                        if cleaned:
                            cues.append(SubtitleCue(start, end, cleaned))
        output_cursor += video.retained_duration_seconds

    return sorted(cues, key=lambda cue: (cue.start_seconds, cue.end_seconds))


def _srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{millis:03d}"


def render_srt(cues: list[SubtitleCue]) -> str:
    """Serialize cues using SubRip timestamps."""
    blocks = []
    for index, cue in enumerate(cues, start=1):
        if cue.end_seconds <= cue.start_seconds:
            continue
        blocks.append(
            f"{index}\n{_srt_timestamp(cue.start_seconds)} --> "
            f"{_srt_timestamp(cue.end_seconds)}\n{cue.text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def generate_subtitles(
    folder: Path,
    timeline_path: Path | None = None,
    transcription_path: Path | None = None,
    output_path: Path | None = None,
) -> tuple[Path, int]:
    """Generate a UTF-8 SRT sidecar using the reviewed timeline and Whisper artifact."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project folder does not exist: {root}")
    timeline, timeline_file = _load_timeline(root, timeline_path)
    transcript_file = (
        (transcription_path or root / ".video-agent" / "transcription.json")
        .expanduser()
        .resolve()
    )
    if not transcript_file.is_file():
        raise FileNotFoundError(
            f"Missing {transcript_file}; run 'video-agent transcribe' first."
        )
    transcription = ProjectTranscription.model_validate_json(
        transcript_file.read_text(encoding="utf-8")
    )
    if Path(transcription.project_path).resolve() != root:
        raise SubtitleError("transcription belongs to a different project folder")

    destination = (
        output_path or root / ".video-agent" / "subtitles.srt"
    ).expanduser().resolve()
    if destination in {timeline_file, transcript_file}:
        raise SubtitleError("subtitle output cannot overwrite an input artifact")
    source_paths = {(root / video.path).resolve() for video in timeline.videos}
    if destination in source_paths:
        raise SubtitleError("subtitle output cannot overwrite a source video")
    cues = build_subtitle_cues(timeline, transcription)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_srt(cues), encoding="utf-8")
    return destination, len(cues)
