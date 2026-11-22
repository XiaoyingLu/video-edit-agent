from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from video_edit_agent.artifact_tracking import register_artifact
from video_edit_agent.rendering.audio_mix import (
    AudioMixError,
    append_music_input,
    append_music_mix,
    resolve_selected_music,
)


def test_append_music_input_loops_track_and_returns_next_input_index(tmp_path: Path) -> None:
    music = tmp_path / "music.wav"
    music.write_bytes(b"audio")
    command = ["ffmpeg", "-i", "video.mp4"]

    index = append_music_input(command, music)

    assert index == 1
    assert command[-4:] == ["-stream_loop", "-1", "-i", str(music)]


def test_blend_graph_includes_fades_and_sidechain_ducking() -> None:
    filters: list[str] = []

    output = append_music_mix(
        filters,
        base_label="baseaudio",
        music_input_index=2,
        duration_seconds=12,
        mode="blend",
        volume=0.18,
        fade_in_seconds=0.8,
        fade_out_seconds=1.2,
        ducking=True,
    )

    graph = ";".join(filters)
    assert output == "audiomix"
    assert "atrim=duration=12.000000" in graph
    assert "volume=0.1800" in graph
    assert "afade=t=in:st=0:d=0.800000" in graph
    assert "afade=t=out:st=10.800000:d=1.200000" in graph
    assert "sidechaincompress=threshold=0.025:ratio=8" in graph
    assert "[baseaudio][duckedmusic]amix=inputs=2" in graph


def test_music_only_graph_does_not_mix_original_audio() -> None:
    filters: list[str] = []

    output = append_music_mix(
        filters,
        base_label="baseaudio",
        music_input_index=1,
        duration_seconds=5,
        mode="music_only",
    )

    assert output == "audiomix"
    assert filters[-1] == "[musicbed]anull[audiomix]"
    assert not any("amix=" in item for item in filters)


def test_original_mode_preserves_original_audio() -> None:
    filters: list[str] = []

    output = append_music_mix(
        filters,
        base_label="baseaudio",
        music_input_index=1,
        duration_seconds=5,
        mode="original",
    )

    assert output == "baseaudio"
    assert filters == []


def test_music_only_requires_a_selected_track() -> None:
    with pytest.raises(AudioMixError, match="requires a selected soundtrack"):
        append_music_mix(
            [],
            base_label="baseaudio",
            music_input_index=None,
            duration_seconds=5,
            mode="music_only",
        )


def test_resolve_selected_music_accepts_only_project_local_audio(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    music = root / "music.wav"
    music.write_bytes(b"audio")

    assert resolve_selected_music(root, music_path=Path("music.wav")) == music.resolve()

    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"audio")
    with pytest.raises(AudioMixError, match="inside the project"):
        resolve_selected_music(root, music_path=outside)


def test_resolve_selected_music_reads_tracked_generated_selection(tmp_path: Path) -> None:
    root = tmp_path / "project"
    artifact_dir = root / ".video-agent"
    artifact_dir.mkdir(parents=True)
    music = artifact_dir / "generated.wav"
    music.write_bytes(b"audio")
    selection_file = artifact_dir / "soundtrack-selection.json"
    selection_file.write_text(
        json.dumps(
            {
                "project_path": str(root.resolve()),
                "candidate": {"asset_path": ".video-agent/generated.wav"},
            }
        ),
        encoding="utf-8",
    )
    register_artifact(root, selection_file)

    assert resolve_selected_music(root) == music.resolve()


def test_resolve_selected_catalog_track_requires_local_download(tmp_path: Path) -> None:
    root = tmp_path / "project"
    artifact_dir = root / ".video-agent"
    artifact_dir.mkdir(parents=True)
    selection_file = artifact_dir / "soundtrack-selection.json"
    selection_file.write_text(
        json.dumps(
            {
                "project_path": str(root.resolve()),
                "candidate": {"asset_path": None},
            }
        ),
        encoding="utf-8",
    )
    register_artifact(root, selection_file)

    with pytest.raises(AudioMixError, match="not downloaded"):
        resolve_selected_music(root)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_music_mix_filter_graph_renders_a_duration_bounded_audio_file(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    source = tmp_path / "source.wav"
    music = tmp_path / "music.wav"
    output = tmp_path / "mixed.wav"
    for destination, frequency, duration in (
        (source, 440, 2),
        (music, 660, 0.7),
    ):
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={frequency}:sample_rate=48000:duration={duration}",
                str(destination),
            ],
            check=True,
            capture_output=True,
        )

    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
    music_index = append_music_input(command, music)
    filters = ["[0:a]atrim=duration=2,asetpts=PTS-STARTPTS[baseaudio]"]
    output_label = append_music_mix(
        filters,
        base_label="baseaudio",
        music_input_index=music_index,
        duration_seconds=2,
        mode="blend",
        volume=0.2,
        fade_in_seconds=0.1,
        fade_out_seconds=0.2,
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            f"[{output_label}]",
            "-t",
            "2",
            str(output),
        ]
    )
    subprocess.run(command, check=True, capture_output=True)

    assert output.is_file()
    probe = subprocess.run(
        [
            shutil.which("ffprobe") or "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert float(probe.stdout.strip()) == pytest.approx(2, abs=0.1)
