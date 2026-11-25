"""Shared soundtrack resolution and FFmpeg mixing filters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from ..artifact_tracking import validate_artifact

MusicMode = Literal["original", "music_only", "blend"]


class AudioMixError(ValueError):
    """Raised when a selected soundtrack cannot safely be used for rendering."""


def resolve_selected_music(
    folder: Path,
    *,
    music_path: Path | None = None,
    selection_path: Path | None = None,
) -> Path | None:
    """Resolve an explicitly selected, project-local soundtrack; never guess catalog assets."""
    root = folder.expanduser().resolve()
    selection_file = selection_path or root / ".video-agent" / "soundtrack-selection.json"
    selection_file = selection_file.expanduser()
    if not selection_file.is_absolute():
        selection_file = (root / selection_file).resolve()
    else:
        selection_file = selection_file.resolve()

    if music_path is None and not selection_file.is_file():
        return None

    if music_path is not None:
        selected = music_path.expanduser()
        if selected.is_absolute():
            selected = selected.resolve()
        else:
            working_directory_path = selected.resolve()
            selected = (
                working_directory_path
                if working_directory_path.is_relative_to(root)
                else (root / selected).resolve()
            )
    else:
        if not validate_artifact(root, selection_file):
            raise AudioMixError(
                "Soundtrack selection is untracked; regenerate it before rendering."
            )
        try:
            selection = json.loads(selection_file.read_text(encoding="utf-8"))
            if Path(selection["project_path"]).resolve() != root:
                raise AudioMixError("Soundtrack selection belongs to a different project.")
            asset_path = selection["candidate"].get("asset_path")
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise AudioMixError(f"Invalid soundtrack selection: {exc}") from exc
        if not asset_path:
            raise AudioMixError(
                "The selected catalog track is not downloaded. Supply its licensed local file "
                "with --music."
            )
        selected = (root / Path(asset_path)).resolve()

    try:
        selected.relative_to(root)
    except ValueError as exc:
        raise AudioMixError("Soundtrack audio must be inside the project folder.") from exc
    if not selected.is_file():
        raise FileNotFoundError(f"Selected soundtrack audio does not exist: {selected}")
    if selected.suffix.lower() not in {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}:
        raise AudioMixError(f"Unsupported soundtrack format: {selected.suffix}")
    return selected


def append_music_input(command: list[str], music_path: Path | None) -> int | None:
    """Append a looping soundtrack input and return its FFmpeg input index."""
    if music_path is None:
        return None
    index = sum(1 for item in command if item == "-i")
    command.extend(["-stream_loop", "-1", "-i", str(music_path)])
    return index


def append_music_mix(
    filters: list[str],
    *,
    base_label: str,
    music_input_index: int | None,
    duration_seconds: float,
    mode: MusicMode = "blend",
    volume: float = 0.22,
    fade_in_seconds: float = 1.0,
    fade_out_seconds: float = 1.5,
    ducking: bool = False,
    loudness_normalization: bool = False,
    target_lufs: float = -16.0,
    true_peak_db: float = -1.5,
    loudness_range: float = 11.0,
) -> str:
    """Append a duration-bounded music mix and return its output label."""
    if mode not in {"original", "music_only", "blend"}:
        raise AudioMixError("music mode must be original, music_only, or blend")
    if not 0 < volume <= 2:
        raise AudioMixError("music volume must be greater than 0 and at most 2")
    if fade_in_seconds < 0 or fade_out_seconds < 0:
        raise AudioMixError("music fade durations must be non-negative")
    if not -24 <= target_lufs <= -12:
        raise AudioMixError("target_lufs must be between -24 and -12 LUFS")
    if not -9 <= true_peak_db <= -1:
        raise AudioMixError("true_peak_db must be between -9 and -1 dBTP")
    if not 5 <= loudness_range <= 20:
        raise AudioMixError("loudness_range must be between 5 and 20 LU")
    if mode == "original":
        return base_label
    if music_input_index is None:
        if mode == "music_only":
            raise AudioMixError("music_only mode requires a selected soundtrack; use --music.")
        return base_label
    if duration_seconds <= 0:
        raise AudioMixError("render duration must be positive")

    fade_in = min(fade_in_seconds, duration_seconds / 2)
    fade_out = min(fade_out_seconds, duration_seconds / 2)
    chain = (
        f"[{music_input_index}:a:0]atrim=duration={duration_seconds:.6f},"
        "asetpts=PTS-STARTPTS,aresample=48000,"
        "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
        f"volume={volume:.4f}"
    )
    if fade_in > 0:
        chain += f",afade=t=in:st=0:d={fade_in:.6f}"
    if fade_out > 0:
        chain += f",afade=t=out:st={max(0.0, duration_seconds - fade_out):.6f}:d={fade_out:.6f}"
    filters.append(chain + "[musicbed]")

    loudnorm = (
        f"loudnorm=I={target_lufs:.1f}:TP={true_peak_db:.1f}:LRA={loudness_range:.1f}:linear=false"
    )
    if mode == "music_only":
        if loudness_normalization:
            filters.append(f"[musicbed]{loudnorm}[audiomix]")
        else:
            filters.append("[musicbed]anull[audiomix]")
        return "audiomix"
    if ducking:
        filters.append(
            f"[musicbed][{base_label}]sidechaincompress=threshold=0.025:ratio=8:"
            "attack=20:release=450[duckedmusic]"
        )
        music_label = "duckedmusic"
    else:
        music_label = "musicbed"
    filters.append(
        f"[{base_label}][{music_label}]amix=inputs=2:duration=first:"
        "dropout_transition=2:normalize=0,alimiter=limit=0.95[audiomix]"
    )
    if loudness_normalization:
        filters.append(f"[audiomix]{loudnorm}[normalized_audio]")
        return "normalized_audio"
    return "audiomix"
