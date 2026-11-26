"""Runtime capability discovery and fail-closed tool/skill registration."""

from __future__ import annotations

import importlib.util
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config import get_setting
from .media import discover_videos


@dataclass(frozen=True)
class CapabilityReport:
    """Snapshot of runtime prerequisites available to a single pipeline run."""

    project_root: str
    media_files: tuple[str, ...] = ()
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None
    ffmpeg_filters: frozenset[str] = frozenset()
    ffmpeg_encoders: frozenset[str] = frozenset()
    fonts: frozenset[str] = frozenset()
    external_keys: frozenset[str] = frozenset()
    available_tools: frozenset[str] = frozenset()
    available_skills: frozenset[str] = frozenset()
    unavailable: dict[str, str] = field(default_factory=dict)

    def has(self, capability: str) -> bool:
        """Return whether a named tool or skill is available."""
        return capability in self.available_tools or capability in self.available_skills

    def require(self, *capabilities: str) -> None:
        """Fail before planning when any requested capability is unavailable."""
        missing = [name for name in capabilities if not self.has(name)]
        if missing:
            details = "; ".join(
                f"{name}: {self.unavailable.get(name, 'capability unavailable')}"
                for name in missing
            )
            raise RuntimeCapabilityError(f"Required runtime capability unavailable: {details}")

    def prompt_context(self) -> dict:
        """Return a compact, secret-free capability snapshot for planning prompts."""
        return {
            "available_tools": sorted(self.available_tools),
            "available_skills": sorted(self.available_skills),
            "unavailable": dict(sorted(self.unavailable.items())),
            "media_file_count": len(self.media_files),
            "ffmpeg_filters": sorted(self.ffmpeg_filters),
            "ffmpeg_encoders": sorted(self.ffmpeg_encoders),
            "fonts": sorted(self.fonts),
        }


class RuntimeCapabilityError(RuntimeError):
    """Raised when a pipeline step is requested without its runtime prerequisites."""


def _run_help(executable: str | None, option: str) -> str:
    if executable is None:
        return ""
    try:
        result = subprocess.run(
            [executable, "-hide_banner", option],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (result.stdout or "") + "\n" + (result.stderr or "")


def _listed_names(output: str) -> frozenset[str]:
    names = set()
    for line in output.splitlines():
        match = re.match(r"^\s*[TSC.]?[.]{5,}\s+([A-Za-z0-9_]+)\s+", line)
        if match:
            names.add(match.group(1))
            continue
        # FFmpeg filters use three flag columns; encoders use six.
        match = re.match(r"^\s*[A-Z.]{3}\s+([A-Za-z0-9_]+)\s+", line)
        if match:
            names.add(match.group(1))
            continue
        match = re.match(r"^\s*[A-Z.]{6}\s+([A-Za-z0-9_]+)\s+", line)
        if match:
            names.add(match.group(1))
    return frozenset(names)


def _available_fonts() -> frozenset[str]:
    fonts: set[str] = set()
    fc_match = shutil.which("fc-match")
    if fc_match:
        for name in ("DejaVu Sans", "Lato"):
            try:
                result = subprocess.run(
                    [fc_match, "-f", "%{family}", name],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if result.returncode == 0 and name.casefold() in result.stdout.casefold():
                fonts.add(name)
    bundled = Path(__file__).parent / "rendering" / "fonts" / "Lato-Regular.ttf"
    if bundled.is_file():
        fonts.add("Lato")
    if Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf").is_file():
        fonts.add("DejaVu Sans")
    return frozenset(fonts)


def probe_runtime_capabilities(project_root: Path) -> CapabilityReport:
    """Probe media, FFmpeg, fonts, optional model dependencies, and provider keys."""
    root = project_root.expanduser().resolve()
    try:
        media_files = tuple(path.relative_to(root).as_posix() for path in discover_videos(root))
    except (FileNotFoundError, NotADirectoryError):
        media_files = ()

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    filters = _listed_names(_run_help(ffmpeg, "-filters"))
    encoders = _listed_names(_run_help(ffmpeg, "-encoders"))
    fonts = _available_fonts()

    keys = frozenset(
        name
        for name in ("SOUNDSTRIPE_API_KEY", "STABILITY_API_KEY")
        if (get_setting(name) or "").strip()
    )
    unavailable: dict[str, str] = {}
    tools: set[str] = set()
    skills: set[str] = set()

    if media_files and ffprobe:
        tools.add("inspect_media")
    else:
        unavailable["inspect_media"] = (
            "No supported video files were found."
            if not media_files
            else "ffprobe is not installed or not on PATH."
        )

    if media_files and ffprobe and importlib.util.find_spec("cv2") is not None:
        skills.add("scene_understanding")
    else:
        unavailable["scene_understanding"] = (
            "Requires supported video files, ffprobe, and OpenCV."
        )

    vision_provider = (get_setting("VISION_PROVIDER") or "ollama").strip().lower()
    if vision_provider == "ollama":
        skills.add("visual_analysis")
    elif vision_provider in {"mlx", "qwen3-vl"}:
        if platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}:
            package = "mlx_vlm"
            if importlib.util.find_spec(package) is not None:
                skills.add("visual_analysis")
            else:
                unavailable["visual_analysis"] = "The configured MLX provider package is missing."
        else:
            unavailable["visual_analysis"] = (
                "The configured MLX vision provider requires Apple Silicon macOS."
            )
    else:
        unavailable["visual_analysis"] = f"Unsupported vision provider: {vision_provider}."

    if media_files and importlib.util.find_spec("faster_whisper") is not None:
        skills.add("transcription")
    else:
        unavailable["transcription"] = (
            "Requires supported video files and the optional faster-whisper dependency."
        )

    required_render_filters = {
        "scale", "pad", "setsar", "fps", "format", "trim", "setpts", "concat",
        "aresample", "aformat", "anullsrc",
    }
    if (
        ffmpeg
        and ffprobe
        and {"libx264", "aac"} <= encoders
        and required_render_filters <= filters
    ):
        tools.add("render_video")
    else:
        unavailable["render_video"] = (
            "Requires ffmpeg/ffprobe, the scale filter, and libx264/aac encoders."
        )

    if ffmpeg and "subtitles" in filters and "DejaVu Sans" in fonts:
        tools.add("render_subtitles")
    else:
        unavailable["render_subtitles"] = (
            "Requires the FFmpeg subtitles filter and the configured DejaVu Sans font."
        )

    if ffmpeg and "drawtext" in filters and fonts:
        tools.add("render_titles")
    else:
        unavailable["render_titles"] = (
            "Requires the FFmpeg drawtext filter and a supported title font."
        )

    if "SOUNDSTRIPE_API_KEY" in keys:
        tools.add("search_soundtrack_catalog")
        skills.add("soundtrack_discovery")
    else:
        unavailable["search_soundtrack_catalog"] = "SOUNDSTRIPE_API_KEY is not configured."

    if "STABILITY_API_KEY" in keys:
        tools.add("generate_soundtrack")
        skills.add("soundtrack_generation")
    else:
        unavailable["generate_soundtrack"] = "STABILITY_API_KEY is not configured."

    return CapabilityReport(
        project_root=str(root),
        media_files=media_files,
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
        ffmpeg_filters=filters,
        ffmpeg_encoders=encoders,
        fonts=fonts,
        external_keys=keys,
        available_tools=frozenset(tools),
        available_skills=frozenset(skills),
        unavailable=unavailable,
    )
