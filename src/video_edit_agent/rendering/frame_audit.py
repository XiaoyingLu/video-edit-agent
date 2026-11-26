"""Post-render audit for deterministic random-access decoding and visual edge cases."""

from __future__ import annotations

import hashlib
import random
import shutil
import subprocess
from pathlib import Path


class FrameAuditError(RuntimeError):
    """Raised when a rendered file fails deterministic frame decoding."""


def validate_title_font(configured_font: str, bundled_font: Path) -> Path | None:
    """Resolve and verify the concrete font file used by approved title overlays."""
    configured = configured_font.strip()
    candidate = Path(configured).expanduser()
    looks_like_path = candidate.suffix.lower() in {".ttf", ".otf", ".ttc"} or "/" in configured
    if looks_like_path:
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise FrameAuditError(f"configured title font file does not exist: {resolved}")
        return resolved
    if configured.casefold() in {"lato", "bundled"} and bundled_font.is_file():
        return bundled_font.resolve()
    matcher = shutil.which("fc-match")
    if matcher:
        try:
            result = subprocess.run(
                [matcher, "-f", "%{file}", configured or "DejaVu Sans"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise FrameAuditError(f"could not resolve configured title font: {exc}") from exc
        font_file = Path((result.stdout or "").strip()).expanduser()
        if result.returncode == 0 and font_file.is_file():
            return font_file.resolve()
    fallback = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if fallback.is_file():
        return fallback
    raise FrameAuditError(f"configured title font could not be resolved: {configured}")


def _sample_times(
    duration: float, boundaries: list[float], frame_rate: float = 30.0
) -> list[float]:
    epsilon = 1.0 / frame_rate
    times = {min(max(0.25, duration / 2), duration - 0.01), max(0.0, duration - 0.25)}
    for boundary in boundaries:
        for offset in (-epsilon, 0.0, epsilon):
            value = boundary + offset
            if 0.0 <= value < duration:
                times.add(round(value, 6))
    return sorted(times)


def _extract_frame(ffmpeg: str, video: Path, timestamp: float, timeout: float) -> bytes:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{timestamp:.6f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-an",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FrameAuditError(
            f"random-access frame decode failed at {timestamp:.3f}s: {exc}"
        ) from exc
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace")[-500:]
        raise FrameAuditError(
            f"random-access frame decode failed at {timestamp:.3f}s: "
            f"{detail or 'no frame returned'}"
        )
    return result.stdout


def _text_edge_warnings(
    tesseract: str, frames: dict[float, bytes], width: int, height: int
) -> list[str]:
    warnings: list[str] = []
    margin_x = max(2, round(width * 0.01))
    margin_y = max(2, round(height * 0.01))
    for timestamp, frame in frames.items():
        try:
            result = subprocess.run(
                [tesseract, "stdin", "stdout", "tsv"],
                input=frame,
                capture_output=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            warnings.append("Text-clipping OCR audit could not run.")
            break
        if result.returncode != 0:
            warnings.append("Text-clipping OCR audit could not run.")
            break
        lines = result.stdout.decode("utf-8", errors="replace").splitlines()[1:]
        for line in lines:
            columns = line.split("\t")
            if len(columns) < 12:
                continue
            try:
                confidence = float(columns[10])
                left, top, box_width, box_height = map(int, columns[6:10])
            except ValueError:
                continue
            if confidence < 35 or not columns[11].strip():
                continue
            if (
                left <= margin_x
                or top <= margin_y
                or left + box_width >= width - margin_x
                or top + box_height >= height - margin_y
            ):
                warnings.append(
                    f"Possible text clipping near the frame edge at {timestamp:.2f}s."
                )
                break
    return warnings


def audit_rendered_frames(
    ffmpeg: str,
    video: Path,
    duration: float,
    boundaries: list[float],
    *,
    width: int,
    height: int,
    has_titles: bool,
    timeout: float = 30.0,
) -> dict:
    """Decode sampled frames in two different orders and inspect cuts/title safe areas."""
    if not video.is_file() or video.stat().st_size == 0:
        raise FrameAuditError(f"render output is missing or empty: {video}")
    timestamps = _sample_times(duration, boundaries)
    if not timestamps:
        raise FrameAuditError("frame audit produced no sample timestamps")

    shuffled = timestamps.copy()
    random.Random(20261126).shuffle(shuffled)
    orders = (shuffled, list(reversed(shuffled)))
    frame_hashes: dict[float, str] = {}
    frame_bytes: dict[float, bytes] = {}
    for order in orders:
        for timestamp in order:
            frame = _extract_frame(ffmpeg, video, timestamp, timeout)
            digest = hashlib.sha256(frame).hexdigest()
            previous = frame_hashes.get(timestamp)
            if previous is not None and previous != digest:
                raise FrameAuditError(
                    f"non-deterministic frame output detected at {timestamp:.3f}s"
                )
            frame_hashes[timestamp] = digest
            frame_bytes[timestamp] = frame

    warnings: list[str] = []
    # A black frame exactly at a cut is often an accidental one-frame gap. Probe only
    # the boundary samples; intentional black footage remains a reviewable warning.
    blackdetect = shutil.which("ffmpeg")
    if blackdetect:
        for timestamp in boundaries:
            try:
                result = subprocess.run(
                    [
                        ffmpeg,
                        "-hide_banner",
                        "-loglevel",
                        "info",
                        "-nostdin",
                        "-ss",
                        f"{timestamp:.6f}",
                        "-i",
                        str(video),
                        "-frames:v",
                        "1",
                        "-vf",
                        "signalstats,metadata=print:file=-",
                        "-f",
                        "null",
                        "-",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            metadata = (result.stdout or "") + (result.stderr or "")
            for line in metadata.splitlines():
                if "lavfi.signalstats.YAVG=" in line:
                    try:
                        if float(line.rsplit("=", 1)[1]) < 3.0:
                            warnings.append(
                                f"Near-black frame at cut/transition boundary {timestamp:.2f}s."
                            )
                    except ValueError:
                        pass
                    break

    ocr_checked = False
    if has_titles:
        tesseract = shutil.which("tesseract")
        if tesseract:
            warnings.extend(_text_edge_warnings(tesseract, frame_bytes, width, height))
            ocr_checked = True
        else:
            warnings.append("Text-clipping check skipped: Tesseract OCR is not installed.")

    return {
        "sample_count": len(timestamps),
        "boundary_count": len(boundaries),
        "deterministic": True,
        "text_clipping_checked": ocr_checked,
        "warnings": sorted(set(warnings)),
    }
