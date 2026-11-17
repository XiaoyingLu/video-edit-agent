"""Lightweight local soundtrack analysis using FFmpeg and NumPy."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .schemas import MusicAnalysis, MusicSection

_SAMPLE_RATE = 11_025
_FRAME_SIZE = 2_048
_HOP_SIZE = 256


class MusicAnalysisError(RuntimeError):
    """Raised when soundtrack decoding or analysis fails."""


def _decode_audio(path: Path) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(_SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=300)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise MusicAnalysisError(f"Could not decode soundtrack with FFmpeg: {exc}") from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace")[-1200:]
        raise MusicAnalysisError(f"FFmpeg could not decode {path.name}: {detail}")
    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if samples.size < _SAMPLE_RATE * 3:
        raise MusicAnalysisError("Soundtrack must contain at least 3 seconds of decodable audio.")
    if not np.isfinite(samples).all():
        raise MusicAnalysisError("Decoded soundtrack contains invalid audio samples.")
    return samples


def _onset_envelope(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate positive spectral flux on overlapping Hann-windowed frames."""
    starts = np.arange(0, max(1, samples.size - _FRAME_SIZE), _HOP_SIZE)
    if starts.size < 8:
        raise MusicAnalysisError("Soundtrack is too short for tempo estimation.")
    window = np.hanning(_FRAME_SIZE).astype(np.float32)
    spectra = np.empty((starts.size, _FRAME_SIZE // 2 + 1), dtype=np.float32)
    for index, start in enumerate(starts):
        frame = samples[start : start + _FRAME_SIZE]
        spectra[index] = np.abs(np.fft.rfft(frame * window))
    positive_flux = np.maximum(0, np.diff(spectra, axis=0)).mean(axis=1)
    envelope = np.concatenate(([0.0], positive_flux)).astype(np.float64)
    if envelope.max() > 0:
        envelope /= envelope.max()
    times = (starts + _FRAME_SIZE / 2) / _SAMPLE_RATE
    return times, envelope


def _estimate_tempo(
    times: np.ndarray, onset: np.ndarray
) -> tuple[float | None, float, float | None]:
    centered = onset - onset.mean()
    if np.linalg.norm(centered) < 1e-8:
        return None, 0.0, None
    frame_rate = 1 / float(np.median(np.diff(times)))
    min_lag = max(1, int(frame_rate * 60 / 200))
    max_lag = min(len(centered) - 1, int(frame_rate * 60 / 40))
    if max_lag <= min_lag:
        return None, 0.0, None
    fft_size = 1 << (2 * len(centered) - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size)
    correlations = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)[: len(centered)]
    candidates = correlations[min_lag : max_lag + 1]
    lag = min_lag + int(np.argmax(candidates))
    baseline = float(np.mean(np.abs(candidates))) + 1e-9
    confidence = float(np.clip((correlations[lag] / baseline - 1) / 8, 0, 1))
    interval = lag / frame_rate
    bpm = 60 / interval
    if confidence < 0.08:
        return None, confidence, None
    return round(float(bpm), 2), confidence, interval


def _beat_grid(
    times: np.ndarray, onset: np.ndarray, interval: float | None, duration: float
) -> list[float]:
    if interval is None:
        return []
    candidates = np.arange(0, min(interval, float(times[-1])), max(0.01, interval / 32))
    best_phase, best_score = float(candidates[0]), -1.0
    for phase in candidates:
        grid = np.arange(phase, duration, interval)
        indices = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
        score = float(onset[indices].mean()) if len(indices) else 0.0
        if score > best_score:
            best_phase, best_score = float(phase), score
    beats = []
    for target in np.arange(best_phase, duration, interval):
        left = int(np.searchsorted(times, target - interval * 0.12))
        right = int(np.searchsorted(times, target + interval * 0.12))
        if right > left:
            beat = float(times[left + int(np.argmax(onset[left:right]))])
        else:
            beat = float(target)
        if 0 <= beat < duration and (not beats or beat - beats[-1] > interval * 0.5):
            beats.append(round(beat, 3))
    return beats


def _energy_sections(samples: np.ndarray, duration: float) -> list[MusicSection]:
    window = _SAMPLE_RATE * 8
    energies = [
        float(np.sqrt(np.mean(samples[start : min(start + window, len(samples))] ** 2)))
        for start in range(0, len(samples), window)
    ]
    positive = [value for value in energies if value > 1e-6]
    if not positive:
        return []
    low, high = np.quantile(positive, [0.33, 0.67])
    sections = []
    for index, energy in enumerate(energies):
        start = float(index * 8)
        end = min(duration, float(index * 8 + 8))
        level = "low" if energy <= low else "high" if energy >= high else "medium"
        sections.append(
            MusicSection(
                start_seconds=start,
                end_seconds=end,
                energy=level,
                energy_rms=round(energy, 5),
            )
        )
    return sections


def analyze_music(folder: Path, music_path: Path) -> MusicAnalysis:
    """Analyze an explicit soundtrack and suggest beat-aligned edit boundaries."""
    root = folder.expanduser().resolve()
    source = music_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Soundtrack does not exist: {source}")
    samples = _decode_audio(source)
    duration = len(samples) / _SAMPLE_RATE
    times, onset = _onset_envelope(samples)
    bpm, confidence, interval = _estimate_tempo(times, onset)
    beats = _beat_grid(times, onset, interval, duration)
    notes = [
        "Tempo and beat positions are signal-based estimates; inspect them against the music.",
        (
            "Downbeats assume four-beat grouping and may not match unusual meters "
            "or syncopated tracks."
        ),
        (
            "Energy sections describe relative loudness, not semantic labels such as "
            "a chorus or drop."
        ),
        (
            "Use beats as optional cut anchors; preserve natural shot length and "
            "editorial breathing room."
        ),
    ]
    if bpm is None:
        notes.append("No stable tempo was detected; use the energy map and manual cut placement.")
    return MusicAnalysis(
        project_path=str(root),
        music_path=str(source),
        duration_seconds=round(duration, 3),
        sample_rate=_SAMPLE_RATE,
        bpm=bpm,
        beat_confidence=round(confidence, 3),
        beat_interval_seconds=round(interval, 4) if interval else None,
        beat_times_seconds=beats,
        downbeat_times_seconds=beats[::4],
        sections=_energy_sections(samples, duration),
        suggested_cut_points_seconds=beats,
        notes=notes,
    )
