"""Generate a self-contained, local HTML storyboard for visual review."""

import hashlib
import html
import os
from pathlib import Path

from ..artifact_tracking import validate_artifact
from ..media import discover_videos
from ..understanding.schemas import VideoUnderstanding
from .planning_schemas import VideoProjectPlan


def _project_asset(root: Path, value: str | None) -> str | None:
    if not value:
        return None
    candidate = Path(value)
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return None
    if not resolved.is_file():
        return None
    return relative.as_posix()


def generate_storyboard_review(
    folder: Path,
    *,
    plan_path: Path | None = None,
    understanding_path: Path | None = None,
    output_path: Path | None = None,
) -> tuple[Path, int, int]:
    """Write a read-only HTML storyboard with representative source-frame thumbnails."""
    root = folder.expanduser().resolve()
    artifact_dir = root / ".video-agent"
    plan_file = (plan_path or artifact_dir / "video-project-plan.json").expanduser().resolve()
    understanding_file = (
        (understanding_path or artifact_dir / "understanding.json").expanduser().resolve()
    )
    for path, label in ((plan_file, "project plan"), (understanding_file, "video understanding")):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{label} must be inside the project folder") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; generate it first.")
        validate_artifact(root, path)

    plan = VideoProjectPlan.model_validate_json(plan_file.read_text(encoding="utf-8"))
    understanding = VideoUnderstanding.model_validate_json(
        understanding_file.read_text(encoding="utf-8")
    )
    if Path(plan.project_path).resolve() != root or Path(understanding.project_path).resolve() != root:
        raise ValueError("storyboard artifacts belong to a different project folder")
    if plan.planning_stage not in {"storyboard_ready", "footage_mapped", "production_ready"}:
        raise ValueError("visual storyboard requires a generated storyboard")

    destination = (output_path or artifact_dir / "storyboard.html").expanduser().resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("storyboard output must be inside the project folder") from exc

    video_keys = {}
    for video in discover_videos(root):
        relative = video.relative_to(root).as_posix()
        video_keys[hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]] = relative

    observations_by_video = {}
    for scene in understanding.scenes:
        key = scene.scene_id.split("-scene-", maxsplit=1)[0]
        video_path = video_keys.get(key)
        if video_path:
            observations_by_video.setdefault(video_path, []).extend(scene.observations)

    cards = []
    image_count = 0
    shot_count = 0
    for scene in plan.scenes:
        shot_cards = []
        for shot in scene.shots:
            shot_count += 1
            asset = None
            if shot.footage_path:
                candidates = observations_by_video.get(shot.footage_path, [])
                eligible = [
                    item for item in candidates
                    if shot.footage_start_seconds is not None
                    and shot.footage_end_seconds is not None
                    and shot.footage_start_seconds <= item.timestamp_seconds <= shot.footage_end_seconds
                ]
                if eligible:
                    chosen = min(
                        eligible,
                        key=lambda item: abs(
                            item.timestamp_seconds
                            - ((shot.footage_start_seconds + shot.footage_end_seconds) / 2)
                        ),
                    )
                    asset = _project_asset(root, chosen.frame_path)
            if asset:
                image_count += 1
                image_src = Path(os.path.relpath(root / asset, destination.parent)).as_posix()
                media = (
                    f'<img loading="lazy" src="{html.escape(image_src, quote=True)}" '
                    f'alt="Representative frame for {html.escape(shot.shot_id, quote=True)}">'
                )
            else:
                media = '<div class="placeholder">No representative frame available</div>'
            assignment = (
                f"{shot.footage_path} · {shot.footage_start_seconds:.2f}–"
                f"{shot.footage_end_seconds:.2f}s"
                if shot.footage_path and shot.footage_start_seconds is not None
                and shot.footage_end_seconds is not None
                else "Unassigned"
            )
            status = "Assigned" if shot.assignment_status != "unassigned" else "Needs footage"
            shot_cards.append(
                f"""<article class="shot">
                  <div class="frame">{media}</div>
                  <div class="shot-body">
                    <div class="eyebrow">{html.escape(shot.shot_id)} · {status}</div>
                    <h3>{html.escape(shot.purpose)}</h3>
                    <p>{html.escape(shot.visual_description)}</p>
                    <dl><dt>Source</dt><dd>{html.escape(assignment)}</dd>
                    <dt>Target</dt><dd>{shot.target_duration_seconds:.1f}s</dd>
                    <dt>Shot</dt><dd>{html.escape(shot.shot_scale.replace("_", " "))} · {html.escape(shot.visual_role.replace("_", " "))}</dd></dl>
                    {f'<p class="cue">Text: {html.escape(shot.on_screen_text)}</p>' if shot.on_screen_text else ''}
                    {f'<p class="cue">Narration: {html.escape(shot.narration)}</p>' if shot.narration else ''}
                  </div>
                </article>"""
            )
        cards.append(
            f'<section><header><div class="eyebrow">SCENE {scene.sequence_index:02d}</div>'
            f'<h2>{html.escape(scene.title)}</h2><p>{html.escape(scene.purpose)}</p>'
            f'<span>{scene.target_duration_seconds:.1f}s · {html.escape(scene.narrative_function)}</span>'
            f'</header><div class="shots">{"".join(shot_cards)}</div></section>'
        )

    title = plan.story_treatment.working_title if plan.story_treatment else "Storyboard"
    logline = plan.story_treatment.logline if plan.story_treatment else ""
    stage = plan.planning_stage.replace("_", " ")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} — Storyboard</title>
<style>
:root{{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#101318;color:#edf0f4}}
body{{margin:0}}main{{max-width:1180px;margin:auto;padding:40px 24px 80px}}
h1{{font-size:clamp(2rem,5vw,3.8rem);margin:.2em 0}}h2{{margin:.2em 0;font-size:1.7rem}}
h3{{font-size:1.05rem;margin:.4em 0}}p{{color:#b8c0cc;line-height:1.5}}
.hero{{padding:28px 0 42px;border-bottom:1px solid #303741;margin-bottom:32px}}
.meta,.eyebrow{{color:#8fa7c5;font-size:.76rem;letter-spacing:.12em;text-transform:uppercase}}
.pill{{display:inline-block;border:1px solid #46566a;border-radius:99px;padding:6px 10px;margin-top:10px;font-size:.8rem}}
section{{margin:42px 0}}section>header{{margin-bottom:16px}}section>header span{{color:#9aa6b5;font-size:.85rem}}
.shots{{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:16px}}
.shot{{background:#1a2029;border:1px solid #303946;border-radius:14px;overflow:hidden}}
.frame{{aspect-ratio:16/9;background:#0b0e12}}img{{width:100%;height:100%;object-fit:cover;display:block}}
.placeholder{{height:100%;display:grid;place-items:center;color:#7e8998;font-size:.85rem}}
.shot-body{{padding:16px}}dl{{display:grid;grid-template-columns:65px 1fr;gap:6px 10px;font-size:.82rem}}
dt{{color:#8290a2}}dd{{margin:0;overflow-wrap:anywhere;color:#d8dee7}}.cue{{font-size:.82rem;color:#c5b9a0}}
footer{{border-top:1px solid #303741;padding-top:18px;color:#8994a3;font-size:.8rem}}
</style></head><body><main>
<div class="hero"><div class="meta">Local storyboard review · {shot_count} shots · {len(plan.scenes)} scenes</div>
<h1>{html.escape(title)}</h1><p>{html.escape(logline)}</p><span class="pill">Planning stage: {html.escape(stage)}</span>
<p>Creative review only. This page does not approve the project, alter source media, or render video.</p></div>
{"".join(cards)}
<footer>Generated by video-agent · Images are local project frame-cache references · Re-run after changing the storyboard or footage assignments.</footer>
</main></body></html>"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination, shot_count, image_count
