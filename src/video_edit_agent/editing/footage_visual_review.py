"""Interactive browser review for storyboard-to-footage matching."""

import hashlib
import html
import json
import os
from pathlib import Path

from ..artifact_tracking import validate_artifact
from ..understanding.schemas import VideoUnderstanding
from .footage_schemas import ProjectFootageSuggestions
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
    return relative.as_posix() if resolved.is_file() else None


def generate_footage_visual_review(
    folder: Path,
    *,
    plan_path: Path | None = None,
    suggestions_path: Path | None = None,
    understanding_path: Path | None = None,
    output_path: Path | None = None,
    submission_url: str | None = None,
) -> tuple[Path, int, int]:
    """Write an interactive HTML page that exports explicit candidate selections."""
    root = folder.expanduser().resolve()
    artifact_dir = root / ".video-agent"
    files = (
        (plan_path or artifact_dir / "video-project-plan.json", "project plan"),
        (suggestions_path or artifact_dir / "footage-suggestions.json", "footage suggestions"),
        (understanding_path or artifact_dir / "understanding.json", "video understanding"),
    )
    loaded = []
    for path, label in files:
        resolved = path.expanduser().resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{label} must be inside the project folder") from exc
        if not resolved.is_file():
            raise FileNotFoundError(f"Missing {resolved}; generate it first.")
        validate_artifact(root, resolved)
        loaded.append(resolved)

    plan = VideoProjectPlan.model_validate_json(loaded[0].read_text(encoding="utf-8"))
    suggestions = ProjectFootageSuggestions.model_validate_json(
        loaded[1].read_text(encoding="utf-8")
    )
    understanding = VideoUnderstanding.model_validate_json(loaded[2].read_text(encoding="utf-8"))
    if any(
        Path(item).resolve() != root
        for item in (plan.project_path, suggestions.project_path, understanding.project_path)
    ):
        raise ValueError("review artifacts belong to a different project folder")
    if suggestions.source_plan != loaded[0].relative_to(root).as_posix():
        raise ValueError("footage suggestions were generated from a different storyboard")

    observations = {}
    for scene in understanding.scenes:
        observations[scene.scene_id] = scene.observations
    destination = (output_path or artifact_dir / "footage-review.html").expanduser().resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("review output must be inside the project folder") from exc

    by_shot = {item.shot_id: item for item in suggestions.suggestions}
    sections = []
    candidate_count = 0
    shot_count = 0
    for scene in plan.scenes:
        cards = []
        for shot in scene.shots:
            shot_count += 1
            match = by_shot.get(shot.shot_id)
            options = []
            if match:
                for number, candidate in enumerate(match.candidates, start=1):
                    candidate_count += 1
                    frames = [
                        item
                        for item in observations.get(candidate.source_scene_id, [])
                        if candidate.source_start_seconds
                        <= item.timestamp_seconds
                        <= candidate.source_end_seconds
                    ]
                    frame = (
                        min(
                            frames,
                            key=lambda item: abs(
                                item.timestamp_seconds
                                - (candidate.source_start_seconds + candidate.source_end_seconds)
                                / 2
                            ),
                        )
                        if frames
                        else None
                    )
                    asset = _project_asset(root, frame.frame_path) if frame else None
                    if asset:
                        image_src = html.escape(
                            os.path.relpath(root / asset, destination.parent), quote=True
                        )
                        media = (
                            f'<img loading="lazy" src="{image_src}" alt="Candidate {number} frame">'
                        )
                    else:
                        media = '<div class="placeholder">No preview frame</div>'
                    reasons = "".join(
                        f"<li>{html.escape(reason)}</li>" for reason in candidate.reasons
                    )
                    options.append(
                        f'<label class="candidate"><input type="radio" '
                        f'name="{html.escape(shot.shot_id, quote=True)}" '
                        f'value="{number}" {"checked" if number == 1 else ""}>'
                        f'<div class="preview">{media}</div><div class="candidate-body">'
                        f"<strong>Option {number} · {candidate.score:.1f}/100</strong>"
                        f"<p>{html.escape(candidate.video_path)} · "
                        f"{candidate.source_start_seconds:.2f}–"
                        f"{candidate.source_end_seconds:.2f}s</p>"
                        f"<p>{html.escape(candidate.scene_summary)}</p>"
                        f"<ul>{reasons}</ul></div></label>"
                    )
            else:
                options.append('<p class="empty">No candidate footage was found for this shot.</p>')
            cards.append(
                f'<article class="shot"><div class="shot-heading">'
                f"<span>{html.escape(shot.shot_id)}</span>"
                f"<h3>{html.escape(shot.purpose)}</h3>"
                f"<p>{html.escape(shot.visual_description)}</p></div>"
                f'<div class="options">{"".join(options)}</div>'
                f'<label class="skip"><input type="radio" '
                f'name="{html.escape(shot.shot_id, quote=True)}" value="0">'
                f"Leave this shot unassigned</label></article>"
            )
        sections.append(
            f"<section><h2>{scene.sequence_index:02d}. {html.escape(scene.title)}</h2>"
            f"<p>{html.escape(scene.purpose)}</p>"
            f'<div class="shots">{"".join(cards)}</div></section>'
        )

    title = plan.story_treatment.working_title if plan.story_treatment else "Storyboard"
    suggestions_sha256 = hashlib.sha256(loaded[1].read_bytes()).hexdigest()
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} — Footage Review</title>
<style>
:root{{color-scheme:dark;font-family:Inter,system-ui,sans-serif;background:#101318;color:#edf0f4}}
body{{margin:0}}main{{max-width:1200px;margin:auto;padding:32px 22px 70px}}
h1{{font-size:2.4rem;margin:.2em 0}}h2{{font-size:1.5rem}}h3{{font-size:1.05rem;margin:.2em 0}}
p{{color:#b8c0cc;line-height:1.45}}.hero{{border-bottom:1px solid #39414c;padding-bottom:24px}}
.actions{{position:sticky;top:0;background:#101318ed;padding:14px 0;
z-index:2;border-bottom:1px solid #39414c}}
button{{background:#d8e6f7;color:#111923;border:0;border-radius:8px;
padding:11px 16px;font-weight:700;cursor:pointer}}
#status{{margin-left:12px;color:#b8c0cc}}section{{margin:34px 0}}.shots{{display:grid;gap:18px}}
.shot{{background:#1a2029;border:1px solid #35404e;border-radius:12px;padding:16px}}
.shot-heading{{margin-bottom:12px}}
.shot-heading span{{color:#8fa7c5;font-size:.8rem;text-transform:uppercase}}
.options{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}}
.candidate{{display:block;border:1px solid #465366;border-radius:10px;
overflow:hidden;cursor:pointer}}
.candidate:has(input:checked){{border:2px solid #83b7f5}}.candidate input{{margin:12px}}
.preview{{aspect-ratio:16/9;background:#0b0e12}}
.preview img{{width:100%;height:100%;object-fit:cover}}
.placeholder{{height:100%;display:grid;place-items:center;color:#8793a2}}
.candidate-body{{padding:0 12px 12px}}.candidate-body p{{font-size:.88rem;margin:.5em 0}}
.candidate-body ul{{padding-left:18px;color:#9eacbc;font-size:.82rem}}
.skip{{display:inline-block;margin-top:14px;color:#d2d8e0}}
.empty{{color:#e6bd79}}
</style></head><body><main><div class="hero">
<p>VISUAL FOOTAGE REVIEW · {shot_count} SHOTS · {candidate_count} CANDIDATES</p>
<h1>{html.escape(title)}</h1>
<p>Choose the footage that best fits each storyboard shot. The first-ranked match is preselected,
but you can change it or leave a shot unassigned.
Nothing is approved or rendered by this page.</p></div>
<div class="actions"><button id="export">Download selections</button>
<span id="status" role="status">Review each shot, then download your choices.</span></div>
{"".join(sections)}</main>
<script>
document.getElementById("export").addEventListener("click", async () => {{
 const selections = {{}};
 document.querySelectorAll(".shot").forEach(card => {{
   const checked = card.querySelector('input[type="radio"]:checked');
   const shotId = card.querySelector(".shot-heading span").textContent;
   selections[shotId] = checked ? Number(checked.value) : 0;
 }});
 const payload = {{schema_version:"1.0", suggestions_sha256:"__SUGGESTIONS_SHA256__", selections}};
 const status = document.getElementById("status");
 if ("__SUBMISSION_URL__") {{
   try {{
     const response = await fetch("__SUBMISSION_URL__", {{
       method: "POST", headers: {{"Content-Type": "text/plain"}},
       body: JSON.stringify(payload)
     }});
     if (!response.ok) throw new Error("The local review session rejected the selections.");
     status.textContent = "Selections received. You can return to the terminal."; 
     document.getElementById("export").disabled = true;
   }} catch (error) {{
     status.textContent = error.message + " Please retry or use the CLI file import.";
   }}
 }} else {{
   const blob = new Blob([JSON.stringify(payload, null, 2)], {{type:"application/json"}});
   const url = URL.createObjectURL(blob);
   const link = document.createElement("a");
   link.href = url; link.download = "footage-review-selections.json"; link.click();
   URL.revokeObjectURL(url);
   status.textContent = "Selections downloaded. Place the JSON file in your project folder."; 
 }}
}});
</script></body></html>"""
    document = document.replace("__SUGGESTIONS_SHA256__", suggestions_sha256)
    document = document.replace("__SUBMISSION_URL__", submission_url or "")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination, shot_count, candidate_count


def load_footage_review_selections(
    path: Path, *, expected_shot_ids: set[str], suggestions_path: Path
) -> tuple[dict[str, int], set[str]]:
    """Load the browser export and reject stale or incomplete shot decisions."""
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read footage review selections: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("Unsupported footage review selection schema version")
    try:
        expected_hash = hashlib.sha256(suggestions_path.expanduser().read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"Could not verify footage suggestions: {exc}") from exc
    if payload.get("suggestions_sha256") != expected_hash:
        raise ValueError(
            "Selection file was created from different footage suggestions; regenerate it"
        )
    selections = payload.get("selections")
    if not isinstance(selections, dict) or set(selections) != expected_shot_ids:
        raise ValueError("Selection file must contain exactly one choice for every storyboard shot")
    selected = {}
    rejected = set()
    for shot_id, number in selections.items():
        if isinstance(number, bool) or not isinstance(number, int) or not 0 <= number <= 10:
            raise ValueError(f"Invalid candidate selection for {shot_id}")
        if number == 0:
            rejected.add(shot_id)
        else:
            selected[shot_id] = number
    return selected, rejected
