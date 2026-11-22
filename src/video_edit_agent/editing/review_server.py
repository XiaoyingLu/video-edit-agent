"""Loopback-only HTTP server for saving visual timeline edits from the review workspace."""

from __future__ import annotations

import hashlib
import json
import secrets
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from ..artifact_tracking import validate_artifact
from .timeline_pipeline import apply_timeline_review, save_draft_timeline
from .timeline_schemas import ProjectDraftTimeline
from .timeline_visual_editor import TimelineVisualEditExport, apply_timeline_visual_edits

_MEDIA_SUFFIXES = {
    ".aac",
    ".avi",
    ".flac",
    ".jpeg",
    ".jpg",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".png",
    ".wav",
    ".webm",
    ".webp",
}
_BLOCKED_PARTS = {".git", ".venv", "venv", "__pycache__", "node_modules"}

_SAVE_CONTROLS = """
<section id="live-save"
  style="margin:.75rem 0;padding:.75rem;border:1px solid #8888;border-radius:8px">
  <button id="save-draft-revision" type="button">Save as draft revision</button>
  <button id="save-clip-decisions" type="button">Save clip decisions</button>
  <button id="approve-complete-timeline" type="button">Approve complete timeline</button>
  <span id="live-save-status" role="status" aria-live="polite">
    Save clip decisions first; final timeline approval is a separate action.
  </span>
</section>
<script>
(() => {
  const button = document.getElementById("save-draft-revision");
  const status = document.getElementById("live-save-status");
  const decisionsButton = document.getElementById("save-clip-decisions");
  const approveButton = document.getElementById("approve-complete-timeline");
  if (!button) return;
  async function postReview(approveTimeline) {
    const token = new URL(window.parent.location.href).searchParams.get("token");
    if (!token) throw new Error("Open this page from the running review workspace.");
    const decisions = {};
    for (const clip of report.clips) {
      const selected = document.querySelector('input[name="decision-' + clip.sequence_index + '"]:checked');
      decisions[clip.clip_id] = selected ? selected.value : "pending";
    }
    const response = await fetch("/__video_agent__/timeline-review", {
      method:"POST", headers:{"Content-Type":"application/json","X-Video-Agent-Token":token},
      body:JSON.stringify({schema_version:"1.0",timeline_path:report.timeline_path,
        timeline_sha256:report.timeline_sha256,decisions,approve_timeline:approveTimeline})
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Review save failed");
    return result;
  }
  decisionsButton.addEventListener("click", async () => {
    decisionsButton.disabled=true; status.textContent=" Saving clip decisions…";
    try {
      const result=await postReview(false);
      status.textContent=" Saved: "+result.approved_count+" approved, "+
        result.rejected_count+" rejected, "+result.pending_count+" pending. Timeline not approved.";
    } catch(error) { status.textContent=" Could not save: "+error.message; }
    finally { decisionsButton.disabled=false; }
  });
  approveButton.addEventListener("click", async () => {
    if (!window.confirm("Approve this complete timeline? This enables the existing render workflow.")) return;
    approveButton.disabled=true; status.textContent=" Approving complete timeline…";
    try {
      const result=await postReview(true);
      status.textContent=" Timeline approved: "+result.approved_count+
        " approved, "+result.rejected_count+" rejected.";
    } catch(error) { status.textContent=" Could not approve: "+error.message; }
    finally { approveButton.disabled=false; }
  });
  button.addEventListener("click", async () => {
    const token = new URL(window.parent.location.href).searchParams.get("token");
    if (!token) {
      status.textContent = "Open this page from the running review-workspace server to save.";
      return;
    }
    const clips = Array.from(document.querySelectorAll("main .clip")).map(card => ({
      clip_id: card.dataset.clipId,
      source_start_seconds: Number(card.querySelector(".trim-start").value),
      source_end_seconds: Number(card.querySelector(".trim-end").value)
    }));
    button.disabled = true;
    status.textContent = "Validating and saving draft revision…";
    try {
      const response = await fetch("/__video_agent__/timeline-edits", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Video-Agent-Token": token
        },
        body: JSON.stringify({
          schema_version: "1.0",
          timeline_path: report.timeline_path,
          timeline_sha256: report.timeline_sha256,
          clips
        })
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Save failed");
      status.textContent = "Saved unapproved revision: " + result.revision_path +
        ". Review and approve it before rendering.";
    } catch (error) {
      status.textContent = "Could not save revision: " + error.message;
    } finally {
      button.disabled = false;
    }
  });
})();
</script>
"""


class _ReviewWorkspaceServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def create_review_server(
    folder: Path, *, host: str = "127.0.0.1", port: int = 0
) -> ThreadingHTTPServer:
    """Create a review server; 0.0.0.0 supports explicitly port-published containers."""
    if host not in {"127.0.0.1", "0.0.0.0"}:
        raise ValueError(f"Unsupported bind host: {host}")
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project folder does not exist: {root}")
    token = secrets.token_urlsafe(32)

    class Handler(SimpleHTTPRequestHandler):
        server_version = "VideoAgentReview/1.0"
        sys_version = ""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, _format, *args):
            # Do not log request query strings, which may contain the session token.
            return

        def translate_path(self, path):
            translated = Path(super().translate_path(path)).resolve()
            try:
                relative = translated.relative_to(root)
            except ValueError:
                return str(root / "__video_agent_not_found__")
            parts = relative.parts
            if any(part in _BLOCKED_PARTS or part.startswith(".env") for part in parts):
                return str(root / "__video_agent_not_found__")
            if ".video-agent" not in parts and translated.suffix.lower() not in _MEDIA_SUFFIXES:
                return str(root / "__video_agent_not_found__")
            return str(translated)

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path in {"/__video_agent__/timeline-edits", "/__video_agent__/timeline-review"}:
                self.send_error(405, "Method not allowed")
                return
            target = Path(self.translate_path(parsed.path))
            if target.is_dir():
                self.send_error(404, "Not found")
                return
            if target.is_file() and target.suffix.lower() in {".html", ".htm"}:
                try:
                    content = target.read_text(encoding="utf-8")
                except (OSError, UnicodeError):
                    self.send_error(404, "Not found")
                    return
                if "<title>Visual Timeline Review</title>" in content:
                    content = content.replace("</header>", _SAVE_CONTROLS + "</header>", 1)
                body = content.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def do_POST(self):
            parsed = urlsplit(self.path)
            if parsed.path not in {"/__video_agent__/timeline-edits", "/__video_agent__/timeline-review"}:
                self.send_error(404, "Not found")
                return
            expected_host = f"127.0.0.1:{self.server.server_port}"
            allowed_hosts = {expected_host, f"localhost:{self.server.server_port}"}
            if self.headers.get("Host", "").lower() not in allowed_hosts:
                self.send_error(403, "Invalid host")
                return
            allowed_origins = {
                f"http://127.0.0.1:{self.server.server_port}",
                f"http://localhost:{self.server.server_port}",
            }
            if self.headers.get("Origin") not in allowed_origins:
                self.send_error(403, "Invalid origin")
                return
            supplied_token = self.headers.get("X-Video-Agent-Token", "")
            if not secrets.compare_digest(supplied_token, token):
                self.send_error(403, "Invalid session token")
                return
            if self.headers.get_content_type() != "application/json":
                self.send_error(415, "Expected application/json")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(400, "Invalid content length")
                return
            if length <= 0 or length > 256 * 1024:
                self.send_error(413, "Request body must be between 1 byte and 256 KiB")
                return
            try:
                payload = json.loads(self.rfile.read(length))
                edits = TimelineVisualEditExport.model_validate(payload)
                if not edits.clips:
                    raise ValueError("At least one clip is required")
                ids = [clip.clip_id for clip in edits.clips]
                if len(ids) != len(set(ids)):
                    raise ValueError("Duplicate clip IDs are not allowed")
                if parsed.path == "/__video_agent__/timeline-edits":
                    revision = apply_timeline_visual_edits(root, edits)
                    self._send_json(200, {"revision_path": revision.relative_to(root).as_posix()})
                    return
                self._save_timeline_review(payload)
                return
            except (ValueError, OSError, TypeError) as exc:
                self._send_json(400, {"error": str(exc)})
                return
            self._send_json(200, {"revision_path": revision.relative_to(root).as_posix()})

        def _save_timeline_review(self, payload):
            try:
                if payload.get("schema_version") != "1.0":
                    raise ValueError("Unsupported timeline review schema version")
                relative = Path(payload.get("timeline_path", ""))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Timeline path must be project-relative")
                timeline_file = (root / relative).resolve()
                timeline_file.relative_to(root)
                if not timeline_file.is_file() or not validate_artifact(root, timeline_file):
                    raise ValueError("Timeline is missing, stale, or untracked")
                if hashlib.sha256(timeline_file.read_bytes()).hexdigest() != payload.get("timeline_sha256"):
                    raise ValueError("Timeline changed after review; reload the review page")
                timeline = ProjectDraftTimeline.model_validate_json(timeline_file.read_text(encoding="utf-8"))
                if Path(timeline.project_path).resolve() != root:
                    raise ValueError("Timeline belongs to a different project folder")
                decisions = payload.get("decisions")
                if not isinstance(decisions, dict) or set(decisions) != {c.clip_id for c in timeline.clips}:
                    raise ValueError("Decisions must include every timeline clip exactly once")
                if any(value not in {"approved", "rejected", "pending"} for value in decisions.values()):
                    raise ValueError("Decisions must be approved, rejected, or pending")
                updated = apply_timeline_review(
                    timeline,
                    {key for key,value in decisions.items() if value == "approved"},
                    {key for key,value in decisions.items() if value == "rejected"},
                )
                approve = payload.get("approve_timeline") is True
                pending = sum(c.decision == "pending" for c in updated.clips)
                approved = sum(c.decision == "approved" for c in updated.clips)
                rejected = sum(c.decision == "rejected" for c in updated.clips)
                if approve and (pending or not approved):
                    raise ValueError("Resolve every clip and approve at least one before approving the timeline")
                updated.timeline_approved = approve
                updated = ProjectDraftTimeline.model_validate(updated.model_dump())
                save_draft_timeline(updated, timeline_file)
            except (ValueError, OSError, TypeError) as exc:
                self._send_json(400, {"error": str(exc)})
                return
            self._send_json(200, {"timeline_path":relative.as_posix(),"timeline_approved":approve,
                "approved_count":approved,"rejected_count":rejected,"pending_count":pending})

        def _send_json(self, status: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

    server = _ReviewWorkspaceServer((host, port), Handler)
    server.session_token = token
    server.project_root = root
    return server
