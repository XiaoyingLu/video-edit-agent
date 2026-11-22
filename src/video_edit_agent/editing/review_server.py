"""Loopback-only HTTP server for saving visual timeline edits from the review workspace."""

from __future__ import annotations

import json
import secrets
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

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
  <span id="live-save-status" role="status" aria-live="polite">
    Saves a separate, unapproved timeline revision in this project.
  </span>
</section>
<script>
(() => {
  const button = document.getElementById("save-draft-revision");
  const status = document.getElementById("live-save-status");
  if (!button) return;
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
            if parsed.path == "/__video_agent__/timeline-edits":
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
            if parsed.path != "/__video_agent__/timeline-edits":
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
                revision = apply_timeline_visual_edits(root, edits)
            except (ValueError, OSError, TypeError) as exc:
                self._send_json(400, {"error": str(exc)})
                return
            self._send_json(200, {"revision_path": revision.relative_to(root).as_posix()})

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
