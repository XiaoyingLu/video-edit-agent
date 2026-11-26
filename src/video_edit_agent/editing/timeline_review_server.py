"""Short-lived loopback server for direct browser timeline-edit submissions."""

import json
import secrets
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def start_timeline_review_server(
    project_root: Path, edits_path: Path
) -> tuple[ThreadingHTTPServer, threading.Event, str, str]:
    """Serve project media and accept one version-bound timeline edit proposal."""
    root = project_root.expanduser().resolve()
    destination = edits_path.expanduser().resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("timeline edit output must be inside the project folder") from exc

    token = secrets.token_urlsafe(24)
    submitted = threading.Event()
    endpoint = f"/__video-agent/timeline/{token}"

    class TimelineReviewHandler(SimpleHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != endpoint:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or not isinstance(payload.get("clips"), list):
                    raise ValueError("timeline edit payload must contain a clips array")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except (ValueError, OSError) as exc:
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            body = b'{"received":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            submitted.set()

        def log_message(self, format: str, *args: object) -> None:
            return

    handler = partial(TimelineReviewHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    review_url = f"http://127.0.0.1:{port}/.video-agent/timeline-live-review.html"
    submission_url = f"http://127.0.0.1:{port}{endpoint}"
    return server, submitted, review_url, submission_url
