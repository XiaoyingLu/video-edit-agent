"""Short-lived loopback server for receiving browser footage-review decisions."""

import json
import secrets
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def start_footage_review_server(
    project_root: Path, selections_path: Path
) -> tuple[ThreadingHTTPServer, threading.Event, str, str]:
    """Serve project previews and accept one local browser selection submission."""
    root = project_root.expanduser().resolve()
    destination = selections_path.expanduser().resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("selection output must be inside the project folder") from exc

    token = secrets.token_urlsafe(24)
    submitted = threading.Event()
    endpoint = f"/__video-agent/review/{token}"

    class ReviewHandler(SimpleHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != endpoint:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("selection payload must be a JSON object")
                selections = payload.get("selections")
                if not isinstance(selections, dict):
                    raise ValueError("selection payload is missing its selections")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except (ValueError, OSError, json.JSONDecodeError) as exc:
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

    handler = partial(ReviewHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    review_path = ".video-agent/footage-review.html"
    review_url = f"http://127.0.0.1:{port}/{review_path}"
    submission_url = f"http://127.0.0.1:{port}{endpoint}"
    return server, submitted, review_url, submission_url
