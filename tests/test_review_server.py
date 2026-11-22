import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

from video_edit_agent.editing.review_server import create_review_server


def _post(server, payload, *, token=None):
    host, port = server.server_address
    request = urllib.request.Request(
        f"http://{host}:{port}/__video_agent__/timeline-edits",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Origin": f"http://{host}:{port}",
            "X-Video-Agent-Token": token or server.session_token,
        },
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=3)


def test_review_server_injects_save_control_into_timeline_page(tmp_path: Path):
    page = tmp_path / ".video-agent" / "timeline-visual-review" / "review.html"
    page.parent.mkdir(parents=True)
    page.write_text(
        '<html><head><title>Visual Timeline Review</title></head>'
        '<body><header>Review</header></body></html>',
        encoding="utf-8",
    )
    server = create_review_server(tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        with urllib.request.urlopen(
            f"http://{host}:{port}/.video-agent/timeline-visual-review/review.html",
            timeout=3,
        ) as response:
            content = response.read().decode()
        assert "Save as draft revision" in content
        assert "/__video_agent__/timeline-edits" in content
        assert response.headers["Cache-Control"] == "no-store"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_review_server_rejects_missing_or_invalid_session_token(tmp_path: Path):
    server = create_review_server(tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        try:
            _post(server, {}, token="invalid")
        except urllib.error.HTTPError as exc:
            assert exc.code == 403
        else:
            raise AssertionError("Expected invalid token to be rejected")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_review_server_rejects_requests_from_other_origins(tmp_path: Path):
    server = create_review_server(tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        request = urllib.request.Request(
            f"http://{host}:{port}/__video_agent__/timeline-edits",
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "Origin": "http://example.com",
                "X-Video-Agent-Token": server.session_token,
            },
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=3)
        except urllib.error.HTTPError as exc:
            assert exc.code == 403
        else:
            raise AssertionError("Expected cross-origin request to be rejected")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
