import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from video_edit_agent.editing.timeline_review_server import start_timeline_review_server


def test_local_timeline_server_receives_browser_edits(tmp_path: Path) -> None:
    review_dir = tmp_path / ".video-agent"
    review_dir.mkdir()
    (review_dir / "timeline-live-review.html").write_text("<h1>Timeline</h1>", encoding="utf-8")
    edits_path = review_dir / "timeline-live-edits.json"

    server, submitted, review_url, submission_url = start_timeline_review_server(
        tmp_path, edits_path
    )
    try:
        with urlopen(review_url) as response:
            assert response.status == 200
            assert b"Timeline" in response.read()

        payload = {
            "schema_version": "1.0",
            "timeline_path": ".video-agent/optimized-timeline.json",
            "timeline_sha256": "a" * 64,
            "clips": [
                {
                    "clip_id": "shot-1",
                    "source_start_seconds": 1,
                    "source_end_seconds": 4,
                }
            ],
        }
        request = Request(
            submission_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "text/plain"},
            method="POST",
        )
        with urlopen(request) as response:
            assert response.status == 200
        assert submitted.wait(timeout=1)
        assert json.loads(edits_path.read_text(encoding="utf-8")) == payload
    finally:
        server.shutdown()
        server.server_close()


def test_timeline_server_rejects_unknown_path_and_invalid_payload(tmp_path: Path) -> None:
    server, _, _, submission_url = start_timeline_review_server(
        tmp_path, tmp_path / ".video-agent" / "edits.json"
    )
    try:
        for url, payload, expected_status in (
            (submission_url + "-invalid", b"{}", 404),
            (submission_url, b'{"clips":{}}', 400),
        ):
            request = Request(
                url,
                data=payload,
                headers={"Content-Type": "text/plain"},
                method="POST",
            )
            try:
                urlopen(request)
            except HTTPError as exc:
                assert exc.code == expected_status
            else:
                raise AssertionError("unexpectedly accepted invalid timeline submission")
    finally:
        server.shutdown()
        server.server_close()
