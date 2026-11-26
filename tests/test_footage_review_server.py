import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from video_edit_agent.editing.footage_review_server import start_footage_review_server


def test_local_review_server_receives_browser_selections(tmp_path: Path) -> None:
    review_dir = tmp_path / ".video-agent"
    review_dir.mkdir()
    (review_dir / "footage-review.html").write_text("<h1>Review</h1>", encoding="utf-8")
    selections_path = review_dir / "footage-review-selections.json"

    server, submitted, review_url, submission_url = start_footage_review_server(
        tmp_path, selections_path
    )
    try:
        with urlopen(review_url) as response:
            assert response.status == 200
            assert b"Review" in response.read()

        payload = {
            "schema_version": "1.0",
            "suggestions_sha256": "test-hash",
            "selections": {"shot-001": 2},
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
        assert json.loads(selections_path.read_text(encoding="utf-8")) == payload
    finally:
        server.shutdown()
        server.server_close()


def test_local_review_server_rejects_unknown_submission_path(tmp_path: Path) -> None:
    server, _, _, submission_url = start_footage_review_server(
        tmp_path, tmp_path / ".video-agent" / "selections.json"
    )
    try:
        request = Request(
            submission_url + "-invalid",
            data=b"{}",
            headers={"Content-Type": "text/plain"},
            method="POST",
        )
        try:
            urlopen(request)
        except HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("unexpectedly accepted an unknown submission path")
    finally:
        server.shutdown()
        server.server_close()
