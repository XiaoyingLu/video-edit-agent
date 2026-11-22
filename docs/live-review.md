# Live project review

Start a local review server for a project that already has its review artifacts:

```bash
video-agent review-serve ./my-video-project
```

Open the printed URL in your browser. The server hosts the existing tabbed review workspace and adds a **Save as draft revision** action to the visual timeline page. Reordering shots and changing source in/out points can then be saved directly into a new file under `.video-agent/timeline-revisions/` without exporting JSON or returning to the terminal.

By default, the server binds only to `127.0.0.1` and chooses an available port. It uses a per-run session token and same-origin checks for its write endpoint. It serves generated `.video-agent` review artifacts and common source-media formats from the selected project folder, while blocking environment files and common development directories.

### Using Docker

Publish a fixed container port to the Mac's loopback interface, and bind the server to all container interfaces:

```bash
docker compose run --rm -p 127.0.0.1:8765:8765 video-agent \
  review-serve ./travel-test --host 0.0.0.0 --port 8765
```

Open the URL printed by the command (it will use `http://127.0.0.1:8765`). Keep the terminal running while reviewing. The port is published only on the host's loopback interface; do not change the host-side `127.0.0.1` binding unless you intend to expose the review server to your network.

Saving timeline edits creates a separate, unapproved revision; the source timeline and media remain unchanged. In the live timeline page, **Save clip decisions** persists per-shot approve/reject/pending choices without approving the timeline. **Approve complete timeline** is a separate confirmed action and is rejected unless every clip has an explicit decision and at least one clip is approved. Rendering remains a separate CLI action. The static `review-workspace` command remains available.
