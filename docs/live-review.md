# Live project review

Start a local review server for a project that already has its review artifacts:

```bash
video-agent review-serve ./my-video-project
```

Open the printed URL. The server hosts the tabbed review workspace and adds **Save as draft revision** to the visual timeline page. Reordering shots and changing source in/out points saves a new revision under `.video-agent/timeline-revisions/`.

By default, the server binds only to `127.0.0.1` and chooses an available port. It uses a per-run session token and same-origin checks for writes. It serves generated `.video-agent` artifacts and common media formats from the project folder, while blocking environment files and common development directories.

### Using Docker

Publish a fixed container port to the Mac's loopback interface, and bind the server to all container interfaces:

```bash
docker compose run --rm -p 127.0.0.1:8765:8765 video-agent \
  review-serve ./travel-test --host 0.0.0.0 --port 8765
```

Open the URL printed by the command (it will use `http://127.0.0.1:8765`). Keep the terminal running while reviewing. The port is published only on the host's loopback interface; do not change the host-side `127.0.0.1` binding unless you intend to expose the review server to your network.

Saving creates a separate, unapproved revision. The source timeline and source media remain unchanged. Clip approve/reject decisions, final timeline approval, and rendering remain explicit steps in the existing workflow. The static `review-workspace` command remains available.
