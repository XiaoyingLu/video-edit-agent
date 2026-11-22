# Live project review

Start a loopback-only review server for a project that already has its review artifacts:

```bash
video-agent review-serve ./my-video-project
```

Open the printed URL in your browser. The server hosts the existing tabbed review workspace and adds a **Save as draft revision** action to the visual timeline page. Reordering shots and changing source in/out points can then be saved directly into a new file under `.video-agent/timeline-revisions/` without exporting JSON or returning to the terminal.

The server binds only to `127.0.0.1`, chooses an available port by default, and uses a per-run session token plus same-origin checks for its write endpoint. It serves generated `.video-agent` review artifacts and common source-media formats from the selected project folder, while blocking environment files and common development directories.

Saving creates a separate, unapproved revision. The source timeline and source media are unchanged. This first slice does not apply clip approve/reject decisions, approve the complete timeline, or render video; use the existing review and render commands for those explicit steps. The existing static `review-workspace` command remains available.
