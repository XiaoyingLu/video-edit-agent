# Animated asset overlays

Reviewed title-overlay documents can also contain `overlay_kind: "asset"` entries. This lets a project composite animated logos, badges, or stickers without a separate animation runtime.

Place assets under `templates/overlays/` in the project:

- **WebM:** a transparent animated `.webm` clip. The renderer loops it for the overlay's active interval.
- **PNG sequence:** a numbered pattern such as `templates/overlays/spark-%04d.png`. Frames loop at `asset_fps` (default 30).

Add an entry to `.video-agent/title-overlays.json` and approve it with the existing review command:

```json
{
  "overlay_id": "brand-mark",
  "overlay_kind": "asset",
  "asset_path": "templates/overlays/brand-mark.webm",
  "asset_format": "webm",
  "asset_x": 0.92,
  "asset_y": 0.08,
  "asset_scale": 0.12,
  "start_seconds": 0.5,
  "end_seconds": 12.0,
  "decision": "pending"
}
```

For a PNG sequence, use `"asset_format": "png_sequence"`, a numbered `.png` pattern, and optionally set `"asset_fps": 24`.

Position values are normalized from 0 to 1: (0, 0) is top-left, (0.5, 0.5) centers the asset, and (1, 1) is bottom-right. `asset_scale` is the overlay width as a fraction of the output width. The asset keeps its aspect ratio.

Review and render through the same workflow as text overlays:

```bash
video-agent review-title-overlays ./my-project --approve brand-mark
video-agent render-draft ./my-project --title-overlays .video-agent/title-overlays.json
```

The renderer rejects paths outside `templates/overlays/`, missing assets, invalid sequence patterns, and overlays extending beyond the rendered timeline. Approved asset files are recorded as render dependencies.
