# Video model quality and stability benchmark

The first candidate benchmark is useful for a quick one-scene latency check. Use
`scripts/benchmark_scene_suite.py` for the follow-up comparison across multiple scenes.
It keeps each model loaded for the complete suite and validates outputs against the
production Pydantic schemas.

## 1. Prepare a scene manifest

Choose 20–30 representative scenes from the same source video. Include two or three
ordered frames per scene, with their actual source-video timestamps. Use the same
manifest for both models.

Paths are relative to the manifest file:

```json
{
  "scenes": [
    {
      "scene_id": "scene-001",
      "frames": [
        {"path": "frames/scene-001-0.jpg", "timestamp_seconds": 12.4},
        {"path": "frames/scene-001-1.jpg", "timestamp_seconds": 15.1},
        {"path": "frames/scene-001-2.jpg", "timestamp_seconds": 18.0}
      ],
      "review_notes": "Performer moves across a waterfront plaza",
      "expected_events": [
        "Performer is visible",
        "Audience watches the performance"
      ]
    }
  ]
}
```

Include varied footage: landscapes, walking, driving, camera movement, people,
object interaction, rapid action, and visually similar scenes. Keep the frame count
and sampling positions identical between candidates.

## 2. Run the suite

Run from the repository root in the native macOS MLX environment:

```bash
python scripts/benchmark_scene_suite.py ./benchmark-scenes/manifest.json \
  --providers smolvlm2 qwen3-vl \
  --output .video-agent/scene-suite-benchmark.json
```

The model is loaded once per candidate and remains resident while all scenes are
processed. The report includes per-frame and per-scene timings, mean/p50/p95 latency,
schema-validation rates, per-scene memory snapshots, raw model outputs, and human-review
annotations. Qwen records prompt-template formatting time separately from the high-level
MLX-VLM generation call, plus output character count, for each frame and temporal request.
The generation-call duration includes image preprocessing and token generation; the current
high-level API does not expose a reliable split between those internal phases. Failed schema
validation is recorded per output instead of discarding the rest of the suite.

## 3. Review quality separately from schema validity

The benchmark validates frames with `FrameObservation` and temporal outputs with
`SceneDynamics`. This catches missing fields, invalid enum values, wrong types, and
out-of-range confidence values. It does not prove that a description is visually true.

Review the saved outputs against the source frames and the `review_notes` /
`expected_events` annotations. In particular, check whether temporal action progression
is supported by the sampled frames, whether camera motion is distinguished from subject
motion, and whether the model invents events between samples. The annotations are
human-review aids; the tool does not calculate semantic accuracy automatically.

## 4. Interpret performance carefully

- Compare warm per-frame and temporal latency across the full suite; model loading is
  reported separately.
- Use p50 and p95 to spot slow scenes rather than relying only on a three-frame average.
- Inspect memory snapshots for growth over the run. MLX allocation metrics are not the
  same as total macOS unified-memory pressure.
- Run candidates separately, with the same machine state and manifest, to reduce
  cross-model memory and thermal interference.
- Repeat the suite if results vary materially. Do not reduce frame samples or disable
  temporal analysis to improve a candidate's score.

This is an isolated experiment. It does not change production provider selection,
sampling, checkpoints, or understanding behavior.
