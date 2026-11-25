# Color consistency analysis

Run metadata analysis first, then sample representative frames from each source video:

```bash
video-agent analyze ./my-project
video-agent analyze-color ./my-project --samples 5
```

The report is written to `.video-agent/color-analysis.json`. It records FFmpeg
signalstats measurements (average luma and chroma U/V) and bounded offsets that
would move each sample toward the project-wide median measured across all source videos.

This stage is diagnostic only. It does not apply color filters, modify source
media, approve corrections, or render a video. Review the measurements before
using any suggested values in a later grading workflow. The measurements are
frame-wide averages and are not semantic or perceptual color judgments.
