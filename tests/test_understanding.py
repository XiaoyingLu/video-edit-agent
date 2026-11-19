import json
from pathlib import Path

import pytest

from travel_video_agent.understanding.frame_sampler import extract_representative_frames
from travel_video_agent.understanding.pipeline import understand_folder
from travel_video_agent.understanding.scene_detection import detect_scenes
from travel_video_agent.understanding.visual_analyzer import OllamaVisionAnalyzer


@pytest.fixture(autouse=True)
def disable_environment_dependent_memory_preflight(monkeypatch):
    monkeypatch.setattr(
        "travel_video_agent.understanding.visual_analyzer.ensure_memory_available",
        lambda required_gb, task: None,
    )


def test_scene_detection_falls_back_to_one_scene_when_detector_finds_none(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeVideo:
        frame_rate = 30

    class FakeManager:
        def add_detector(self, detector):
            pass

        def detect_scenes(self, video, show_progress=False):
            pass

        def get_scene_list(self, start_in_scene=True):
            return []

    import scenedetect

    monkeypatch.setattr(scenedetect, "open_video", lambda path: FakeVideo())
    monkeypatch.setattr(scenedetect, "SceneManager", FakeManager)
    assert detect_scenes(tmp_path / "clip.mp4", 8.0) == [(0.0, 8.0)]


def test_frame_sampler_rejects_invalid_scene(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "travel_video_agent.understanding.frame_sampler.shutil.which", lambda _: "ffmpeg"
    )
    assert extract_representative_frames(tmp_path / "clip.mp4", 4.0, 4.0, tmp_path / "frames") == []


def test_ollama_analyzer_uses_environment_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434/")
    analyzer = OllamaVisionAnalyzer()
    assert analyzer.base_url == "http://ollama:11434"


def test_ollama_analyzer_parses_structured_response(monkeypatch, tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-image")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            content = {
                "description": "A lake beneath mountains.",
                "objects": ["lake", "mountains"],
                "activities": [],
                "confidence": 0.9,
            }
            return json.dumps({"message": {"content": json.dumps(content)}}).encode()

    monkeypatch.setattr(
        "travel_video_agent.understanding.visual_analyzer.urlopen",
        lambda request, timeout: FakeResponse(),
    )
    result = OllamaVisionAnalyzer().analyze_frame(frame)
    assert result["description"] == "A lake beneath mountains."
    assert result["confidence"] == 0.9


def test_ollama_analyzer_compares_rendered_frame_to_storyboard(monkeypatch, tmp_path: Path):
    frame = tmp_path / "rendered-frame.jpg"
    frame.write_bytes(b"fake-image")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            assessment = {
                "match": "partial",
                "score": 0.62,
                "confidence": 0.88,
                "evidence": "The image shows a mountain lake, but no visible trail.",
                "suggestion": "Check whether a hiking shot better supports the intended beat.",
            }
            return json.dumps({"message": {"content": json.dumps(assessment)}}).encode()

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("travel_video_agent.understanding.visual_analyzer.urlopen", fake_urlopen)
    analyzer = OllamaVisionAnalyzer(model="qwen2.5vl:3b")
    result = analyzer.compare_frame_to_storyboard(
        frame,
        purpose="Show the journey into the mountains",
        visual_description="A hiker following a trail toward a mountain lake",
        primary_subject="hiker",
        expected_role="action",
        expected_scale="wide",
    )

    assert result["match"] == "partial"
    assert result["score"] == 0.62
    assert result["confidence"] == 0.88
    assert captured["payload"]["format"]["type"] == "object"
    assert set(captured["payload"]["format"]["required"]) == {
        "match",
        "score",
        "confidence",
        "evidence",
        "suggestion",
    }
    prompt = captured["payload"]["messages"][0]["content"]
    assert "A hiker following a trail toward a mountain lake" in prompt
    assert "Expected editorial role: action" in prompt


def test_understanding_pipeline_builds_timestamped_scenes(monkeypatch, tmp_path: Path) -> None:
    from travel_video_agent.understanding import pipeline

    video = tmp_path / "clip.mp4"
    video.touch()
    info = type("Info", (), {"path": "clip.mp4", "duration_seconds": 5.0})()
    analysis = type("Analysis", (), {"video_count": 1, "videos": [info]})()
    monkeypatch.setattr(pipeline, "analyze_folder", lambda root: analysis)
    monkeypatch.setattr(pipeline, "detect_scenes", lambda *args: [(0.0, 5.0)])
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake")
    monkeypatch.setattr(
        pipeline,
        "extract_representative_frames",
        lambda *args, **kwargs: [(2.5, frame)],
    )
    monkeypatch.setattr(pipeline.OllamaVisionAnalyzer, "unload_model", lambda self: True)
    monkeypatch.setattr(
        pipeline.OllamaVisionAnalyzer,
        "analyze_frame",
        lambda self, path: {
            "description": "A scenic lake.",
            "objects": ["lake"],
            "activities": [],
            "confidence": 0.8,
        },
    )
    result = understand_folder(tmp_path)
    assert result.scene_count == 1
    assert result.scenes[0].start_seconds == 0.0
    assert result.scenes[0].observations[0].timestamp_seconds == 2.5


def test_ollama_analyzer_requests_json_schema(monkeypatch, tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-image")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            content = {
                "description": "A lake beneath mountains.",
                "objects": ["lake", "mountains"],
                "activities": [],
                "confidence": 0.9,
            }
            return json.dumps({"message": {"content": json.dumps(content)}}).encode()

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("travel_video_agent.understanding.visual_analyzer.urlopen", fake_urlopen)
    result = OllamaVisionAnalyzer(model="qwen2.5vl:3b").analyze_frame(frame)

    assert result["description"] == "A lake beneath mountains."
    assert captured["payload"]["format"]["type"] == "object"
    assert set(captured["payload"]["format"]["required"]) == {
        "description",
        "objects",
        "activities",
        "confidence",
        "shot_scale",
        "camera_angle",
        "composition",
        "lighting",
        "mood",
        "visual_role",
        "setting",
        "primary_subject",
    }


def test_ollama_analyzer_accepts_markdown_wrapped_json(monkeypatch, tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-image")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            content = (
                "```json\n"
                '{"description":"A coastal road.","objects":["road"],'
                '"activities":["driving"],"confidence":0.8}'
                "\n```"
            )
            return json.dumps({"message": {"content": content}}).encode()

    monkeypatch.setattr(
        "travel_video_agent.understanding.visual_analyzer.urlopen",
        lambda request, timeout: FakeResponse(),
    )
    result = OllamaVisionAnalyzer().analyze_frame(frame)

    assert result["description"] == "A coastal road."
    assert result["activities"] == ["driving"]


def test_ollama_analyzer_reports_invalid_json_with_response_preview(
    monkeypatch, tmp_path: Path
) -> None:
    from travel_video_agent.understanding.visual_analyzer import VisionModelError

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-image")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"message": {"content": "not json"}}).encode()

    monkeypatch.setattr(
        "travel_video_agent.understanding.visual_analyzer.urlopen",
        lambda request, timeout: FakeResponse(),
    )

    import pytest

    with pytest.raises(VisionModelError, match="Response preview"):
        OllamaVisionAnalyzer().analyze_frame(frame)


def test_ollama_analyzer_limits_structured_output_and_generation_tokens(
    monkeypatch, tmp_path: Path
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-image")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            content = {
                "description": "A person prepares coffee.",
                "objects": ["person", "coffee machine"],
                "activities": ["preparing coffee"],
                "confidence": 0.9,
            }
            return json.dumps({"message": {"content": json.dumps(content)}}).encode()

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("travel_video_agent.understanding.visual_analyzer.urlopen", fake_urlopen)
    OllamaVisionAnalyzer(model="qwen2.5vl:3b").analyze_frame(frame)

    schema = captured["payload"]["format"]
    assert schema["properties"]["description"]["maxLength"] == 200
    assert schema["properties"]["objects"]["maxItems"] == 8
    assert schema["properties"]["activities"]["maxItems"] == 4
    assert captured["payload"]["options"]["num_predict"] == 256
    assert captured["payload"]["format"]["properties"]["visual_role"]["enum"] == [
        "establishing",
        "action",
        "detail",
        "reaction",
        "transition",
        "portrait",
        "scenic_breath",
        "unknown",
    ]


def test_ollama_analyzer_retries_truncated_json_with_shorter_output(
    monkeypatch, tmp_path: Path
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-image")
    requests = []

    class FakeResponse:
        def __init__(self, content):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"message": {"content": self.content}}).encode()

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        requests.append(payload)
        if len(requests) == 1:
            return FakeResponse('{"description":"A person at a coffee machine","objects":["person"')
        return FakeResponse(
            json.dumps(
                {
                    "description": "A person prepares coffee.",
                    "objects": ["person", "coffee machine"],
                    "activities": ["preparing coffee"],
                    "confidence": 0.9,
                }
            )
        )

    monkeypatch.setattr("travel_video_agent.understanding.visual_analyzer.urlopen", fake_urlopen)
    result = OllamaVisionAnalyzer(model="qwen2.5vl:3b").analyze_frame(frame)

    assert result["description"] == "A person prepares coffee."
    assert len(requests) == 2
    assert requests[0]["options"]["num_predict"] == 256
    assert requests[1]["options"]["num_predict"] == 192
    assert "at most 5 objects and 2 activities" in requests[1]["messages"][0]["content"]


def test_ollama_analyzer_rejects_too_many_objects(monkeypatch, tmp_path: Path) -> None:
    import pytest

    from travel_video_agent.understanding.visual_analyzer import VisionModelError

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-image")
    content = json.dumps(
        {
            "description": "A busy counter.",
            "objects": [f"object {index}" for index in range(9)],
            "activities": [],
            "confidence": 0.8,
        }
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"message": {"content": content}}).encode()

    monkeypatch.setattr(
        "travel_video_agent.understanding.visual_analyzer.urlopen",
        lambda request, timeout: FakeResponse(),
    )

    with pytest.raises(VisionModelError, match="after one short-output retry"):
        OllamaVisionAnalyzer().analyze_frame(frame)


def test_ollama_analyzer_sends_keep_alive_and_can_unload(monkeypatch, tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-image")
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            content = {
                "description": "A coastal road.",
                "objects": ["road"],
                "activities": [],
                "confidence": 0.9,
            }
            return json.dumps({"message": {"content": json.dumps(content)}}).encode()

    def fake_urlopen(request, timeout):
        requests.append((request.full_url, json.loads(request.data.decode("utf-8"))))
        return FakeResponse()

    monkeypatch.setattr("travel_video_agent.understanding.visual_analyzer.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "travel_video_agent.understanding.visual_analyzer.ensure_memory_available",
        lambda required_gb, task: None,
    )
    analyzer = OllamaVisionAnalyzer(model="qwen2.5vl:3b")
    analyzer.analyze_frame(frame)
    assert requests[0][1]["keep_alive"] == "5m"
    assert analyzer.unload_model() is True
    assert requests[1][0].endswith("/api/generate")
    assert requests[1][1] == {"model": "qwen2.5vl:3b", "keep_alive": 0}


def test_vision_memory_preflight_stops_before_ollama_request(monkeypatch, tmp_path: Path) -> None:
    from travel_video_agent.resource_manager import ResourceError
    from travel_video_agent.understanding.visual_analyzer import VisionModelError

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-image")

    def reject(required_gb, task):
        raise ResourceError("Not enough available memory")

    monkeypatch.setattr(
        "travel_video_agent.understanding.visual_analyzer.ensure_memory_available", reject
    )
    monkeypatch.setattr(
        "travel_video_agent.understanding.visual_analyzer.urlopen",
        lambda *args, **kwargs: pytest.fail("Ollama must not be called after failed preflight"),
    )
    with pytest.raises(VisionModelError, match="Not enough available memory"):
        OllamaVisionAnalyzer(model="qwen2.5vl:7b").analyze_frame(frame)


def test_vision_analyzer_normalizes_missing_editorial_fields_to_unknown(
    monkeypatch, tmp_path: Path
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-image")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            content = {
                "description": "A lake beneath mountains.",
                "objects": ["lake", "mountains"],
                "activities": [],
                "confidence": 0.9,
            }
            return json.dumps({"message": {"content": json.dumps(content)}}).encode()

    monkeypatch.setattr(
        "travel_video_agent.understanding.visual_analyzer.urlopen",
        lambda request, timeout: FakeResponse(),
    )
    result = OllamaVisionAnalyzer().analyze_frame(frame)
    assert result["shot_scale"] == "unknown"
    assert result["visual_role"] == "unknown"
    assert result["setting"] == ""
    assert result["primary_subject"] == ""


def test_vision_analyzer_returns_cinematic_attributes(monkeypatch, tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-image")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            content = {
                "description": "A wide coastal landscape at sunset.",
                "objects": ["ocean", "cliffs"],
                "activities": [],
                "confidence": 0.95,
                "shot_scale": "wide",
                "camera_angle": "eye_level",
                "composition": "leading_lines",
                "lighting": "golden_hour",
                "mood": "serene",
                "visual_role": "establishing",
                "setting": "rocky coastline",
                "primary_subject": "ocean and cliffs",
            }
            return json.dumps({"message": {"content": json.dumps(content)}}).encode()

    monkeypatch.setattr(
        "travel_video_agent.understanding.visual_analyzer.urlopen",
        lambda request, timeout: FakeResponse(),
    )
    result = OllamaVisionAnalyzer().analyze_frame(frame)
    assert result["shot_scale"] == "wide"
    assert result["composition"] == "leading_lines"
    assert result["lighting"] == "golden_hour"
    assert result["visual_role"] == "establishing"
    assert result["setting"] == "rocky coastline"


def test_scene_editorial_labels_use_most_common_known_value() -> None:
    from travel_video_agent.understanding.pipeline import _dominant_label
    from travel_video_agent.understanding.schemas import FrameObservation

    observations = [
        FrameObservation(
            timestamp_seconds=1, frame_path="a.jpg", description="a", visual_role="establishing"
        ),
        FrameObservation(
            timestamp_seconds=2, frame_path="b.jpg", description="b", visual_role="unknown"
        ),
        FrameObservation(
            timestamp_seconds=3, frame_path="c.jpg", description="c", visual_role="establishing"
        ),
        FrameObservation(
            timestamp_seconds=4, frame_path="d.jpg", description="d", visual_role="detail"
        ),
    ]
    assert _dominant_label(observations, "visual_role") == "establishing"
    assert _dominant_label(observations, "mood") == "unknown"


def test_temporal_analyzer_uses_ordered_frames_and_validates_dynamics(monkeypatch, tmp_path):
    from travel_video_agent.understanding.visual_analyzer import OllamaVisionAnalyzer

    frames = []
    for index in range(3):
        frame = tmp_path / f"frame-{index}.jpg"
        frame.write_bytes(b"image")
        frames.append((float(index * 2), frame))
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            result = {
                "temporal_summary": "A hiker walks toward the mountain viewpoint.",
                "action_progression": [
                    "Hiker enters the frame",
                    "Hiker approaches the viewpoint",
                ],
                "subject_motion": "approaching",
                "camera_motion": "static",
                "confidence": 0.88,
            }
            return json.dumps({"message": {"content": json.dumps(result)}}).encode()

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("travel_video_agent.understanding.visual_analyzer.urlopen", fake_urlopen)
    analyzer = OllamaVisionAnalyzer(model="qwen2.5vl:3b")
    result = analyzer.analyze_scene(frames)

    assert result["subject_motion"] == "approaching"
    assert result["camera_motion"] == "static"
    assert result["confidence"] == pytest.approx(0.88)
    message = captured["payload"]["messages"][0]
    assert len(message["images"]) == 3
    assert "0.00s, 2.00s, 4.00s" in message["content"]
    assert captured["payload"]["options"]["temperature"] == 0


def test_temporal_analyzer_returns_unknown_for_single_frame(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"image")
    result = OllamaVisionAnalyzer().analyze_scene([(1.0, frame)])
    assert result["camera_motion"] == "unknown"
    assert result["subject_motion"] == "unknown"
    assert result["confidence"] == 0


def test_understanding_pipeline_attaches_temporal_scene_dynamics(monkeypatch, tmp_path):
    from travel_video_agent.understanding import pipeline

    video = tmp_path / "clip.mp4"
    video.touch()
    info = type("Info", (), {"path": "clip.mp4", "duration_seconds": 5.0})()
    analysis = type("Analysis", (), {"video_count": 1, "videos": [info]})()
    monkeypatch.setattr(pipeline, "analyze_folder", lambda root: analysis)
    monkeypatch.setattr(pipeline, "detect_scenes", lambda *args: [(0.0, 5.0)])
    frames = []
    for index, timestamp in enumerate((1.0, 4.0)):
        frame = tmp_path / f"frame-{index}.jpg"
        frame.write_bytes(b"fake")
        frames.append((timestamp, frame))
    monkeypatch.setattr(pipeline, "extract_representative_frames", lambda *args, **kwargs: frames)
    monkeypatch.setattr(pipeline.OllamaVisionAnalyzer, "unload_model", lambda self: True)
    monkeypatch.setattr(
        pipeline.OllamaVisionAnalyzer,
        "analyze_frame",
        lambda self, path: {
            "description": "A hiker on a mountain trail.",
            "objects": ["hiker", "trail"],
            "activities": ["walking"],
            "confidence": 0.8,
        },
    )
    monkeypatch.setattr(
        pipeline.OllamaVisionAnalyzer,
        "analyze_scene",
        lambda self, scene_frames: {
            "temporal_summary": "The hiker walks uphill and moves farther into the landscape.",
            "action_progression": [
                "Hiker walks uphill",
                "Hiker recedes into the distance",
            ],
            "subject_motion": "receding",
            "camera_motion": "static",
            "confidence": 0.9,
        },
    )

    result = understand_folder(tmp_path)
    dynamics = result.scenes[0].dynamics
    assert dynamics is not None
    assert dynamics.subject_motion == "receding"
    assert dynamics.camera_motion == "static"
    assert dynamics.action_progression[0] == "Hiker walks uphill"

def test_understanding_resumes_cached_frame_observations_after_interruption(
    monkeypatch, tmp_path: Path
) -> None:
    from travel_video_agent.understanding import pipeline
    from travel_video_agent.understanding.visual_analyzer import VisionModelError

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"source-video")
    info = type("Info", (), {"path": "clip.mp4", "duration_seconds": 5.0})()
    analysis = type("Analysis", (), {"video_count": 1, "videos": [info]})()
    monkeypatch.setattr(pipeline, "analyze_folder", lambda root: analysis)
    monkeypatch.setattr(pipeline, "detect_scenes", lambda *args: [(0.0, 5.0)])

    frames = []
    for index, timestamp in enumerate((1.0, 4.0)):
        frame = tmp_path / f"frame-{index}.jpg"
        frame.write_bytes(f"frame-{index}".encode())
        frames.append((timestamp, frame))
    monkeypatch.setattr(
        pipeline, "extract_representative_frames", lambda *args, **kwargs: frames
    )

    calls = []
    fail_once = True

    class FakeAnalyzer:
        def __init__(self, model):
            self.model = model

        def analyze_frame(self, frame_path):
            nonlocal fail_once
            calls.append(frame_path.name)
            if frame_path.name == "frame-1.jpg" and fail_once:
                fail_once = False
                raise VisionModelError("simulated interruption during frame analysis")
            return {
                "description": f"Description for {frame_path.name}",
                "objects": [],
                "activities": [],
                "confidence": 0.9,
            }

        def analyze_scene(self, scene_frames):
            return {
                "temporal_summary": "The scene progresses across the frames.",
                "action_progression": ["The scene begins", "The scene continues"],
                "subject_motion": "unknown",
                "camera_motion": "static",
                "confidence": 0.8,
            }

        def unload_model(self):
            return True

    monkeypatch.setattr(pipeline, "OllamaVisionAnalyzer", FakeAnalyzer)

    with pytest.raises(VisionModelError, match="simulated interruption"):
        understand_folder(tmp_path, samples_per_scene=2)

    checkpoint = tmp_path / ".video-agent" / "understanding-checkpoint.json"
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert len(next(iter(saved["scenes"].values()))["frames"]) == 1

    result = understand_folder(tmp_path, samples_per_scene=2)

    assert calls == ["frame-0.jpg", "frame-1.jpg", "frame-1.jpg"]
    assert result.scene_count == 1
    assert len(result.scenes[0].observations) == 2
    assert result.scenes[0].dynamics is not None

