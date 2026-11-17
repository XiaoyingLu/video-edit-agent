import json
from pathlib import Path

from travel_video_agent.understanding.frame_sampler import extract_representative_frames
from travel_video_agent.understanding.pipeline import understand_folder
from travel_video_agent.understanding.scene_detection import detect_scenes
from travel_video_agent.understanding.visual_analyzer import OllamaVisionAnalyzer


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
    assert "no more than 5 key objects" in requests[1]["messages"][0]["content"]


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
