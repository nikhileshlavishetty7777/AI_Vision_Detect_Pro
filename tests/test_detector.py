"""
Tests for utils/detector.py

YOLO inference itself is mocked (via monkeypatching `detector.load_model`)
so these tests run fast and don't require downloading real weights or
installing PyTorch. What IS tested for real: detection extraction logic,
DetectionResult's derived properties, the image-detection pipeline, and
the full video-processing loop against a real (synthetic) video file.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from utils import detector


class FakeScalar:
    """Stands in for a torch scalar tensor (`.item()`)."""

    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class FakeArray:
    """Stands in for a torch tensor (`.tolist()`)."""

    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class FakeBox:
    def __init__(self, class_id: int, confidence: float, xyxy: list[float]):
        self.cls = FakeScalar(class_id)
        self.conf = FakeScalar(confidence)
        self.xyxy = [FakeArray(xyxy)]


class FakeResult:
    def __init__(self, boxes: list[FakeBox], names: dict[int, str], plot_image: np.ndarray):
        self.boxes = boxes
        self.names = names
        self._plot_image = plot_image

    def plot(self) -> np.ndarray:
        return self._plot_image


class FakeModel:
    """Stands in for a loaded YOLO model; `predict()` returns a preset FakeResult."""

    def __init__(self, result: FakeResult):
        self._result = result
        self.predict_calls: list[dict] = []

    def predict(self, source, conf, iou, verbose=False):
        self.predict_calls.append({"conf": conf, "iou": iou})
        return [self._result]


NAMES = {0: "person", 2: "car"}


def _make_fake_result(image_shape=(48, 64, 3)) -> FakeResult:
    boxes = [
        FakeBox(0, 0.9, [1.0, 1.0, 10.0, 10.0]),
        FakeBox(2, 0.6, [5.0, 5.0, 20.0, 20.0]),
    ]
    plot_image = np.ones(image_shape, dtype="uint8") * 255
    return FakeResult(boxes, NAMES, plot_image)


class TestExtractDetections:
    def test_extracts_all_boxes_with_correct_fields(self):
        result = _make_fake_result()
        detections = detector._extract_detections(result)
        assert len(detections) == 2
        assert detections[0].class_name == "person"
        assert detections[0].confidence == pytest.approx(0.9)
        assert detections[0].box_xyxy == (1.0, 1.0, 10.0, 10.0)
        assert detections[1].class_name == "car"

    def test_no_boxes_returns_empty_list(self):
        result = FakeResult(boxes=None, names=NAMES, plot_image=np.zeros((10, 10, 3), dtype="uint8"))
        assert detector._extract_detections(result) == []

    def test_unknown_class_id_falls_back_to_str(self):
        result = FakeResult(
            boxes=[FakeBox(99, 0.5, [0, 0, 1, 1])], names=NAMES, plot_image=np.zeros((10, 10, 3), dtype="uint8")
        )
        detections = detector._extract_detections(result)
        assert detections[0].class_name == "99"


class TestDetectionResultProperties:
    def _build(self):
        return detector.DetectionResult(
            annotated_image=np.zeros((10, 10, 3), dtype="uint8"),
            detections=[
                detector.Detection(0, "person", 0.9, (0, 0, 1, 1)),
                detector.Detection(0, "person", 0.5, (0, 0, 1, 1)),
                detector.Detection(2, "car", 0.7, (0, 0, 1, 1)),
            ],
            inference_time_ms=12.3,
        )

    def test_object_count(self):
        assert self._build().object_count == 3

    def test_class_counts(self):
        assert self._build().class_counts == {"person": 2, "car": 1}

    def test_average_confidence(self):
        result = self._build()
        assert result.average_confidence == pytest.approx((0.9 + 0.5 + 0.7) / 3)

    def test_average_confidence_with_no_detections(self):
        empty = detector.DetectionResult(annotated_image=np.zeros((5, 5, 3), dtype="uint8"))
        assert empty.average_confidence == 0.0


class TestRunImageDetection:
    def test_raises_on_empty_image(self, monkeypatch):
        monkeypatch.setattr(detector, "load_model", lambda w: FakeModel(_make_fake_result()))
        with pytest.raises(detector.DetectorError, match="empty"):
            detector.run_image_detection(np.array([]), model_weight="yolov8n.pt")

    def test_returns_populated_detection_result(self, monkeypatch):
        fake_model = FakeModel(_make_fake_result())
        monkeypatch.setattr(detector, "load_model", lambda w: fake_model)

        image = np.zeros((48, 64, 3), dtype="uint8")
        result = detector.run_image_detection(image, model_weight="yolov8n.pt", conf=0.3, iou=0.5)

        assert result.object_count == 2
        assert result.class_counts == {"person": 1, "car": 1}
        assert result.inference_time_ms >= 0
        # Confirms conf/iou were actually forwarded to the model
        assert fake_model.predict_calls[0] == {"conf": 0.3, "iou": 0.5}


class TestProcessVideo:
    def _write_synthetic_video(self, path: Path, num_frames: int = 6, size=(64, 48)) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
        for _ in range(num_frames):
            frame = (np.random.rand(size[1], size[0], 3) * 255).astype("uint8")
            writer.write(frame)
        writer.release()

    def test_processes_all_frames_and_produces_output(self, tmp_path, monkeypatch):
        input_path = tmp_path / "input.mp4"
        self._write_synthetic_video(input_path, num_frames=6)

        fake_model = FakeModel(_make_fake_result(image_shape=(48, 64, 3)))
        monkeypatch.setattr(detector, "load_model", lambda w: fake_model)
        monkeypatch.setattr(detector, "OUTPUTS_DIR", tmp_path)

        progress_calls = []
        result = detector.process_video(
            input_path=input_path,
            model_weight="yolov8n.pt",
            frame_skip=2,
            progress_callback=lambda idx, total, fps: progress_calls.append((idx, total, fps)),
        )

        assert result.total_frames == 6
        assert result.output_path.exists()
        assert result.average_fps > 0
        assert len(progress_calls) == 6
        # frame_skip=2, but frame 1 always runs inference too (so the very
        # first output frame is never a raw pass-through) -> frames 1,2,4,6
        # get real inference = 4 calls x 2 detections/call = 8 rows.
        assert len(result.detections_df) == 8

    def test_raises_on_unopenable_video(self, tmp_path, monkeypatch):
        monkeypatch.setattr(detector, "load_model", lambda w: FakeModel(_make_fake_result()))
        bogus_path = tmp_path / "does_not_exist.mp4"
        with pytest.raises(detector.DetectorError, match="could not open"):
            detector.process_video(input_path=bogus_path, model_weight="yolov8n.pt")
