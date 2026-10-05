import json
from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from detection import CONFIDENCE_THRESHOLD, PERSON_CLASS_ID, Detector, InvalidImageError


def build_detection(class_id, confidence, x=0, y=0):
    return {
        "class": class_id,
        "confidence": confidence,
        "box": {"x1": x, "y1": y, "x2": x + 10, "y2": y + 10},
    }


def build_image_bytes(width=80, height=50):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes()


@dataclass(frozen=True)
class FakeResults:
    detections: list
    inference_ms: float

    def to_json(self):
        return json.dumps(self.detections)

    @property
    def speed(self):
        return {"inference": self.inference_ms}


class FakeModel:
    def __init__(self, detections, inference_ms):
        self._results = [FakeResults(detections, inference_ms)]
        self.calls = []

    def predict(self, image):
        self.calls.append(image)
        return self._results


def install_fake_model(monkeypatch, detections, inference_ms):
    model = FakeModel(detections, inference_ms)
    monkeypatch.setattr("detection.YOLO", lambda weights: model)
    return model


def test_detect_counts_thresholded_persons_and_averages_all_confidences(monkeypatch):
    # given
    detections = [
        build_detection(PERSON_CLASS_ID, 0.87),
        build_detection(PERSON_CLASS_ID, 0.30),
        build_detection(2, 0.99),
    ]
    model = install_fake_model(monkeypatch, detections, 29.9)
    detector = Detector("yolov8n.pt")

    # when
    result = detector.detect(build_image_bytes())

    # then
    assert len(model.calls) == 1
    assert model.calls[0].shape == (50, 80, 3)
    assert result.detections == detections
    assert result.persons == 1
    assert result.avg_confidence == pytest.approx((0.87 + 0.30 + 0.99) / 3)
    assert result.inference_ms == 29.9


def test_detect_without_detections_returns_zeros(monkeypatch):
    # given
    install_fake_model(monkeypatch, [], 29.9)
    detector = Detector("yolov8n.pt")

    # when
    result = detector.detect(build_image_bytes())

    # then
    assert result.persons == 0
    assert result.detections == []
    assert result.avg_confidence == 0.0


def test_detect_rejects_invalid_image(monkeypatch):
    # given
    install_fake_model(monkeypatch, [], 29.9)
    detector = Detector("yolov8n.pt")

    # when
    # then
    with pytest.raises(InvalidImageError):
        detector.detect(b"not-an-image")