import json
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

import config


class InvalidImageError(ValueError):
    pass


@dataclass(frozen=True)
class DetectionResult:
    detections: list
    persons: int
    inference_ms: float
    avg_confidence: float


class Detector:
    def __init__(self, weights: str = "yolov8n.pt"):
        self._model = YOLO(weights)

    def detect(self, image_bytes: bytes) -> DetectionResult:
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImageError("Invalid or unreadable image file")

        results = self._model.predict(image)
        detections = json.loads(results[0].to_json())
        speed = results[0].speed or {}

        return DetectionResult(
            detections=detections,
            persons=self._count_persons(detections),
            inference_ms=float(speed.get("inference", 0.0)),
            avg_confidence=self._avg_confidence(detections),
        )

    @staticmethod
    def _count_persons(detections: list) -> int:
        return sum(
            1
            for detection in detections
            if detection["class"] == config.PERSON_CLASS_ID
            and detection["confidence"] >= config.CONFIDENCE_THRESHOLD
        )

    @staticmethod
    def _avg_confidence(detections: list) -> float:
        confidences = [detection["confidence"] for detection in detections]
        return sum(confidences) / len(confidences) if confidences else 0.0