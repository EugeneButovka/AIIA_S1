import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import config
from detection import DetectionResult, InvalidImageError


@dataclass(frozen=True)
class TrackingResult(DetectionResult):
    vehicles: int = 0
    unique_persons: int = 0
    unique_vehicles: int = 0


class TrackRegistry:
    def __init__(self, min_confirmed_frames: int):
        self._min_confirmed_frames = max(1, min_confirmed_frames)
        self._streaks: Dict[int, int] = {}
        self._person_ids: Set[int] = set()
        self._vehicle_ids: Set[int] = set()

    def observe_frame(self, observations: List[Tuple[int, int]]) -> None:
        seen: Set[int] = set()
        for track_id, class_id in observations:
            seen.add(track_id)
            streak = self._streaks.get(track_id, 0) + 1
            self._streaks[track_id] = streak
            if streak < self._min_confirmed_frames:
                continue
            if class_id == config.PERSON_CLASS_ID:
                self._person_ids.add(track_id)
            elif class_id in config.VEHICLE_CLASS_IDS:
                self._vehicle_ids.add(track_id)
        for stale_id in [track_id for track_id in self._streaks if track_id not in seen]:
            del self._streaks[stale_id]

    def unique_persons(self) -> int:
        return len(self._person_ids)

    def unique_vehicles(self) -> int:
        return len(self._vehicle_ids)

    def reset(self) -> None:
        self._streaks.clear()
        self._person_ids.clear()
        self._vehicle_ids.clear()


def frame_counts(detections: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    persons = sum(
        1
        for detection in detections
        if detection["class"] == config.PERSON_CLASS_ID
        and detection["confidence"] >= config.CONFIDENCE_THRESHOLD
    )
    vehicles = sum(
        1
        for detection in detections
        if detection["class"] in config.VEHICLE_CLASS_IDS
        and detection["confidence"] >= config.CONFIDENCE_THRESHOLD
    )
    confidences = [detection["confidence"] for detection in detections]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return persons, vehicles, avg_confidence


class TrackingDetector:
    def __init__(self, weights: str = config.TRACKING_WEIGHTS):
        self._model = YOLO(weights)
        self._registry = TrackRegistry(config.UNIQUE_COUNT_MIN_CONFIRMED_FRAMES)
        self._tracked_classes = [config.PERSON_CLASS_ID, *config.VEHICLE_CLASS_IDS]

    def detect(self, image_bytes: bytes) -> TrackingResult:
        image = self._decode(image_bytes)
        results = self._model.track(
            image,
            persist=True,
            tracker=str(config.TRACKER_CONFIG),
            classes=self._tracked_classes,
            verbose=False,
        )
        result = results[0]
        detections = json.loads(result.to_json())
        track_ids = self._extract_track_ids(result.boxes, len(detections))
        for detection, track_id in zip(detections, track_ids):
            detection["track_id"] = track_id
        persons, vehicles, avg_confidence = frame_counts(detections)
        self._registry.observe_frame(
            [
                (detection["track_id"], detection["class"])
                for detection in detections
                if detection["track_id"] is not None
                and detection["confidence"] >= config.CONFIDENCE_THRESHOLD
            ]
        )
        speed = result.speed or {}
        return TrackingResult(
            detections=detections,
            persons=persons,
            vehicles=vehicles,
            inference_ms=float(speed.get("inference", 0.0)),
            avg_confidence=avg_confidence,
            unique_persons=self._registry.unique_persons(),
            unique_vehicles=self._registry.unique_vehicles(),
        )

    def reset_registry(self) -> None:
        self._registry.reset()

    @staticmethod
    def _decode(image_bytes: bytes) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImageError("Invalid or unreadable image file")
        return image

    @staticmethod
    def _extract_track_ids(boxes, expected: int) -> List[Optional[int]]:
        if boxes is None or boxes.id is None:
            return [None] * expected
        track_ids = [int(track_id) for track_id in boxes.id.tolist()]
        return track_ids + [None] * (expected - len(track_ids))