import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import config
from detection import InvalidImageError
from tracking import frame_counts

Predictor = Callable[[np.ndarray], Tuple[int, int, float, float]]


@dataclass(frozen=True)
class VariantSummary:
    scale: float
    quality: int
    kb_per_frame: float
    persons: int
    vehicles: int
    avg_confidence: float
    parity: float
    inference_ms: float

    def to_payload(self) -> Dict[str, Any]:
        return {
            "scale": self.scale,
            "quality": self.quality,
            "kb_per_frame": round(self.kb_per_frame, 1),
            "persons": self.persons,
            "vehicles": self.vehicles,
            "avg_confidence": round(self.avg_confidence, 4),
            "parity": round(self.parity, 3),
            "avg_inference_ms": round(self.inference_ms, 1),
        }


@dataclass(frozen=True)
class FrameMeasurement:
    persons: int
    vehicles: int
    avg_confidence: float
    inference_ms: float
    kb: float


def recommend(baseline: VariantSummary, variants: List[VariantSummary]) -> Tuple[VariantSummary, str]:
    candidates = [
        variant
        for variant in variants
        if variant.parity >= config.CALIBRATION_MIN_PARITY
        and abs(variant.persons - baseline.persons) <= config.CALIBRATION_COUNT_TOLERANCE
        and abs(variant.vehicles - baseline.vehicles) <= config.CALIBRATION_COUNT_TOLERANCE
        and variant.avg_confidence
        >= baseline.avg_confidence * (1 - config.CALIBRATION_CONFIDENCE_DROP_LIMIT)
    ]
    if candidates:
        return (
            min(candidates, key=lambda variant: variant.kb_per_frame),
            "smallest variant passing parity, count and confidence gates",
        )
    fallback = next(
        (
            variant
            for variant in variants
            if variant.scale == config.CALIBRATION_FALLBACK_SCALE
            and variant.quality == config.CALIBRATION_FALLBACK_QUALITY
        ),
        None,
    )
    if fallback is not None:
        return fallback, "no variant passed the gates; falling back to full resolution q75"
    return baseline, "no variant passed the gates and fallback missing; keeping baseline"


class Calibrator:
    def __init__(
        self,
        weights: str,
        predictor: Optional[Predictor] = None,
        persist_path: Path = config.CALIBRATION_FILENAME,
    ):
        self._weights = weights
        self._predictor = predictor
        self._persist_path = persist_path
        self._model: Optional[YOLO] = None

    def calibrate(self, images: List[bytes]) -> Dict[str, Any]:
        if not images:
            raise InvalidImageError("No images provided")
        frames = [self._decode(image_bytes) for image_bytes in images]
        baseline_measurements = [self._measure_baseline(frame) for frame in frames]
        baseline = self._summarize(1.0, 95, baseline_measurements, [None] * len(frames))
        variants = self._evaluate_variants(frames, baseline_measurements)
        recommendation, rule = recommend(baseline, variants)
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "frames": len(frames),
            "weights": self._weights,
            "baseline": baseline.to_payload(),
            "variants": [
                variant.to_payload()
                for variant in sorted(variants, key=lambda item: item.kb_per_frame)
            ],
            "recommendation": {
                "scale": recommendation.scale,
                "quality": recommendation.quality,
                "kb_per_frame": round(recommendation.kb_per_frame, 1),
                "persons": recommendation.persons,
                "vehicles": recommendation.vehicles,
            },
            "rule": rule,
            "bandwidth_vs_baseline_pct": round(
                100 * recommendation.kb_per_frame / baseline.kb_per_frame, 1
            ),
        }
        self._persist(payload)
        return payload

    def load(self) -> Optional[Dict[str, Any]]:
        if not self._persist_path.exists():
            return None
        return json.loads(self._persist_path.read_text())

    def _evaluate_variants(
        self, frames: List[np.ndarray], baseline_measurements: List[FrameMeasurement]
    ) -> List[VariantSummary]:
        summaries = []
        for scale in config.CALIBRATION_RESOLUTION_SCALES:
            resized = frames if scale == 1.0 else [self._resize(frame, scale) for frame in frames]
            for quality in config.CALIBRATION_JPEG_QUALITIES:
                if scale == 1.0 and quality == 95:
                    continue
                measurements = []
                parities = []
                for source, baseline_measurement in zip(resized, baseline_measurements):
                    encoded = self._encode(source, quality)
                    decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                    persons, vehicles, avg_confidence, inference_ms = self._predict(decoded)
                    measurements.append(
                        FrameMeasurement(
                            persons=persons,
                            vehicles=vehicles,
                            avg_confidence=avg_confidence,
                            inference_ms=inference_ms,
                            kb=len(encoded) / 1024,
                        )
                    )
                    parities.append(
                        persons == baseline_measurement.persons
                        and vehicles == baseline_measurement.vehicles
                    )
                summaries.append(
                    self._summarize(scale, quality, measurements, parities)
                )
        return summaries

    def _measure_baseline(self, frame: np.ndarray) -> FrameMeasurement:
        persons, vehicles, avg_confidence, inference_ms = self._predict(frame)
        baseline_kb = len(self._encode(frame, 95)) / 1024
        return FrameMeasurement(
            persons=persons,
            vehicles=vehicles,
            avg_confidence=avg_confidence,
            inference_ms=inference_ms,
            kb=baseline_kb,
        )

    def _summarize(
        self,
        scale: float,
        quality: int,
        measurements: List[FrameMeasurement],
        parities: List[Optional[bool]],
    ) -> VariantSummary:
        return VariantSummary(
            scale=scale,
            quality=quality,
            kb_per_frame=sum(item.kb for item in measurements) / len(measurements),
            persons=sum(item.persons for item in measurements),
            vehicles=sum(item.vehicles for item in measurements),
            avg_confidence=sum(item.avg_confidence for item in measurements) / len(measurements),
            parity=(
                sum(1 for parity in parities if parity) / len(parities) if parities else 1.0
            ),
            inference_ms=sum(item.inference_ms for item in measurements) / len(measurements),
        )

    def _predict(self, image: np.ndarray) -> Tuple[int, int, float, float]:
        if self._predictor is not None:
            return self._predictor(image)
        model = self._model or YOLO(self._weights)
        self._model = model
        result = model.predict(image, verbose=False)[0]
        detections = json.loads(result.to_json())
        persons, vehicles, avg_confidence = frame_counts(detections)
        inference_ms = float((result.speed or {}).get("inference", 0.0))
        return persons, vehicles, avg_confidence, inference_ms

    def _persist(self, payload: Dict[str, Any]) -> None:
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(json.dumps(payload, indent=2))

    @staticmethod
    def _decode(image_bytes: bytes) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImageError("Invalid or unreadable image file")
        return image

    @staticmethod
    def _resize(frame: np.ndarray, scale: float) -> np.ndarray:
        height, width = frame.shape[:2]
        return cv2.resize(
            frame,
            (max(2, round(width * scale)), max(2, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _encode(frame: np.ndarray, quality: int) -> bytes:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise InvalidImageError("JPEG encoding failed")
        return encoded.tobytes()