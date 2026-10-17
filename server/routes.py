import json
import time
from datetime import datetime, timezone
from typing import Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

import config
from calibration import Calibrator
from detection import Detector, InvalidImageError
from metrics import PerformanceTracker
from storage import PredictionRecord, ResultsStore
from tracking import TrackingResult


def build_router(
    detector: Detector,
    store: ResultsStore,
    tracker: PerformanceTracker,
    calibrator: Calibrator,
) -> APIRouter:
    router = APIRouter()

    @router.post("/predict")
    async def predict(file: UploadFile = File(...)):
        started_at = time.time()
        contents = await file.read()

        try:
            result = detector.detect(contents)
        except InvalidImageError as error:
            raise HTTPException(status_code=400, detail=str(error))

        response_ms = (time.time() - started_at) * 1000
        timestamp = datetime.now(timezone.utc).isoformat()

        payload = {
            "detections": result.detections,
            "persons": result.persons,
            "inference_ms": round(result.inference_ms, 1),
            "response_ms": round(response_ms, 1),
        }
        if isinstance(result, TrackingResult):
            payload["vehicles"] = result.vehicles
            payload["unique_persons"] = result.unique_persons
            payload["unique_vehicles"] = result.unique_vehicles

        response_bytes = len(json.dumps(payload).encode())

        record = PredictionRecord(
            timestamp=timestamp,
            persons=result.persons,
            detections=len(result.detections),
            avg_confidence=result.avg_confidence,
            inference_ms=result.inference_ms,
            response_ms=response_ms,
            image_bytes=len(contents),
            response_bytes=response_bytes,
            unique_persons=result.unique_persons if isinstance(result, TrackingResult) else None,
            unique_vehicles=result.unique_vehicles if isinstance(result, TrackingResult) else None,
        )
        store.append(record)

        entry: Dict = {
            "timestamp": timestamp,
            "persons": result.persons,
            "detections": len(result.detections),
            "avg_confidence": round(result.avg_confidence, 4),
            "inference_ms": round(result.inference_ms, 1),
            "response_ms": round(response_ms, 1),
        }
        if isinstance(result, TrackingResult):
            entry["vehicles"] = result.vehicles
            entry["unique_persons"] = result.unique_persons
            entry["unique_vehicles"] = result.unique_vehicles
        tracker.record_request(entry, image_bytes=len(contents), response_bytes=response_bytes)

        return payload

    @router.post("/calibrate")
    async def calibrate(files: List[UploadFile] = File(...)):
        if len(files) > config.CALIBRATION_MAX_FRAMES:
            raise HTTPException(
                status_code=400,
                detail=f"At most {config.CALIBRATION_MAX_FRAMES} frames per calibration run",
            )
        images = [await file.read() for file in files]
        try:
            return calibrator.calibrate(images)
        except InvalidImageError as error:
            raise HTTPException(status_code=400, detail=str(error))

    @router.get("/calibration")
    def calibration():
        payload = calibrator.load()
        if payload is None:
            raise HTTPException(status_code=404, detail="No calibration has been run yet")
        return payload

    @router.post("/reset")
    def reset():
        if not hasattr(detector, "reset_registry"):
            raise HTTPException(
                status_code=400,
                detail="Unique counting is not active: DETECTION_MODULE must be 'tracking'",
            )
        detector.reset_registry()
        return {"unique_persons": 0, "unique_vehicles": 0}

    @router.get("/history")
    def history():
        return tracker.history()

    @router.get("/stats")
    def stats():
        return tracker.stats()

    @router.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(config.STATIC_DIR / "index.html")

    return router