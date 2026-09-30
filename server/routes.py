import json
import time
from datetime import datetime, timezone
from typing import Dict

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

import config
from detection import Detector, InvalidImageError
from metrics import PerformanceTracker
from storage import PredictionRecord, ResultsStore


def build_router(detector: Detector, store: ResultsStore, tracker: PerformanceTracker) -> APIRouter:
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
        tracker.record_request(entry, image_bytes=len(contents), response_bytes=response_bytes)

        return payload

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