import csv
import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import psutil
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from ultralytics import YOLO

PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5
HISTORY_SIZE = 2000
RESOURCE_SAMPLE_INTERVAL_SECONDS = 4
COST_PER_HOUR = float(os.environ.get("COST_PER_HOUR", "0.096"))

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
CSV_FILENAME = DATA_DIR / "results.csv"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()
model = YOLO("yolov8n.pt")

started_at = time.time()
lock = threading.Lock()
history = deque(maxlen=HISTORY_SIZE)
resource_history = deque(maxlen=600)
totals = {"requests": 0, "persons": 0, "detections": 0, "bytes_in": 0, "bytes_out": 0}


def percentile(values, percent):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(percent / 100 * len(ordered)))
    return ordered[index]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def append_result(timestamp, persons, detections, avg_confidence, inference_ms, response_ms, image_bytes, response_bytes):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_FILENAME.exists()
    with open(CSV_FILENAME, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(["timestamp_utc", "persons", "detections", "avg_confidence", "inference_ms", "response_ms", "image_bytes", "response_bytes"])
        writer.writerow([timestamp, persons, detections, round(avg_confidence, 4), round(inference_ms, 1), round(response_ms, 1), image_bytes, response_bytes])
    with lock:
        history.append({
            "timestamp": timestamp,
            "persons": persons,
            "detections": detections,
            "avg_confidence": round(avg_confidence, 4),
            "inference_ms": round(inference_ms, 1),
            "response_ms": round(response_ms, 1),
        })
        totals["requests"] += 1
        totals["persons"] += persons
        totals["detections"] += detections
        totals["bytes_in"] += image_bytes
        totals["bytes_out"] += response_bytes


def sample_resources():
    psutil.cpu_percent(interval=None)
    while True:
        cpu = psutil.cpu_percent(interval=RESOURCE_SAMPLE_INTERVAL_SECONDS)
        memory = psutil.virtual_memory()
        with lock:
            resource_history.append({
                "t": datetime.now(timezone.utc).isoformat(),
                "cpu": cpu,
                "memory": memory.percent,
            })


threading.Thread(target=sample_resources, daemon=True).start()


@app.post("/predict")
async def run_prediction(file: UploadFile = File(...)):
    request_started_at = time.time()
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid or unreadable image file")

    results = model.predict(img)
    parsed_results = json.loads(results[0].to_json())
    speed = results[0].speed or {}
    inference_ms = float(speed.get("inference", 0.0))
    response_ms = (time.time() - request_started_at) * 1000

    persons = sum(
        1
        for detection in parsed_results
        if detection["class"] == PERSON_CLASS_ID and detection["confidence"] >= CONFIDENCE_THRESHOLD
    )
    confidences = [detection["confidence"] for detection in parsed_results]
    avg_confidence = mean(confidences)

    payload = {
        "detections": parsed_results,
        "persons": persons,
        "inference_ms": round(inference_ms, 1),
        "response_ms": round(response_ms, 1),
    }
    response_bytes = len(json.dumps(payload).encode())

    append_result(
        timestamp=datetime.now(timezone.utc).isoformat(),
        persons=persons,
        detections=len(parsed_results),
        avg_confidence=avg_confidence,
        inference_ms=inference_ms,
        response_ms=response_ms,
        image_bytes=len(contents),
        response_bytes=response_bytes,
    )
    return payload


@app.get("/history")
def get_history():
    with lock:
        return list(history)


@app.get("/stats")
def get_stats():
    with lock:
        response_times = [entry["response_ms"] for entry in history]
        inference_times = [entry["inference_ms"] for entry in history]
        latest_resources = resource_history[-1] if resource_history else {"cpu": 0.0, "memory": 0.0}
        uptime = time.time() - started_at
        requests = totals["requests"]
        memory = psutil.virtual_memory()
        return {
            "uptime_seconds": round(uptime, 1),
            "requests": requests,
            "persons_total": totals["persons"],
            "detections_total": totals["detections"],
            "detections_per_frame": round(totals["detections"] / requests, 2) if requests else 0.0,
            "frames_per_second": round(requests / uptime, 3) if uptime else 0.0,
            "avg_response_ms": round(mean(response_times), 1),
            "p95_response_ms": round(percentile(response_times, 95), 1),
            "avg_inference_ms": round(mean(inference_times), 1),
            "bytes_in": totals["bytes_in"],
            "bytes_out": totals["bytes_out"],
            "cpu_percent": latest_resources["cpu"],
            "memory_percent": latest_resources["memory"],
            "memory_used_mb": round(memory.used / 1e6, 1),
            "memory_total_mb": round(memory.total / 1e6, 1),
            "cost_per_hour": COST_PER_HOUR,
            "cost_total": round(COST_PER_HOUR * uptime / 3600, 4),
            "resource_history": list(resource_history),
        }


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")