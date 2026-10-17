import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import requests
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContentSettings

PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5
POLL_INTERVAL_SECONDS = 0.5
BLOB_SEND_INTERVAL_SECONDS = 5.0
REQUEST_TIMEOUT_SECONDS = 30

SERVER_MODE = "server"
AZURE_MODE = "azure"
AZURE_BLOB_MODE = "azure-blob"

OUTPUT_DIR = Path(__file__).resolve().parent
DEPLOY_ENV_FILE = OUTPUT_DIR.parent / "deploy.env"

# STREAM_URL = os.environ.get("STREAM_URL", "http://47.181.86.62:8082/mjpg/video.mjpg") # main variant
STREAM_URL = os.environ.get("STREAM_URL", "http://79.3.91.147:9002/mjpg/video.mjpg")


def load_deploy_env():
    if not DEPLOY_ENV_FILE.exists():
        return
    for line in DEPLOY_ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def env_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes"}


load_deploy_env()

API_URL = os.environ.get("YOLO_API_URL", "http://<YOUR_VM_PUBLIC_IP>/predict")
AZURE_PREDICT_URL = os.environ.get("AZURE_PREDICT_URL", "")
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_UPLOAD_CONTAINER = os.environ.get("AZURE_UPLOAD_CONTAINER", "uploads")
SEND_MODE = os.environ.get("SEND_MODE", SERVER_MODE).lower()
CALIBRATION_ENABLED = env_flag("SENDER_CALIBRATION", True)
RESET_ON_START = env_flag("SENDER_RESET_ON_START", False)


def resolve_predict_url(send_mode, server_url, azure_url):
    if send_mode == AZURE_MODE:
        return azure_url
    return server_url


PREDICT_URL = resolve_predict_url(SEND_MODE, API_URL, AZURE_PREDICT_URL)

CSV_FILENAME = OUTPUT_DIR / "database.csv"
FRAME_FILENAME = OUTPUT_DIR / "processed_frame.jpg"


@dataclass(frozen=True)
class CalibrationSettings:
    scale: float
    quality: int


def server_base_url(predict_url):
    if not predict_url.endswith("/predict"):
        return None
    return predict_url[: -len("/predict")]


def fetch_calibration(base_url):
    try:
        response = requests.get(f"{base_url}/calibration", timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    recommendation = response.json().get("recommendation") or {}
    scale = recommendation.get("scale")
    quality = recommendation.get("quality")
    if scale is None or quality is None or not 0 < float(scale) <= 1 or not 1 <= int(quality) <= 100:
        return None
    return CalibrationSettings(float(scale), int(quality))


def reset_server_counts(base_url):
    try:
        response = requests.post(f"{base_url}/reset", timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException:
        return False
    return response.status_code == 200


def encode_frame(frame, calibration):
    if calibration is None:
        success, encoded = cv2.imencode(".jpg", frame)
    else:
        height, width = frame.shape[:2]
        target = (max(2, round(width * calibration.scale)), max(2, round(height * calibration.scale)))
        resized = frame if target == (width, height) else cv2.resize(frame, target, interpolation=cv2.INTER_AREA)
        success, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), calibration.quality])
    if not success:
        return None
    return encoded.tobytes()


def predict(frame, calibration=None):
    image_bytes = encode_frame(frame, calibration)
    if image_bytes is None:
        return {}
    response = requests.post(
        PREDICT_URL,
        files={"file": ("frame.jpg", image_bytes, "image/jpeg")},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def draw_person_boxes(frame, detections, box_scale=1.0):
    person_count = 0
    for detection in detections:
        if detection["class"] != PERSON_CLASS_ID or detection["confidence"] < CONFIDENCE_THRESHOLD:
            continue
        box = detection["box"]
        top_left = (int(box["x1"] * box_scale), int(box["y1"] * box_scale))
        bottom_right = (int(box["x2"] * box_scale), int(box["y2"] * box_scale))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        person_count += 1
    return person_count


def save_person_count(person_count):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_header = not CSV_FILENAME.exists()
    with open(CSV_FILENAME, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(["Timestamp", "Person_Count"])
        writer.writerow([timestamp, person_count])


def build_blob_name(timestamp=None):
    stamp = timestamp or datetime.now()
    return f"frame_{stamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"


def build_uploads_container():
    container = BlobServiceClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    ).get_container_client(AZURE_UPLOAD_CONTAINER)
    if not container.exists():
        try:
            container.create_container()
        except ResourceExistsError:
            pass
    return container


def upload_frame(container, frame):
    try:
        success, encoded = cv2.imencode(".jpg", frame)
    except cv2.error:
        return None
    if not success:
        return None
    blob_name = build_blob_name()
    container.upload_blob(
        name=blob_name,
        data=encoded.tobytes(),
        content_settings=ContentSettings(content_type="image/jpeg"),
    )
    return blob_name


def run_blob_cycle(frame, container):
    blob_name = upload_frame(container, frame)
    if blob_name is None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] failed to encode frame")
        return
    print(f"[{datetime.now().strftime('%H:%M:%S')}] uploaded {blob_name} to '{AZURE_UPLOAD_CONTAINER}' container")


def main():
    if SEND_MODE == AZURE_MODE and not AZURE_PREDICT_URL:
        print("Error: SEND_MODE=azure requires AZURE_PREDICT_URL to be set")
        return
    if SEND_MODE == AZURE_BLOB_MODE and not AZURE_STORAGE_CONNECTION_STRING:
        print("Error: SEND_MODE=azure-blob requires AZURE_STORAGE_CONNECTION_STRING to be set")
        return
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print(f"Error: Could not open the stream: {STREAM_URL}")
        return
    uploads_container = build_uploads_container() if SEND_MODE == AZURE_BLOB_MODE else None
    calibration = None
    if uploads_container is None and SEND_MODE == SERVER_MODE:
        base_url = server_base_url(PREDICT_URL)
        if base_url and CALIBRATION_ENABLED:
            calibration = fetch_calibration(base_url)
            if calibration is not None:
                print(
                    f"calibration applied: scale {calibration.scale}, "
                    f"JPEG quality {calibration.quality} (from {base_url}/calibration)"
                )
            else:
                print("server has no calibration; sending full-resolution frames")
        if base_url and RESET_ON_START:
            if reset_server_counts(base_url):
                print("server counters reset")
            else:
                print("server reset unavailable (legacy module or request failed); counting continues")
    destinations = {
        AZURE_MODE: "azure function",
        AZURE_BLOB_MODE: f"'{AZURE_UPLOAD_CONTAINER}' container",
    }
    destination = destinations.get(SEND_MODE, "server")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            height, width = frame.shape[:2]
            print(f"[{datetime.now().strftime('%H:%M:%S')}] new image {width}x{height} prepared, sending to {destination}")
            if uploads_container is not None:
                run_blob_cycle(frame, uploads_container)
            else:
                started_at = time.time()
                payload = predict(frame, calibration)
                elapsed = time.time() - started_at
                detections = payload.get("detections", [])
                box_scale = 1.0 / calibration.scale if calibration is not None else 1.0
                person_count = draw_person_boxes(frame, detections, box_scale=box_scale)
                unique_note = (
                    f", unique {payload['unique_persons']} person(s), {payload['unique_vehicles']} vehicle(s)"
                    if "unique_persons" in payload
                    else ""
                )
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] {destination} response: "
                    f"{len(detections)} detections, {person_count} person(s){unique_note}, {elapsed:.2f}s"
                )
                save_person_count(person_count)
                cv2.imwrite(str(FRAME_FILENAME), frame)
            interval = (
                BLOB_SEND_INTERVAL_SECONDS
                if uploads_container is not None
                else POLL_INTERVAL_SECONDS
            )
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    main()