import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import requests
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContentSettings

PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5
POLL_INTERVAL_SECONDS = 2.0
BLOB_SEND_INTERVAL_SECONDS = 5.0
REQUEST_TIMEOUT_SECONDS = 30

SERVER_MODE = "server"
AZURE_MODE = "azure"
AZURE_BLOB_MODE = "azure-blob"

OUTPUT_DIR = Path(__file__).resolve().parent
DEPLOY_ENV_FILE = OUTPUT_DIR.parent / "deploy.env"

STREAM_URL = os.environ.get("STREAM_URL", "http://47.181.86.62:8082/mjpg/video.mjpg")


def load_deploy_env():
    if not DEPLOY_ENV_FILE.exists():
        return
    for line in DEPLOY_ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


load_deploy_env()

API_URL = os.environ.get("YOLO_API_URL", "http://<YOUR_VM_PUBLIC_IP>/predict")
AZURE_PREDICT_URL = os.environ.get("AZURE_PREDICT_URL", "")
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_UPLOAD_CONTAINER = os.environ.get("AZURE_UPLOAD_CONTAINER", "uploads")
SEND_MODE = os.environ.get("SEND_MODE", SERVER_MODE).lower()


def resolve_predict_url(send_mode, server_url, azure_url):
    if send_mode == AZURE_MODE:
        return azure_url
    return server_url


PREDICT_URL = resolve_predict_url(SEND_MODE, API_URL, AZURE_PREDICT_URL)

CSV_FILENAME = OUTPUT_DIR / "database.csv"
FRAME_FILENAME = OUTPUT_DIR / "processed_frame.jpg"


def predict(frame):
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        return []
    response = requests.post(
        PREDICT_URL,
        files={"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["detections"]


def draw_person_boxes(frame, detections):
    person_count = 0
    for detection in detections:
        if detection["class"] != PERSON_CLASS_ID or detection["confidence"] < CONFIDENCE_THRESHOLD:
            continue
        box = detection["box"]
        top_left = (int(box["x1"]), int(box["y1"]))
        bottom_right = (int(box["x2"]), int(box["y2"]))
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
                detections = predict(frame)
                elapsed = time.time() - started_at
                person_count = draw_person_boxes(frame, detections)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {destination} response: {len(detections)} detections, {person_count} person(s), {elapsed:.2f}s")
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