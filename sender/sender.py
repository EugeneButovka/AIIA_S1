import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import requests

PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5
POLL_INTERVAL_SECONDS = 0.5
REQUEST_TIMEOUT_SECONDS = 30

STREAM_URL = "http://79.3.91.147:9002/mjpg/video.mjpg"
API_URL = os.environ.get("YOLO_API_URL", "http://<YOUR_VM_PUBLIC_IP>/predict")

OUTPUT_DIR = Path(__file__).resolve().parent
CSV_FILENAME = OUTPUT_DIR / "database.csv"
FRAME_FILENAME = OUTPUT_DIR / "processed_frame.jpg"


def predict(frame):
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        return []
    response = requests.post(
        API_URL,
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


def main():
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print(f"Error: Could not open the stream: {STREAM_URL}")
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            detections = predict(frame)
            person_count = draw_person_boxes(frame, detections)
            save_person_count(person_count)
            cv2.imwrite(str(FRAME_FILENAME), frame)
            time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    main()