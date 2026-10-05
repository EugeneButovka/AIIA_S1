from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from detection import CONFIDENCE_THRESHOLD, PERSON_CLASS_ID

PEOPLE_BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2


def build_processed_blob_name(blob_name: str) -> str:
    return f"processed_{Path(blob_name).stem}.jpg"


# Azure Vision (Image Analysis 4.0) drawing — disabled in favour of the local YOLO model:
#
# OBJECT_BOX_COLOR = (0, 0, 255)
#
#
# def draw_box(image, box, color) -> None:
#     top_left = (box.x, box.y)
#     bottom_right = (box.x + box.width, box.y + box.height)
#     cv2.rectangle(image, top_left, bottom_right, color, BOX_THICKNESS)
#
#
# def draw_detection_boxes(image_data: bytes, people_boxes, object_boxes) -> Optional[bytes]:
#     image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
#     if image is None:
#         return None
#     for box in people_boxes:
#         draw_box(image, box, PEOPLE_BOX_COLOR)
#     for box in object_boxes:
#         draw_box(image, box, OBJECT_BOX_COLOR)
#     success, encoded = cv2.imencode(".jpg", image)
#     return encoded.tobytes() if success else None


def draw_detection_boxes(image_data: bytes, detections) -> Optional[bytes]:
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    for detection in detections:
        if detection["class"] != PERSON_CLASS_ID or detection["confidence"] < CONFIDENCE_THRESHOLD:
            continue
        box = detection["box"]
        top_left = (int(box["x1"]), int(box["y1"]))
        bottom_right = (int(box["x2"]), int(box["y2"]))
        cv2.rectangle(image, top_left, bottom_right, PEOPLE_BOX_COLOR, BOX_THICKNESS)
    success, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes() if success else None