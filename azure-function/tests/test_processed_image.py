import cv2
import numpy as np

from detection import PERSON_CLASS_ID
from processed_image import build_processed_blob_name, draw_detection_boxes


def build_test_image(width=80, height=50):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes()


def decode(image_data):
    return cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)


def build_detection(class_id, confidence, x1, y1, x2, y2):
    return {
        "class": class_id,
        "confidence": confidence,
        "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }


# Azure Vision (Image Analysis 4.0) — the ImageBoundingBox stubs and the
# people/object drawing tests were removed together with the disabled code path.


def test_build_processed_blob_name_uses_stem_and_jpeg_extension():
    assert build_processed_blob_name("uploads/frame.jpg") == "processed_frame.jpg"
    assert build_processed_blob_name("uploads/cam1/snapshot.png") == "processed_snapshot.jpg"
    assert build_processed_blob_name("plain.jpg") == "processed_plain.jpg"


def test_draw_detection_boxes_draws_green_person_boxes():
    # given
    image_data = build_test_image()
    detections = [build_detection(PERSON_CLASS_ID, 0.87, 20, 10, 60, 40)]

    # when
    processed = draw_detection_boxes(image_data, detections)

    # then
    decoded = decode(processed)
    assert decoded is not None
    border = decoded[10, 20]
    assert border[1] > 200
    assert border[0] < 50
    assert border[2] < 50
    center = decoded[25, 40]
    assert center[0] < 50
    assert center[1] < 50
    assert center[2] < 50


def test_draw_detection_boxes_skips_persons_below_threshold():
    # given
    image_data = build_test_image()
    detections = [build_detection(PERSON_CLASS_ID, 0.30, 20, 10, 60, 40)]

    # when
    processed = draw_detection_boxes(image_data, detections)

    # then
    decoded = decode(processed)
    border = decoded[10, 20]
    assert border[0] < 50
    assert border[1] < 50
    assert border[2] < 50


def test_draw_detection_boxes_skips_non_person_classes():
    # given
    image_data = build_test_image()
    detections = [build_detection(2, 0.99, 20, 10, 60, 40)]

    # when
    processed = draw_detection_boxes(image_data, detections)

    # then
    decoded = decode(processed)
    border = decoded[10, 20]
    assert border[0] < 50
    assert border[1] < 50
    assert border[2] < 50


def test_draw_detection_boxes_returns_none_for_invalid_image():
    # given
    # when
    processed = draw_detection_boxes(b"not-an-image", [])

    # then
    assert processed is None