import json

import azure.functions as func
import cv2
import numpy as np

import function_app
from detection import DetectionResult, InvalidImageError


def build_image_bytes(width=40, height=30):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes()


def build_multipart_body(image_data, boundary="----testboundary"):
    head = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="frame.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    return head + image_data + f"\r\n--{boundary}--\r\n".encode()


def build_multipart_request(image_data):
    return func.HttpRequest(
        method="POST",
        url="/api/predict",
        headers={"Content-Type": "multipart/form-data; boundary=----testboundary"},
        body=build_multipart_body(image_data),
    )


def build_raw_request(image_data):
    return func.HttpRequest(
        method="POST",
        url="/api/predict",
        headers={"Content-Type": "application/octet-stream"},
        body=image_data,
    )


def build_result():
    detections = [
        {"class": 0, "name": "person", "confidence": 0.87, "box": {"x1": 10, "y1": 10, "x2": 20, "y2": 40}}
    ]
    return DetectionResult(detections=detections, persons=1, inference_ms=29.87, avg_confidence=0.87)


class FakeDetector:
    def __init__(self, result=None, error=None):
        self.calls = []
        self.result = result
        self.error = error

    def detect(self, image_data):
        self.calls.append(image_data)
        if self.error:
            raise self.error
        return self.result


def install_pipeline(monkeypatch, detector):
    appended = []
    uploaded = []
    drawn = []

    def fake_append(connection_string, container_name, csv_name, record):
        appended.append((connection_string, container_name, csv_name, record))

    def fake_upload(connection_string, container_name, blob_name, image_data):
        uploaded.append((connection_string, container_name, blob_name, image_data))

    def fake_draw(image_data, detections):
        drawn.append((image_data, detections))
        return b"processed-bytes"

    monkeypatch.setenv("AzureWebJobsStorage", "fake-connection-string")
    monkeypatch.delenv("RESULTS_CONTAINER_NAME", raising=False)
    monkeypatch.delenv("RESULTS_CSV_NAME", raising=False)
    monkeypatch.setattr("function_app.get_detector", lambda: detector)
    monkeypatch.setattr("analysis.append_analysis_row", fake_append)
    monkeypatch.setattr("analysis.upload_processed_image", fake_upload)
    monkeypatch.setattr("analysis.draw_detection_boxes", fake_draw)
    return appended, uploaded, drawn


def test_predict_returns_server_format_payload(monkeypatch):
    # given
    image_data = build_image_bytes()
    detector = FakeDetector(result=build_result())
    appended, uploaded, _ = install_pipeline(monkeypatch, detector)

    # when
    response = function_app.predict(build_multipart_request(image_data))

    # then
    assert response.status_code == 200
    payload = json.loads(response.get_body())
    assert payload["detections"] == build_result().detections
    assert payload["persons"] == 1
    assert payload["inference_ms"] == 29.9
    assert payload["response_ms"] >= 0
    assert detector.calls == [image_data]
    assert len(appended) == 1
    record = appended[0][3]
    assert record.blob_name.startswith("predict_")
    assert record.response_bytes == len(response.get_body())
    assert len(uploaded) == 1
    assert uploaded[0][2].startswith("processed_predict_")
    assert uploaded[0][3] == b"processed-bytes"


def test_predict_accepts_raw_image_body(monkeypatch):
    # given
    image_data = build_image_bytes()
    detector = FakeDetector(result=build_result())
    install_pipeline(monkeypatch, detector)

    # when
    response = function_app.predict(build_raw_request(image_data))

    # then
    assert response.status_code == 200
    assert detector.calls == [image_data]
    assert json.loads(response.get_body())["persons"] == 1


def test_predict_rejects_missing_image(monkeypatch):
    # given
    detector = FakeDetector(result=build_result())
    install_pipeline(monkeypatch, detector)

    # when
    response = function_app.predict(build_raw_request(b""))

    # then
    assert response.status_code == 400
    assert detector.calls == []


def test_predict_rejects_invalid_image(monkeypatch):
    # given
    detector = FakeDetector(error=InvalidImageError("invalid image"))
    appended, _, _ = install_pipeline(monkeypatch, detector)

    # when
    response = function_app.predict(build_raw_request(b"not-an-image"))

    # then
    assert response.status_code == 400
    assert detector.calls == [b"not-an-image"]
    assert appended == []