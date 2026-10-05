import json

from analysis import analyze_blob, build_response_payload, to_analysis_record
from detection import DetectionResult, InvalidImageError


def build_detection(class_id, confidence, x=0, y=0):
    return {
        "class": class_id,
        "confidence": confidence,
        "box": {"x1": x, "y1": y, "x2": x + 10, "y2": y + 10},
    }


def build_result():
    detections = [
        build_detection(0, 0.87, 120, 80),
        build_detection(2, 0.99, 10, 10),
    ]
    return DetectionResult(detections=detections, persons=1, inference_ms=29.87, avg_confidence=0.93)


# Azure Vision (Image Analysis 4.0) — the vision-specific tests were removed together
# with the disabled code path (stubs for caption/read/tags/people/objects results).


def test_to_analysis_record_extracts_detection_metrics():
    # given
    result = build_result()

    # when
    record = to_analysis_record(result, "uploads/frame.jpg", b"image-bytes", 180.456)

    # then
    assert record.persons == 1
    assert record.detections == 2
    assert record.avg_confidence == 0.93
    assert record.inference_ms == 29.87
    assert record.response_ms == 180.456
    assert record.image_bytes == len(b"image-bytes")
    assert record.blob_name == "uploads/frame.jpg"
    assert record.caption == ""
    assert record.caption_confidence == 0.0
    assert record.tags == ""
    assert record.ocr_text == ""
    assert record.response_bytes == len(
        json.dumps(build_response_payload(result, 180.456)).encode("utf-8")
    )


def test_to_analysis_record_handles_empty_detections():
    # given
    result = DetectionResult(detections=[], persons=0, inference_ms=0.0, avg_confidence=0.0)

    # when
    record = to_analysis_record(result, "uploads/frame.jpg", b"image-bytes", 20.0)

    # then
    assert record.persons == 0
    assert record.detections == 0
    assert record.avg_confidence == 0.0
    assert record.response_bytes > 0


def test_build_response_payload_mirrors_server_shape():
    # given
    result = build_result()

    # when
    payload = build_response_payload(result, 180.456)

    # then
    assert payload == {
        "detections": result.detections,
        "persons": 1,
        "inference_ms": 29.9,
        "response_ms": 180.5,
    }


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

    def fake_append(service, container_name, csv_name, record):
        appended.append((service, container_name, csv_name, record))

    def fake_upload(service, container_name, blob_name, image_data):
        uploaded.append((service, container_name, blob_name, image_data))

    def fake_draw(image_data, detections):
        drawn.append((image_data, detections))
        return b"processed-bytes"

    monkeypatch.delenv("RESULTS_CONTAINER_NAME", raising=False)
    monkeypatch.delenv("RESULTS_CSV_NAME", raising=False)
    monkeypatch.setattr("analysis.build_blob_service", lambda: "fake-service")
    monkeypatch.setattr("analysis.get_detector", lambda: detector)
    monkeypatch.setattr("analysis.append_analysis_row", fake_append)
    monkeypatch.setattr("analysis.upload_processed_image", fake_upload)
    monkeypatch.setattr("analysis.draw_detection_boxes", fake_draw)
    return appended, uploaded, drawn


def test_analyze_blob_appends_record_and_uploads_processed_image(monkeypatch):
    # given
    detector = FakeDetector(result=build_result())
    appended, uploaded, drawn = install_pipeline(monkeypatch, detector)

    # when
    analyze_blob("uploads/frame.jpg", b"image-bytes")

    # then
    assert detector.calls == [b"image-bytes"]
    assert len(appended) == 1
    service, container_name, csv_name, record = appended[0]
    assert service == "fake-service"
    assert container_name == "results"
    assert csv_name == "analysis.csv"
    assert record.persons == 1
    assert record.detections == 2
    assert record.image_bytes == len(b"image-bytes")
    assert drawn == [(b"image-bytes", build_result().detections)]
    assert uploaded == [("fake-service", "results", "processed_frame.jpg", b"processed-bytes")]


def test_analyze_blob_skips_upload_when_render_fails(monkeypatch):
    # given
    detector = FakeDetector(result=build_result())
    appended, uploaded, _ = install_pipeline(monkeypatch, detector)
    monkeypatch.setattr("analysis.draw_detection_boxes", lambda image_data, detections: None)

    # when
    analyze_blob("uploads/frame.jpg", b"image-bytes")

    # then
    assert len(appended) == 1
    assert uploaded == []


def test_analyze_blob_skips_invalid_image(monkeypatch):
    # given
    detector = FakeDetector(error=InvalidImageError("invalid image"))
    appended, uploaded, drawn = install_pipeline(monkeypatch, detector)

    # when
    analyze_blob("uploads/note.txt", b"not-an-image")

    # then
    assert appended == []
    assert uploaded == []
    assert drawn == []