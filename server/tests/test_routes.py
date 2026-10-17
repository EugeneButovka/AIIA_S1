from fastapi import FastAPI
from fastapi.testclient import TestClient

from cost import CostEstimator
from detection import DetectionResult
from metrics import PerformanceTracker
from routes import build_router
from storage import COLUMNS, ResultsStore
from tracking import TrackingResult


class StubLegacyDetector:
    def detect(self, image_bytes):
        return DetectionResult(detections=[], persons=1, inference_ms=1.0, avg_confidence=0.6)


class StubTrackingDetector:
    def detect(self, image_bytes):
        return TrackingResult(
            detections=[],
            persons=1,
            vehicles=2,
            inference_ms=1.0,
            avg_confidence=0.6,
            unique_persons=3,
            unique_vehicles=2,
        )

    def reset_registry(self):
        pass


class StubCalibrator:
    def calibrate(self, images):
        return {"frames": len(images), "recommendation": {"scale": 0.5, "quality": 75}}

    def load(self):
        return {"recommendation": {"scale": 0.5, "quality": 75}}


def make_client(detector, tmp_path):
    tracker = PerformanceTracker(CostEstimator(0.096))
    store = ResultsStore(tmp_path / "results.csv")
    app = FastAPI()
    app.include_router(build_router(detector, store, tracker, StubCalibrator()))
    return TestClient(app)


def test_legacy_payload_is_unchanged(tmp_path):
    client = make_client(StubLegacyDetector(), tmp_path)
    response = client.post(
        "/predict", files={"file": ("frame.jpg", b"bytes", "image/jpeg")}
    )
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"detections", "persons", "inference_ms", "response_ms"}
    assert payload["persons"] == 1


def test_tracking_payload_has_unique_counts(tmp_path):
    client = make_client(StubTrackingDetector(), tmp_path)
    response = client.post(
        "/predict", files={"file": ("frame.jpg", b"bytes", "image/jpeg")}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["vehicles"] == 2
    assert payload["unique_persons"] == 3
    assert payload["unique_vehicles"] == 2


def test_csv_schema_is_extended(tmp_path):
    client = make_client(StubTrackingDetector(), tmp_path)
    client.post("/predict", files={"file": ("frame.jpg", b"bytes", "image/jpeg")})
    lines = (tmp_path / "results.csv").read_text().splitlines()
    assert lines[0].split(",") == COLUMNS
    assert len(lines[1].split(",")) == len(COLUMNS)
    assert lines[1].split(",")[-2:] == ["3", "2"]


def test_reset_rejected_for_legacy_module(tmp_path):
    client = make_client(StubLegacyDetector(), tmp_path)
    response = client.post("/reset")
    assert response.status_code == 400


def test_reset_works_for_tracking_module(tmp_path):
    client = make_client(StubTrackingDetector(), tmp_path)
    response = client.post("/reset")
    assert response.status_code == 200
    assert response.json() == {"unique_persons": 0, "unique_vehicles": 0}


def test_stats_expose_unique_totals_for_tracking(tmp_path):
    client = make_client(StubTrackingDetector(), tmp_path)
    client.post("/predict", files={"file": ("frame.jpg", b"bytes", "image/jpeg")})
    stats = client.get("/stats").json()
    assert stats["unique_persons"] == 3
    assert stats["unique_vehicles"] == 2


def test_stats_have_no_unique_totals_for_legacy(tmp_path):
    client = make_client(StubLegacyDetector(), tmp_path)
    client.post("/predict", files={"file": ("frame.jpg", b"bytes", "image/jpeg")})
    stats = client.get("/stats").json()
    assert "unique_persons" not in stats


def test_calibrate_accepts_multiple_frames(tmp_path):
    client = make_client(StubLegacyDetector(), tmp_path)
    response = client.post(
        "/calibrate",
        files=[
            ("files", ("a.jpg", b"a", "image/jpeg")),
            ("files", ("b.jpg", b"b", "image/jpeg")),
        ],
    )
    assert response.status_code == 200
    assert response.json()["frames"] == 2


def test_calibrate_rejects_too_many_frames(tmp_path):
    client = make_client(StubLegacyDetector(), tmp_path)
    files = [("files", (f"{index}.jpg", b"x", "image/jpeg")) for index in range(9)]
    response = client.post("/calibrate", files=files)
    assert response.status_code == 400


def test_calibration_endpoint_returns_persisted_payload(tmp_path):
    client = make_client(StubLegacyDetector(), tmp_path)
    response = client.get("/calibration")
    assert response.status_code == 200
    assert response.json()["recommendation"] == {"scale": 0.5, "quality": 75}