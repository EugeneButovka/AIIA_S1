import cv2
import numpy as np
import requests

import sender
from sender import (
    CalibrationSettings,
    draw_person_boxes,
    encode_frame,
    env_flag,
    fetch_calibration,
    server_base_url,
)


def build_frame(width=80, height=60, noise=False):
    if noise:
        rng = np.random.default_rng(7)
        return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    return np.zeros((height, width, 3), dtype=np.uint8)


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def build_detection(cls=0, confidence=0.9, x1=10, y1=10, x2=20, y2=20):
    return {
        "class": cls,
        "confidence": confidence,
        "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }


def test_server_base_url_strips_predict_suffix():
    # given
    # when
    base_url = server_base_url("http://vm/predict")

    # then
    assert base_url == "http://vm"


def test_server_base_url_returns_none_without_predict_suffix():
    # given
    # when
    base_url = server_base_url("http://vm/other")

    # then
    assert base_url is None


def test_fetch_calibration_reads_recommendation(monkeypatch):
    # given
    monkeypatch.setattr(
        "sender.requests.get",
        lambda url, timeout: FakeResponse(payload={"recommendation": {"scale": 1.0, "quality": 60}}),
    )

    # when
    calibration = fetch_calibration("http://vm")

    # then
    assert calibration == CalibrationSettings(scale=1.0, quality=60)


def test_fetch_calibration_reads_scaled_recommendation(monkeypatch):
    # given
    monkeypatch.setattr(
        "sender.requests.get",
        lambda url, timeout: FakeResponse(payload={"recommendation": {"scale": 0.5, "quality": 75}}),
    )

    # when
    calibration = fetch_calibration("http://vm")

    # then
    assert calibration == CalibrationSettings(scale=0.5, quality=75)


def test_fetch_calibration_returns_none_when_absent(monkeypatch):
    # given
    monkeypatch.setattr("sender.requests.get", lambda url, timeout: FakeResponse(status_code=404))

    # when
    calibration = fetch_calibration("http://vm")

    # then
    assert calibration is None


def test_fetch_calibration_returns_none_without_recommendation(monkeypatch):
    # given
    monkeypatch.setattr("sender.requests.get", lambda url, timeout: FakeResponse(payload={}))

    # when
    calibration = fetch_calibration("http://vm")

    # then
    assert calibration is None


def test_fetch_calibration_returns_none_for_invalid_values(monkeypatch):
    # given
    payloads = [
        {"recommendation": {"scale": 0, "quality": 60}},
        {"recommendation": {"scale": 1.5, "quality": 60}},
        {"recommendation": {"scale": 0.5, "quality": 0}},
    ]
    for payload in payloads:
        monkeypatch.setattr("sender.requests.get", lambda url, timeout: FakeResponse(payload=payload))

        # when
        calibration = fetch_calibration("http://vm")

        # then
        assert calibration is None


def test_fetch_calibration_returns_none_on_network_error(monkeypatch):
    # given
    def raise_error(url, timeout):
        raise requests.ConnectionError("unreachable")

    monkeypatch.setattr("sender.requests.get", raise_error)

    # when
    calibration = fetch_calibration("http://vm")

    # then
    assert calibration is None


def test_encode_frame_without_calibration_keeps_resolution():
    # given
    frame = build_frame()

    # when
    image_bytes = encode_frame(frame, None)

    # then
    assert image_bytes is not None
    decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (60, 80, 3)


def test_encode_frame_applies_calibration_scale():
    # given
    frame = build_frame()
    calibration = CalibrationSettings(scale=0.5, quality=60)

    # when
    image_bytes = encode_frame(frame, calibration)

    # then
    decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (30, 40, 3)


def test_encode_frame_keeps_resolution_at_full_scale():
    # given
    frame = build_frame()
    calibration = CalibrationSettings(scale=1.0, quality=60)

    # when
    image_bytes = encode_frame(frame, calibration)

    # then
    decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (60, 80, 3)


def test_encode_frame_lower_quality_shrinks_bytes():
    # given
    frame = build_frame(noise=True)

    # when
    heavy = encode_frame(frame, CalibrationSettings(scale=1.0, quality=95))
    light = encode_frame(frame, CalibrationSettings(scale=1.0, quality=10))

    # then
    assert heavy is not None and light is not None
    assert len(light) < len(heavy)


def test_draw_person_boxes_scales_coordinates():
    # given
    frame = build_frame()
    detections = [build_detection()]

    # when
    person_count = draw_person_boxes(frame, detections, box_scale=2.0)

    # then
    assert person_count == 1
    assert frame[30, 20].tolist() == [0, 255, 0]
    assert frame[30, 40].tolist() == [0, 255, 0]
    assert frame[30, 15].tolist() == [0, 0, 0]


def test_draw_person_boxes_skips_low_confidence_and_other_classes():
    # given
    frame = build_frame()
    detections = [build_detection(confidence=0.3), build_detection(cls=2)]

    # when
    person_count = draw_person_boxes(frame, detections, box_scale=1.0)

    # then
    assert person_count == 0
    assert frame[15, 10].tolist() == [0, 0, 0]


def test_env_flag_reads_environment(monkeypatch):
    # given
    monkeypatch.setenv("SENDER_TEST_FLAG", "true")

    # when
    # then
    assert env_flag("SENDER_TEST_FLAG", False) is True
    monkeypatch.setenv("SENDER_TEST_FLAG", "0")
    assert env_flag("SENDER_TEST_FLAG", True) is False


def test_env_flag_falls_back_to_default(monkeypatch):
    # given
    monkeypatch.delenv("SENDER_TEST_FLAG", raising=False)

    # when
    # then
    assert env_flag("SENDER_TEST_FLAG", True) is True
    assert env_flag("SENDER_TEST_FLAG", False) is False