import pytest

import config
from detection import Detector
from main import active_weights, select_detector_class
from tracking import TrackingDetector


def test_legacy_module_is_selected_by_default(monkeypatch):
    monkeypatch.setattr(config, "DETECTION_MODULE", config.LEGACY_MODULE)
    assert select_detector_class() is Detector
    assert active_weights() == config.LEGACY_WEIGHTS


def test_tracking_module_is_selected_when_configured(monkeypatch):
    monkeypatch.setattr(config, "DETECTION_MODULE", config.TRACKING_MODULE)
    assert select_detector_class() is TrackingDetector
    assert active_weights() == config.TRACKING_WEIGHTS


def test_unknown_module_fails_fast(monkeypatch):
    monkeypatch.setattr(config, "DETECTION_MODULE", "bogus")
    with pytest.raises(RuntimeError):
        select_detector_class()