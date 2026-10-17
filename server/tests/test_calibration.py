import cv2
import numpy as np
import pytest

from calibration import Calibrator, VariantSummary, recommend


def summary(scale, quality, kb, persons, vehicles, conf, parity):
    return VariantSummary(
        scale=scale,
        quality=quality,
        kb_per_frame=kb,
        persons=persons,
        vehicles=vehicles,
        avg_confidence=conf,
        parity=parity,
        inference_ms=10.0,
    )


def test_recommend_picks_smallest_passing_variant():
    baseline = summary(1.0, 95, 300, 6, 8, 0.40, 1.0)
    variants = [
        summary(1.0, 75, 150, 6, 8, 0.41, 1.0),
        summary(0.5, 75, 50, 6, 8, 0.40, 0.9),
        summary(0.5, 60, 40, 5, 8, 0.40, 0.6),
        summary(0.25, 75, 15, 1, 8, 0.36, 0.6),
    ]
    choice, rule = recommend(baseline, variants)
    assert (choice.scale, choice.quality) == (0.5, 75)
    assert "smallest" in rule


def test_recommend_rejects_confidence_drop_and_falls_back():
    baseline = summary(1.0, 95, 300, 6, 8, 0.40, 1.0)
    variants = [
        summary(1.0, 75, 150, 6, 8, 0.41, 1.0),
        summary(0.5, 75, 50, 6, 8, 0.30, 1.0),
    ]
    choice, rule = recommend(baseline, variants)
    assert (choice.scale, choice.quality) == (1.0, 75)


def test_recommend_rejects_count_drift():
    baseline = summary(1.0, 95, 300, 6, 8, 0.40, 1.0)
    variants = [
        summary(1.0, 75, 150, 6, 8, 0.41, 1.0),
        summary(0.5, 75, 50, 9, 8, 0.40, 1.0),
    ]
    choice, _ = recommend(baseline, variants)
    assert (choice.scale, choice.quality) == (1.0, 75)


def test_recommend_falls_back_to_baseline_without_q75():
    baseline = summary(1.0, 95, 300, 6, 8, 0.40, 1.0)
    variants = [summary(0.5, 75, 50, 1, 8, 0.40, 1.0)]
    choice, rule = recommend(baseline, variants)
    assert (choice.scale, choice.quality) == (1.0, 95)
    assert "baseline" in rule


def test_calibrator_end_to_end_with_fake_predictor(tmp_path):
    predictor = lambda image: (1, 1, 0.5, 10.0)
    calibrator = Calibrator(
        weights="unused", predictor=predictor, persist_path=tmp_path / "calibration.json"
    )
    ok, encoded = cv2.imencode(".jpg", np.zeros((120, 160, 3), np.uint8))
    assert ok
    payload = calibrator.calibrate([encoded.tobytes()])

    assert payload["frames"] == 1
    assert len(payload["variants"]) == 19
    assert payload["baseline"]["persons"] == 1
    assert payload["recommendation"]["scale"] == 0.25
    assert payload["recommendation"]["quality"] == 40
    assert payload["bandwidth_vs_baseline_pct"] < 100.0

    loaded = calibrator.load()
    assert loaded is not None
    assert loaded["recommendation"] == payload["recommendation"]


def test_calibrator_rejects_empty_input(tmp_path):
    calibrator = Calibrator(
        weights="unused", predictor=lambda image: (0, 0, 0.0, 1.0), persist_path=tmp_path / "c.json"
    )
    with pytest.raises(ValueError):
        calibrator.calibrate([])