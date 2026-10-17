import threading

from fastapi import FastAPI

import config
from calibration import Calibrator
from cost import CostEstimator
from detection import Detector
from metrics import PerformanceTracker
from routes import build_router
from storage import ResultsStore
from tracking import TrackingDetector


def select_detector_class():
    if config.DETECTION_MODULE == config.TRACKING_MODULE:
        return TrackingDetector
    if config.DETECTION_MODULE == config.LEGACY_MODULE:
        return Detector
    raise RuntimeError("DETECTION_MODULE must be 'legacy' or 'tracking'")


def active_weights() -> str:
    if config.DETECTION_MODULE == config.TRACKING_MODULE:
        return config.TRACKING_WEIGHTS
    return config.LEGACY_WEIGHTS


def create_app() -> FastAPI:
    cost_estimator = CostEstimator(config.DEFAULT_COST_PER_HOUR)
    tracker = PerformanceTracker(cost_estimator)
    store = ResultsStore(config.CSV_FILENAME)
    detector = select_detector_class()()
    calibrator = Calibrator(active_weights())

    app = FastAPI()
    app.include_router(build_router(detector, store, tracker, calibrator))

    tracker.start_background()
    threading.Thread(target=cost_estimator.resolve, daemon=True).start()
    return app


app = create_app()