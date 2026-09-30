import threading

from fastapi import FastAPI

import config
from cost import CostEstimator
from detection import Detector
from metrics import PerformanceTracker
from routes import build_router
from storage import ResultsStore


def create_app() -> FastAPI:
    cost_estimator = CostEstimator(config.DEFAULT_COST_PER_HOUR)
    tracker = PerformanceTracker(cost_estimator)
    store = ResultsStore(config.CSV_FILENAME)
    detector = Detector()

    app = FastAPI()
    app.include_router(build_router(detector, store, tracker))

    tracker.start_background()
    threading.Thread(target=cost_estimator.resolve, daemon=True).start()
    return app


app = create_app()