import os
from pathlib import Path
from typing import Optional

PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = (2, 3, 5, 7)
CONFIDENCE_THRESHOLD = 0.5
HISTORY_SIZE = 2000
RESOURCE_HISTORY_SIZE = 600
RESOURCE_SAMPLE_INTERVAL_SECONDS = 4
HOURS_PER_MONTH = 730
DEFAULT_COST_PER_HOUR = 0.096

LEGACY_MODULE = "legacy"
TRACKING_MODULE = "tracking"
DETECTION_MODULE = os.environ.get("DETECTION_MODULE", LEGACY_MODULE).strip().lower()
LEGACY_WEIGHTS = "yolov8n.pt"
TRACKING_WEIGHTS = os.environ.get("TRACKING_WEIGHTS", "yolo11n.pt")
UNIQUE_COUNT_MIN_CONFIRMED_FRAMES = int(os.environ.get("UNIQUE_COUNT_MIN_CONFIRMED_FRAMES", "2"))

CALIBRATION_RESOLUTION_SCALES = (1.0, 0.75, 0.5, 0.375, 0.25)
CALIBRATION_JPEG_QUALITIES = (95, 75, 60, 40)
CALIBRATION_MIN_PARITY = float(os.environ.get("CALIBRATION_MIN_PARITY", "0.75"))
CALIBRATION_CONFIDENCE_DROP_LIMIT = float(os.environ.get("CALIBRATION_CONFIDENCE_DROP_LIMIT", "0.10"))
CALIBRATION_COUNT_TOLERANCE = 1
CALIBRATION_MAX_FRAMES = 8
CALIBRATION_FALLBACK_SCALE = 1.0
CALIBRATION_FALLBACK_QUALITY = 75

IMDS_METADATA_URL = "http://169.254.169.31/metadata/instance?api-version=2021-02-01"
RETAIL_PRICES_URL = "https://prices.azure.com/api/retail/prices"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
CSV_FILENAME = DATA_DIR / "results.csv"
STATIC_DIR = BASE_DIR / "static"
TRACKER_CONFIG = BASE_DIR / "trackers" / "bytetrack.yaml"
CALIBRATION_FILENAME = DATA_DIR / "calibration.json"


def configured_cost_per_hour() -> Optional[float]:
    value = os.environ.get("COST_PER_HOUR", "").strip()
    return float(value) if value else None