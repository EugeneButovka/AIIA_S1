import os
from pathlib import Path
from typing import Optional

PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5
HISTORY_SIZE = 2000
RESOURCE_HISTORY_SIZE = 600
RESOURCE_SAMPLE_INTERVAL_SECONDS = 4
HOURS_PER_MONTH = 730
DEFAULT_COST_PER_HOUR = 0.096

IMDS_METADATA_URL = "http://169.254.169.31/metadata/instance?api-version=2021-02-01"
RETAIL_PRICES_URL = "https://prices.azure.com/api/retail/prices"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
CSV_FILENAME = DATA_DIR / "results.csv"
STATIC_DIR = BASE_DIR / "static"


def configured_cost_per_hour() -> Optional[float]:
    value = os.environ.get("COST_PER_HOUR", "").strip()
    return float(value) if value else None