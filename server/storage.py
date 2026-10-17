import csv
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

COLUMNS = [
    "timestamp_utc",
    "persons",
    "detections",
    "avg_confidence",
    "inference_ms",
    "response_ms",
    "image_bytes",
    "response_bytes",
    "unique_persons",
    "unique_vehicles",
]


@dataclass(frozen=True)
class PredictionRecord:
    timestamp: str
    persons: int
    detections: int
    avg_confidence: float
    inference_ms: float
    response_ms: float
    image_bytes: int
    response_bytes: int
    unique_persons: Optional[int] = None
    unique_vehicles: Optional[int] = None

    def to_row(self) -> list:
        return [
            self.timestamp,
            self.persons,
            self.detections,
            round(self.avg_confidence, 4),
            round(self.inference_ms, 1),
            round(self.response_ms, 1),
            self.image_bytes,
            self.response_bytes,
            "" if self.unique_persons is None else self.unique_persons,
            "" if self.unique_vehicles is None else self.unique_vehicles,
        ]


class ResultsStore:
    def __init__(self, csv_path: Path):
        self._csv_path = csv_path
        self._lock = threading.Lock()

    def append(self, record: PredictionRecord) -> None:
        with self._lock:
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_schema_changed()
            write_header = not self._csv_path.exists()
            with open(self._csv_path, mode="a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                if write_header:
                    writer.writerow(COLUMNS)
                writer.writerow(record.to_row())

    def _rotate_if_schema_changed(self) -> None:
        if not self._csv_path.exists():
            return
        with open(self._csv_path, mode="r", newline="") as csv_file:
            header = next(csv.reader(csv_file), [])
        if header == COLUMNS:
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._csv_path.rename(self._csv_path.with_name(f"{self._csv_path.stem}_{stamp}.csv"))