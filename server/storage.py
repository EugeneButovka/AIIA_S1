import csv
import threading
from dataclasses import dataclass
from pathlib import Path

COLUMNS = [
    "timestamp_utc",
    "persons",
    "detections",
    "avg_confidence",
    "inference_ms",
    "response_ms",
    "image_bytes",
    "response_bytes",
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
        ]


class ResultsStore:
    def __init__(self, csv_path: Path):
        self._csv_path = csv_path
        self._lock = threading.Lock()

    def append(self, record: PredictionRecord) -> None:
        with self._lock:
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not self._csv_path.exists()
            with open(self._csv_path, mode="a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                if write_header:
                    writer.writerow(COLUMNS)
                writer.writerow(record.to_row())