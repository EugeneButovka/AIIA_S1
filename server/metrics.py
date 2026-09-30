import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List

import psutil

import config
from cost import CostEstimator


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: List[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(percent / 100 * len(ordered)))
    return ordered[index]


class PerformanceTracker:
    def __init__(self, cost_estimator: CostEstimator):
        self._cost = cost_estimator
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._history = deque(maxlen=config.HISTORY_SIZE)
        self._resource_history = deque(maxlen=config.RESOURCE_HISTORY_SIZE)
        self._totals = {"requests": 0, "persons": 0, "detections": 0, "bytes_in": 0, "bytes_out": 0}

    def record_request(self, entry: Dict[str, Any], image_bytes: int, response_bytes: int) -> None:
        with self._lock:
            self._history.append(entry)
            self._totals["requests"] += 1
            self._totals["persons"] += entry["persons"]
            self._totals["detections"] += entry["detections"]
            self._totals["bytes_in"] += image_bytes
            self._totals["bytes_out"] += response_bytes

    def history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            response_times = [entry["response_ms"] for entry in self._history]
            inference_times = [entry["inference_ms"] for entry in self._history]
            confidences = [entry["avg_confidence"] for entry in self._history]
            latest_resources = self._resource_history[-1] if self._resource_history else {"cpu": 0.0, "memory": 0.0}
            totals = dict(self._totals)
            resource_history = list(self._resource_history)

        uptime = time.time() - self._started_at
        requests = totals["requests"]
        memory = psutil.virtual_memory()
        per_hour = self._cost.per_hour
        return {
            "uptime_seconds": round(uptime, 1),
            "requests": requests,
            "persons_total": totals["persons"],
            "detections_total": totals["detections"],
            "detections_per_frame": round(totals["detections"] / requests, 2) if requests else 0.0,
            "frames_per_second": round(requests / uptime, 3) if uptime else 0.0,
            "avg_response_ms": round(mean(response_times), 1),
            "p95_response_ms": round(percentile(response_times, 95), 1),
            "avg_inference_ms": round(mean(inference_times), 1),
            "avg_confidence": round(mean(confidences), 3),
            "bytes_in": totals["bytes_in"],
            "bytes_out": totals["bytes_out"],
            "cpu_percent": latest_resources["cpu"],
            "memory_percent": latest_resources["memory"],
            "memory_used_mb": round(memory.used / 1e6, 1),
            "memory_total_mb": round(memory.total / 1e6, 1),
            "cost_per_hour": per_hour,
            "cost_per_month": round(per_hour * config.HOURS_PER_MONTH, 2),
            "cost_source": self._cost.source,
            "cost_total": round(per_hour * uptime / 3600, 4),
            "resource_history": resource_history,
        }

    def start_background(self) -> None:
        threading.Thread(target=self._sample_resources, daemon=True).start()

    def _sample_resources(self) -> None:
        psutil.cpu_percent(interval=None)
        while True:
            cpu = psutil.cpu_percent(interval=config.RESOURCE_SAMPLE_INTERVAL_SECONDS)
            memory = psutil.virtual_memory()
            with self._lock:
                self._resource_history.append(
                    {
                        "t": datetime.now(timezone.utc).isoformat(),
                        "cpu": cpu,
                        "memory": memory.percent,
                    }
                )