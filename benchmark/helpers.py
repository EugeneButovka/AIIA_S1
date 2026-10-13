import math
from dataclasses import dataclass, replace
from typing import Optional

FLEX_EXECUTIONS_PRICE_PER_MILLION = 0.20
FLEX_GB_SECONDS_PRICE = 0.0000111
DEFAULT_INSTANCE_MEMORY_GB = 2.0

FRAMES_CSV_COLUMNS = (
    "timestamp_utc",
    "level",
    "cameras",
    "camera",
    "latency_ms",
    "status",
    "http_code",
    "detections",
    "persons",
    "inference_ms",
    "sent_bytes",
)
LEVELS_CSV_COLUMNS = ("level", "cameras", "started_at", "ended_at", "frames_sent")
VMSTATS_CSV_COLUMNS = (
    "timestamp_utc",
    "level",
    "cameras",
    "cpu_percent",
    "memory_percent",
    "memory_used_mb",
    "bytes_in",
    "bytes_out",
    "cost_total",
    "uptime_seconds",
    "requests",
)

SUMMARY_CSV_COLUMNS = (
    "cameras",
    "frames_sent",
    "ok",
    "errors",
    "error_rate",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_max_ms",
    "sent_mb",
    "cpu_percent",
    "memory_percent",
    "bandwidth_in_mb",
    "bandwidth_out_mb",
    "cost_delta",
    "cloud_rows",
    "cloud_avg_response_ms",
    "cloud_avg_inference_ms",
    "cloud_detections_per_frame",
    "cloud_avg_confidence",
    "cloud_cost_estimate",
)


@dataclass(frozen=True)
class LevelSummary:
    cameras: int
    frames_sent: int
    ok: int
    errors: int
    error_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_max_ms: float
    sent_bytes: int
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    bandwidth_in_bytes: Optional[float] = None
    bandwidth_out_bytes: Optional[float] = None
    cost_delta: Optional[float] = None
    cloud_rows: int = 0
    cloud_avg_response_ms: Optional[float] = None
    cloud_avg_inference_ms: Optional[float] = None
    cloud_detections_per_frame: Optional[float] = None
    cloud_avg_confidence: Optional[float] = None


@dataclass(frozen=True)
class LimitVerdict:
    max_cameras: int
    saturated_at: Optional[int]
    reason: str


def parse_levels(text):
    levels = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise ValueError("camera counts must be >= 1")
        levels.append(value)
    if not levels:
        raise ValueError("at least one camera count is required")
    return levels


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def rounded_mean(values, digits=1):
    if not values:
        return None
    return round(sum(values) / len(values), digits)


def summarize_level(frames_rows, stats_rows=None, cloud_rows=None):
    if frames_rows:
        cameras = int(frames_rows[0]["cameras"])
    elif stats_rows:
        cameras = int(stats_rows[0]["cameras"])
    else:
        cameras = 0
    ok_rows = [row for row in frames_rows if row["status"] == "ok"]
    errors = len(frames_rows) - len(ok_rows)
    latencies = [float(row["latency_ms"]) for row in ok_rows]
    summary = LevelSummary(
        cameras=cameras,
        frames_sent=len(frames_rows),
        ok=len(ok_rows),
        errors=errors,
        error_rate=round(errors / len(frames_rows), 4) if frames_rows else 0.0,
        latency_p50_ms=round(percentile(latencies, 0.5), 1),
        latency_p95_ms=round(percentile(latencies, 0.95), 1),
        latency_max_ms=round(max(latencies), 1) if latencies else 0.0,
        sent_bytes=sum(int(row["sent_bytes"]) for row in ok_rows),
    )
    if stats_rows:
        cpu_values = [float(row["cpu_percent"]) for row in stats_rows if row.get("cpu_percent") not in (None, "")]
        memory_values = [float(row["memory_percent"]) for row in stats_rows if row.get("memory_percent") not in (None, "")]
        bytes_in = [float(row["bytes_in"]) for row in stats_rows if row.get("bytes_in") not in (None, "")]
        bytes_out = [float(row["bytes_out"]) for row in stats_rows if row.get("bytes_out") not in (None, "")]
        cost_values = [float(row["cost_total"]) for row in stats_rows if row.get("cost_total") not in (None, "")]
        summary = replace(
            summary,
            cpu_percent=rounded_mean(cpu_values),
            memory_percent=rounded_mean(memory_values),
            bandwidth_in_bytes=max(bytes_in) - min(bytes_in) if len(bytes_in) > 1 else None,
            bandwidth_out_bytes=max(bytes_out) - min(bytes_out) if len(bytes_out) > 1 else None,
            cost_delta=round(max(cost_values) - min(cost_values), 6) if len(cost_values) > 1 else None,
        )
    if cloud_rows:
        response_values = [float(row["response_ms"]) for row in cloud_rows if row.get("response_ms") not in (None, "")]
        inference_values = [float(row["inference_ms"]) for row in cloud_rows if row.get("inference_ms") not in (None, "")]
        detection_values = [float(row["detections"]) for row in cloud_rows if row.get("detections") not in (None, "")]
        confidence_values = [float(row["avg_confidence"]) for row in cloud_rows if row.get("detections") not in (None, "", "0")]
        summary = replace(
            summary,
            cloud_rows=len(cloud_rows),
            cloud_avg_response_ms=rounded_mean(response_values),
            cloud_avg_inference_ms=rounded_mean(inference_values),
            cloud_detections_per_frame=rounded_mean(detection_values, 2),
            cloud_avg_confidence=rounded_mean(confidence_values, 4),
        )
    return summary


def level_is_saturated(
    summary,
    error_rate_threshold=0.02,
    latency_p95_threshold_ms=10000.0,
    cpu_percent_threshold=95.0,
    memory_percent_threshold=95.0,
):
    if summary.frames_sent and summary.error_rate > error_rate_threshold:
        return "error rate {:.1%} exceeds {:.1%}".format(summary.error_rate, error_rate_threshold)
    if summary.latency_p95_ms > latency_p95_threshold_ms:
        return "p95 latency {:.0f} ms exceeds {:.0f} ms".format(summary.latency_p95_ms, latency_p95_threshold_ms)
    if summary.cpu_percent is not None and summary.cpu_percent > cpu_percent_threshold:
        return "cpu {:.1f}% exceeds {:.0f}%".format(summary.cpu_percent, cpu_percent_threshold)
    if summary.memory_percent is not None and summary.memory_percent > memory_percent_threshold:
        return "memory {:.1f}% exceeds {:.0f}%".format(summary.memory_percent, memory_percent_threshold)
    return None


def identify_limit(summaries, **thresholds):
    if not summaries:
        return LimitVerdict(max_cameras=0, saturated_at=None, reason="no data")
    max_cameras = 0
    for summary in summaries:
        reason = level_is_saturated(summary, **thresholds)
        if reason is not None:
            return LimitVerdict(max_cameras=max_cameras, saturated_at=summary.cameras, reason=reason)
        max_cameras = summary.cameras
    return LimitVerdict(max_cameras=max_cameras, saturated_at=None, reason="no saturation within the tested range")


def count_in_window(timestamps, start, end):
    return sum(1 for value in timestamps if start <= value <= end)


def estimate_flex_cost(executions, execution_seconds, memory_gb=DEFAULT_INSTANCE_MEMORY_GB):
    execution_cost = executions * FLEX_EXECUTIONS_PRICE_PER_MILLION / 1_000_000
    compute_cost = execution_seconds * memory_gb * FLEX_GB_SECONDS_PRICE
    return {
        "executions": executions,
        "execution_seconds": round(execution_seconds, 3),
        "execution_cost": round(execution_cost, 6),
        "compute_cost": round(compute_cost, 6),
        "total": round(execution_cost + compute_cost, 6),
    }


def summary_to_row(summary, cloud_cost_total=None):
    return {
        "cameras": summary.cameras,
        "frames_sent": summary.frames_sent,
        "ok": summary.ok,
        "errors": summary.errors,
        "error_rate": summary.error_rate,
        "latency_p50_ms": summary.latency_p50_ms,
        "latency_p95_ms": summary.latency_p95_ms,
        "latency_max_ms": summary.latency_max_ms,
        "sent_mb": round(summary.sent_bytes / 1e6, 2),
        "cpu_percent": summary.cpu_percent if summary.cpu_percent is not None else "",
        "memory_percent": summary.memory_percent if summary.memory_percent is not None else "",
        "bandwidth_in_mb": round(summary.bandwidth_in_bytes / 1e6, 2) if summary.bandwidth_in_bytes is not None else "",
        "bandwidth_out_mb": round(summary.bandwidth_out_bytes / 1e6, 2) if summary.bandwidth_out_bytes is not None else "",
        "cost_delta": summary.cost_delta if summary.cost_delta is not None else "",
        "cloud_rows": summary.cloud_rows,
        "cloud_avg_response_ms": summary.cloud_avg_response_ms if summary.cloud_avg_response_ms is not None else "",
        "cloud_avg_inference_ms": summary.cloud_avg_inference_ms if summary.cloud_avg_inference_ms is not None else "",
        "cloud_detections_per_frame": summary.cloud_detections_per_frame if summary.cloud_detections_per_frame is not None else "",
        "cloud_avg_confidence": summary.cloud_avg_confidence if summary.cloud_avg_confidence is not None else "",
        "cloud_cost_estimate": cloud_cost_total if cloud_cost_total is not None else "",
    }