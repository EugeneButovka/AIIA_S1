import pytest

from dataclasses import replace

from helpers import (
    LevelSummary,
    count_in_window,
    estimate_flex_cost,
    identify_limit,
    level_is_saturated,
    parse_levels,
    percentile,
    rounded_mean,
    summarize_level,
    summary_to_row,
)


def make_summary(**overrides):
    base = LevelSummary(
        cameras=4,
        frames_sent=12,
        ok=12,
        errors=0,
        error_rate=0.0,
        latency_p50_ms=100.0,
        latency_p95_ms=200.0,
        latency_max_ms=300.0,
        sent_bytes=600000,
    )
    return replace(base, **overrides)


class TestParseLevels:
    def test_parses_comma_separated_counts(self):
        assert parse_levels("1,2,4,8") == [1, 2, 4, 8]

    def test_ignores_blank_segments_and_spaces(self):
        assert parse_levels(" 1 , 2 ,, ") == [1, 2]

    def test_rejects_zero(self):
        with pytest.raises(ValueError):
            parse_levels("0,1")

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            parse_levels("")


class TestPercentile:
    def test_empty_returns_zero(self):
        assert percentile([], 0.5) == 0.0

    def test_single_value(self):
        assert percentile([42], 0.95) == 42.0

    def test_median_of_even_count_interpolates(self):
        assert percentile([1, 2, 3, 4], 0.5) == 2.5

    def test_p95_of_flat_series(self):
        assert percentile([10] * 20, 0.95) == 10.0


class TestRoundedMean:
    def test_empty_returns_none(self):
        assert rounded_mean([]) is None

    def test_averages_and_rounds(self):
        assert rounded_mean([1, 2, 3], digits=2) == 2.0


class TestSummarizeLevel:
    def test_counts_ok_and_errors(self):
        rows = [dict(status="ok", latency_ms=100, sent_bytes=50000, cameras=4) for _ in range(8)]
        rows += [dict(status="network_error", latency_ms=100, sent_bytes=0, cameras=4) for _ in range(2)]
        summary = summarize_level(rows)
        assert summary.cameras == 4
        assert summary.frames_sent == 10
        assert summary.ok == 8
        assert summary.errors == 2
        assert summary.error_rate == 0.2

    def test_latency_percentiles_exclude_failures(self):
        rows = [
            dict(status="ok", latency_ms=100, sent_bytes=10, cameras=1),
            dict(status="ok", latency_ms=200, sent_bytes=10, cameras=1),
            dict(status="ok", latency_ms=300, sent_bytes=10, cameras=1),
            dict(status="timeout", latency_ms=60000, sent_bytes=10, cameras=1),
        ]
        summary = summarize_level(rows)
        assert summary.latency_max_ms == 300.0
        assert summary.latency_p50_ms == 200.0

    def test_vm_stats_are_averaged(self):
        frames = [dict(status="ok", latency_ms=50, sent_bytes=1, cameras=2)]
        stats = [
            dict(cameras=2, cpu_percent=10, memory_percent=40, bytes_in=1000, bytes_out=500, cost_total=0.1),
            dict(cameras=2, cpu_percent=30, memory_percent=50, bytes_in=3000, bytes_out=1500, cost_total=0.3),
        ]
        summary = summarize_level(frames, stats_rows=stats)
        assert summary.cpu_percent == 20.0
        assert summary.memory_percent == 45.0
        assert summary.bandwidth_in_bytes == 2000
        assert summary.bandwidth_out_bytes == 1000
        assert summary.cost_delta == pytest.approx(0.2)

    def test_cloud_rows_are_averaged(self):
        frames = [dict(status="ok", latency_ms=50, sent_bytes=1, cameras=4)]
        cloud = [
            dict(response_ms="500", inference_ms="400", detections="2", avg_confidence="0.8"),
            dict(response_ms="700", inference_ms="600", detections="4", avg_confidence="0.9"),
        ]
        summary = summarize_level(frames, cloud_rows=cloud)
        assert summary.cloud_rows == 2
        assert summary.cloud_avg_response_ms == 600.0
        assert summary.cloud_avg_inference_ms == 500.0
        assert summary.cloud_detections_per_frame == 3.0
        assert summary.cloud_avg_confidence == pytest.approx(0.85)

    def test_cloud_confidence_skips_zero_detection_rows(self):
        frames = [dict(status="ok", latency_ms=50, sent_bytes=1, cameras=1)]
        cloud = [
            dict(response_ms="500", inference_ms="400", detections="0", avg_confidence=""),
            dict(response_ms="500", inference_ms="400", detections="3", avg_confidence="0.9"),
        ]
        summary = summarize_level(frames, cloud_rows=cloud)
        assert summary.cloud_avg_confidence == 0.9

    def test_missing_optional_values_are_kept_none(self):
        summary = summarize_level([dict(status="ok", latency_ms=10, sent_bytes=1, cameras=1)])
        assert summary.cpu_percent is None
        assert summary.cloud_rows == 0

    def test_single_stats_sample_has_no_bandwidth_delta(self):
        frames = [dict(status="ok", latency_ms=10, sent_bytes=1, cameras=1)]
        stats = [dict(cameras=1, cpu_percent=5, memory_percent=30, bytes_in=10, bytes_out=5, cost_total=0.0)]
        summary = summarize_level(frames, stats_rows=stats)
        assert summary.bandwidth_in_bytes is None
        assert summary.cost_delta is None


class TestLevelIsSaturated:
    def test_healthy_level_is_not_saturated(self):
        assert level_is_saturated(make_summary()) is None

    def test_error_rate_trips(self):
        assert (level_is_saturated(make_summary(errors=1, error_rate=0.0833)) or "").startswith("error rate")

    def test_latency_trips(self):
        assert (level_is_saturated(make_summary(latency_p95_ms=12000.0)) or "").startswith("p95 latency")

    def test_cpu_trips(self):
        assert (level_is_saturated(make_summary(cpu_percent=97.0)) or "").startswith("cpu")

    def test_memory_trips(self):
        assert (level_is_saturated(make_summary(memory_percent=98.0)) or "").startswith("memory")

    def test_none_cpu_is_ignored(self):
        assert level_is_saturated(make_summary(cpu_percent=None)) is None


class TestIdentifyLimit:
    def test_empty_input(self):
        verdict = identify_limit([])
        assert verdict.max_cameras == 0
        assert verdict.reason == "no data"

    def test_returns_last_healthy_level_when_saturated(self):
        verdict = identify_limit(
            [
                make_summary(cameras=1),
                make_summary(cameras=2, error_rate=0.5, errors=6),
            ]
        )
        assert verdict.max_cameras == 1
        assert verdict.saturated_at == 2
        assert "error rate" in verdict.reason

    def test_first_level_saturated_means_zero(self):
        verdict = identify_limit([make_summary(cameras=1, error_rate=0.9, errors=9)])
        assert verdict.max_cameras == 0
        assert verdict.saturated_at == 1

    def test_no_saturation_reports_last_level(self):
        verdict = identify_limit([make_summary(cameras=c) for c in (1, 2, 4)])
        assert verdict.max_cameras == 4
        assert verdict.saturated_at is None

    def test_thresholds_can_be_overridden(self):
        summary = make_summary(cameras=1, errors=1, error_rate=0.01)
        assert identify_limit([summary]).max_cameras == 1
        assert identify_limit([summary], error_rate_threshold=0.005).max_cameras == 0


class TestCountInWindow:
    def test_counts_rows_between_bounds(self):
        timestamps = ["2026-01-01T00:00:01", "2026-01-01T00:00:05", "2026-01-01T00:00:09"]
        assert count_in_window(timestamps, "2026-01-01T00:00:02", "2026-01-01T00:00:08") == 1

    def test_bounds_are_inclusive(self):
        assert count_in_window(["a", "b"], "a", "b") == 2


class TestEstimateFlexCost:
    def test_known_value(self):
        result = estimate_flex_cost(executions=1_000_000, execution_seconds=1000, memory_gb=2.0)
        assert result["execution_cost"] == pytest.approx(0.20)
        assert result["compute_cost"] == pytest.approx(1000 * 2.0 * 0.0000111)
        assert result["total"] == pytest.approx(0.20 + 0.0222)

    def test_zero_executions(self):
        assert estimate_flex_cost(executions=0, execution_seconds=0)["total"] == 0


class TestSummaryToRow:
    def test_none_fields_become_blank(self):
        row = summary_to_row(make_summary())
        assert row["cpu_percent"] == ""
        assert row["cloud_avg_response_ms"] == ""
        assert row["cloud_cost_estimate"] == ""

    def test_cost_passthrough(self):
        row = summary_to_row(make_summary(), cloud_cost_total=0.123)
        assert row["cloud_cost_estimate"] == 0.123

    def test_sent_bytes_become_mb(self):
        row = summary_to_row(make_summary(sent_bytes=2_500_000))
        assert row["sent_mb"] == 2.5