import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

from helpers import (
    SUMMARY_CSV_COLUMNS,
    estimate_flex_cost,
    identify_limit,
    summarize_level,
    summary_to_row,
)


def read_rows(path):
    with open(path, newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def split_by_level(rows):
    grouped = {}
    for row in rows:
        level = row.get("level")
        if level not in (None, ""):
            grouped.setdefault(level, []).append(row)
    return grouped


def in_window(timestamp, level_row):
    return level_row["started_at"] <= timestamp <= level_row["ended_at"]


def rows_for_level(rows, level_row):
    return [row for row in rows if in_window(row.get("timestamp_utc", ""), level_row)]


def response_seconds(rows):
    return sum(
        float(row["response_ms"]) / 1000
        for row in rows
        if row.get("response_ms") not in (None, "")
    )


def print_table(headers, rows):
    widths = [
        max([len(str(header))] + [len(str(row[i])) for row in rows])
        for header, i in zip(headers, range(len(headers)))
    ]
    line = " | ".join(str(header).ljust(width) for header, width in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(width) for i, width in zip(range(len(headers)), widths)))


def parse_timestamp(text):
    return datetime.fromisoformat(text)


def style_axis(ax, title, ylabel, xlabel="simultaneous cameras"):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def ms_axis(ax, values):
    positive = [value for value in values if value > 0]
    if not positive:
        return
    low = min(positive) * 0.8
    high = max(positive) * 1.4
    ticks = []
    for exponent in range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1):
        for mult in (1, 2, 5):
            candidate = mult * 10 ** exponent
            if low <= candidate <= high:
                ticks.append(candidate)
    if not ticks:
        return
    labels = ["{:.0f}".format(t) if t < 1000 else "{:g}k".format(t / 1000) for t in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.yaxis.set_minor_locator(NullLocator())


def camera_axis(ax, xmax, ticks):
    ax.set_xscale("log")
    ax.set_xlim(0.85, xmax)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())


def render_plots(summaries, levels_rows, frames_rows, stats_rows, summary_rows, run_label, out_dir, verdict, thresholds):
    frames_by_level = split_by_level(frames_rows)
    stats_by_level = split_by_level(stats_rows)
    cameras = [summary.cameras for summary in summaries]
    durations = [
        max((parse_timestamp(level_row["ended_at"]) - parse_timestamp(level_row["started_at"])).total_seconds(), 1.0)
        for level_row in levels_rows
    ]
    sent_fps = [summary.frames_sent / duration for summary, duration in zip(summaries, durations)]
    ok_fps = [summary.ok / duration for summary, duration in zip(summaries, durations)]
    server_fps = []
    inference_by_level = []
    for level_row, duration in zip(levels_rows, durations):
        requests = [float(row["requests"]) for row in stats_by_level.get(level_row["level"], []) if row.get("requests") not in (None, "")]
        server_fps.append((max(requests) - min(requests)) / duration if len(requests) > 1 else None)
        values = [float(row["inference_ms"]) for row in frames_by_level.get(level_row["level"], []) if row.get("status") == "ok" and row.get("inference_ms") not in (None, "")]
        inference_by_level.append(values if values else None)
    xmax = max(cameras) * 1.6

    fig, axes = plt.subplots(4, 3, figsize=(20, 17))
    lat, dec, thr = axes[0]
    err, box, res = axes[1]
    band, timeline, infer = axes[2]
    cost, det, cloud = axes[3]
    visible = []

    lat.plot(cameras, [s.latency_p50_ms for s in summaries], marker="o", label="p50")
    lat.plot(cameras, [s.latency_p95_ms for s in summaries], marker="s", label="p95")
    lat.plot(cameras, [s.latency_max_ms for s in summaries], marker="^", label="max")
    lat.axhline(thresholds["latency"], color="red", linestyle="--", linewidth=1, label="limit {:.0f}s".format(thresholds["latency"] / 1000))
    style_axis(lat, "Client response time", "milliseconds (log)")
    lat.set_yscale("log")
    ms_axis(lat, [s.latency_p50_ms for s in summaries] + [s.latency_p95_ms for s in summaries] + [s.latency_max_ms for s in summaries] + [thresholds["latency"]])
    lat.legend()
    camera_axis(lat, xmax, cameras)
    visible.append(lat)

    with_inference = [(s.cameras, values) for s, values in zip(summaries, inference_by_level) if values]
    if with_inference:
        p50_by_cameras = {s.cameras: s.latency_p50_ms for s in summaries}
        inf_avg = [sum(v) / len(v) for _, v in with_inference]
        p50_matched = [p50_by_cameras[c] for c, _ in with_inference]
        dec.plot([c for c, _ in with_inference], p50_matched, marker="o", label="client p50 response")
        dec.plot([c for c, _ in with_inference], inf_avg, marker="s", label="model inference (server-reported)")
        dec.fill_between([c for c, _ in with_inference], inf_avg, p50_matched, alpha=0.25, label="queueing + I/O overhead")
        style_axis(dec, "Latency decomposition (p50)", "milliseconds (log)")
        dec.set_yscale("log")
        ms_axis(dec, p50_matched + inf_avg)
        dec.legend()
        camera_axis(dec, xmax, cameras)
    else:
        dec.set_visible(False)
    visible.append(dec)

    thr.plot(cameras, sent_fps, marker="o", label="sent fps")
    thr.plot(cameras, ok_fps, marker="s", label="completed fps (client)")
    with_server = [(c, f) for c, f in zip(cameras, server_fps) if f is not None]
    if with_server:
        thr.plot([c for c, _ in with_server], [f for _, f in with_server], marker="^", label="completed fps (server)")
    style_axis(thr, "Throughput", "frames per second")
    thr.legend()
    camera_axis(thr, xmax, cameras)
    visible.append(thr)

    err.plot(cameras, [s.error_rate * 100 for s in summaries], marker="o", color="tab:red", label="error rate")
    err.axhline(thresholds["error"] * 100, color="red", linestyle="--", linewidth=1, label="limit {:.1f}%".format(thresholds["error"] * 100))
    style_axis(err, "Error rate", "percent of frames")
    err.legend()
    camera_axis(err, xmax, cameras)
    visible.append(err)

    box_data = []
    box_positions = []
    box_labels = []
    for level_row in levels_rows:
        values = [float(row["latency_ms"]) for row in frames_by_level.get(level_row["level"], []) if row.get("status") == "ok"]
        if values:
            box_data.append(values)
            box_positions.append(int(level_row["cameras"]))
            box_labels.append(str(level_row["cameras"]))
    if box_data:
        box.boxplot(box_data, positions=box_positions, showfliers=False)
        box.set_xticks(box_positions)
        box.set_xticklabels(box_labels)
        style_axis(box, "Latency distribution per level", "milliseconds (log)")
        box.set_yscale("log")
        ms_axis(box, [min(values) for values in box_data] + [max(values) for values in box_data])
        camera_axis(box, xmax, box_positions)
        visible.append(box)
    else:
        box.set_visible(False)

    with_cpu = [summary for summary in summaries if summary.cpu_percent is not None]
    if with_cpu:
        res.plot([s.cameras for s in with_cpu], [s.cpu_percent for s in with_cpu], marker="o", label="cpu %")
        res.plot([s.cameras for s in with_cpu], [s.memory_percent for s in with_cpu], marker="s", label="memory %")
        res.axhline(thresholds["cpu"], color="red", linestyle="--", linewidth=1, label="limit {:.0f}%".format(thresholds["cpu"]))
        style_axis(res, "VM resources", "percent")
        res.legend()
        camera_axis(res, xmax, cameras)
        visible.append(res)
    else:
        res.set_visible(False)

    with_bandwidth = [(summary, duration) for summary, duration in zip(summaries, durations) if summary.bandwidth_in_bytes is not None]
    if with_bandwidth:
        band.plot([s.cameras for s, _ in with_bandwidth], [s.bandwidth_in_bytes / d / 1e6 for s, d in with_bandwidth], marker="o", label="in MB/s")
        band.plot([s.cameras for s, _ in with_bandwidth], [s.bandwidth_out_bytes / d / 1e6 for s, d in with_bandwidth], marker="s", label="out MB/s")
        style_axis(band, "VM bandwidth", "megabytes per second")
        band.legend()
        camera_axis(band, xmax, cameras)
        visible.append(band)
    else:
        band.set_visible(False)

    if levels_rows:
        run_start = parse_timestamp(levels_rows[0]["started_at"])
        run_end = parse_timestamp(levels_rows[-1]["ended_at"])
        total = max((run_end - run_start).total_seconds(), 1.0)
        bucket = 10.0
        n_buckets = int(total // bucket) + 1
        sent_counts = [0] * n_buckets
        ok_counts = [0] * n_buckets
        for row in frames_rows:
            index = min(int((parse_timestamp(row["timestamp_utc"]) - run_start).total_seconds() // bucket), n_buckets - 1)
            sent_counts[index] += 1
            if row["status"] == "ok":
                ok_counts[index] += 1
        bucket_centers = [i * bucket + bucket / 2 for i in range(n_buckets)]
        timeline.plot(bucket_centers, [c / bucket for c in sent_counts], label="sent fps")
        timeline.plot(bucket_centers, [c / bucket for c in ok_counts], label="completed fps")
        for index, level_row in enumerate(levels_rows):
            started = (parse_timestamp(level_row["started_at"]) - run_start).total_seconds()
            ended = (parse_timestamp(level_row["ended_at"]) - run_start).total_seconds()
            timeline.axvspan(started, ended, color="gray", alpha=0.15 if index % 2 == 0 else 0.05)
        cpu_points = [
            ((parse_timestamp(row["timestamp_utc"]) - run_start).total_seconds(), float(row["cpu_percent"]))
            for row in stats_rows
            if row.get("cpu_percent") not in (None, "")
        ]
        if cpu_points:
            cpu_axis = timeline.twinx()
            cpu_axis.plot([t for t, _ in cpu_points], [c for _, c in cpu_points], color="tab:red", alpha=0.6, label="cpu %")
            cpu_axis.set_ylabel("cpu %", color="tab:red")
            cpu_axis.legend(loc="upper left")
        style_axis(timeline, "Load timeline (gray bands = levels)", "frames per second", xlabel="seconds since run start")
        visible.append(timeline)
    else:
        timeline.set_visible(False)

    if any(values for values in inference_by_level if values):
        infer.plot([c for c, v in zip(cameras, inference_by_level) if v], [sum(v) / len(v) for v in inference_by_level if v], marker="o", label="avg inference ms")
        infer.plot([c for c, v in zip(cameras, inference_by_level) if v], [max(v) for v in inference_by_level if v], marker="^", label="max inference ms")
        style_axis(infer, "Model inference time (server-reported)", "milliseconds")
        infer.legend()
        camera_axis(infer, xmax, cameras)
        visible.append(infer)
    else:
        infer.set_visible(False)

    vm_costs = [(summary.cameras, summary.cost_delta) for summary in summaries if summary.cost_delta is not None]
    cloud_costs = [(row["cameras"], float(row["cloud_cost_estimate"])) for row in summary_rows if row.get("cloud_cost_estimate") not in (None, "")]
    if vm_costs or cloud_costs:
        if vm_costs:
            cost.plot([c for c, _ in vm_costs], [v for _, v in vm_costs], marker="o", label="VM cost $")
            cost.text(0.03, 0.92, "VM: linear in uptime, load-independent\nexcludes disk, IP, egress; B-series burst\ncredits throttle sustained 24/7 load", transform=cost.transAxes, fontsize=8, color="dimgray", va="top")
        if cloud_costs:
            cost.plot([c for c, _ in cloud_costs], [v for _, v in cloud_costs], marker="s", label="function cost $")
            cost.text(0.03, 0.5, "function: billed per execution + GB-s\nscales with frames, not uptime", transform=cost.transAxes, fontsize=8, color="dimgray", va="top")
        style_axis(cost, "Cost per level (compute-time only)", "US dollars")
        cost.legend(loc="lower right")
        camera_axis(cost, xmax, cameras)
        visible.append(cost)
    else:
        cost.set_visible(False)

    det_data = []
    for level_row in levels_rows:
        rows = frames_by_level.get(level_row["level"], [])
        detections = [float(row["detections"]) for row in rows if row.get("status") == "ok" and row.get("detections") not in (None, "")]
        persons = [float(row["persons"]) for row in rows if row.get("status") == "ok" and row.get("persons") not in (None, "")]
        det_data.append((int(level_row["cameras"]), sum(detections) / len(detections), sum(persons) / len(persons)) if detections else None)
    with_det = [entry for entry in det_data if entry]
    if with_det:
        det.plot([c for c, _, _ in with_det], [d for _, d, _ in with_det], marker="o", label="detections per frame")
        det.plot([c for c, _, _ in with_det], [p for _, _, p in with_det], marker="s", label="persons per frame")
        style_axis(det, "Detection efficiency", "average per frame")
        det.legend()
        camera_axis(det, xmax, cameras)
        visible.append(det)
    else:
        det.set_visible(False)

    with_cloud = [summary for summary in summaries if summary.cloud_rows]
    if with_cloud:
        cloud.plot([s.cameras for s in with_cloud], [s.cloud_avg_response_ms for s in with_cloud], marker="o", label="function response ms")
        cloud.plot([s.cameras for s in with_cloud], [s.cloud_avg_inference_ms for s in with_cloud], marker="s", label="model inference ms")
        style_axis(cloud, "Azure Function timings", "milliseconds")
        cloud.legend()
        camera_axis(cloud, xmax, cameras)
        visible.append(cloud)
    else:
        cloud.set_visible(False)

    if verdict.saturated_at is not None:
        for ax in visible:
            if ax is not timeline:
                ax.axvline(verdict.saturated_at, color="red", linestyle=":", alpha=0.6)

    fig.suptitle("{} — processing limit: {} cameras ({})".format(run_label, verdict.max_cameras, verdict.reason), fontsize=15)
    fig.tight_layout()
    plot_path = out_dir / "summary.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def main():
    parser = argparse.ArgumentParser(description="aggregate a loadgen run into a per-level summary and the processing limit")
    parser.add_argument("run_dir", help="directory produced by loadgen.py (frames.csv, levels.csv, vmstats.csv)")
    parser.add_argument("--cloud-csv", dest="cloud_csv", default="", help="function results analysis.csv to attribute cloud rows per level")
    parser.add_argument("--latency-threshold-ms", dest="latency_threshold_ms", type=float, default=10000.0)
    parser.add_argument("--error-threshold", dest="error_threshold", type=float, default=0.02)
    parser.add_argument("--cpu-threshold", dest="cpu_threshold", type=float, default=95.0)
    parser.add_argument("--memory-threshold", dest="memory_threshold", type=float, default=95.0)
    parser.add_argument("--no-plot", dest="no_plot", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    frames_rows = read_rows(run_dir / "frames.csv")
    levels_rows = read_rows(run_dir / "levels.csv")
    vmstats_path = run_dir / "vmstats.csv"
    stats_rows = read_rows(vmstats_path) if vmstats_path.exists() else []
    cloud_rows = read_rows(args.cloud_csv) if args.cloud_csv and Path(args.cloud_csv).exists() else []

    frames_by_level = split_by_level(frames_rows)
    stats_by_level = split_by_level(stats_rows)
    summaries = []
    level_cloud_rows = []
    for level_row in levels_rows:
        cloud_window = rows_for_level(cloud_rows, level_row)
        level_cloud_rows.append(cloud_window)
        summaries.append(
            summarize_level(
                frames_by_level.get(level_row["level"], []),
                stats_rows=stats_by_level.get(level_row["level"]),
                cloud_rows=cloud_window,
            )
        )

    verdict = identify_limit(
        summaries,
        error_rate_threshold=args.error_threshold,
        latency_p95_threshold_ms=args.latency_threshold_ms,
        cpu_percent_threshold=args.cpu_threshold,
        memory_percent_threshold=args.memory_threshold,
    )

    summary_rows = []
    for summary, cloud_window in zip(summaries, level_cloud_rows):
        level_cost = estimate_flex_cost(len(cloud_window), response_seconds(cloud_window)) if cloud_window else None
        summary_rows.append(summary_to_row(summary, cloud_cost_total=level_cost["total"] if level_cost else ""))

    headers = [
        "cameras", "sent", "ok", "err%", "p50 ms", "p95 ms", "max ms",
        "cpu %", "mem %", "cost $", "cloud rows", "cloud resp ms", "cloud det/frame",
    ]
    table_rows = []
    for summary, row in zip(summaries, summary_rows):
        table_rows.append([
            summary.cameras,
            summary.frames_sent,
            summary.ok,
            "{:.1%}".format(summary.error_rate),
            summary.latency_p50_ms,
            summary.latency_p95_ms,
            summary.latency_max_ms,
            row["cpu_percent"],
            row["memory_percent"],
            row["cost_delta"],
            summary.cloud_rows,
            row["cloud_avg_response_ms"],
            row["cloud_detections_per_frame"],
        ])
    print_table(headers, table_rows)
    print()
    print("processing limit: {} cameras".format(verdict.max_cameras))
    if verdict.saturated_at is not None:
        print("saturation detected at {} cameras: {}".format(verdict.saturated_at, verdict.reason))
    else:
        print(verdict.reason)
    if cloud_rows:
        total_cost = estimate_flex_cost(len(cloud_rows), response_seconds(cloud_rows))
        print(
            "cloud cost estimate: ${total:.4f} total (${execution_cost:.4f} for {executions} executions + ${compute_cost:.4f} for {execution_seconds}s of 2 GB compute)".format(**total_cost)
        )
        print("pricing assumptions: $0.20 / 1M executions, $0.0000111 / GB-s — adjust constants in helpers.py if pricing changed")

    summary_path = run_dir / "summary.csv"
    with open(summary_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    print("summary written: {}".format(summary_path))
    if not args.no_plot:
        thresholds = {
            "latency": args.latency_threshold_ms,
            "error": args.error_threshold,
            "cpu": args.cpu_threshold,
        }
        plot_path = render_plots(summaries, levels_rows, frames_rows, stats_rows, summary_rows, run_dir.name, run_dir, verdict, thresholds)
        print("chart written: {}".format(plot_path))


if __name__ == "__main__":
    main()