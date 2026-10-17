import argparse
import csv
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import requests
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceExistsError

from helpers import FRAMES_CSV_COLUMNS, LEVELS_CSV_COLUMNS, VMSTATS_CSV_COLUMNS, parse_levels

OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
DEPLOY_ENV_FILE = Path(__file__).resolve().parent.parent / "deploy.env"
REQUEST_TIMEOUT_SECONDS = 30


def load_deploy_env():
    if not DEPLOY_ENV_FILE.exists():
        return
    for line in DEPLOY_ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class FrameHub:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._produce, daemon=True).start()

    def stop(self):
        self._stop.set()

    def _produce(self):
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.stream_url)
            if not cap.isOpened():
                time.sleep(2)
                continue
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                with self._lock:
                    self._frame = frame
            cap.release()
            if not self._stop.is_set():
                time.sleep(2)

    def grab(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()


class CameraWorker(threading.Thread):
    def __init__(self, camera_id, level, cameras, interval, offset, hub, send, rows, stop_event):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.level = level
        self.cameras = cameras
        self.interval = interval
        self.offset = offset
        self.hub = hub
        self.send = send
        self.rows = rows
        self.stop_event = stop_event

    def run(self):
        started = time.monotonic() + self.offset
        tick = 0
        while not self.stop_event.is_set():
            target = started + tick * self.interval
            delay = target - time.monotonic()
            if delay > 0 and self.stop_event.wait(delay):
                break
            frame = self.hub.grab()
            if frame is not None:
                row = self.send(self.camera_id, frame)
                row["timestamp_utc"] = utc_now()
                row["level"] = self.level
                row["cameras"] = self.cameras
                row["camera"] = self.camera_id
                self.rows.append(row)
            tick += 1


class StatsPoller(threading.Thread):
    def __init__(self, stats_url, every_seconds, level_holder, rows, stop_event):
        super().__init__(daemon=True)
        self.stats_url = stats_url
        self.every_seconds = every_seconds
        self.level_holder = level_holder
        self.rows = rows
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            if self.stop_event.wait(self.every_seconds):
                break
            try:
                response = requests.get(self.stats_url, timeout=REQUEST_TIMEOUT_SECONDS)
                response.raise_for_status()
                stats = response.json()
            except (requests.RequestException, ValueError):
                continue
            level, cameras = self.level_holder()
            self.rows.append(
                {
                    "timestamp_utc": utc_now(),
                    "level": level,
                    "cameras": cameras,
                    "cpu_percent": stats.get("cpu_percent", ""),
                    "memory_percent": stats.get("memory_percent", ""),
                    "memory_used_mb": stats.get("memory_used_mb", ""),
                    "bytes_in": stats.get("bytes_in", ""),
                    "bytes_out": stats.get("bytes_out", ""),
                    "cost_total": stats.get("cost_total", ""),
                    "uptime_seconds": stats.get("uptime_seconds", ""),
                    "requests": stats.get("requests", ""),
                }
            )


def encode_frame(frame):
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        return None
    return encoded.tobytes()


def build_http_sender(url):
    def send(camera_id, frame):
        data = encode_frame(frame)
        if data is None:
            return {"latency_ms": 0.0, "status": "encode_failed", "http_code": "", "detections": "", "persons": "", "inference_ms": "", "sent_bytes": 0}
        started = time.perf_counter()
        try:
            response = requests.post(url, files={"file": ("frame.jpg", data, "image/jpeg")}, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException:
            return {"latency_ms": (time.perf_counter() - started) * 1000, "status": "network_error", "http_code": "", "detections": "", "persons": "", "inference_ms": "", "sent_bytes": len(data)}
        latency_ms = (time.perf_counter() - started) * 1000
        payload = {}
        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError:
                payload = {}
        detections = payload.get("detections", [])
        return {
            "latency_ms": latency_ms,
            "status": "ok" if response.status_code == 200 else "http_{}".format(response.status_code),
            "http_code": response.status_code,
            "detections": len(detections) if isinstance(detections, list) else detections,
            "persons": payload.get("persons", ""),
            "inference_ms": payload.get("inference_ms", ""),
            "sent_bytes": len(data),
        }

    return send


def build_blob_sender(connection_string, container_name, run_tag):
    service = BlobServiceClient.from_connection_string(connection_string)
    container = service.get_container_client(container_name)
    if not container.exists():
        try:
            container.create_container()
        except ResourceExistsError:
            pass

    def send(camera_id, frame):
        data = encode_frame(frame)
        if data is None:
            return {"latency_ms": 0.0, "status": "encode_failed", "http_code": "", "detections": "", "persons": "", "inference_ms": "", "sent_bytes": 0}
        blob_name = "bench_{}_{}_{}.jpg".format(run_tag, camera_id, datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        started = time.perf_counter()
        try:
            container.upload_blob(
                name=blob_name,
                data=data,
                content_settings=ContentSettings(content_type="image/jpeg"),
            )
        except Exception as error:
            return {"latency_ms": (time.perf_counter() - started) * 1000, "status": type(error).__name__, "http_code": "", "detections": "", "persons": "", "inference_ms": "", "sent_bytes": len(data)}
        return {"latency_ms": (time.perf_counter() - started) * 1000, "status": "ok", "http_code": "", "detections": "", "persons": "", "inference_ms": "", "sent_bytes": len(data)}

    return send


def build_dry_sender():
    def send(camera_id, frame):
        data = encode_frame(frame)
        if data is None:
            return {"latency_ms": 0.0, "status": "encode_failed", "http_code": "", "detections": "", "persons": "", "inference_ms": "", "sent_bytes": 0}
        return {"latency_ms": 0.0, "status": "ok", "http_code": "", "detections": "", "persons": "", "inference_ms": "", "sent_bytes": len(data)}

    return send


def write_csv(path, columns, rows):
    with open(path, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_benchmark(args):
    levels = parse_levels(args.cameras)
    run_tag = "{}_{}".format(args.target, datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir = OUTPUT_ROOT / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    hub = FrameHub(args.stream)
    hub.start()
    print("[{}] run {} -> {}".format(utc_now(), args.target, out_dir))
    print("[{}] waiting for the first frame from the stream".format(utc_now()))
    for _ in range(60):
        if hub.grab() is not None:
            break
        time.sleep(0.5)
    else:
        hub.stop()
        raise SystemExit("camera stream unreachable: {}".format(args.stream))

    if args.target == "vm":
        sender = build_http_sender(args.vm_url)
    elif args.target == "azure":
        sender = build_http_sender(args.azure_url)
    elif args.target == "azure-blob":
        sender = build_blob_sender(args.connection_string, args.container, run_tag)
    else:
        sender = build_dry_sender()

    for _ in range(args.warmup):
        frame = hub.grab()
        if frame is not None:
            sender("warmup", frame)
        time.sleep(1)
    if args.warmup:
        print("[{}] warm-up: {} frame(s) sent".format(utc_now(), args.warmup))

    frames_rows = []
    stats_rows = []
    levels_rows = []
    interrupted = threading.Event()
    level_holder_value = [0, 0]

    def level_holder():
        return level_holder_value[0], level_holder_value[1]

    poller = None
    if args.target == "vm" and args.stats_url:
        poller = StatsPoller(args.stats_url, args.stats_every, level_holder, stats_rows, threading.Event())
        poller.start()

    try:
        for level, cameras in enumerate(levels, start=1):
            level_holder_value[0] = level
            level_holder_value[1] = cameras
            started_at = utc_now()
            print("[{}] level {}/{}: {} camera(s) for {}s".format(utc_now(), level, len(levels), cameras, args.per_level))
            level_stop = threading.Event()
            workers = []
            for index in range(cameras):
                worker = CameraWorker(
                    camera_id="cam_{:02d}".format(index + 1),
                    level=level,
                    cameras=cameras,
                    interval=args.interval,
                    offset=index * args.interval / cameras,
                    hub=hub,
                    send=sender,
                    rows=frames_rows,
                    stop_event=level_stop,
                )
                worker.start()
                workers.append(worker)
            deadline = time.monotonic() + args.per_level
            try:
                while time.monotonic() < deadline:
                    time.sleep(0.25)
            except KeyboardInterrupt:
                interrupted.set()
            level_stop.set()
            for worker in workers:
                worker.join(timeout=args.interval + REQUEST_TIMEOUT_SECONDS)
            sent_at_level = sum(1 for row in frames_rows if row["level"] == level)
            levels_rows.append(
                {
                    "level": level,
                    "cameras": cameras,
                    "started_at": started_at,
                    "ended_at": utc_now(),
                    "frames_sent": sent_at_level,
                }
            )
            errors_at_level = sum(1 for row in frames_rows if row["level"] == level and row["status"] != "ok")
            print("[{}] level {} done: {} frames sent, {} errors".format(utc_now(), cameras, sent_at_level, errors_at_level))
            if interrupted.is_set():
                break
            time.sleep(args.pause)
    except KeyboardInterrupt:
        interrupted.set()

    hub.stop()
    if poller is not None:
        poller.stop_event.set()
        poller.join(timeout=5)
    write_csv(out_dir / "frames.csv", FRAMES_CSV_COLUMNS, frames_rows)
    write_csv(out_dir / "levels.csv", LEVELS_CSV_COLUMNS, levels_rows)
    if stats_rows:
        write_csv(out_dir / "vmstats.csv", VMSTATS_CSV_COLUMNS, stats_rows)
    if interrupted.is_set():
        print("[{}] interrupted — partial results written to {}".format(utc_now(), out_dir))
    print("[{}] done: {} frames, {} stats samples -> {}".format(utc_now(), len(frames_rows), len(stats_rows), out_dir))
    return out_dir


def build_parser():
    parser = argparse.ArgumentParser(description="multi-camera load generator: ramps camera counts against a detection backend")
    parser.add_argument("--target", default="dry", choices=["vm", "azure", "azure-blob", "dry"])
    parser.add_argument("--cameras", default="1,2,4,8,16,32,64", help="comma-separated camera counts, e.g. 1,2,4,8,16,32,64")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between frames per camera")
    parser.add_argument("--per-level", dest="per_level", type=float, default=60.0, help="seconds to run each camera level")
    parser.add_argument("--pause", type=float, default=5.0, help="seconds to rest between levels")
    parser.add_argument("--stats-every", dest="stats_every", type=float, default=5.0, help="seconds between VM /stats polls")
    parser.add_argument("--warmup", type=int, default=0, help="sequential warm-up frames sent before level 1 (discarded from results)")
    parser.add_argument("--stream", default="", help="MJPEG stream URL")
    parser.add_argument("--vm-url", dest="vm_url", default="", help="server /predict URL")
    parser.add_argument("--stats-url", dest="stats_url", default="", help="server /stats URL")
    parser.add_argument("--azure-url", dest="azure_url", default="", help="function /api/predict URL")
    parser.add_argument("--connection-string", dest="connection_string", default="", help="storage connection string for azure-blob target")
    parser.add_argument("--container", default="uploads", help="upload container for azure-blob target")
    return parser


def main():
    load_deploy_env()
    args = build_parser().parse_args()
    args.stream = args.stream or os.environ.get("STREAM_URL", "")
    args.vm_url = args.vm_url or os.environ.get("YOLO_API_URL", "")
    args.azure_url = args.azure_url or os.environ.get("AZURE_PREDICT_URL", "")
    args.connection_string = args.connection_string or os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
    if args.target == "vm" and not args.vm_url:
        raise SystemExit("target vm requires --vm-url or YOLO_API_URL in deploy.env")
    if args.target == "vm" and not args.stats_url:
        args.stats_url = args.vm_url.rsplit("/", 1)[0] + "/stats"
    if args.target == "azure" and not args.azure_url:
        raise SystemExit("target azure requires --azure-url or AZURE_PREDICT_URL in deploy.env")
    if args.target == "azure-blob" and not args.connection_string:
        raise SystemExit("target azure-blob requires --connection-string or AZURE_STORAGE_CONNECTION_STRING in deploy.env")
    if not args.stream:
        raise SystemExit("a camera stream URL is required (--stream or STREAM_URL in deploy.env)")
    run_benchmark(args)


if __name__ == "__main__":
    main()