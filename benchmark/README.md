# AIIA — Multi-camera load benchmark (Part 4)

Measures how many simultaneous cameras each deployment can process:

- **Task 4.2 (VM)** — ramp camera counts against the Azure VM server and record CPU, memory, bandwidth, cost and response time per level (`/stats` is polled while the load runs).
- **Task 4.3 (Cloud)** — the same ramp against the Azure Function, in HTTP mode (`/api/predict`, end-to-end latency measured by the client) and blob mode (upload to the `uploads` container; processing is asynchronous and observed through `analysis.csv` in the `results` container).

Each simulated camera sends one frame every `--interval` seconds (default 5 s, the sender's real cadence). The generator ramps through camera counts (`--cameras 1,2,4,8,12`), runs each level for `--per-level` seconds and pauses briefly between levels.

## How it works

The camera is an MJPEG server that accepts a **single client**, so it cannot be opened N times. A `FrameHub` thread keeps one connection open, stores the latest frame in memory, and every camera worker grabs a copy of it (`benchmark/loadgen.py`). Workers are staggered — worker *i* of *N* fires at offset `i * interval / N` — so load spreads evenly inside each interval instead of arriving in bursts.

Raw data lands in `benchmark/results/<target>_<timestamp>/` (gitignored). `report.py` aggregates it into per-level summaries, a processing-limit verdict, a cost estimate and a chart.

## Setup

```bash
cd benchmark
uv sync
uv run pytest
```

The last line runs the unit tests for the aggregation logic. `loadgen.py` reads `deploy.env` from the repo root automatically, so `YOLO_API_URL`, `AZURE_PREDICT_URL` and `AZURE_STORAGE_CONNECTION_STRING` come from there unless overridden by CLI flags or exported environment variables.

## Sending load — `loadgen.py`

```bash
uv run python loadgen.py --target vm --cameras 1,2,4,8,12 --interval 5 --per-level 300
```

| Target | Sends | Requires | Collects |
|---|---|---|---|
| `dry` | nothing (JPEG-encode only) | camera stream | `frames.csv` — smoke-test the pipeline end to end |
| `vm` | POST to the server `/predict` | `YOLO_API_URL` | `frames.csv` + `vmstats.csv` (polls `/stats` every `--stats-every` s) |
| `azure` | POST to the function `/api/predict` | `AZURE_PREDICT_URL` | `frames.csv` — client-measured end-to-end latency |
| `azure-blob` | upload to the `uploads` container | `AZURE_STORAGE_CONNECTION_STRING` | `frames.csv` — upload time only; processing happens asynchronously |

| Flag | Default | Meaning |
|---|---|---|
| `--target` | `dry` | one of the targets above |
| `--cameras` | `1,2,4,8,16,32,64` | comma-separated camera counts, one ramp level each |
| `--interval` | `5.0` | seconds between frames per camera |
| `--per-level` | `60.0` | seconds each level runs |
| `--pause` | `5.0` | rest between levels |
| `--stats-every` | `5.0` | `/stats` poll period (VM target) |
| `--stream` | `STREAM_URL` env | MJPEG stream URL |
| `--vm-url` | `YOLO_API_URL` env | server `/predict` URL |
| `--stats-url` | derived | server `/stats` URL — defaults to the `/predict` URL with `predict` replaced by `stats` |
| `--azure-url` | `AZURE_PREDICT_URL` env | function `/api/predict` URL |
| `--connection-string` | `AZURE_STORAGE_CONNECTION_STRING` env | storage connection string (`azure-blob` target) |
| `--container` | `uploads` | upload container (`azure-blob` target) |

`Ctrl+C` stops the run early and writes partial results with a note in the console output.

Blob-mode uploads are named `bench_<run>_<camera>_<timestamp>.jpg`, so they trigger the real blob-trigger pipeline like any sender frame — that is the point of the exercise.

## Aggregating — `report.py`

```bash
uv run python report.py results/vm_20261013_181532
```

Prints a per-level table (frames, errors, p50/p95/max latency, VM CPU/memory, cost delta, cloud rows/timings), the **processing limit** verdict, and writes `summary.csv` + `summary.png`.

| Flag | Default | Meaning |
|---|---|---|
| `run_dir` (positional) | — | directory produced by `loadgen.py` |
| `--cloud-csv` | — | downloaded `analysis.csv` from the function's `results` container |
| `--error-threshold` | `0.02` | error rate that marks a level saturated |
| `--latency-threshold-ms` | `10000` | p95 latency that marks a level saturated |
| `--cpu-threshold` | `95` | VM CPU % that marks a level saturated |
| `--memory-threshold` | `95` | VM memory % that marks a level saturated |
| `--no-plot` | off | skip writing `summary.png` |

**Verdict logic** — a level is *saturated* when any threshold trips; the reported limit is the highest camera count whose level stayed healthy. When nothing trips, the last tested level is reported as the limit with a "no saturation within the tested range" note (extend `--cameras` to find the ceiling).

**Cloud windowing** — `--cloud-csv` rows are attributed to levels by comparing `timestamp_utc` with each level's `started_at`/`ended_at` bounds from `levels.csv`. Both sides write UTC ISO-8601 with the `+00:00` offset, so plain string comparison orders timestamps correctly across differing fractional-second precision.

**Cost model** — cloud cost uses Flex Consumption pricing: executions ($0.20 / 1M) plus compute (GB-seconds at $0.0000111 / GB-s, 2 GB per instance assumed). Constants live in `benchmark/helpers.py` — update them if the pricing changes. The VM's cost comes from the server's own cost tracking via `/stats` (cost delta within a level).

## Output files

All in `benchmark/results/<run>/`:

| File | Contents |
|---|---|
| `frames.csv` | one row per frame sent: `timestamp_utc, level, cameras, camera, latency_ms, status, http_code, detections, persons, inference_ms, sent_bytes` |
| `levels.csv` | level boundaries: `level, cameras, started_at, ended_at, frames_sent` |
| `vmstats.csv` | `/stats` samples tagged with their level: `timestamp_utc, level, cameras, cpu_percent, memory_percent, memory_used_mb, bytes_in, bytes_out, cost_total, uptime_seconds, requests` |
| `summary.csv` | per-level aggregates written by `report.py` |
| `summary.png` | up-to-twelve-panel chart, camera-count axes on log scale with ticks at the tested levels, millisecond axes on log scale with 1-2-5 ticks, threshold lines and a vertical saturation marker on every panel; panels without data for the target are omitted: client response time (p50/p95/max), latency decomposition (server-reported model inference vs the queueing + I/O overhead gap), throughput (sent vs completed fps from client and server counters), error rate, latency distribution box plot, VM resources, load timeline (sent/completed fps per 10 s bucket with level shading and CPU trace), VM bandwidth, model inference time, cost per level, detection efficiency (detections/persons per frame), Azure Function timings |

`status` values in `frames.csv`: `ok`, `http_<code>` (non-200 response), `network_error` (connection failure/timeout), `encode_failed`, or the Azure SDK exception class name for blob uploads.

## Task 4.2 — VM limit

```bash
uv run python loadgen.py --target vm --cameras 1,2,4,8,16,32,64 --interval 5 --per-level 300
uv run python report.py results/vm_<timestamp>
```

Five minutes per level gives each camera ~60 frames and the `/stats` poller ~60 samples — enough for stable percentiles. The default ramp (`1,2,4,8,16,32,64`) brackets the expected ceiling: with ~230 ms CPU inference per frame and a 5 s interval, the single-worker server saturates around 20 cameras, so the limit should fall between the 16 and 32 levels.

## Task 4.3 — Cloud limit

```bash
uv run python loadgen.py --target azure --cameras 1,2,4,8,16,32,64 --interval 5 --per-level 300
uv run python loadgen.py --target azure-blob --cameras 1,2,4,8,16,32,64 --interval 5 --per-level 300
az storage blob download --account-name <STORAGE_ACCOUNT> --container-name results --name analysis.csv --file analysis.csv --auth-mode login
uv run python report.py results/azure_<timestamp> --cloud-csv analysis.csv
uv run python report.py results/azure-blob_<timestamp> --cloud-csv analysis.csv
```

HTTP mode reports client-measured end-to-end latency; blob mode reports upload time, while the cloud-side response/inference times come from `analysis.csv` rows windowed into levels. Detection efficiency per level is `cloud_detections_per_frame` / `cloud_avg_confidence` in `summary.csv`.

Requires the function app running and (for blob mode) the Event Grid subscription wired with the subject filter — see sections 3 and 7 of the root [`README.md`](../README.md).

## Cleanup after a blob-mode run

Benchmark blobs trigger real processing, so both containers accumulate artifacts:

```bash
az storage blob delete-batch --account-name <STORAGE_ACCOUNT> --source uploads --pattern "bench_*" --auth-mode login
az storage blob delete-batch --account-name <STORAGE_ACCOUNT> --source results --pattern "processed_bench_*" --auth-mode login
```

`results/analysis.csv` keeps its rows (they are the measurement); delete the downloaded local `analysis.csv` copy when the report has been generated.