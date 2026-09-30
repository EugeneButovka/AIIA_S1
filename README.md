# S1 — Remote Person Detection

End-to-end person-counting pipeline split across two machines:

- **`sender/`** (local machine) — grabs frames from an MJPEG camera stream, sends them to the remote YOLO API, and feeds a local dashboard.
- **`server/`** (Azure VM, Docker) — FastAPI service that runs YOLOv8 inference, stores results in a CSV, and hosts a public dashboard with live performance metrics.

```
Camera (MJPEG stream)
        │
        ▼
sender/sender.py ──── POST /predict ────▶ server (Azure VM, Docker) ────▶ detections (JSON)
        │                                        │
        ▼                                        ▼
sender dashboard (local)              results.csv + public dashboard at http://<VM_PUBLIC_IP>/
```

## Project layout

```
├── deploy.sh             # one-command deploy to the Azure VM
├── deploy.env            # gitignored config (VM host, repo URL, API URL) — used by everything
├── docker-compose.yml    # builds ./server, maps port 80 → 8000, restart: always
├── sender/
│   ├── sender.py         # frame grabber + API client + CSV/annotated-frame writer
│   ├── index.html        # local dashboard (annotated frame + person-count chart)
│   └── pyproject.toml
└── server/
    ├── main.py           # app assembly (create_app)
    ├── config.py         # settings, paths, env parsing
    ├── routes.py         # endpoints: /predict, /stats, /history, /
    ├── detection.py      # YOLO wrapper (Detector, DetectionResult)
    ├── storage.py        # CSV results store (ResultsStore, PredictionRecord)
    ├── metrics.py        # performance tracker (stats, resource sampling)
    ├── cost.py           # Azure cost auto-resolution (IMDS + Retail Prices API)
    ├── static/index.html # public dashboard
    └── pyproject.toml
```

## Quick start

```bash
cp deploy.env.example deploy.env   # or create it manually — see the deploy.env section
# edit deploy.env with your VM IP, SSH user and repo URL
./deploy.sh                       # deploy the server to the VM
cd sender && uv run python sender.py   # start collecting counts locally
```

Requirements: Python 3.9+ with [uv](https://docs.astral.sh/uv/); an Azure Ubuntu VM with port 80 open.

---

## deploy.env

Single source of configuration, read by `deploy.sh`, the sender, and `docker-compose.yml`. Gitignored, so keep real IPs here — not in the README or code.

```bash
VM_HOST=<azure-vm-public-ip>
VM_USER=<ssh-user>
REPO_URL=<your-git-repo-url>
YOLO_API_URL=http://<azure-vm-public-ip>/predict
```

| Variable | Required | Default | Used by | Meaning |
|---|---|---|---|---|
| `VM_HOST` | yes | — | `deploy.sh` | Azure VM public IP |
| `REPO_URL` | yes | — | `deploy.sh` | Git repository the VM pulls from |
| `VM_USER` | no | `azureuser` | `deploy.sh` | SSH user; deploys to that user home `yolo-api` folder on the VM |
| `REMOTE_DIR` | no | `~/yolo-api` on the VM | `deploy.sh` | absolute path, `~`-relative, or bare name — resolved on the VM side |
| `YOLO_API_URL` | no | placeholder | sender | `/predict` endpoint URL |
| `STREAM_URL` | no | hardcoded camera | sender | MJPEG camera stream URL |
| `COST_PER_HOUR` | no | Azure auto-lookup | server | hourly rate override for cost estimation |

Exported environment variables always override `deploy.env` values.

---

## 1. Server (Azure VM)

### Prerequisites (one-time, Azure Portal)

1. Create an **Ubuntu VM** (the Data Science image ships with Docker; `deploy.sh` installs Docker automatically otherwise — requires passwordless `sudo`, the default for Azure VM users).
2. Pick a size with **at least 4 GB RAM** (e.g. `Standard_B2s`) — PyTorch CPU inference freezes smaller instances.
3. In **Networking**, add an **Inbound Security Rule** allowing TCP on port **80**.

### Deploy / update

```bash
./deploy.sh
```

The script SSHes into the VM and:
- clones the repo on the first run, `git pull`s on subsequent runs,
- installs Docker if missing (via the official `get.docker.com` script),
- builds the image and (re)starts the service with `docker compose up -d --build`,
- prints the container status.

It is safe to re-run at any time; Docker Compose only recreates the container when something changed. Password authentication is fine — the script prompts for the VM password when needed. For passwordless deploys, run once:

```bash
ssh-copy-id <ssh-user>@<VM_PUBLIC_IP>
```

On a GPU VM, add `gpus: all` to the `yolo-api` service in `docker-compose.yml`.

### Verify

```bash
curl -X POST -F "file=@some_image.jpg" http://<VM_PUBLIC_IP>/predict
```

```json
{
  "detections": [
    {"class": 0, "name": "person", "confidence": 0.87,
     "box": {"x1": 120.5, "y1": 80.2, "x2": 300.1, "y2": 480.9}}
  ],
  "persons": 1,
  "inference_ms": 31.2,
  "response_ms": 180.4
}
```

Interactive API docs: `http://<VM_PUBLIC_IP>/docs`.

### Results storage (CSV)

Every prediction is appended to `server/data/results.csv` on the VM (mounted as a Docker volume — persists across container restarts and redeploys):

```csv
timestamp_utc,persons,detections,avg_confidence,inference_ms,response_ms,image_bytes,response_bytes
2026-09-30T00:23:15.765452+00:00,1,1,0.6054,29.9,180.9,493218,205
```

### Dashboard & performance data

The server hosts a web dashboard at **`http://<VM_PUBLIC_IP>/`** (public IP, port 80). It auto-refreshes every 5 seconds and shows:

- **Histogram** of person counts per frame — how many frames saw 0, 1, 2, … people
- **Persons over time** line chart
- **Average detection confidence over time** chart
- **CPU / memory usage** chart of the VM
- Cards: throughput, response times, confidence, bandwidth totals, uptime, estimated cost

| Endpoint | Returns |
|---|---|
| `GET /stats` | aggregate performance metrics (JSON) |
| `GET /history` | recent per-request records (JSON) |
| `GET /docs` | interactive OpenAPI docs |
| `GET /` | dashboard page |

Performance data mapping (Task 2.5):

| Metric | Fields | Implementation |
|---|---|---|
| Detection efficiency | `frames_per_second`, `detections_per_frame`, `avg_inference_ms` | requests / uptime, YOLO inference timings |
| Detection confidence | `avg_confidence` | mean confidence across recent detections |
| Memory usage | `memory_percent`, `memory_used_mb`, `memory_total_mb` | `psutil` (VM-wide) |
| CPU usage | `cpu_percent`, `resource_history` chart | `psutil`, sampled every 4 s |
| Bandwidth consumption | `bytes_in`, `bytes_out` | sum of uploaded image and response payload sizes |
| Monetary cost | `cost_per_hour`, `cost_per_month`, `cost_total`, `cost_source` | see below |
| Response time | `avg_response_ms`, `p95_response_ms` | full request handling time, avg and 95th percentile |

**Cost resolution order:** on startup the server asks the Azure Instance Metadata Service (IMDS) for its own VM size and region, then looks up the Linux pay-as-you-go hourly rate in the public Azure Retail Prices API — `cost_source` shows the resolved SKU, e.g. `Azure Retail Prices API (Standard_B2s, italynorth)`. If `COST_PER_HOUR` is set in `deploy.env`, it always wins; outside Azure the default rate is used. The retail rate covers compute only — disk and bandwidth are not included, so match against the Azure Portal invoice.

### Manage over SSH

```bash
ssh <ssh-user>@<VM_PUBLIC_IP>
cd ~/yolo-api
docker compose ps          # status
docker compose logs -f     # logs
docker compose down        # stop
```

---

## 2. Sender (local machine)

### Setup & run

```bash
cd sender
uv sync
uv run python sender.py
```

Configuration comes from `deploy.env` (`YOLO_API_URL`, optionally `STREAM_URL`) — no exports needed. Stop with `Ctrl+C`. Each cycle:

1. reads a frame from the camera stream (prints `[HH:MM:SS] new image WxH prepared, sending to server`),
2. POSTs it to `YOLO_API_URL` (prints `[HH:MM:SS] server response: N detections, M person(s), X.XXs`),
3. draws green boxes around detected persons,
4. appends `timestamp, person_count` to `sender/database.csv`,
5. writes the annotated frame to `sender/processed_frame.jpg`.

Tuning constants at the top of `sender/sender.py`:

| Constant | Default | Meaning |
|---|---|---|
| `PERSON_CLASS_ID` | `0` | COCO class id for "person" |
| `CONFIDENCE_THRESHOLD` | `0.5` | minimum detection confidence |
| `POLL_INTERVAL_SECONDS` | `0.5` | delay between frames |

### Local dashboard

With the sender running, open `sender/index.html`. It reloads every second, showing the annotated frame and a live person-count chart from `sender/database.csv`.

Note: the camera is a single-client MJPEG server — while the sender is running, do not open the same stream URL in a browser or another client, or frames will stall.

---

## 3. Local testing (no VM)

Run the server on the local machine:

```bash
cd server
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

or with Docker, from the repo root:

```bash
docker compose up --build
```

Then point the sender at the local instance (overrides `deploy.env`):

```bash
cd sender
YOLO_API_URL=http://127.0.0.1:8000/predict uv run python sender.py
```

---

## 4. Full workflow

1. Provision the Azure VM and open port 80 (once).
2. Commit and push code changes to Git.
3. `./deploy.sh` — deploy or update the server.
4. `cd sender && uv run python sender.py` — start collecting counts.
5. Open `sender/index.html` locally, and `http://<VM_PUBLIC_IP>/` for the public dashboard and metrics.

---

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `failed to bind host port 0.0.0.0:80/tcp: address already in use` | Apache preinstalled on Ubuntu holds port 80 | `sudo systemctl disable --now apache2` (or purge it) |
| VM freezes / SSH times out after first request | VM too small for PyTorch CPU inference | resize to ≥ 4 GB RAM (`Standard_B2s` or larger) and restart |
| `Download failure ... Environment may be offline` in container logs | transient `yolov8n.pt` download failure | `docker compose up -d --force-recreate`; if it persists, check container DNS |
| Sender hangs with no output | camera allows only one client, or stream slow to start | close other consumers of the stream; first frames may take a few seconds |
| `Unit apache2.service could not be found` in deploy output | expected after Apache was removed | harmless |
| Dashboard `database.csv` chart empty | sender not running or no rows yet | start the sender; the chart skips the CSV header row |