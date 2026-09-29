# S1 — Remote Person Detection

Splits person detection into two parts:

- **`sender/`** — runs on the local machine. Grabs frames from an MJPEG camera stream, sends them to the remote YOLO API, counts detected persons, and feeds a local dashboard.
- **`server/`** — runs on an Azure VM inside Docker. Exposes a FastAPI `/predict` endpoint that runs YOLOv8 inference on uploaded images.

```
Camera (MJPEG stream)
        │
        ▼
sender/sender.py ──── POST /predict ────▶ server (Azure VM, Docker) ────▶ YOLOv8 detections (JSON)
        │
        ▼
database.csv + processed_frame.jpg ────▶ index.html dashboard (Chart.js)
```

---

## 1. Server (Azure VM)

### Prerequisites (one-time, Azure Portal)

1. Create an **Ubuntu Data Science Virtual Machine (DSVM)** — Docker is pre-installed.
   - On a plain Ubuntu VM, `deploy.sh` installs Docker automatically (requires passwordless `sudo`, default for Azure VM users).
   - GPU size (e.g. `Standard_NC4as_T4_v3`) for fast inference, or a CPU size for testing.
2. In the VM's **Networking** settings, add an **Inbound Security Rule** allowing TCP on port **80** (or use an existing rule).

### Deploy / update

Configuration lives in the gitignored `deploy.env` file in the repo root — both `deploy.sh` and the sender read it automatically:

```bash
VM_HOST=<azure-vm-public-ip>
VM_USER=<ssh-user>
REPO_URL=<your-git-repo-url>
YOLO_API_URL=http://<azure-vm-public-ip>/predict
```

Push your code to Git, then deploy from the local machine:

```bash
cd server
../deploy.sh
```

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `VM_HOST` | yes | — | Azure VM public IP |
| `REPO_URL` | yes | — | Git repository URL |
| `VM_USER` | no | `azureuser` | SSH user; deploys to that user home `yolo-api` folder on the VM |
| `REMOTE_DIR` | no | `~/yolo-api` on the VM | absolute path, `~`-relative, or name — resolved on the VM side |
| `YOLO_API_URL` | used by sender | placeholder | URL of the `/predict` endpoint |

Exported environment variables override `deploy.env` values.

`deploy.sh` SSHes into the VM and:
- clones the repo on the first run,
- runs `git pull` on subsequent runs,
- builds the image and starts the service with `docker compose up -d --build`,
- prints the container status.

**Auth:** with password authentication, `deploy.sh` prompts for the VM password when run. For passwordless deploys, set up an SSH key once:

```bash
ssh-copy-id azureuser@<VM_PUBLIC_IP>
```

On a GPU VM, add `gpus: all` to the `yolo-api` service in `docker-compose.yml`.

### Verify

```bash
curl -X POST -F "file=@some_image.jpg" http://<VM_PUBLIC_IP>/predict
```

Response:

```json
{
  "detections": [
    {
      "class": 0,
      "name": "person",
      "confidence": 0.87,
      "box": {"x1": 120.5, "y1": 80.2, "x2": 300.1, "y2": 480.9}
    }
  ],
  "persons": 1,
  "inference_ms": 31.2,
  "response_ms": 180.4
}
```

### Results storage (CSV)

Every prediction is appended to `server/data/results.csv` on the VM (mounted as a Docker volume, so it persists across container restarts and redeployments):

```csv
timestamp_utc,persons,detections,avg_confidence,inference_ms,response_ms,image_bytes,response_bytes
2026-09-30T00:23:15.765452+00:00,1,1,0.6054,29.9,180.9,493218,205
```

### Dashboard & performance data

The server hosts a web dashboard at **`http://<VM_PUBLIC_IP>/`** (public, port 80). It auto-refreshes every 5 seconds and shows:

- **Histogram** of person counts per frame
- **Persons over time** line chart
- **CPU / memory usage** chart of the VM
- Cards with throughput, response times, bandwidth totals, uptime, and estimated cost

Backing endpoints:

- `GET /stats` — aggregate performance metrics (JSON)
- `GET /history` — recent per-request records (JSON)

Performance data (Task 2.5) mapping:

| Metric | Where | Implementation |
|---|---|---|
| Detection efficiency | `frames_per_second`, `detections_per_frame`, `avg_inference_ms` | requests / uptime, YOLO inference timings |
| Memory usage | `memory_percent`, `memory_used_mb` / `memory_total_mb` | `psutil` (VM-wide) |
| CPU usage | `cpu_percent`, `resource_history` chart | `psutil`, sampled every 4 s |
| Bandwidth consumption | `bytes_in`, `bytes_out` | sum of uploaded image and response payload sizes |
| Monetary cost | `cost_per_hour`, `cost_total` | `COST_PER_HOUR` env (default `$0.096`/h) × uptime |
| Response time | `avg_response_ms`, `p95_response_ms` | full request handling time, avg and 95th percentile |

Set your VM's actual price with `COST_PER_HOUR=...` in `deploy.env` — `docker-compose.yml` passes it through to the container.

### Manage

Run over SSH (`ssh azureuser@<VM_PUBLIC_IP>`):

```bash
cd ~/yolo-api
docker compose ps          # status
docker compose logs -f    # logs
docker compose down        # stop
```

### Manual run on the VM (without Docker)

For debugging, run the server directly on the VM. Stop the Docker service first (`docker compose down`), otherwise both fight over port 8000:

```bash
cd ~/yolo-api/server
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

If `uv` is missing on the VM, install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The service is then reachable on port 8000 directly (no Docker port mapping), so the sender URL becomes `http://<VM_PUBLIC_IP>:8000/predict` with port 8000 open in the NSG.

### Run locally (no VM)

For testing without an Azure VM, run the server on the local machine.

With Docker (from the repo root):

```bash
docker compose up --build
```

Without Docker (Python 3.9+ with [uv](https://docs.astral.sh/uv/)):

```bash
cd server
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Then point the sender at it:

```bash
export YOLO_API_URL=http://127.0.0.1:8000/predict
```

The API is available at `http://127.0.0.1:8000/predict` (interactive docs at `http://127.0.0.1:8000/docs`).

---

## 2. Sender (local machine)

### Setup

Python 3.9+ with [uv](https://docs.astral.sh/uv/):

```bash
cd sender
uv sync
```

### Configure

Point the sender at your VM:

```bash
export YOLO_API_URL=http://<VM_PUBLIC_IP>/predict
```

(The VM exposes the service on port **80**; the container still listens on 8000 internally, mapped by `docker-compose.yml`.)

Camera stream URL and detection thresholds are constants at the top of `sender/sender.py`:

| Constant | Default | Meaning |
|---|---|---|
| `STREAM_URL` | `http://79.3.91.147:9002/mjpg/video.mjpg` | MJPEG camera stream |
| `PERSON_CLASS_ID` | `0` | COCO class id for "person" |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum detection confidence |
| `POLL_INTERVAL_SECONDS` | `0.5` | Delay between frames |

### Run

```bash
cd sender
uv run python sender.py
```

Stop with `Ctrl+C`. While running it:
1. reads a frame from the camera stream,
2. sends it to `YOLO_API_URL`,
3. draws green boxes around detected persons,
4. appends `timestamp, person_count` to `sender/database.csv`,
5. writes the annotated frame to `sender/processed_frame.jpg`.

### Dashboard

With the sender running, open `sender/index.html` (or serve the folder with any static server). It reloads every second, showing the annotated frame and a live person-count chart.

---

## 3. Full workflow

1. Provision the Azure DSVM and open port 8000 (once).
2. Commit and push code changes to Git.
3. `./deploy.sh` — deploy or update the server.
4. `export YOLO_API_URL=...` and `uv run python sender/sender.py` on the local machine.
5. Open `sender/index.html` to watch the counts.