# S1 — Remote Person Detection

End-to-end person-counting pipeline split across two machines:

- **`sender/`** (local machine) — grabs frames from an MJPEG camera stream, sends them for person detection, and feeds a local dashboard; `SEND_MODE` picks the destination: VM server (`server`), the function's HTTP endpoint (`azure`), or fire-and-forget blob upload (`azure-blob`).
- **`server/`** (Azure VM, Docker) — FastAPI service that runs YOLOv8 inference, stores results in a CSV, and hosts a public dashboard with live performance metrics.
- **`azure-function/`** (serverless, Flex Consumption, optional) — the same YOLO model without a GPU VM or Docker: a blob trigger analyzes uploaded images, and an HTTP endpoint mirrors the server's `/predict`. Writes CSV rows in the server's schema plus annotated images in the sender style. Deployed automatically by GitHub Actions.

```
Camera (MJPEG stream)
        │
        ▼
sender/sender.py ─── SEND_MODE ─┬─ server      ── POST /predict ─────▶ server (Azure VM, Docker) ──▶ results.csv + public dashboard
                               ├─ azure       ── POST /api/predict ─▶ azure-function (Flex) ─┐ JSON response (server format)
                               └─ azure-blob  ── blob upload ────────▶ uploads container ─────┘
                                                                                               │ blob trigger
                                                                                               ▼
                                                                        results container: analysis.csv + processed_*.jpg
```

The sender's local dashboard (annotated frame + person-count chart) is populated in `server` and `azure` modes; `azure-blob` is fire-and-forget with cloud-only results.

## Project layout

```
├── deploy.sh             # one-command deploy to the Azure VM
├── deploy.env            # gitignored config (VM host, repo URL, API URL) — used by everything
├── docker-compose.yml    # builds ./server, maps port 80 → 8000, restart: always
├── sender/
│   ├── sender.py         # frame grabber: VM server, function /api/predict, or blob upload (SEND_MODE)
│   ├── index.html        # local dashboard (annotated frame + person-count chart)
│   ├── tests/            # pytest suite (mode resolution + upload path)
│   └── pyproject.toml
├── server/
│   ├── main.py           # app assembly (create_app)
│   ├── config.py         # settings, paths, env parsing
│   ├── routes.py         # endpoints: /predict, /stats, /history, /
│   ├── detection.py      # YOLO wrapper (Detector, DetectionResult)
│   ├── storage.py        # CSV results store (ResultsStore, PredictionRecord)
│   ├── metrics.py        # performance tracker (stats, resource sampling)
│   ├── cost.py           # Azure cost auto-resolution (IMDS + Retail Prices API)
│   ├── static/index.html # public dashboard
│   └── pyproject.toml
└── azure-function/
    ├── function_app.py    # hello (HTTP sample) + predict (HTTP, server format) + analyze_image (blob trigger)
    ├── detection.py       # YOLOv8n detector (mirrors server/detection.py)
    ├── analysis.py        # pipeline: detect → CSV row → annotated image (Vision path commented)
    ├── processed_image.py # detection-box drawing (JPEG re-encode, sender style)
    ├── results.py         # CSV append-blob writer + processed-image upload
    ├── local.settings.json# gitignored local config (Azurite connection, weights path)
    └── tests/             # pytest suite (no network needed)
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
| `SEND_MODE` | no | `server` | sender | `server` = POST to `YOLO_API_URL`; `azure` = POST to the function's `/api/predict`; `azure-blob` = fire-and-forget upload to the function's `uploads` container |
| `AZURE_PREDICT_URL` | yes when `SEND_MODE=azure` | — | sender | Azure Function predict endpoint — note: **Flex Consumption apps use a unique default domain** shown in the portal (e.g. `https://<name>-<hash>.<region>.azurewebsites.net/api/predict`), not `<name>.azurewebsites.net` |
| `AZURE_STORAGE_CONNECTION_STRING` | yes when `SEND_MODE=azure-blob` | — | sender | storage account connection string of the function app (account-key based — requires "Allow storage account key access" enabled on the account, see section 3) |
| `AZURE_UPLOAD_CONTAINER` | no | `uploads` | sender | input container watched by the `analyze_image` blob trigger |

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
| `POLL_INTERVAL_SECONDS` | `2.0` | delay between frames (server / azure HTTP modes) |
| `BLOB_SEND_INTERVAL_SECONDS` | `5.0` | delay between uploads in `azure-blob` mode — slower than the HTTP poll because every upload becomes a trigger invocation on Azure |

### Azure modes (optional)

Two `SEND_MODE` values reroute frames from the VM server to the Azure Function (section 3). Both end up writing the same cloud records (`results/analysis.csv` + `results/processed_*.jpg`); they differ in how the frame travels and where the results show up.

**`SEND_MODE=azure` — HTTP request/response, full local parity**

```bash
SEND_MODE=azure
AZURE_PREDICT_URL=https://<default-domain>/api/predict
```

- The sender POSTs the frame as multipart (`file` field) to the function's `/api/predict` — the same request the VM server's `/predict` accepts — and gets back the identical server-format JSON: `detections`, `persons`, `inference_ms`, `response_ms`.
- Because the response comes back synchronously, the sender keeps all local behavior: green boxes on the frame, `database.csv` rows, `processed_frame.jpg`, live dashboard.
- The function records each request in the cloud as well: one CSV row (blob name `predict_<timestamp>.jpg`) plus the annotated image in the `results` container.
- No storage credentials needed on the sender — plain HTTPS to an anonymous endpoint.
- Latency: the first request after a cold start pays the torch import + YOLO load (5–15 s on Flex); later requests are ~1 s or less.

On the Flex Consumption plan the default domain is unique per app (`<name>-<hash>.<region>.azurewebsites.net`) — copy it from the function app's Overview page in the portal.

**`SEND_MODE=azure-blob` — fire-and-forget upload**

```bash
SEND_MODE=azure-blob
AZURE_STORAGE_CONNECTION_STRING=<storage-account-connection-string>
# AZURE_UPLOAD_CONTAINER=uploads
```

- Each frame is uploaded to the `uploads` container as `frame_YYYYMMDD_HHMMSS_<microseconds>.jpg` (container auto-created if missing) and the sender moves on immediately — no response is awaited.
- The `analyze_image` blob trigger picks the blob up — typically within seconds, up to about a minute — and runs the same detect → CSV → annotated-image pipeline.
- Results exist **only in the cloud** (`results` container): the sender skips the local draw/CSV/frame steps and the local dashboard stays empty.
- The connection string is account-key based, so the storage account must have key access enabled (see "Storage & containers" in section 3 — the Flex application-storage flow disables it by default).
- The blob trigger needs **Always ready instances ≥ 1** on the function app (portal → Scale) — non-HTTP triggers don't fire reliably from a zero-instance Flex app.

### Local dashboard

With the sender running, open `sender/index.html`. It reloads every second, showing the annotated frame and a live person-count chart from `sender/database.csv`.

Note: the camera is a single-client MJPEG server — while the sender is running, do not open the same stream URL in a browser or another client, or frames will stall.

---

## 3. Azure Function (serverless image AI)

Serverless alternative to the VM server: images uploaded to Azure Storage are analyzed by the **same YOLOv8n model as the server** (`detection.py` mirrors `server/detection.py`) — no GPU VM, no Docker, scale-to-zero. The Azure Vision (Image Analysis 4.0) implementation is kept commented in the code for reference.

```
blob upload to uploads/             POST /api/predict (multipart `file` / raw body)
        │                                         │
        ▼                                         ▼
analyze_image (blob trigger)              predict (HTTP function)
        └────────────────────┬────────────────────┘
                               ▼
                  YOLOv8n inference (same model as the server)
                               │
                               ├── results/analysis.csv          # one row per image, header written once
                               └── results/processed_<name>.jpg  # image with detection boxes drawn
```

Detection runs the **same YOLO model as the server** (`detection.py` mirrors `server/detection.py`, weights `yolov8n.pt` — bundled into the deployment package by CI, auto-downloaded locally on first run). The Azure AI Vision (Image Analysis 4.0) code is kept commented in `analysis.py` / `processed_image.py` for reference — uncomment it to switch back.

### Functions

| Function | Trigger | Purpose |
|---|---|---|
| `analyze_image` | new blob in `uploads/{name}` | analyze image → append CSV row → upload annotated image |
| `predict` | HTTP `POST /api/predict` | same request/response format as the server's `/predict` — also records CSV row + annotated image; used by the sender in `SEND_MODE=azure` |
| `hello` | HTTP `GET /api/hello?name=...` | sample health-check endpoint (anonymous) |

Input images can enter the pipeline two ways: a blob dropped into the `uploads` container (Storage Explorer, `az storage blob upload`, or the sender in `SEND_MODE=azure-blob`) fires `analyze_image`, while an HTTP POST to `/api/predict` (multipart `file` field or raw body — used by the sender in `SEND_MODE=azure`) returns the server-compatible JSON payload synchronously. Both feed the same recording flow.

### CSV schema (aligned with the server)

`server/storage.py` is the source of format truth: the first 8 columns are identical to `server/data/results.csv`, with extras appended. The last four (`blob_name`, `caption`, `caption_confidence`, `tags`, `ocr_text`) keep the schema stable — the vision-only fields stay empty while the YOLO path is active:

```csv
timestamp_utc,persons,detections,avg_confidence,inference_ms,response_ms,image_bytes,response_bytes,blob_name,caption,caption_confidence,tags,ocr_text
```

`persons` counts person-class detections with confidence ≥ 0.5 (same threshold as server and sender). Annotated images mirror the sender: green boxes around detected persons, thickness 2.

### Local development

```bash
cd azure-function && uv sync && uv run pytest   # 34 tests, no network needed
docker run -d -p 10000:10000 mcr.microsoft.com/azure-storage/azurite azurite-blob --blobHost 0.0.0.0
func start
```

`local.settings.json` points `AzureWebJobsStorage` at local Azurite; the YOLO weights download automatically on first invocation. Upload a test image (e.g. via [Azure Storage Explorer](https://azure.microsoft.com/products/storage/storage-explorer/)) into the `uploads` container and watch the CSV row and annotated image appear in `results`.

### Deploy (one-time)

**1. Create the function app** — portal: Function App → Create, runtime **Python**, plan **Flex Consumption** (region e.g. `spaincentral`), and attach **application storage** during creation: this configures the identity-based `AzureWebJobsStorage__*` settings and grants the app's managed identity `Storage Blob Data Contributor` on the account. CLI equivalent:

```bash
az group create --name <RESOURCE_GROUP> --location spaincentral
az functionapp create --name <FUNCTION_APP_NAME> --resource-group <RESOURCE_GROUP> \
  --flexconsumption-location spaincentral --runtime python --runtime-version 3.13 \
  --functions-version 4
```

**2. App settings** (weights + results location):

```bash
az functionapp config appsettings set --name <FUNCTION_APP_NAME> --resource-group <RESOURCE_GROUP> \
  --settings YOLO_WEIGHTS=yolov8n.pt RESULTS_CONTAINER_NAME=results RESULTS_CSV_NAME=analysis.csv
```

**3. Storage & containers** (account `<STORAGE_ACCOUNT>` in this deployment). The pipeline uses two containers: `uploads` (trigger input) and `results` (CSV + annotated images); the `azure-webjobs-*` / `app-package-*` containers belong to the platform — leave them alone. Flex application-storage accounts are created with **shared key access disabled** (identity-only), which is fine for the function itself:

```bash
# ad-hoc az storage commands work with your AAD identity:
az storage container create --name uploads --account-name <STORAGE_ACCOUNT> --auth-mode login
az storage container create --name results --account-name <STORAGE_ACCOUNT> --auth-mode login

# only needed for the sender's azure-blob mode (account-key connection string):
az storage account update --name <STORAGE_ACCOUNT> --resource-group <RESOURCE_GROUP> --allow-shared-key-access true
CONN=$(az storage account show-connection-string --name <STORAGE_ACCOUNT> --resource-group <RESOURCE_GROUP> -o tsv)
```

`results` is also auto-created by the function on first write; `uploads` must exist before blob-trigger traffic starts (the sender auto-creates it too, once key auth works).

**4. Blob trigger wiring** (blob-trigger mode only). Flex Consumption runs **only the event-based (Event Grid) blob trigger** — the default polling trigger silently never fires (it's declared as `source="EventGrid"` on the `@app.blob_trigger` decorator in `function_app.py`). One-time setup:

- **Always ready instances = 1** (portal → function app → Scale) — non-HTTP triggers don't fire reliably from a zero-instance Flex app; one 2048 MB always-ready instance costs ~$2/month.
- Register the `Microsoft.EventGrid` resource provider on the subscription if it isn't already: `az provider register --namespace Microsoft.EventGrid`.
- The **Event Grid event subscription** on the `uploads` container (event type *Blob Created*): Flex may auto-provision it on deploy; if not, create it manually — storage account → Events → + Event Subscription → Web Hook endpoint `https://<app-domain>/runtime/webhooks/blobs?functionName=Host.Functions.analyze_image&code=<blobs_extension key>` (key: portal → function app → Functions → App keys → System keys).
- **Identity roles**: the portal's application-storage flow grants only `Storage Blob Data Contributor`, but the Event Grid blob trigger also uses the account's **Queue** service internally — without `Storage Queue Data Contributor` (and `Storage Table Data Contributor` for host bookkeeping) the function app **crash-loops at host startup** with `403 AuthorizationPermissionMismatch` from the Queue service. Grant both to the app's user-assigned identity (`az functionapp identity show` → `principalId`):
  ```bash
  az role assignment create --assignee-object-id <PRINCIPAL_ID> --assignee-principal-type ServicePrincipal --role "Storage Queue Data Contributor" --scope "/subscriptions/<sub>/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.Storage/storageAccounts/<STORAGE_ACCOUNT>"
  az role assignment create --assignee-object-id <PRINCIPAL_ID> --assignee-principal-type ServicePrincipal --role "Storage Table Data Contributor" --scope "/subscriptions/<sub>/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.Storage/storageAccounts/<STORAGE_ACCOUNT>"
  ```

Flex hosts each function group on separate instances (the Scale blade sets always-ready counts per group: `http`, `blob`, …), so the HTTP endpoints can be healthy while the blob group is wedged — check the group you're actually testing. Failed Event Grid deliveries are retried with backoff for up to 24 hours, so blobs uploaded while the app was down still process once it recovers — but blobs created *before* the event subscription existed never generate events and stay unprocessed forever.

End-to-end latency in this mode is seconds: upload → Event Grid delivery → trigger → YOLO → CSV + annotated image in `results`.

**5. Deploy** — handled by CI/CD (below); `func azure functionapp publish <FUNCTION_APP_NAME>` from Core Tools works as a manual alternative.

Settings (managed in `local.settings.json` locally, app settings in Azure):

| Variable | Default | Meaning |
|---|---|---|
| `AzureWebJobsStorage` | — | trigger + `results` container: a connection string locally (Azurite); on Flex Consumption prefer identity-based settings (`AzureWebJobsStorage__blobServiceUri` + `AzureWebJobsStorage__credential=managedIdentity` + `AzureWebJobsStorage__clientId`) — this is what the portal's "application storage" flow configures, and the results writer supports both forms |
| `YOLO_WEIGHTS` | `yolov8n.pt` | YOLO weights path (same `yolov8n.pt` model as the server); CI bundles the file into the deployment package |
| `RESULTS_CONTAINER_NAME` | `results` | output container (auto-created on first write) |
| `RESULTS_CSV_NAME` | `analysis.csv` | CSV blob name in the results container |
| `VISION_ENDPOINT` / `VISION_KEY` | — | unused — only needed if the commented Azure Vision path is re-enabled |

Deployment notes:

- **Plan**: PyTorch + ultralytics exceed the 500 MB app size limit of the classic Consumption plan — use **Flex Consumption** or Premium, not `--consumption-plan-location`.
- **CI/CD** (`.github/workflows/deploy-azure-function.yml`): on push to `main` touching `azure-function/**` → runs the test suite (uv) → vendors dependencies with `pip --target .python_packages/lib/site-packages` — the **Flex layout**, which has no `python3.x` path segment, unlike classic plans — using the **CPU-only torch** pins from `requirements.txt` (plain Linux torch would pull CUDA libs and add gigabytes; the `+cpu` wheels are Linux/Windows-only, so that file only resolves on the Linux runner) → downloads `yolov8n.pt` into the package → deploys the zip via `functions-action` with **`sku: flexconsumption`** (required for publish-profile auth on Flex — without it the action calls the classic Kudu `zipdeploy`, which Flex doesn't support, and fails with `502 Bad Gateway`). `ultralytics` is pinned to the tested version: unpinned, pip can backtrack to a legacy release with a different package layout under wheel-only constraints.
- **CI/CD GitHub configuration** (repo → Settings → Secrets and variables → Actions):
    - `AZURE_FUNCTIONAPP_PUBLISH_PROFILE` — **secret** stored in the `AIIA_S1_ENV` **environment** (the workflow job declares `environment: AIIA_S1_ENV`, which is what makes environment secrets visible to it); value = a **freshly downloaded** publish profile — a profile downloaded while SCM basic auth was disabled keeps failing with `401` even after the policy is re-enabled. SCM basic auth must be enabled on the app (`basicPublishingCredentialsPolicies` → `scm`).
    - `AZURE_FUNCTIONAPP_NAME` — **repository variable** (the *Variables* tab, not Secrets); value = the function app name. The name is not hardcoded in the public workflow file (`${{ vars.AZURE_FUNCTIONAPP_NAME }}`) — without this variable the deploy fails with an empty app name.
- **Python version**: the workflow's `PYTHON_VERSION` must match the app's worker runtime (`FUNCTIONS_WORKER_RUNTIME_VERSION`, 3.13 here) so the vendored cp313 wheels line up with the interpreter on Azure.
- **Storage auth**: with identity-based storage the function app's managed identity needs `Storage Blob Data Contributor` on the account — the portal assigns this when you attach application storage during creation. The blob trigger additionally needs `Storage Queue Data Contributor` + `Storage Table Data Contributor` (see step 4).
- The Azure AI Vision resource is no longer needed; create one (ComputerVision, `F0`) only if you re-enable the commented vision path. Microsoft announced Image Analysis 4.0 retirement for September 2028.

---

## 4. Local testing (no VM)

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

## 5. Full workflow

1. Provision the Azure VM and open port 80 (once).
2. *(optional, serverless mode)* Set up the Azure Function, its containers and the CI/CD secret (section 3), then pick a `SEND_MODE` in `deploy.env` (section 2).
3. Commit and push code changes to Git — pushing `azure-function/**` also redeploys the function via GitHub Actions.
4. `./deploy.sh` — deploy or update the server.
5. `cd sender && uv run python sender.py` — start collecting counts.
6. Open `sender/index.html` locally, and `http://<VM_PUBLIC_IP>/` for the public dashboard and metrics. In `azure-blob` mode check the `results` container in the storage account instead — the local dashboard stays empty.

---

## 6. Troubleshooting

Live-debug commands (App Insights queries, Event Grid subscription checks, health probes, blob container inspection) are collected in [`azure-debug.md`](azure-debug.md), including a symptom → meaning table for the failure modes below.

| Symptom | Cause | Fix |
|---|---|---|
| `failed to bind host port 0.0.0.0:80/tcp: address already in use` | Apache preinstalled on Ubuntu holds port 80 | `sudo systemctl disable --now apache2` (or purge it) |
| VM freezes / SSH times out after first request | VM too small for PyTorch CPU inference | resize to ≥ 4 GB RAM (`Standard_B2s` or larger) and restart |
| `Download failure ... Environment may be offline` in container logs | transient `yolov8n.pt` download failure | `docker compose up -d --force-recreate`; if it persists, check container DNS |
| Sender hangs with no output | camera allows only one client, or stream slow to start | close other consumers of the stream; first frames may take a few seconds |
| `Unit apache2.service could not be found` in deploy output | expected after Apache was removed | harmless |
| Dashboard `database.csv` chart empty | sender not running or no rows yet | start the sender; the chart skips the CSV header row |
| CI deploy fails with `502 Bad Gateway` at zipdeploy | `functions-action` used the classic Kudu zipdeploy, unsupported on Flex | keep `sku: flexconsumption` on the deploy step |
| CI deploy fails with `401` fetching Kudu settings | stale publish profile (downloaded while SCM basic auth was disabled) | re-enable basic auth, re-download the profile, update the GitHub secret |
| `ModuleNotFoundError` on Azure (e.g. `cv2`) but tests pass locally | dependencies missing from the package or in the wrong path | deps must be vendored into `.python_packages/lib/site-packages` (Flex layout, no `python3.x` segment) before the deploy step |
| `No module named 'pkg_resources'` + old ultralytics layout in Azure logs | pip resolved a legacy `ultralytics` release instead of 8.4.x | `ultralytics` is pinned in `requirements.txt` — keep it pinned |
| `KeyBasedAuthenticationNotPermitted` from `az storage` commands | shared key access disabled on the account (Flex default) | use `--auth-mode login` for az commands; enable keys only for the sender's `azure-blob` mode |
| Blob uploads sit unprocessed | polling-based blob trigger (Flex runs only the Event Grid one), or zero always-ready instances, or missing event subscription | ensure `source="EventGrid"` on the trigger, Always ready instances ≥ 1, and the Blob Created event subscription exists (see §3 step 4) |
| Function app 503s on everything, `An unhandled exception has occurred. Host is shutting down.` in App Insights right after host start | app identity lacks `Storage Queue Data Contributor` — the Event Grid blob trigger hits the Queue service at startup and 403s (`AuthorizationPermissionMismatch` from `Windows-Azure-Queue`) | grant the queue/table roles to the app identity, then restart (see §3 step 4) |