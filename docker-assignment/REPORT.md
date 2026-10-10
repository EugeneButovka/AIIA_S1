# Deployment of Deep Learning Architectures in Docker Containers

**Course:** AIIA — Deep Learning Deployment Assignment
## Student: Evgenii Butovka
### Date: October 10th, 2026
### Code: https://github.com/EugeneButovka/AIIA_S1/tree/main/docker-assignment

---

## Table of Contents

1. [Objective and Scope](#1-objective-and-scope)
2. [Selected Model](#2-selected-model)
3. [System Architecture](#3-system-architecture)
4. [Inference Container](#4-inference-container)
5. [GUI Container](#5-gui-container)
6. [Docker Compose and Run Parameters](#6-docker-compose-and-run-parameters)
7. [Deployment and Results](#7-deployment-and-results)
8. [Improvements Beyond the Requirements](#8-improvements-beyond-the-requirements)
9. [Conclusions](#9-conclusions)

---

## 1. Objective and Scope

This report documents the assembly, configuration and deployment of a Deep Learning
inference service inside Docker containers. All code, Dockerfiles and Compose files
described here are available in the public repository:
`https://github.com/EugeneButovka/AIIA_S1/tree/main/docker-assignment`.
The deliverable is a Docker Compose stack
made of **two logically separated containers**:

- an **inference container** that serves a pre-trained model through a REST API
  (FastAPI), and
- a **GUI container** (Gradio) that contains **no ML dependencies** and performs
  inference exclusively by calling the inference container over an internal Docker
  network.

Beyond the mandatory local + GUI inference, the inference image also ships a
`train.py` script and the `datasets`/`accelerate` libraries, so the **same container
was used to fine-tune the model for one epoch** and to serve the resulting checkpoint.
A GPU override file makes the stack NVIDIA-ready.

## 2. Selected Model

**`distilbert/distilbert-base-uncased-finetuned-sst-2-english`** — DistilBERT fine-tuned
for binary sentiment classification on the SST-2 (Stanford Sentiment Treebank) corpus.

DistilBERT is a 6-layer, 66M-parameter Transformer distilled from BERT-base using
knowledge distillation, retaining ~97 % of BERT's language-understanding performance
while being ~40 % smaller and ~60 % faster (Sanh et al., 2019). It was chosen because
it is small enough for fast CPU inference, is one of the most widely used text
classification models, and its supervised head returns `POSITIVE`/`NEGATIVE`
probabilities, which makes end-to-end verification easy. The tokenizer uses WordPiece
sub-word tokens with a 512-position limit; inputs are truncated accordingly.

## 3. System Architecture

```
            build time (docker compose build)          runtime (docker compose up)
  ┌─────────────────────────────────────┐
  │ 1. OS packages (curl)               │        browser / curl (host)
  │ 2. pip: torch → requirements        │            │ :7860            │ :8000 (tests only)
  │ 3. DistilBERT weights ── baked in ──┼──►  ┌──────▼─────────┐  ┌─────▼──────────────┐
  │ 4. application code (last)          │     │ aiia-gui       │  │ aiia-inference     │
  └─────────────────────────────────────┘     │ Gradio 6       │  │ FastAPI + uvicorn  │
                                              │ gradio+httpx   │  │ DistilBERT SST-2   │
                                              │ no ML deps     │  │ + train.py         │
                                              └──────┬─────────┘  └──────────┬─────────┘
                                                     │ POST /predict (internal DNS "inference")
                                                     ▼                       │
                            ═══════════════ bridge network "dl-net" ══════════╪═══
                                                                             ▼
                                            named volume "models:/models" (persistent checkpoints)
```

- **`aiia-inference`** — owns the model and all ML dependencies (PyTorch,
  Transformers). Exposes `GET /health`, `POST /predict`, `POST /predict/batch` and
  auto-generated Swagger docs at `/docs`. The pre-trained weights are **baked into the
  image** at build time, so the container starts offline in ~5 s.
- **`aiia-gui`** — a thin presentation layer. Its Dockerfile installs only
  `gradio` and `httpx`; there is no model or ML library in this image. It talks to
  the inference container through the service name `inference` resolved by Docker's
  internal DNS on the `dl-net` bridge network.
- **`models` named volume** — persists fine-tuned checkpoints across container
  recreation, so a model trained once can be served by any later container.

## 4. Inference Container

`inference/Dockerfile` (complete):

```dockerfile
FROM python:3.12-slim

ARG TORCH_VERSION=2.9.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG MODEL_ID=distilbert/distilbert-base-uncased-finetuned-sst-2-english

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    MODEL_ID=${MODEL_ID}

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}" \
    && pip install -r requirements.txt

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /opt/hf /models \
    && chown -R appuser:appuser /opt/hf /models

USER appuser

RUN python -c "import os; from transformers import pipeline; pipeline('sentiment-analysis', model=os.environ['MODEL_ID'])"

COPY train.py ./
COPY app ./app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Key design decisions:

- **Order of layers for cache efficiency.** Instructions are sorted by change
  frequency: OS packages → dependency files → heavy downloads → application code.
  `requirements.txt` is copied *before* the code, so editing the app never reinstalls
  dependencies; the model download layer sits *before* `COPY app`, so a code change
  never re-downloads the 268 MB checkpoint. PyTorch is installed in its own `RUN`
  (from a configurable index) before `requirements.txt`, so the largest wheel is
  re-downloaded only when the pinned version changes. In the measured session, an
  empty rebuild reported **12 layers `CACHED` and finished in ~2 s**.
- **Model baked into the image** (`RUN python -c "...pipeline(...)"`) — the
  resulting container needs no internet access at start-up and the ~5 s cold start
  loads weights from `/opt/hf` inside the image.
- **Non-interactive installs** — `DEBIAN_FRONTEND=noninteractive`,
  `apt-get -y --no-install-recommends`, `rm -rf /var/lib/apt/lists/*` (image size),
  `PIP_NO_CACHE_DIR=1`, and fully pinned dependency versions
  (`torch==2.9.1`, `transformers==5.19.0`, `fastapi==0.143.0`, `uvicorn[standard]==0.54.0`,
  `datasets==5.1.0`, `accelerate==1.15.0`) for reproducible builds.
- **CPU/GPU portability via build args** — by default the slim CPU-only PyTorch wheel
  is used; passing `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130` builds the
  identical stack against CUDA 13.0 (see §6).
- **Runs as unprivileged user** (`appuser`), and a `.dockerignore` excludes
  `.git`, caches and the Dockerfiles themselves from the build context.
- **`HEALTHCHECK`** using `curl` (the only extra OS package) — enables the
  `service_healthy` gate in Compose.

The API (`inference/app/main.py`) loads the pipeline **once** in FastAPI's lifespan
handler, selects `cuda`/`cpu` automatically, validates input with Pydantic
(`text: 1..2048` chars, batch ≤ 16) and returns the label, the score and the
inference latency. `MODEL_PATH` overrides `MODEL_ID`, so any local checkpoint
(e.g. from the training demo) can be served without rebuilding. Core excerpt:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state["pipe"] = pipeline("sentiment-analysis", model=MODEL_PATH or MODEL_ID, device=device)
    yield
    state.clear()

app = FastAPI(title="DistilBERT SST-2 Inference Service", lifespan=lifespan)

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    started = time.perf_counter()
    result = state["pipe"]([request.text], truncation=True)[0]
    return PredictResponse(label=result["label"], score=round(result["score"], 4),
                           elapsed_ms=round((time.perf_counter() - started) * 1000, 1))
```

## 5. GUI Container

`gui/Dockerfile` (complete) mirrors the inference conventions but deliberately
contains **no ML dependencies**, resulting in a 642 MB image versus 2.23 GB:

```dockerfile
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY app ./app

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/', timeout=5)" || exit 1

CMD ["python", "app/main.py"]
```

The only pinned dependencies are `gradio==6.30.0` and `httpx==0.28.1`; the
healthcheck reuses the Python interpreter instead of installing extra OS packages. The interface
(`gui/app/main.py`) is a single `gr.Interface` with a text box, worked examples and a
`gr.Label` confidence display; `classify()` simply does:

```python
response = httpx.post(f"{INFERENCE_URL}/predict", json={"text": text}, timeout=60.0)
```

where `INFERENCE_URL` defaults to `http://inference:8000` (Docker's service DNS).
Failures are surfaced to the user through `gr.Error`, so stopping the inference
container produces a clear message in the GUI instead of a crash.

## 6. Docker Compose and Run Parameters

`docker-compose.yml` (complete):

```yaml
name: aiia-docker-assignment

services:
  inference:
    build:
      context: ./inference
      args:
        TORCH_VERSION: "2.9.1"
        TORCH_INDEX_URL: https://download.pytorch.org/whl/cpu
        MODEL_ID: distilbert/distilbert-base-uncased-finetuned-sst-2-english
    image: aiia/dl-inference:latest
    container_name: aiia-inference
    environment:
      MODEL_ID: distilbert/distilbert-base-uncased-finetuned-sst-2-english
    shm_size: 512m
    deploy:
      resources:
        limits:
          memory: 3g
          cpus: "2.0"
    ports:
      - "8000:8000"
    volumes:
      - models:/models
    networks:
      - dl-net
    restart: unless-stopped

  gui:
    build:
      context: ./gui
    image: aiia/dl-gui:latest
    container_name: aiia-gui
    environment:
      INFERENCE_URL: http://inference:8000
    shm_size: 256m
    deploy:
      resources:
        limits:
          memory: 1g
          cpus: "1.0"
    ports:
      - "7860:7860"
    depends_on:
      inference:
        condition: service_healthy
    networks:
      - dl-net
    restart: unless-stopped

networks:
  dl-net:
    driver: bridge

volumes:
  models:
```

Run parameters explained:

| Parameter | Value | Purpose |
|---|---|---|
| `deploy.resources.limits.memory` | 3 GB (inference), 1 GB (GUI) | hard RAM cap per container |
| `deploy.resources.limits.cpus` | 2.0 (inference), 1.0 (GUI) | CPU quota; the GUI cannot starve inference |
| `shm_size` | 512 MB / 256 MB | shared memory for PyTorch's dataloader workers, `/dev/shm` is only 64 MB by default |
| `depends_on: service_healthy` | — | the GUI starts **only after** the model is loaded and `/health` returns 200 |
| `restart: unless-stopped` | — | automatic recovery after crashes or host reboots |
| `networks: dl-net` | bridge | isolated internal network; containers resolve each other by service name |
| `volumes: models:/models` | named volume | trained checkpoints survive container recreation |
| `ports` | 8000, 7860 | 8000 is published only for the terminal tests; the GUI uses the internal network, so it could be unpublished |

`docker-compose.gpu.yml` is an override for NVIDIA hosts (NVIDIA Container Toolkit):

```yaml
services:
  inference:
    build:
      args:
        TORCH_INDEX_URL: https://download.pytorch.org/whl/cu130
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

`docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build` builds
the CUDA PyTorch wheel, reserves **all GPUs** for the inference container, and both
inference and training switch to GPU automatically (`device="cuda"`,
`use_cpu=False`). The merged configuration was validated with
`docker compose config` (the host used for this report has no NVIDIA GPU, so the
override is documented and syntax-verified rather than executed).

## 7. Deployment and Results

Environment: Docker 29.4.1 with Compose v5.1.3 on Ubuntu 24.04 (Linux VM on macOS via
colima, aarch64, 12 GB RAM). Commands are run from the project root.

**Build and start** (~6 min first build; the GUI waits for the model to load):

```console
$ docker compose build
$ docker compose up -d
 Container aiia-inference Healthy
 Container aiia-gui Started
$ docker compose ps
NAME             SERVICE     STATUS                   PORTS
aiia-gui         gui         Up 4 minutes (healthy)   0.0.0.0:7860->7860/tcp
aiia-inference   inference   Up 17 seconds (healthy)  0.0.0.0:8000->8000/tcp
```

**Inference from the terminal (local API):**

```console
$ curl -s http://localhost:8000/health
{"status":"ok","model":"distilbert/distilbert-base-uncased-finetuned-sst-2-english","device":"cpu"}

$ curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "Docker containers make deploying deep learning models incredibly easy!"}'
{"label":"POSITIVE","score":0.9899,"elapsed_ms":43.4}

$ curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "The deployment failed and the documentation is confusing."}'
{"label":"NEGATIVE","score":0.9998,"elapsed_ms":60.3}
```

A batch endpoint (`/predict/batch`, ≤ 16 texts) behaves equivalently and returns
one result per text (~28 ms each in the test run).

**Inference through the GUI.** The browser interface is at `http://localhost:7860`
(text box + examples + confidence bars). Using Gradio's HTTP API to prove that the
GUI container really forwards requests to the inference container:

```console
$ curl -s -X POST http://localhost:7860/gradio_api/call/predict \
  -H "Content-Type: application/json" \
  -d '{"data": ["The Docker Compose setup works flawlessly!"]}'
{"event_id": "c50d9c50d33144bab265d1d1d877d1e5"}

$ curl -s -N http://localhost:7860/gradio_api/call/predict/c50d9c50d33144bab265d1d1d877d1e5
event: complete
data: [{"label": "POSITIVE", "confidences": [{"label": "POSITIVE", "confidence": 0.9998},
       {"label": "NEGATIVE", "confidence": 0.0002}]}, "26.9 ms"]
```

The same request made **from inside the GUI container** over the internal network
(service DNS name `inference`) confirms the logical separation:

```console
$ docker compose exec gui python -c "import httpx; print(httpx.post('http://inference:8000/predict',
  json={'text': 'request routed through the internal bridge network'}).json())"
{'label': 'NEGATIVE', 'score': 0.8192, 'elapsed_ms': 18.7}
```

**Optional training (few epochs) in the same container.** The image ships
`train.py`, which fine-tunes the served model on SST-2 (`nyu-mll/glue`) with the
Transformers `Trainer` and stores the checkpoint on the `models` volume:

```console
$ docker compose exec inference python train.py --epochs 1 --train-samples 2000
{'loss': '0.08863', 'grad_norm': '0.0343', 'learning_rate': '1.216e-05', 'epoch': '0.4'}
...
{'train_runtime': '137.8', 'train_samples_per_second': '14.51',
 'train_steps_per_second': '0.907', 'train_loss': '0.06396', 'epoch': '1'}
fine-tuned model saved to /models/finetuned-sst2
```

The checkpoint was verified and then **served by a fresh container** using the
`MODEL_PATH` environment variable, demonstrating that the volume keeps trained
artifacts across container recreation:

```console
$ docker compose run -d --name aiia-inference-ft --service-ports \
  -e MODEL_PATH=/models/finetuned-sst2 inference
$ curl -s http://localhost:8000/health
{"status":"ok","model":"/models/finetuned-sst2","device":"cpu"}
$ curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "Serving a freshly fine-tuned checkpoint from a named volume."}'
{"label":"POSITIVE","score":0.9978,"elapsed_ms":48.6}
```

**Resource usage under the configured limits** (`docker stats --no-stream`):

```console
NAME             MEM USAGE / LIMIT   CPU %
aiia-inference   274.5MiB / 3GiB     0.29 %
aiia-gui         109.7MiB / 1GiB     9.25 %
```

## 8. Improvements Beyond the Requirements

1. **Layer ordering for cache and size** — torch / requirements / model-download /
   code are separate layers sorted by change frequency (measured: empty rebuild
   fully `CACHED`, ~2 s); `.dockerignore` keeps the context minimal.
2. **CPU/GPU-portable build** — a single `TORCH_INDEX_URL` build argument switches
   between the slim CPU wheel and the CUDA 13.0 build, combined with a
   `docker-compose.gpu.yml` override that reserves all NVIDIA GPUs.
3. **Baked-in model weights** — the container starts offline in ~5 s; no
   network dependency or cold-download latency at runtime.
4. **Healthcheck-gated start-up** — the GUI cannot start before the model is
   loaded (`condition: service_healthy`), eliminating start-order race conditions.
5. **Security** — both containers run as non-root; the inference port can be
   unpublished since the GUI communicates over the internal bridge network.
6. **Named volume for checkpoints** — models trained inside the container persist
   and can be served by any later container via `MODEL_PATH`.
7. **Batch endpoint, Swagger UI (`/docs`), latency reporting, restart policies,
   `shm_size` sized for PyTorch dataloaders.**
8. **Hugging Face Spaces readiness** — Spaces supports Docker SDK images; the
   inference container is deployable there almost as-is (HF's default app port is
   7860, overridden to 8000 via the documented `app_port` key in the Space's README
   metadata, so no code change is required). Note that publishing on the Docker SDK
   now requires a paid HF plan (PRO / Team): since the platform's policy change,
   Gradio and Docker Spaces run on metered compute and are no longer free to create,
   with only the free-tier exception of up to two Gradio Spaces on ZeroGPU hardware.
   A cost-free alternative is therefore a Gradio SDK Space with the pipeline embedded,
   which still exposes public API inference through the same
   `/gradio_api/call/predict` flow used in §7. Publishing was ultimately not
   performed because the assignment allows local inference.

## 9. Conclusions

The assignment's goals were met in full: a pre-trained DistilBERT sentiment model
performs inference inside a Docker container both through a REST API and through a
graphical interface, with strict container-level separation between the GUI and the
model. Docker Compose orchestrates the two services with health-gated start-up,
memory/CPU limits, shared-memory sizing and an isolated bridge network. The optional
training step was executed inside the same image, its checkpoint persisted on a
volume and served back through the same API, and a documented override file makes
the stack ready for NVIDIA GPUs and Hugging Face Spaces deployment.

---

*References: V. Sanh, L. Debut, J. Champlain, T. Wolf et al., "DistilBERT, a distilled
version of BERT: smaller, faster, cheaper and lighter", NeurIPS EMC² Workshop, 2019.
Docker documentation, https://docs.docker.com. Docker Compose documentation,
https://docs.docker.com/compose.*