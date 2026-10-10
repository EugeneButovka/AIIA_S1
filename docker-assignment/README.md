# AIIA — Deployment of Deep Learning Architectures in Docker Containers

Two logically separated containers orchestrated with Docker Compose:

- **`inference`** — FastAPI service running the pre-trained
  `distilbert/distilbert-base-uncased-finetuned-sst-2-english` sentiment model.
  The model is baked into the image at build time. Includes a `train.py` script
  so the same image can fine-tune the model for a few epochs.
- **`gui`** — Gradio web interface. Contains no ML dependencies; it only
  forwards HTTP requests to the inference container.

## Quick start

```bash
docker compose build
docker compose up -d
docker compose ps          # wait until aiia-inference is (healthy)
```

- API docs (Swagger): http://localhost:8000/docs
- GUI: http://localhost:7860

## Test inference from the terminal

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Docker containers are awesome!"}'
```

## Optional fine-tuning (few epochs) inside the inference container

```bash
docker compose exec inference python train.py --epochs 1 --train-samples 2000
docker compose up -d --force-recreate inference   # nothing else restarts
# serve the fine-tuned model:
MODEL_PATH=/models/finetuned-sst2 docker compose up -d --force-recreate inference
```
(the `MODEL_PATH` line needs `MODEL_PATH=/models/finetuned-sst2` added under the
`inference.environment` section of `docker-compose.yml` or exported via an
override file)

## GPU hosts (NVIDIA Container Toolkit)

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The override swaps the CPU-only PyTorch wheel for the CUDA build
(`https://download.pytorch.org/whl/cu130`) and reserves all NVIDIA GPUs for
the inference container. Inference and training then run on GPU automatically.

## Layout

```
docker-assignment/
├── docker-compose.yml         # resource limits, healthchecks, network
├── docker-compose.gpu.yml     # NVIDIA GPU override
├── inference/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── train.py               # optional few-epoch fine-tuning
│   └── app/main.py           # FastAPI inference server
└── gui/
    ├── Dockerfile
    ├── requirements.txt
    └── app/main.py           # Gradio interface
```

The full assignment report is in [REPORT.md](REPORT.md).