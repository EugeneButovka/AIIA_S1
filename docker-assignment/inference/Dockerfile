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