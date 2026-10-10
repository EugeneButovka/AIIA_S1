import os
import time
from contextlib import asynccontextmanager
from typing import Annotated
from typing import Literal

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic import Field
from transformers import pipeline

MODEL_ID = os.getenv("MODEL_ID", "distilbert/distilbert-base-uncased-finetuned-sst-2-english")
MODEL_PATH = os.getenv("MODEL_PATH")
MAX_BATCH = int(os.getenv("MAX_BATCH", "16"))

state: dict = {}


class PredictRequest(BaseModel):
    text: Annotated[str, Field(min_length=1, max_length=2048)]


class BatchPredictRequest(BaseModel):
    texts: Annotated[list[Annotated[str, Field(min_length=1, max_length=2048)]], Field(min_length=1, max_length=MAX_BATCH)]


class PredictResponse(BaseModel):
    label: Literal["POSITIVE", "NEGATIVE"]
    score: float
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_source = MODEL_PATH or MODEL_ID
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state["pipe"] = pipeline("sentiment-analysis", model=model_source, device=device)
    state["model"] = model_source
    state["device"] = device
    yield
    state.clear()


app = FastAPI(title="DistilBERT SST-2 Inference Service", version="1.0.0", lifespan=lifespan)


def run_inference(texts: list[str]) -> list[PredictResponse]:
    started = time.perf_counter()
    results = state["pipe"](texts, truncation=True)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return [
        PredictResponse(label=r["label"], score=round(r["score"], 4), elapsed_ms=round(elapsed_ms, 1))
        for r in results
    ]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model=state.get("model", "loading"), device=state.get("device", "unknown"))


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    return run_inference([request.text])[0]


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(request: BatchPredictRequest) -> list[PredictResponse]:
    return run_inference(request.texts)