"""FastAPI app serving the v2 sentiment model (fine-tuned DistilBERT)."""

import os
import time
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI, HTTPException

from common.request_log import log_prediction
from common.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from v2.model import SentimentModel

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_URI = os.environ.get("MODEL_URI", "models:/imdb-sentiment-v2/2")
MODEL_VERSION = "v2"
MODEL_TYPE = "distilbert"

model: SentimentModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = SentimentModel(MODEL_URI)
    yield


app = FastAPI(title="Sentiment API v2", version="1.0.0", lifespan=lifespan)


def get_model() -> SentimentModel:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return model


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    status = "ok" if model is not None else "loading"
    return HealthResponse(status=status, model_version=MODEL_VERSION)


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    get_model()
    return ModelInfoResponse(
        model_version=MODEL_VERSION,
        model_uri=MODEL_URI,
        model_type=MODEL_TYPE,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    active_model = get_model()

    start = time.perf_counter()
    sentiment, confidence = active_model.predict(request.text)
    latency_ms = (time.perf_counter() - start) * 1000
    log_prediction(request.text, sentiment, confidence, MODEL_VERSION, latency_ms)

    return PredictResponse(
        sentiment=sentiment,
        confidence=confidence,
        model_version=MODEL_VERSION,
        latency_ms=latency_ms,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    active_model = get_model()

    predictions = []
    for text in request.texts:
        start = time.perf_counter()
        sentiment, confidence = active_model.predict(text)
        latency_ms = (time.perf_counter() - start) * 1000
        log_prediction(text, sentiment, confidence, MODEL_VERSION, latency_ms)
        predictions.append(
            PredictResponse(
                sentiment=sentiment,
                confidence=confidence,
                model_version=MODEL_VERSION,
                latency_ms=latency_ms,
            )
        )

    return BatchPredictResponse(predictions=predictions)
