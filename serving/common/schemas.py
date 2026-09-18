"""Pydantic request/response schemas shared by the serving APIs."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # IMDB reviews go up to ~13k characters; cap well above that to only
    # guard against abusive payloads, not reject real reviews.
    text: str = Field(..., min_length=1, max_length=20000)


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    model_version: str
    latency_ms: float


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_version: str


class ModelInfoResponse(BaseModel):
    model_version: str
    model_uri: str
    model_type: str
