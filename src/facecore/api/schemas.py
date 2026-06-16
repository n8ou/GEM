"""Pydantic request/response models — the external API contract."""
from __future__ import annotations

from pydantic import BaseModel, Field


class BoxModel(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


class IdentityModel(BaseModel):
    person_id: str
    similarity: float
    is_known: bool


class FaceResult(BaseModel):
    bbox: BoxModel
    identity: IdentityModel
    is_live: bool = True
    live_score: float = 1.0


class RecognizeResponse(BaseModel):
    faces: list[FaceResult]
    count: int


class EnrollResponse(BaseModel):
    person_id: str
    embeddings_added: int
    index_size: int


class VerifyResponse(BaseModel):
    similarity: float
    is_match: bool
    threshold: float


class HealthResponse(BaseModel):
    status: str = "ok"
    device: str
    index_size: int
    version: str = Field(default="1.0.0")
