"""Centralized, environment-driven configuration.

All tunables live here. Nothing is hardcoded in business logic; secrets and
paths come from the environment (12-factor). Import `get_settings()` everywhere.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DevicePref = Literal["auto", "cuda", "cpu"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FACECORE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["development", "production", "test"] = "production"
    device: DevicePref = "auto"

    # --- Models ---
    model_dir: Path = Path("./artifacts/models")
    detector: str = "retinaface"
    embedder: str = "arcface_r100"
    embedding_dim: int = 512

    # --- Recognition thresholds ---
    match_threshold: float = Field(0.42, ge=-1.0, le=1.0)
    detect_score_threshold: float = Field(0.6, ge=0.0, le=1.0)
    blur_threshold: float = Field(45.0, ge=0.0)
    min_face_size: int = Field(32, ge=8)

    # --- Vector store ---
    index_path: Path = Path("./artifacts/index/faces.faiss")
    index_meta_path: Path = Path("./artifacts/index/faces_meta.json")

    # --- API security ---
    api_key: str = "change-me-in-real-deployment"
    max_upload_mb: int = Field(8, ge=1, le=64)
    max_faces_per_image: int = Field(20, ge=1, le=200)
    rate_limit_per_min: int = Field(120, ge=1)

    # --- Training ---
    data_dir: Path = Path("./dataset")
    checkpoint_dir: Path = Path("./artifacts/checkpoints")
    tensorboard_dir: Path = Path("./artifacts/runs")

    @field_validator("match_threshold")
    @classmethod
    def _warn_threshold(cls, v: float) -> float:
        # Cosine sim on L2-normalized ArcFace embeddings is in [-1, 1].
        # Typical operating point for r100/glint360k is ~0.35-0.50.
        return v

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton. Reads env once per process."""
    return Settings()
