"""Training callbacks: checkpointing (save/resume) and early stopping."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from facecore.logging_conf import get_logger

log = get_logger(__name__)


class CheckpointManager:
    """Saves `last.pt` every epoch and `best.pt` on metric improvement.

    Checkpoints carry full training state for exact resume: backbone, head,
    optimizer, scaler, scheduler, epoch, and best metric.
    """

    def __init__(self, ckpt_dir: Path, monitor_mode: str = "max") -> None:
        self._dir = ckpt_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._mode = monitor_mode
        self._best = -float("inf") if monitor_mode == "max" else float("inf")

    def _is_better(self, value: float) -> bool:
        return value > self._best if self._mode == "max" else value < self._best

    def save(self, state: dict[str, Any], metric: float) -> None:
        torch.save(state, self._dir / "last.pt")
        if self._is_better(metric):
            self._best = metric
            torch.save(state, self._dir / "best.pt")
            log.info("New best checkpoint", extra={"extra_fields": {"metric": round(metric, 4)}})

    def load(self, name: str = "last.pt") -> dict[str, Any] | None:
        path = self._dir / name
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu")


class EarlyStopping:
    def __init__(self, patience: int = 8, mode: str = "max", min_delta: float = 1e-4) -> None:
        self._patience = patience
        self._mode = mode
        self._min_delta = min_delta
        self._best = -float("inf") if mode == "max" else float("inf")
        self._wait = 0
        self.should_stop = False

    def step(self, value: float) -> None:
        improved = (
            value > self._best + self._min_delta
            if self._mode == "max"
            else value < self._best - self._min_delta
        )
        if improved:
            self._best = value
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self._patience:
                self.should_stop = True
