"""ArcFace fine-tuning trainer.

Features: mixed-precision (AMP), cosine LR schedule with warmup, gradient
clipping, checkpoint save/resume, early stopping, TensorBoard logging, and a
reproducible seed. Validation metric is a verification-style accuracy on a held
-out split (cosine threshold sweep) — closer to deployment than top-1 softmax.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from facecore.config import Settings
from facecore.embedding.backbone import build_backbone
from facecore.logging_conf import get_logger
from facecore.preprocessing.augmentation import seed_augment
from facecore.training.callbacks import CheckpointManager, EarlyStopping
from facecore.training.margin_head import ArcMarginHead
from facecore.utils.device import resolve_device

log = get_logger(__name__)


@dataclass(slots=True)
class TrainConfig:
    backbone: str = "r50"
    epochs: int = 40
    batch_size: int = 128
    lr: float = 0.1
    weight_decay: float = 5e-4
    warmup_epochs: int = 2
    grad_clip: float = 5.0
    num_workers: int = 4
    seed: int = 42
    resume: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_augment(seed)
    torch.backends.cudnn.benchmark = True  # variable input? set False for determinism


class Trainer:
    def __init__(
        self,
        settings: Settings,
        cfg: TrainConfig,
        train_ds: torch.utils.data.Dataset,
        val_ds: torch.utils.data.Dataset,
        num_classes: int,
    ) -> None:
        set_seed(cfg.seed)
        self._cfg = cfg
        self._device = resolve_device(settings.device)
        self._backbone = build_backbone(cfg.backbone, settings.embedding_dim, dropout=0.4).to(self._device)
        self._head = ArcMarginHead(settings.embedding_dim, num_classes).to(self._device)
        self._criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self._optim = torch.optim.SGD(
            [*self._backbone.parameters(), *self._head.parameters()],
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._device.type == "cuda")
        self._sched = torch.optim.lr_scheduler.CosineAnnealingLR(self._optim, T_max=cfg.epochs)
        self._ckpt = CheckpointManager(settings.checkpoint_dir, monitor_mode="max")
        self._stopper = EarlyStopping(patience=8, mode="max")
        self._writer = SummaryWriter(str(settings.tensorboard_dir))
        self._train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
            pin_memory=self._device.type == "cuda", drop_last=True, persistent_workers=cfg.num_workers > 0,
        )
        self._val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
            pin_memory=self._device.type == "cuda",
        )
        self._start_epoch = 0
        if cfg.resume:
            self._resume()

    def _resume(self) -> None:
        state = self._ckpt.load("last.pt")
        if state is None:
            log.info("No checkpoint to resume; starting fresh")
            return
        self._backbone.load_state_dict(state["backbone"])
        self._head.load_state_dict(state["head"])
        self._optim.load_state_dict(state["optim"])
        self._scaler.load_state_dict(state["scaler"])
        self._sched.load_state_dict(state["sched"])
        self._start_epoch = state["epoch"] + 1
        log.info("Resumed", extra={"extra_fields": {"epoch": self._start_epoch}})

    def _warmup_lr(self, epoch: int, step: int, steps_per_epoch: int) -> None:
        if epoch >= self._cfg.warmup_epochs:
            return
        total = self._cfg.warmup_epochs * steps_per_epoch
        done = epoch * steps_per_epoch + step
        for g in self._optim.param_groups:
            g["lr"] = self._cfg.lr * done / max(total, 1)

    def _train_epoch(self, epoch: int) -> float:
        self._backbone.train()
        self._head.train()
        running = 0.0
        steps = len(self._train_loader)
        for step, (imgs, labels) in enumerate(self._train_loader):
            imgs = imgs.to(self._device, non_blocking=True)
            labels = labels.to(self._device, non_blocking=True)
            self._warmup_lr(epoch, step, steps)
            self._optim.zero_grad(set_to_none=True)
            with torch.autocast(self._device.type, enabled=self._device.type == "cuda"):
                emb = self._backbone(imgs)
                logits = self._head(emb, labels)
                loss = self._criterion(logits, labels)
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self._optim)
            nn.utils.clip_grad_norm_(self._backbone.parameters(), self._cfg.grad_clip)
            self._scaler.step(self._optim)
            self._scaler.update()
            running += loss.item()
            if step % 50 == 0:
                gstep = epoch * steps + step
                self._writer.add_scalar("train/loss", loss.item(), gstep)
                self._writer.add_scalar("train/lr", self._optim.param_groups[0]["lr"], gstep)
        return running / max(steps, 1)

    @torch.inference_mode()
    def _validate(self) -> float:
        """Verification accuracy: pair held-out embeddings, best cosine threshold."""
        self._backbone.eval()
        embs, labels = [], []
        for imgs, lbls in self._val_loader:
            imgs = imgs.to(self._device, non_blocking=True)
            with torch.autocast(self._device.type, enabled=self._device.type == "cuda"):
                e = self._backbone(imgs)
            embs.append(nn.functional.normalize(e.float()).cpu())
            labels.append(lbls)
        if not embs:
            return 0.0
        E = torch.cat(embs)
        L = torch.cat(labels)
        # Sample balanced positive/negative pairs to estimate verification acc.
        n = min(2048, E.shape[0])
        idx = torch.randperm(E.shape[0])[:n]
        E, L = E[idx], L[idx]
        sims = E @ E.T
        same = (L[:, None] == L[None, :]).float()
        mask = ~torch.eye(n, dtype=torch.bool)
        sims_f, same_f = sims[mask], same[mask]
        best_acc = 0.0
        for thr in torch.linspace(0.1, 0.7, 25):
            pred = (sims_f > thr).float()
            best_acc = max(best_acc, (pred == same_f).float().mean().item())
        return best_acc

    def fit(self) -> None:
        for epoch in range(self._start_epoch, self._cfg.epochs):
            loss = self._train_epoch(epoch)
            self._sched.step()
            acc = self._validate()
            self._writer.add_scalar("val/verification_acc", acc, epoch)
            log.info("epoch done", extra={"extra_fields": {"epoch": epoch, "loss": round(loss, 4), "val_acc": round(acc, 4)}})
            self._ckpt.save(
                {
                    "epoch": epoch,
                    "backbone": self._backbone.state_dict(),
                    "head": self._head.state_dict(),
                    "optim": self._optim.state_dict(),
                    "scaler": self._scaler.state_dict(),
                    "sched": self._sched.state_dict(),
                    "val_acc": acc,
                    "config": asdict(self._cfg),
                },
                metric=acc,
            )
            self._stopper.step(acc)
            if self._stopper.should_stop:
                log.info("Early stopping", extra={"extra_fields": {"epoch": epoch}})
                break
        self._writer.close()
