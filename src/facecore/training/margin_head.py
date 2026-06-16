"""ArcFace (additive angular margin) classification head for training.

During training we attach this head to the IResNet backbone. It enforces an
angular margin between classes, producing embeddings that cluster tightly per
identity and separate across identities — exactly what cosine matching needs.
At inference the head is discarded; only the backbone embeddings are used.

Ref: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face
Recognition" (CVPR 2019).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 64.0, margin: float = 0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)
        self._cos_m = math.cos(margin)
        self._sin_m = math.sin(margin)
        self._th = math.cos(math.pi - margin)
        self._mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi = cosine * self._cos_m - sine * self._sin_m  # cos(theta + m)
        # Keep monotonicity for theta + m > pi (the easy-margin / hard cases).
        phi = torch.where(cosine > self._th, phi, cosine - self._mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale
        return logits
