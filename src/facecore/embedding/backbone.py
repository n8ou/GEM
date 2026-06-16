"""IResNet backbone (ArcFace's reference architecture) used for training & export.

This is the trainable embedding network. For pure inference we can also use the
pre-exported InsightFace ONNX model (see arcface.py); for *custom training* we
fine-tune this backbone with a margin head (training/margin_head.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv3x3(in_c: int, out_c: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c: int, out_c: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c, eps=1e-5)
        self.conv1 = conv3x3(in_c, out_c)
        self.bn2 = nn.BatchNorm2d(out_c, eps=1e-5)
        self.prelu = nn.PReLU(out_c)
        self.conv2 = conv3x3(out_c, out_c, stride)
        self.bn3 = nn.BatchNorm2d(out_c, eps=1e-5)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class IResNet(nn.Module):
    """IResNet-{18,34,50,100}. 112x112 input -> `embedding_dim` vector."""

    def __init__(self, layers: list[int], embedding_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.in_c = 64
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * 7 * 7, embedding_dim)
        self.features = nn.BatchNorm1d(embedding_dim, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
        self._init_weights()

    def _make_layer(self, out_c: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = nn.Sequential(
            nn.Conv2d(self.in_c, out_c, 1, stride, bias=False),
            nn.BatchNorm2d(out_c, eps=1e-5),
        )
        layers = [IBasicBlock(self.in_c, out_c, stride, downsample)]
        self.in_c = out_c
        layers += [IBasicBlock(out_c, out_c) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.features(x)  # un-normalized; matcher normalizes


_CONFIGS = {"r18": [2, 2, 2, 2], "r34": [3, 4, 6, 3], "r50": [3, 4, 14, 3], "r100": [3, 13, 30, 3]}


def build_backbone(name: str = "r50", embedding_dim: int = 512, dropout: float = 0.0) -> IResNet:
    key = name.replace("arcface_", "")
    if key not in _CONFIGS:
        raise ValueError(f"unknown backbone {name!r}; choices: {list(_CONFIGS)}")
    return IResNet(_CONFIGS[key], embedding_dim=embedding_dim, dropout=dropout)
