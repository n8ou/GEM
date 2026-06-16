"""Silent-Face passive anti-spoofing (MiniFASNet ensemble).

Single-RGB-camera presentation-attack detection — no extra hardware, no user
action. Each detected face is cropped at two context scales (2.7 and 4.0) and
fed to the matching MiniFASNet; the two 3-class softmaxes are averaged and the
"real" class (index 1) probability is returned as the live score.

Model + weights from minivision-ai/Silent-Face-Anti-Spoofing (Apache-2.0).
Filenames encode the crop scale / input size / architecture, e.g.
`2.7_80x80_MiniFASNetV2.pth` and `4_0_0_80x80_MiniFASNetV1SE.pth`.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from facecore.domain.entities import BBox
from facecore.liveness.base import LivenessDetector
from facecore.liveness.minifasnet import MODEL_MAPPING
from facecore.logging_conf import get_logger

log = get_logger(__name__)


def _parse_model_name(name: str) -> tuple[int, int, str, float | None]:
    """`2.7_80x80_MiniFASNetV2.pth` -> (80, 80, 'MiniFASNetV2', 2.7)."""
    info = name.split("_")[0:-1]
    h_input, w_input = info[-1].split("x")
    model_type = name.split(".pth", maxsplit=1)[0].rsplit("_", maxsplit=1)[-1]
    scale = None if info[0] == "org" else float(info[0])
    return int(h_input), int(w_input), model_type, scale


def _get_kernel(h: int, w: int) -> tuple[int, int]:
    return (h + 15) // 16, (w + 15) // 16


def _crop(org_img: np.ndarray, bbox_xywh: tuple[int, int, int, int], scale: float,
          out_w: int, out_h: int) -> np.ndarray:
    """Context-aware square-ish crop around the face, clamped to image bounds."""
    src_h, src_w = org_img.shape[:2]
    x, y, box_w, box_h = bbox_xywh
    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
    new_w, new_h = box_w * scale, box_h * scale
    cx, cy = box_w / 2 + x, box_h / 2 + y
    lt_x, lt_y = cx - new_w / 2, cy - new_h / 2
    rb_x, rb_y = cx + new_w / 2, cy + new_h / 2
    if lt_x < 0:
        rb_x -= lt_x
        lt_x = 0
    if lt_y < 0:
        rb_y -= lt_y
        lt_y = 0
    if rb_x > src_w - 1:
        lt_x -= rb_x - src_w + 1
        rb_x = src_w - 1
    if rb_y > src_h - 1:
        lt_y -= rb_y - src_h + 1
        rb_y = src_h - 1
    patch = org_img[int(lt_y):int(rb_y) + 1, int(lt_x):int(rb_x) + 1]
    return cv2.resize(patch, (out_w, out_h))


class SilentFaceLiveness(LivenessDetector):
    def __init__(self, model_dir: Path, device: torch.device) -> None:
        self._device = device
        weights = sorted(Path(model_dir).glob("*.pth"))
        if not weights:
            raise FileNotFoundError(
                f"No anti-spoof .pth weights in {model_dir}. Run scripts/fetch_models.py."
            )
        self._models: list[tuple[float, int, int, torch.nn.Module]] = []
        for w in weights:
            h, wd, mtype, scale = _parse_model_name(w.name)
            if scale is None:
                continue
            net = MODEL_MAPPING[mtype](conv6_kernel=_get_kernel(h, wd)).to(device)
            state = torch.load(str(w), map_location=device)
            # Strip DataParallel 'module.' prefixes if present.
            if next(iter(state)).startswith("module."):
                state = {k[7:]: v for k, v in state.items()}
            net.load_state_dict(state)
            net.eval()
            self._models.append((scale, h, wd, net))
        log.info("SilentFace liveness ready",
                 extra={"extra_fields": {"models": len(self._models), "device": str(device)}})

    @torch.inference_mode()
    def score(self, image_bgr: np.ndarray, bbox: BBox) -> float:
        x1, y1, x2, y2 = bbox.as_int_xyxy()
        bbox_xywh = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
        prob = np.zeros(3, dtype=np.float64)
        for scale, h, w, net in self._models:
            patch = _crop(image_bgr, bbox_xywh, scale, w, h)
            # Silent-Face's ToTensor keeps the [0,255] range (their .div(255) is
            # commented out) — HWC uint8 BGR -> CHW float, NO /255, no color swap.
            t = torch.from_numpy(patch.transpose(2, 0, 1)).float()
            t = t.unsqueeze(0).to(self._device)
            out = net.forward(t)
            prob += F.softmax(out, dim=1).cpu().numpy()[0]
        prob /= max(len(self._models), 1)
        # Class index 1 == "real"/live in the Silent-Face label scheme.
        return float(prob[1])
