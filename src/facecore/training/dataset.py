"""Dataset over `dataset/<person>/<img>` with on-the-fly detect+align+validate.

Built once into a manifest (cached to JSON) so repeated epochs don't re-scan the
filesystem. Corrupt/blurry/faceless images are dropped at build time and logged.

Detection + alignment is the expensive part, so each accepted face is aligned to
the canonical 112x112 template ONCE and the crop is cached to disk (keyed by the
source path). Training then reads the cached crop directly — no detector in the
hot path — which also makes the dataset picklable for `num_workers > 0`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from facecore.config import Settings
from facecore.detection.base import FaceDetector
from facecore.logging_conf import get_logger
from facecore.preprocessing.alignment import align_face
from facecore.preprocessing.augmentation import to_model_tensor, train_transform
from facecore.preprocessing.validation import assess_quality, load_image_safe

log = get_logger(__name__)

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class Sample:
    path: str
    label: int
    cache: str  # path to the pre-aligned 112x112 BGR uint8 .npy crop


class FaceFolderDataset(Dataset):
    def __init__(
        self,
        settings: Settings,
        detector: FaceDetector,
        split: str = "train",
        manifest_path: Path | None = None,
    ) -> None:
        self._settings = settings
        self._detector: FaceDetector | None = detector
        self._train = split == "train"
        self._samples: list[Sample] = []
        self.class_to_idx: dict[str, int] = {}
        manifest_path = manifest_path or (settings.checkpoint_dir / f"manifest_{split}.json")
        if manifest_path.exists():
            self._load_manifest(manifest_path)
        else:
            self._build_manifest(manifest_path)
        # Detector is only needed at build time; drop it so the dataset is
        # picklable across DataLoader workers (ONNXRuntime sessions are not).
        self._detector = None

    # --- manifest ---
    def _cache_dir(self) -> Path:
        return self._settings.checkpoint_dir / "aligned_cache"

    @staticmethod
    def _cache_path(cache_dir: Path, img_path: Path) -> Path:
        key = hashlib.sha1(str(img_path.resolve()).encode("utf-8")).hexdigest()
        return cache_dir / f"{key}.npy"

    def _build_manifest(self, out: Path) -> None:
        assert self._detector is not None  # set in __init__ before build
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        persons = sorted(p for p in self._settings.data_dir.iterdir() if p.is_dir())
        self.class_to_idx = {p.name: i for i, p in enumerate(persons)}
        dropped = 0
        cache_hits = 0
        for person in persons:
            label = self.class_to_idx[person.name]
            for img_path in person.iterdir():
                if img_path.suffix.lower() not in _IMG_EXT:
                    continue
                cache_file = self._cache_path(cache_dir, img_path)
                if cache_file.exists():  # aligned once already (e.g. by the train split)
                    self._samples.append(Sample(str(img_path), label, str(cache_file)))
                    cache_hits += 1
                    continue
                img = load_image_safe(img_path)
                if img is None:
                    dropped += 1
                    continue
                q = assess_quality(
                    img,
                    blur_threshold=self._settings.blur_threshold,
                    min_size=self._settings.min_face_size,
                )
                if not q.ok:
                    dropped += 1
                    continue
                faces = self._detector.detect(img, max_faces=1)
                if not faces:
                    dropped += 1
                    continue
                aligned = align_face(img, faces[0].landmarks)
                np.save(cache_file, aligned)
                self._samples.append(Sample(str(img_path), label, str(cache_file)))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "class_to_idx": self.class_to_idx,
                    "samples": [asdict(s) for s in self._samples],
                }
            ),
            encoding="utf-8",
        )
        log.info(
            "Manifest built",
            extra={
                "extra_fields": {
                    "kept": len(self._samples),
                    "dropped": dropped,
                    "cache_hits": cache_hits,
                    "classes": len(self.class_to_idx),
                }
            },
        )

    def _load_manifest(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.class_to_idx = data["class_to_idx"]
        self._samples = [Sample(**s) for s in data["samples"]]

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self._samples[idx]
        aligned = np.load(sample.cache)  # pre-aligned 112x112 BGR uint8
        tensor = train_transform(aligned) if self._train else to_model_tensor(aligned)
        return tensor, sample.label
