# facecore — Model Evaluation Report

**Date:** 2026-06-16
**Dataset:** `dataset/` — 105 identities (pins celebrity faces), 17,534 images

## Pipeline under test

| Component | Model |
|-----------|-------|
| Detector | YOLO11-pose (`yolo11n-pose_widerface.pt`) — box + 5 facial keypoints |
| Alignment | 5-point similarity transform → 112×112 (ArcFace template) |
| Embedder | Pretrained InsightFace ArcFace **`w600k_r50`** (512-d) |
| Matcher | FAISS inner-product on L2-normalized vectors (cosine) |
| Operating threshold | `FACECORE_MATCH_THRESHOLD = 0.30` |

> Note: the embedder is the **pretrained** InsightFace model. Custom from-scratch
> training was not used — `build_backbone` is random-init and cannot converge on
> 17k images (ArcFace from scratch needs ~millions). See memory notes.

## 1:1 Verification

Sample: **40 identities, 319 images** → 1,113 same-person pairs, 19,542 different-person pairs.

| Metric | Value | Meaning |
|--------|:-----:|---------|
| Overall error @0.30 | **0.11%** | accuracy 99.89% |
| FRR (false rejection) | **1.80%** | same person wrongly rejected |
| FAR (false acceptance) | **0.01%** | different person wrongly accepted |
| **EER (equal error rate)** | **0.12%** | balanced single-number error, at thr ≈ 0.22 |

## 1:N Identification

Sample: **50-identity gallery** (150 enrolled images), **298 probe images**.

| Metric | Value |
|--------|:-----:|
| **Rank-1 accuracy** | **99.33%** |
| Rank-5 accuracy | **100.00%** |
| Rank-1 + above threshold (open-set) | 99.33% |
| Rank-1 error rate | **0.67%** |

## Summary

| Task | Metric | Result |
|------|--------|:------:|
| Verification (1:1) | EER | ~0.12% |
| Verification (1:1) | overall error @0.30 | 0.11% |
| Identification (1:N) | Rank-1 error | 0.67% |

## Caveats

1. Measured on relatively clean celebrity photos → numbers are **optimistic** vs.
   hard real-world conditions (poor lighting, extreme pose, low resolution).
2. Identification error grows as the gallery grows (more distractor identities).
3. With bounding-box-only alignment (a detector lacking keypoints), worst-case
   same-person similarity dropped sharply (min 0.11 vs 0.225 with 5-point) and
   Rank/verification accuracy degrade — the 5-point keypoint model is what
   delivers these results.

## Reproduce

Both benchmarks embed each image via the live config (`get_settings()` →
YOLO11-pose detect → 5-point align → `w600k_r50` embed), then:

- **Verification:** all within-identity pairs (same) vs. 20k sampled cross-identity
  pairs (diff); compute FRR/FAR at threshold and sweep for EER.
- **Identification:** 3 gallery images/identity, remaining as probes; per-identity
  max cosine (mirrors `FaissVectorStore.search`); top-1 / top-5 correctness.

Seed `0` for both. Run from the project root with the `.venv` interpreter.
