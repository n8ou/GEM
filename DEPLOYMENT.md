# facecore — Deployment Guide

Production deployment of the inference API: **YOLO11-pose detection → 5-point
alignment → pretrained ArcFace (`w600k_r50`) embedding → FAISS matching**.

## 1. Prerequisites

- Docker + Docker Compose (or Python 3.12 for a bare run).
- Outbound network access on first run (to download model weights), or
  pre-provisioned `artifacts/models/`.
- ~2 GB disk for models; ~4 GB RAM for the container.

## 2. Configure secrets

The API refuses to serve authenticated routes if the key is unset/default in
production (returns HTTP 500), and the container entrypoint hard-fails at boot.

```bash
cp .env.example .env
# Generate a strong key and set it in .env:
python -c "import secrets; print('FACECORE_API_KEY=' + secrets.token_urlsafe(32))"
```

Set at minimum in `.env`:

| Variable | Notes |
|----------|-------|
| `FACECORE_API_KEY` | **required** — a real secret, not the default |
| `FACECORE_ENV` | `production` |
| `FACECORE_DETECTOR` | `yolo` (or `retinaface`) |
| `FACECORE_YOLO_WEIGHTS` | `yolo11n-pose_widerface.pt` |
| `FACECORE_EMBEDDER` | `w600k_r50` |
| `FACECORE_MATCH_THRESHOLD` | `0.30` (calibrated; see `EVALUATION.md`) |
| `FACECORE_LIVENESS_ENABLED` | `true` — anti-spoofing (rejects photos/screens) |
| `FACECORE_LIVENESS_THRESHOLD` | `0.5` — **must be tuned on your camera + real spoof samples** |

`.env` is gitignored — never commit real secrets.

## 3. Provision models

Reproducible, idempotent, safe to re-run:

```bash
python scripts/fetch_models.py
```

Downloads the YOLO11-pose weights and the InsightFace `buffalo_l` bundle, then
lifts `w600k_r50.onnx` into `artifacts/models/`. The container entrypoint runs
this automatically on first boot if the mounted volume is empty.

## 4. Run with Docker Compose

```bash
docker compose -f docker/docker-compose.yml up --build -d
docker compose -f docker/docker-compose.yml logs -f      # watch for "API ready"
```

The compose file mounts `../artifacts` (models + FAISS index persist across
restarts) and reads `../.env`. The image runs as a non-root user with a
`/health` healthcheck.

### GPU (optional)

The default image is CPU-only (works, slower). For GPU:
1. Install the NVIDIA Container Toolkit on the host.
2. Uncomment `gpus: all` in `docker/docker-compose.yml`.
3. Use a CUDA-enabled base image / ensure `onnxruntime-gpu` finds CUDA libs.

CPU-only hosts can shrink the image by swapping `onnxruntime-gpu` →
`onnxruntime` in `requirements.txt`.

## 5. Smoke test

```bash
KEY=$(grep FACECORE_API_KEY .env | cut -d= -f2-)
curl -s localhost:8000/health
curl -s -H "X-API-Key: $KEY" -F person_id=alice -F files=@a1.jpg -F files=@a2.jpg \
     localhost:8000/enroll
curl -s -H "X-API-Key: $KEY" -F file=@a3.jpg localhost:8000/recognize
```

Expected: `/health` → `{"status":"ok",...}`; enroll → `embeddings_added>0`;
recognize → the enrolled `person_id` with `is_known:true`.

## 6. API reference

| Method | Path | Auth | Purpose |
|--------|------|:----:|---------|
| GET | `/health` | – | liveness, device, index size |
| POST | `/enroll` | ✓ | add a person's face(s): `person_id` + `files[]` |
| POST | `/recognize` | ✓ | identify all faces in one `file` |
| POST | `/verify` | ✓ | 1:1 similarity between `file_a`, `file_b` |

Auth header: `X-API-Key: <key>`. Upload guard: size/type/dimension limits
(`FACECORE_MAX_UPLOAD_MB`, magic-byte sniff, ~40 MP cap).

## 7. Production hardening (recommended before real traffic)

- **TLS**: terminate HTTPS at a reverse proxy (nginx/Traefik); don't expose 8000 directly.
- **Rate limiting**: the built-in limiter is in-process — for multiple replicas use a
  shared store (Redis) instead, or enforce at the proxy.
- **Scaling**: bump `FACECORE_WORKERS` / replicas; the FAISS index is per-process and
  file-backed — for multi-replica writes, move enrollment behind a single writer or a
  shared vector DB.
- **Monitoring**: scrape `/health`, add request/latency metrics and structured-log shipping.
- **Index backups**: persist/back up `artifacts/index/`.
- **Validate on YOUR data**: `EVALUATION.md` numbers are from clean celebrity photos and
  are optimistic. Benchmark and re-tune `FACECORE_MATCH_THRESHOLD` on a held-out set that
  represents your real conditions.
- **Compliance**: face recognition is regulated (GDPR/BIPA/etc.). Ensure lawful basis,
  consent, retention limits, and data-subject rights before deploying.

### Anti-spoofing (liveness)

Passive Silent-Face (MiniFASNet) runs on every enroll and recognize: a printed photo
or phone/screen replay is flagged `is_live: false`, returned as `person_id: "spoof"`,
and **never matched or enrolled** (fail-closed). The `/recognize` response includes
`is_live` and `live_score` per face — your door logic should require BOTH
`is_known: true` AND `is_live: true` before unlocking.

> **Tune the threshold on-site.** The default (0.5) is generic. Capture live members and
> real attack attempts (photo on a phone, printed photo) on the actual entrance camera,
> then pick a threshold that rejects all spoofs while accepting live members. Passive RGB
> liveness is not perfect — pair it with an attended enrollment process and consider a
> camera position/lighting that discourages screen glare.

## 8. Bare (no Docker)

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on Linux
pip install -r requirements.txt && pip install -e .
python scripts/fetch_models.py
uvicorn facecore.api.main:app --host 0.0.0.0 --port 8000
```
