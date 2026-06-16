#!/usr/bin/env sh
# Container entrypoint: provision models (idempotent) then serve.
set -e

# Fail fast if the API key is still the default in production — the API would
# otherwise return HTTP 500 on every authenticated request.
if [ "${FACECORE_ENV:-production}" = "production" ] && \
   [ "${FACECORE_API_KEY:-change-me-in-real-deployment}" = "change-me-in-real-deployment" ]; then
  echo "FATAL: FACECORE_API_KEY is unset/default in production. Set a real key (see .env)." >&2
  exit 1
fi

# Populate <model_dir> if empty (no-op when the mounted volume already has them).
python scripts/fetch_models.py

exec uvicorn facecore.api.main:app \
  --host 0.0.0.0 --port 8000 --workers "${FACECORE_WORKERS:-1}"
