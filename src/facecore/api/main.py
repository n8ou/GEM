"""FastAPI inference service.

Endpoints (all JSON, all behind X-API-Key):
  GET  /health            liveness + model/index status
  POST /recognize         multi-face recognition on one image
  POST /enroll            add a person's face(s) to the gallery
  POST /verify            1:1 similarity between two images

The heavy pipeline is built once at startup (lifespan) and shared across
requests. CPU-bound inference runs in a threadpool so the event loop stays free.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from facecore.api.schemas import (
    BoxModel,
    EnrollResponse,
    FaceResult,
    HealthResponse,
    IdentityModel,
    RecognizeResponse,
    VerifyResponse,
)
from facecore.api.security import RateLimiter, require_api_key, validate_and_decode
from facecore.config import get_settings
from facecore.inference.factory import build_pipeline, build_store
from facecore.logging_conf import configure_logging, get_logger
from facecore.recognition.matcher import Matcher
from facecore.utils.device import resolve_device

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    app.state.settings = settings
    app.state.pipeline = build_pipeline(settings)
    app.state.store = build_store(settings)
    app.state.device = str(resolve_device(settings.device))
    app.state.limiter = RateLimiter(settings.rate_limit_per_min)
    log.info("API ready")
    yield
    app.state.store.save()
    log.info("API shutdown — index persisted")


app = FastAPI(title="facecore", version="1.0.0", lifespan=lifespan)


def _client_id(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    return HealthResponse(device=request.app.state.device, index_size=request.app.state.store.size)


@app.post("/recognize", response_model=RecognizeResponse, dependencies=[Depends(require_api_key)])
async def recognize(request: Request, file: UploadFile = File(...)) -> RecognizeResponse:
    request.app.state.limiter.check(_client_id(request))
    img = validate_and_decode(await file.read())
    results = await run_in_threadpool(request.app.state.pipeline.recognize, img)
    faces = [
        FaceResult(
            bbox=BoxModel(x1=r.bbox.x1, y1=r.bbox.y1, x2=r.bbox.x2, y2=r.bbox.y2, score=r.bbox.score),
            identity=IdentityModel(
                person_id=r.identity.person_id,
                similarity=round(r.identity.similarity, 4),
                is_known=r.identity.is_known,
            ),
        )
        for r in results
    ]
    return RecognizeResponse(faces=faces, count=len(faces))


@app.post("/enroll", response_model=EnrollResponse, dependencies=[Depends(require_api_key)])
async def enroll(
    request: Request,
    person_id: str = Form(..., min_length=1, max_length=128),
    files: list[UploadFile] = File(...),
) -> EnrollResponse:
    request.app.state.limiter.check(_client_id(request))
    pipeline = request.app.state.pipeline
    store = request.app.state.store
    added = 0
    for file in files:
        img = validate_and_decode(await file.read())
        embs = await run_in_threadpool(pipeline.embed_only, img)
        if embs.shape[0]:
            added += await run_in_threadpool(store.add, person_id, embs)
    await run_in_threadpool(store.save)
    return EnrollResponse(person_id=person_id, embeddings_added=added, index_size=store.size)


@app.post("/verify", response_model=VerifyResponse, dependencies=[Depends(require_api_key)])
async def verify(
    request: Request,
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
) -> VerifyResponse:
    request.app.state.limiter.check(_client_id(request))
    pipeline = request.app.state.pipeline
    settings = request.app.state.settings
    img_a = validate_and_decode(await file_a.read())
    img_b = validate_and_decode(await file_b.read())
    emb_a = await run_in_threadpool(pipeline.embed_only, img_a)
    emb_b = await run_in_threadpool(pipeline.embed_only, img_b)
    if emb_a.shape[0] == 0 or emb_b.shape[0] == 0:
        return VerifyResponse(similarity=0.0, is_match=False, threshold=settings.match_threshold)
    sim = Matcher.cosine(emb_a[0], emb_b[0])
    return VerifyResponse(
        similarity=round(sim, 4),
        is_match=sim >= settings.match_threshold,
        threshold=settings.match_threshold,
    )
