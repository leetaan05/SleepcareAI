"""Same FastAPI application is used locally and by Vercel Functions."""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from backend.inference import (
    InputError, MAX_BATCH_EPOCHS, MAX_BODY_BYTES, MODEL_ID, get_session, predict,
)

logger = logging.getLogger("sleepcare")
app = FastAPI(title="SleepCare AI API", version="1.0.0", docs_url=None, redoc_url=None)


@app.middleware("http")
async def no_cache(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/api/health")
async def health():
    try:
        await run_in_threadpool(get_session)
    except Exception:
        logger.exception("Model initialization failed")
        raise HTTPException(503, "Máy chủ AI chưa sẵn sàng. Hãy thử lại sau.")
    return {
        "status": "ok", "model_id": MODEL_ID, "engine": "onnxruntime-cpu",
        "max_batch_epochs": MAX_BATCH_EPOCHS, "max_body_bytes": MAX_BODY_BYTES,
        "stores_uploads": False,
    }


@app.post("/api/analyze")
async def analyze(request: Request):
    content_type = request.headers.get("content-type", "").split(";")[0].lower()
    if content_type not in ("application/octet-stream", "application/x-npy"):
        raise HTTPException(415, "Gửi EEG dạng file .npy với application/octet-stream.")
    size_header = request.headers.get("content-length")
    if size_header is not None:
        try:
            size = int(size_header)
        except ValueError:
            raise HTTPException(400, "Content-Length không hợp lệ.")
        if size < 0:
            raise HTTPException(400, "Content-Length không hợp lệ.")
        if size > MAX_BODY_BYTES:
            raise HTTPException(413, "Mỗi phần EEG không được vượt quá 1 MiB.")
    chunks = bytearray()
    async for part in request.stream():
        if len(chunks) + len(part) > MAX_BODY_BYTES:
            raise HTTPException(413, "Mỗi phần EEG không được vượt quá 1 MiB.")
        chunks.extend(part)
    try:
        return await run_in_threadpool(predict, bytes(chunks))
    except InputError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception:
        # Never log uploaded EEG or return internal model/server paths to callers.
        logger.exception("EEG inference failed")
        raise HTTPException(503, "Máy chủ chưa phân tích được EEG. Hãy thử lại.")
