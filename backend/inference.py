"""Bounded NumPy uploads and explicit model selection for CPU inference."""
from __future__ import annotations

import hashlib
import io
from threading import Lock

import numpy as np
from backend.catalog import DEFAULT_MODEL, MODELS, MODEL_DIR

SAMPLES_PER_EPOCH = 3000
MAX_BATCH_EPOCHS = 64
MAX_BODY_BYTES = 1024 * 1024
_runners: dict = {}
_load_lock = Lock()
_inference_lock = Lock()


class InputError(ValueError):
    """The caller supplied an unsupported or malformed EEG chunk."""


class ModelBusy(RuntimeError):
    """Bound concurrent CPU work instead of exhausting server memory."""


def model_spec(key: str):
    if key not in MODELS:
        raise InputError("Model không hợp lệ. Hãy chọn một model trong danh sách.")
    return MODELS[key]


def get_runner(key: str):
    spec = model_spec(key)
    with _load_lock:
        if key not in _runners:
            path = MODEL_DIR / spec["file"]
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != spec["sha256"]:
                raise RuntimeError("Model checksum mismatch")
            from backend.dinov3_onnx import load_model
            _runners[key] = load_model(path)
    return _runners[key]


def get_session():
    """CNN session retained for parity checks against the original ONNX export."""
    return get_runner("cnn-source")


def read_npy(body: bytes) -> np.ndarray:
    if not body or len(body) > MAX_BODY_BYTES:
        raise InputError("Mỗi phần EEG phải nhỏ hơn hoặc bằng 1 MiB.")
    stream = io.BytesIO(body)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                stream, max_header_size=4096)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(
                stream, max_header_size=4096)
        else:
            raise InputError("API nhận NumPy .npy phiên bản 1 hoặc 2.")
    except InputError:
        raise
    except (ValueError, TypeError, EOFError, UnicodeError) as exc:
        raise InputError("File NumPy bị thiếu hoặc không hợp lệ.") from exc

    valid_shape = (
        shape == (SAMPLES_PER_EPOCH,)
        or (len(shape) == 2 and shape[1] == SAMPLES_PER_EPOCH)
        or (len(shape) == 3 and shape[1:] == (1, SAMPLES_PER_EPOCH))
    )
    if not valid_shape or any(type(n) is not int or n <= 0 for n in shape):
        raise InputError("Shape phải là [3000], [N,3000] hoặc [N,1,3000].")
    epochs = 1 if len(shape) == 1 else shape[0]
    if epochs > MAX_BATCH_EPOCHS:
        raise InputError("Mỗi request nhận tối đa 64 epoch EEG.")
    if fortran or dtype.str not in ("<f4", "<f8"):
        raise InputError("EEG cần float32/float64 little-endian, C-contiguous.")
    offset = stream.tell()
    count = epochs * SAMPLES_PER_EPOCH
    if len(body) != offset + count * dtype.itemsize:
        raise InputError("Kích thước dữ liệu không khớp với header NumPy.")
    # Validate the header and exact size before constructing any data array.
    samples = np.frombuffer(body, dtype=dtype, count=count, offset=offset).reshape(
        epochs, SAMPLES_PER_EPOCH)
    if not np.isfinite(samples).all() or np.any(
        np.abs(samples) > np.finfo(np.float32).max
    ):
        raise InputError("EEG chứa NaN/Inf hoặc vượt miền float32.")
    return samples


def normalize(samples: np.ndarray, floor: float = 1e-8) -> tuple[np.ndarray, int]:
    values = samples.astype(np.float64)
    flat = np.ptp(values, axis=1) == 0
    centered = values - values.mean(axis=1, keepdims=True)
    scale = np.maximum(values.std(axis=1, keepdims=True), floor)
    normalized = (centered / scale).astype(np.float32)
    normalized[flat] = 0
    return normalized[:, None, :], int(flat.sum())


def predict(body: bytes, model: str = DEFAULT_MODEL) -> dict:
    spec = model_spec(model)
    samples = read_npy(body)
    if len(samples) > spec["max_batch_epochs"]:
        raise InputError(f"Model này nhận tối đa {spec['max_batch_epochs']} epoch mỗi request.")
    values, flat = normalize(samples, spec["normalization_floor"])
    if not _inference_lock.acquire(blocking=False):
        raise ModelBusy("Máy chủ đang phân tích một yêu cầu khác. Hãy thử lại sau.")
    try:
        runner = get_runner(model)
        logits = runner.run(["logits"], {"eeg": values})[0]
    finally:
        _inference_lock.release()
    if logits.shape != (len(samples), 5) or not np.isfinite(logits).all():
        raise RuntimeError("Invalid inference output")
    scores = logits.astype(np.float64)
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    predictions = np.argmax(scores, axis=1)
    confidences = probabilities[np.arange(len(samples)), predictions]
    return {
        "schema_version": 1,
        "model_key": model,
        "model_id": spec["id"],
        "model": spec["name"],
        "engine": spec["engine"],
        "smoke_test_only": spec["smoke_test_only"],
        "predictions": predictions.tolist(),
        "confidences": confidences.tolist(),
        "epoch_seconds": 30,
        "sampling_rate_hz": 100,
        "channel": "EEG Fpz-Cz",
        "epochs": len(samples),
        "flat": flat,
    }
