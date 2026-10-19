"""Bounded NumPy uploads and CPU inference using the exported CNN model."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from threading import Lock

import numpy as np
import onnxruntime as ort

SAMPLES_PER_EPOCH = 3000
MAX_BATCH_EPOCHS = 64
MAX_BODY_BYTES = 1024 * 1024
MODEL_SHA256 = "b52c50097fe98800130d7c83385fa432548d818f03a27fcedcc49ab9a61144b1"
MODEL_ID = "sleep-cnn-source:" + MODEL_SHA256[:12]
MODEL_NAME = "EEG_M_SL · source-only (server ONNX)"
MODEL_DIR = Path(__file__).resolve().parent / "models"
_session: ort.InferenceSession | None = None
_session_lock = Lock()


class InputError(ValueError):
    """The caller supplied an unsupported or malformed EEG chunk."""


def get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                model_path = MODEL_DIR / "sleep-cnn-source.onnx"
                digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
                metadata = json.loads((MODEL_DIR / "metadata.json").read_text("utf-8"))
                if digest != MODEL_SHA256 or digest != metadata["onnx_sha256"]:
                    raise RuntimeError("Model checksum mismatch")
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                _session = ort.InferenceSession(
                    str(model_path), sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
    return _session


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


def normalize(samples: np.ndarray) -> tuple[np.ndarray, int]:
    values = samples.astype(np.float64)
    flat = np.ptp(values, axis=1) == 0
    centered = values - values.mean(axis=1, keepdims=True)
    scale = np.maximum(values.std(axis=1, keepdims=True), 1e-8)
    normalized = (centered / scale).astype(np.float32)
    normalized[flat] = 0
    return normalized[:, None, :], int(flat.sum())


def predict(body: bytes) -> dict:
    samples = read_npy(body)
    values, flat = normalize(samples)
    logits = get_session().run(["logits"], {"eeg": values})[0]
    if logits.shape != (len(samples), 5) or not np.isfinite(logits).all():
        raise RuntimeError("Invalid inference output")
    scores = logits.astype(np.float64)
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    predictions = np.argmax(scores, axis=1)
    confidences = probabilities[np.arange(len(samples)), predictions]
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model": MODEL_NAME,
        "engine": "onnxruntime-cpu",
        "predictions": predictions.tolist(),
        "confidences": confidences.tolist(),
        "epoch_seconds": 30,
        "sampling_rate_hz": 100,
        "channel": "EEG Fpz-Cz",
        "epochs": len(samples),
        "flat": flat,
    }
