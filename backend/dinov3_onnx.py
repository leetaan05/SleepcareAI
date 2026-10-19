"""Inference-only DINOv3 QuickRun loader for the lightweight production runtime."""
from pathlib import Path

import onnxruntime as ort


def load_model(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def run_model(session: ort.InferenceSession, values):
    return session.run(["logits"], {"eeg": values})[0]
