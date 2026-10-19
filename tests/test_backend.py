import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from backend.app import app
from backend.inference import (
    InputError, MAX_BATCH_EPOCHS, MAX_BODY_BYTES,
    get_session, get_runner, normalize, predict as predict_model, read_npy, _inference_lock,
)
from backend.catalog import DEFAULT_MODEL, MODELS

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = MODELS["cnn-source"]["id"]


def npy(values):
    stream = io.BytesIO()
    np.save(stream, values, allow_pickle=False)
    return stream.getvalue()


def predict(body):
    return predict_model(body, "cnn-source")


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = np.load(ROOT / "frontend/public/samples/eeg-demo.npy")
        cls.golden = json.loads((ROOT / "tests/fixtures/parity.json").read_text())
        cls.client = TestClient(app)

    def test_real_model_matches_exported_predictions_and_confidences(self):
        result = predict(npy(self.samples))
        self.assertEqual(result["predictions"], self.golden["predictions"])
        logits = np.asarray(self.golden["logits"], dtype=np.float64)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        expected = (exp / exp.sum(axis=1, keepdims=True)).max(axis=1)
        np.testing.assert_allclose(result["confidences"], expected, atol=1e-5)
        self.assertEqual(result["model_id"], MODEL_ID)

    def test_multiple_chunks_preserve_full_record_predictions(self):
        samples = np.tile(self.samples, (17, 1))[:400]
        self.assertGreater(len(npy(samples)), 4_500_000)
        predictions = []
        for start in range(0, len(samples), MAX_BATCH_EPOCHS):
            body = npy(samples[start:start + MAX_BATCH_EPOCHS])
            self.assertLess(len(body), MAX_BODY_BYTES)
            predictions.extend(predict(body)["predictions"])
        values, _ = normalize(samples)
        expected = get_session().run(["logits"], {"eeg": values})[0].argmax(axis=1)
        self.assertEqual(predictions, expected.tolist())

    def test_constant_epoch_is_finite_and_counted(self):
        values, flat = normalize(np.full((1, 3000), 3e-6, dtype=np.float32))
        self.assertEqual(flat, 1)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values == 0).all())
        self.assertEqual(predict(npy(np.ones((1, 3000), dtype=np.float32)))["flat"], 1)

    def test_rejects_bad_shapes_dtypes_nonfinite_and_incomplete_files(self):
        invalid = [
            b"not a NumPy file",
            npy(np.zeros((1, 2999), dtype=np.float32)),
            npy(np.zeros((65, 3000), dtype=np.float32)),
            npy(np.zeros((1, 3000), dtype=np.int32)),
            npy(np.zeros((1, 3000), dtype=">f4")),
            npy(np.full((1, 3000), np.nan, dtype=np.float32)),
            npy(np.full((1, 3000), 1e100, dtype=np.float64)),
            npy(self.samples)[:-4],
            npy(self.samples) + b"extra",
        ]
        for body in invalid:
            with self.subTest(size=len(body)):
                with self.assertRaises(InputError):
                    read_npy(body)

    def test_api_health_upload_and_errors(self):
        health = self.client.get("/api/health?model=cnn-source")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["model_id"], MODEL_ID)
        self.assertFalse(health.json()["stores_uploads"])
        self.assertEqual(health.headers["cache-control"], "no-store")
        headers = {"Content-Type": "application/octet-stream"}
        result = self.client.post("/api/analyze?model=cnn-source", content=npy(self.samples), headers=headers)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["predictions"], self.golden["predictions"])
        self.assertEqual(self.client.post("/api/analyze", json={}).status_code, 415)
        self.assertEqual(self.client.post("/api/analyze", content=b"bad", headers=headers).status_code, 422)
        self.assertEqual(self.client.post(
            "/api/analyze", content=b"x" * (MAX_BODY_BYTES + 1), headers=headers,
        ).status_code, 413)
        self.assertEqual(self.client.get("/api/analyze").status_code, 405)

    def test_dinov3_logits_and_predictions_match_original_quickrun(self):
        from backend.dinov3_onnx import run_model
        golden = json.loads((ROOT / "tests/fixtures/dinov3-parity.json").read_text())
        all_predictions, all_confidences, all_logits = [], [], []
        for start in range(0, len(self.samples), 4):
            chunk = self.samples[start:start + 4]
            result = predict_model(npy(chunk), DEFAULT_MODEL)
            self.assertEqual(result["model_key"], DEFAULT_MODEL)
            self.assertEqual(result["model_id"], MODELS[DEFAULT_MODEL]["id"])
            self.assertEqual(result["engine"], "onnxruntime-cpu")
            self.assertTrue(result["smoke_test_only"])
            all_predictions.extend(result["predictions"])
            all_confidences.extend(result["confidences"])
            values, _ = normalize(chunk, 1e-6)
            all_logits.extend(run_model(get_runner(DEFAULT_MODEL), values).tolist())
        self.assertEqual(all_predictions, golden["predictions"])
        np.testing.assert_allclose(all_logits, golden["logits"], atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(all_confidences, golden["confidences"], atol=1e-5)

    def test_model_catalog_health_default_selection_and_batch_limits(self):
        catalog = self.client.get("/api/models").json()
        self.assertEqual(catalog["default_model"], DEFAULT_MODEL)
        self.assertEqual({m["key"] for m in catalog["models"]}, set(MODELS))
        self.assertTrue(all(m["available"] for m in catalog["models"]))
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["model_key"], DEFAULT_MODEL)
        self.assertEqual(health.json()["max_batch_epochs"], 4)
        headers = {"Content-Type": "application/octet-stream"}
        response = self.client.post("/api/analyze", content=npy(self.samples[:1]), headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_key"], DEFAULT_MODEL)
        self.assertTrue(response.json()["smoke_test_only"])
        self.assertEqual(self.client.post("/api/analyze", content=npy(self.samples[:5]), headers=headers).status_code, 422)
        for key in ("unknown", "../../secret", "cnn-source.onnx"):
            self.assertEqual(self.client.get("/api/health", params={"model": key}).status_code, 422)
            self.assertEqual(self.client.post("/api/analyze", params={"model": key}, content=npy(self.samples[:1]), headers=headers).status_code, 422)

    def test_dinov3_batching_and_flat_signals(self):
        batch = predict_model(npy(self.samples[:4]), DEFAULT_MODEL)
        singles = [predict_model(npy(sample), DEFAULT_MODEL) for sample in self.samples[:4]]
        self.assertEqual(batch["predictions"], [result["predictions"][0] for result in singles])
        np.testing.assert_allclose(batch["confidences"], [result["confidences"][0] for result in singles], atol=1e-5)
        flat = predict_model(npy(np.full((1, 1, 3000), 3e-6, dtype=np.float32)), DEFAULT_MODEL)
        self.assertEqual(flat["flat"], 1)
        self.assertTrue(np.isfinite(flat["confidences"]).all())

    def test_missing_model_returns_503_without_cnn_fallback(self):
        with TemporaryDirectory() as directory, patch("backend.inference.MODEL_DIR", Path(directory)), patch("backend.inference._runners", {}):
            with self.assertLogs("sleepcare", level="ERROR"):
                response = self.client.get("/api/health?model=slarda-dinov3-quick")
            self.assertEqual(response.status_code, 503)
            with self.assertLogs("sleepcare", level="ERROR"):
                response = self.client.post("/api/analyze?model=slarda-dinov3-quick", content=npy(self.samples[:1]), headers={"Content-Type": "application/octet-stream"})
            self.assertEqual(response.status_code, 503)

    def test_concurrent_inference_gets_retryable_response(self):
        with _inference_lock:
            response = self.client.post("/api/analyze?model=cnn-source", content=npy(self.samples[:1]), headers={"Content-Type": "application/octet-stream"})
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "2")


if __name__ == "__main__":
    unittest.main()
