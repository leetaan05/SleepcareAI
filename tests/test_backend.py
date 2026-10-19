import io
import json
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from backend.app import app
from backend.inference import (
    InputError, MAX_BATCH_EPOCHS, MAX_BODY_BYTES, MODEL_ID,
    get_session, normalize, predict, read_npy,
)

ROOT = Path(__file__).resolve().parents[1]


def npy(values):
    stream = io.BytesIO()
    np.save(stream, values, allow_pickle=False)
    return stream.getvalue()


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
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["model_id"], MODEL_ID)
        self.assertFalse(health.json()["stores_uploads"])
        self.assertEqual(health.headers["cache-control"], "no-store")
        headers = {"Content-Type": "application/octet-stream"}
        result = self.client.post("/api/analyze", content=npy(self.samples), headers=headers)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["predictions"], self.golden["predictions"])
        self.assertEqual(self.client.post("/api/analyze", json={}).status_code, 415)
        self.assertEqual(self.client.post("/api/analyze", content=b"bad", headers=headers).status_code, 422)
        self.assertEqual(self.client.post(
            "/api/analyze", content=b"x" * (MAX_BODY_BYTES + 1), headers=headers,
        ).status_code, 413)
        self.assertEqual(self.client.get("/api/analyze").status_code, 405)


if __name__ == "__main__":
    unittest.main()
