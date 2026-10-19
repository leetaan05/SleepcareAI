"""Public model descriptors and pinned server-side artifact identities."""
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL = "slarda-dinov3-quick"
MODELS = {
    "slarda-dinov3-quick": {
        "key": "slarda-dinov3-quick",
        "id": "slarda-dinov3-quick:706e9c1486b1",
        "name": "SLARDA + DINOv3 · QuickRun A→B",
        "description": "Phổ EEG → DINOv3 ViT-S/16 → bộ phân loại 5 giai đoạn.",
        "note": "Bundle thử nghiệm trên tập nhỏ 80 epoch; chưa đại diện chất lượng model huấn luyện đầy đủ.",
        "engine": "pytorch-cpu", "max_batch_epochs": 4,
        "smoke_test_only": True, "normalization_floor": 1e-6,
        "file": "slarda-dinov3-quick.pt",
        "sha256": "706e9c1486b18ad772b5e89a8575f6823a90f427c49a8b26746d105363f0d4e5",
    },
    "cnn-source": {
        "key": "cnn-source", "id": "sleep-cnn-source:b52c50097fe9",
        "name": "EEG_M_SL · CNN source-only",
        "description": "EEG → CNN encoder → bộ phân loại 5 giai đoạn.",
        "note": "Baseline CNN source-only để đối chiếu; model nghiên cứu, không dùng chẩn đoán.",
        "engine": "onnxruntime-cpu", "max_batch_epochs": 64,
        "smoke_test_only": False, "normalization_floor": 1e-8,
        "file": "sleep-cnn-source.onnx",
        "sha256": "b52c50097fe98800130d7c83385fa432548d818f03a27fcedcc49ab9a61144b1",
    },
}


def public_catalog():
    fields = ("key", "id", "name", "description", "note", "engine", "max_batch_epochs", "smoke_test_only")
    return {"default_model": DEFAULT_MODEL, "models": [
        {**{field: spec[field] for field in fields}, "available": (MODEL_DIR / spec["file"]).is_file()}
        for spec in MODELS.values()
    ]}
