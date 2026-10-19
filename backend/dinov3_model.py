"""Inference-only architecture matching QuickRun EEG_DINOv3_S exactly.

The exported bundle contains the entire backbone, projector and classifier.
Construct the architecture locally, then strictly load that state; never download
weights or follow the old dinov3_checkpoint path saved by the training helper.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import DINOv3ViTConfig, DINOv3ViTModel


class EEGToDINOv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DINOv3ViTModel(DINOv3ViTConfig(
            image_size=224, patch_size=16, hidden_size=384,
            intermediate_size=1536, num_hidden_layers=12,
            num_attention_heads=6, num_register_tokens=4,
            query_bias=False, key_bias=False, value_bias=False,
            mlp_bias=True, layerscale_value=1e-5,
        ))
        self.project = nn.Linear(384, 128)
        self.register_buffer("window", torch.hann_window(256), persistent=False)
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, eeg):
        spectrum = torch.stft(eeg[:, 0].float(), n_fft=256, hop_length=64,
                              window=self.window.float(), return_complex=True)
        image = torch.log1p(spectrum.abs()).unsqueeze(1)
        image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        low, high = image.amin(dim=(-2, -1), keepdim=True), image.amax(dim=(-2, -1), keepdim=True)
        image = ((image - low) / (high - low).clamp_min(1e-6)).repeat(1, 3, 1, 1)
        tokens = self.backbone(pixel_values=(image - self.rgb_mean) / self.rgb_std).last_hidden_state
        if tokens.shape[1] != 201:
            raise RuntimeError("Unexpected DINOv3 token layout")
        timeline = tokens[:, 5:].reshape(eeg.shape[0], 14, 14, 384).mean(dim=1)
        features = self.project(timeline).transpose(1, 2)
        return F.interpolate(features, size=22, mode="linear", align_corners=False)


class EEGDINOv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = EEGToDINOv3()
        self.logits = nn.Linear(128 * 22, 5)

    def forward(self, eeg):
        features = self.feature_extractor(eeg)
        return self.logits(features.reshape(features.shape[0], -1))


def load_model(path: Path) -> EEGDINOv3:
    torch.set_num_threads(1)
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "architecture": "EEG_DINOv3_S", "method": "SLARDA",
        "class_names": ["W", "N1", "N2", "N3", "REM"],
        "input_shape": [1, 3000], "sample_rate_hz": 100, "epoch_seconds": 30,
    }
    if any(bundle.get(key) != value for key, value in expected.items()):
        raise RuntimeError("DINOv3 bundle metadata mismatch")
    model = EEGDINOv3()
    model.load_state_dict(bundle["model_state_dict"], strict=True)
    return model.eval().requires_grad_(False)


def run_model(model: EEGDINOv3, values):
    with torch.inference_mode():
        return model(torch.from_numpy(values)).numpy()
