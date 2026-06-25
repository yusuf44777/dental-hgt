"""CNN backbone for tooth ROI feature extraction.

Supports ResNet50 (torchvision) and DINOv2 ViT-S/14 (torch.hub).
Both expose a unified ToothFeatureExtractor interface.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class ToothFeatureExtractor(nn.Module):
    """Wraps ResNet50 or DINOv2 to produce fixed-dim tooth embeddings.

    Input : batch of ROI tensors  [B, 3, H, W]  (ImageNet-normalized)
    Output: feature matrix        [B, feature_dim]
    """

    def __init__(self, backbone: str = "resnet50", feature_dim: int = 128) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.feature_dim = feature_dim

        if backbone == "resnet50":
            import torchvision.models as tvm
            base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.encoder = base
        elif backbone in ("dinov2", "dinov2_vits14"):
            base = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
            in_features = 384
            self.encoder = base
        else:
            raise ValueError(f"Unknown backbone: {backbone!r}. Choose 'resnet50' or 'dinov2'.")

        self.proj = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.backbone_name in ("dinov2", "dinov2_vits14"):
            # DINOv2 patch size is 14 — resize H,W to nearest multiple of 14.
            import torch.nn.functional as F
            h, w = x.shape[-2], x.shape[-1]
            nh = ((h + 13) // 14) * 14
            nw = ((w + 13) // 14) * 14
            if nh != h or nw != w:
                x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        feats = self.encoder(x)
        return self.proj(feats)

    def freeze_encoder(self, up_to_layer: str | None = "layer3") -> None:
        """Freeze backbone layers up to (and including) up_to_layer name."""
        if self.backbone_name not in ("resnet50",):
            for param in self.encoder.parameters():
                param.requires_grad = False
            return
        freeze = True
        for name, module in self.encoder.named_children():
            if freeze:
                for param in module.parameters():
                    param.requires_grad = False
            if name == up_to_layer:
                freeze = False


@torch.no_grad()
def extract_features_batch(
    extractor: ToothFeatureExtractor,
    rois: list[np.ndarray],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Convert ROI list → feature matrix.

    rois: list of uint8 (H, W, 3) arrays (already CLAHE-processed)
    Returns: float32 array of shape [N, feature_dim]
    """
    from src.preprocess import roi_to_tensor

    extractor.eval()
    extractor.to(device)
    all_feats: list[Tensor] = []

    for start in range(0, len(rois), batch_size):
        batch_rois = rois[start : start + batch_size]
        tensors = torch.stack([roi_to_tensor(r) for r in batch_rois]).to(device)
        feats = extractor(tensors)
        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0).numpy()


def build_extractor(config: dict[str, Any]) -> ToothFeatureExtractor:
    bb_cfg = config["backbone"]
    extractor = ToothFeatureExtractor(
        backbone=bb_cfg["name"],
        feature_dim=int(bb_cfg["feature_dim"]),
    )
    extractor.freeze_encoder(up_to_layer=bb_cfg.get("freeze_up_to_layer", "layer3"))
    return extractor


if __name__ == "__main__":
    import json, sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.json"
    with open(cfg_path) as fh:
        config = json.load(fh)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ext = build_extractor(config)
    dummy = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(4)]
    feats = extract_features_batch(ext, dummy, device=device)
    print(f"Backbone={config['backbone']['name']}, feature shape: {feats.shape}")
