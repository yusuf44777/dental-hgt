"""ROI extraction, CLAHE normalization, and augmentation for panoramic X-ray teeth."""
from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np
from PIL import Image


def crop_tooth_roi(
    image: np.ndarray,
    bbox: list[float],
    roi_size: int = 64,
    padding_frac: float = 0.1,
) -> np.ndarray:
    """Crop a single tooth ROI from a panoramic image.

    bbox: COCO format [x, y, w, h] (top-left origin, pixel units)
    Returns: uint8 RGB array of shape (roi_size, roi_size, 3)
    """
    h_img, w_img = image.shape[:2]
    x, y, w, h = bbox
    pad_x = w * padding_frac
    pad_y = h * padding_frac
    x1 = max(0, int(x - pad_x))
    y1 = max(0, int(y - pad_y))
    x2 = min(w_img, int(x + w + pad_x))
    y2 = min(h_img, int(y + h + pad_y))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((roi_size, roi_size, 3), dtype=np.uint8)
    crop = image[y1:y2, x1:x2]
    resized = cv2.resize(crop, (roi_size, roi_size), interpolation=cv2.INTER_AREA)
    return resized


def apply_clahe(image_rgb: np.ndarray, clip_limit: float = 2.0, grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE to the L channel of an RGB image (standard for X-rays)."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def augment_roi(
    roi: np.ndarray,
    brightness_factor: float = 0.2,
    contrast_factor: float = 0.2,
    seed: int | None = None,
) -> np.ndarray:
    """Random brightness and contrast augmentation. No horizontal flip (changes FDI side)."""
    rng = random.Random(seed)
    roi = roi.astype(np.float32)
    b = rng.uniform(-brightness_factor, brightness_factor)
    c = rng.uniform(1.0 - contrast_factor, 1.0 + contrast_factor)
    roi = roi * c + b * 255.0
    return np.clip(roi, 0, 255).astype(np.uint8)


def roi_to_tensor(roi: np.ndarray) -> "torch.Tensor":
    """Convert uint8 HxWxC → float32 CxHxW tensor in [0,1]."""
    import torch
    t = torch.from_numpy(roi).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std


def extract_all_rois(
    panoramic_image: np.ndarray,
    teeth: list,
    config: dict[str, Any],
    augment: bool = False,
    seed: int = 42,
) -> list[np.ndarray]:
    """Extract and preprocess ROIs for all teeth in one panoramic image.

    Returns list of uint8 RGB arrays, one per tooth.
    """
    pre_cfg = config["preprocess"]
    roi_size = int(pre_cfg["roi_size"])
    clip_limit = float(pre_cfg["clahe_clip_limit"])
    grid = tuple(int(g) for g in pre_cfg["clahe_grid"])

    rois = []
    for i, tooth in enumerate(teeth):
        roi = crop_tooth_roi(panoramic_image, tooth.bbox, roi_size=roi_size)
        roi = apply_clahe(roi, clip_limit=clip_limit, grid=grid)
        if augment and pre_cfg.get("augmentation", False):
            roi = augment_roi(
                roi,
                brightness_factor=float(pre_cfg["aug_brightness_factor"]),
                contrast_factor=float(pre_cfg["aug_contrast_factor"]),
                seed=seed + i,
            )
        rois.append(roi)
    return rois


if __name__ == "__main__":
    import sys
    import json

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.json"
    with open(cfg_path) as fh:
        config = json.load(fh)

    from src.data_io import load_dentex_annotations, collect_all_teeth
    images = load_dentex_annotations(config["dataset"]["train_json"])
    image_list = list(images.values())[:1]
    if image_list:
        img_obj = image_list[0]
        raw = img_obj.load_image(config["paths"]["dataset_dir"])
        rois = extract_all_rois(raw, img_obj.teeth, config)
        print(f"Image {img_obj.image_id}: {len(rois)} ROIs extracted, shape={rois[0].shape if rois else 'N/A'}")
