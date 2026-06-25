"""DENTEX dataset loading and annotation parsing."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# FDI: quadrant 1-4 (UR, UL, LL, LR), enumeration 1-8 within quadrant
# Global tooth index: (quadrant_id - 1) * 8 + (enumeration_id - 1)  → 0..31

# DENTEX uses 0-indexed category_id_3 for disease:
#   0=Impacted, 1=Caries, 2=Periapical Lesion, 3=Deep Caries
# We remap to match PATHOLOGY_NAMES order: [caries, deep_caries, periapical, impacted]
DISEASE_CAT_TO_LABEL_IDX = {
    0: 3,  # Impacted    → index 3
    1: 0,  # Caries      → index 0
    2: 2,  # Periapical  → index 2
    3: 1,  # Deep Caries → index 1
}
PATHOLOGY_NAMES_ORDERED = ["caries", "deep_caries", "periapical", "impacted"]
N_PATHOLOGIES = 4

# Legacy alias kept for backward compat with summary reporting
PATHOLOGY_ID_TO_NAME = {0: "impacted", 1: "caries", 2: "periapical", 3: "deep_caries"}
PATHOLOGY_NAME_TO_ID = {v: k for k, v in PATHOLOGY_ID_TO_NAME.items()}

# FDI quadrant → arch membership (0=maxilla, 1=mandible)
QUADRANT_TO_ARCH = {1: 0, 2: 0, 3: 1, 4: 1}

# FDI bilateral pairs: (Q1, Q2) and (Q3, Q4)
BILATERAL_PAIRS = [(1, 2), (3, 4)]

# FDI antagonist pairs: Q1↔Q4, Q2↔Q3
ANTAGONIST_PAIRS = [(1, 4), (2, 3)]


@dataclass
class ToothAnnotation:
    ann_id: int
    image_id: int
    quadrant_id: int        # 1-4
    enumeration_id: int     # 1-8
    disease_ids: list[int]  # list of disease_id (1-4); empty = healthy
    bbox: list[float]       # [x, y, w, h] in COCO format (top-left origin)
    segmentation: list      # raw polygon, may be empty

    @property
    def global_tooth_idx(self) -> int:
        return (self.quadrant_id - 1) * 8 + (self.enumeration_id - 1)

    @property
    def label_vector(self) -> np.ndarray:
        # disease_ids are raw category_id_3 values (0-indexed)
        vec = np.zeros(N_PATHOLOGIES, dtype=np.float32)
        for did in self.disease_ids:
            idx = DISEASE_CAT_TO_LABEL_IDX.get(did)
            if idx is not None:
                vec[idx] = 1.0
        return vec

    @property
    def arch_id(self) -> int:
        return QUADRANT_TO_ARCH[self.quadrant_id]


@dataclass
class PanoramicImage:
    image_id: int
    file_name: str          # relative to dataset root
    width: int
    height: int
    teeth: list[ToothAnnotation] = field(default_factory=list)

    def load_image(self, image_dir: str | Path) -> np.ndarray:
        """Load the panoramic image. image_dir should point to the xrays/ folder."""
        path = Path(image_dir) / self.file_name
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.uint8)


def load_dentex_annotations(
    json_path: str | Path,
    level: str = "disease",
) -> dict[int, PanoramicImage]:
    """Parse a DENTEX annotation JSON file.

    level: one of 'quadrant', 'enumeration', 'disease'
    Returns a dict mapping image_id → PanoramicImage.
    """
    with Path(json_path).open("r", encoding="utf-8") as fh:
        coco = json.load(fh)

    images: dict[int, PanoramicImage] = {}
    for img_info in coco["images"]:
        images[img_info["id"]] = PanoramicImage(
            image_id=img_info["id"],
            file_name=img_info["file_name"],
            width=img_info["width"],
            height=img_info["height"],
        )

    for ann in coco.get("annotations", []):
        iid = ann["image_id"]
        if iid not in images:
            continue

        # DENTEX uses 0-indexed category_id_1/2/3; convert to 1-indexed FDI
        q_id = int(ann.get("category_id_1", 0)) + 1   # 1-4
        e_id = int(ann.get("category_id_2", 0)) + 1   # 1-8

        # category_id_3: 0=Impacted, 1=Caries, 2=Periapical, 3=Deep Caries (0-indexed)
        raw_did = ann.get("category_id_3", None)
        if raw_did is None:
            disease_ids: list[int] = []
        elif isinstance(raw_did, list):
            disease_ids = [int(d) for d in raw_did]
        else:
            disease_ids = [int(raw_did)]

        tooth = ToothAnnotation(
            ann_id=ann["id"],
            image_id=iid,
            quadrant_id=q_id,
            enumeration_id=e_id,
            disease_ids=disease_ids,
            bbox=list(ann.get("bbox", [0, 0, 1, 1])),
            segmentation=ann.get("segmentation", []),
        )
        images[iid].teeth.append(tooth)

    return images


def collect_all_teeth(
    images: dict[int, PanoramicImage],
) -> tuple[list[ToothAnnotation], list[int]]:
    """Flatten all tooth annotations.

    Returns:
        teeth: flat list of ToothAnnotation
        image_ids: corresponding image_id for each tooth
    """
    teeth: list[ToothAnnotation] = []
    image_ids: list[int] = []
    for img in images.values():
        for tooth in img.teeth:
            teeth.append(tooth)
            image_ids.append(img.image_id)
    return teeth, image_ids


def dataset_summary(
    train_images: dict[int, PanoramicImage],
    val_images: dict[int, PanoramicImage],
) -> dict[str, Any]:
    train_teeth, _ = collect_all_teeth(train_images)
    val_teeth, _ = collect_all_teeth(val_images)

    def count_pathologies(teeth: list[ToothAnnotation]) -> dict[str, int]:
        counts: dict[str, int] = {name: 0 for name in PATHOLOGY_ID_TO_NAME.values()}
        for t in teeth:
            for did in t.disease_ids:
                name = PATHOLOGY_ID_TO_NAME.get(did)
                if name:
                    counts[name] += 1
        return counts

    return {
        "train_images": len(train_images),
        "val_images": len(val_images),
        "train_teeth": len(train_teeth),
        "val_teeth": len(val_teeth),
        "train_pathology_counts": count_pathologies(train_teeth),
        "val_pathology_counts": count_pathologies(val_teeth),
    }


if __name__ == "__main__":
    import sys
    import json as _json

    json_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/dentex/train/quadrant_enumeration_disease/xrays.json"
    images = load_dentex_annotations(json_path)
    teeth, _ = collect_all_teeth(images)
    print(f"Loaded {len(images)} images, {len(teeth)} tooth annotations")
    counts: dict[str, int] = {name: 0 for name in PATHOLOGY_ID_TO_NAME.values()}
    for t in teeth:
        for did in t.disease_ids:
            name = PATHOLOGY_ID_TO_NAME.get(did)
            if name:
                counts[name] += 1
    print("Pathology counts:", _json.dumps(counts, indent=2))
