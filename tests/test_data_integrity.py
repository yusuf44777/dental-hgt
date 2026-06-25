"""Data integrity and smoke tests for DentalHGT."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_io import (
    ToothAnnotation,
    PanoramicImage,
    PATHOLOGY_ID_TO_NAME,
    N_PATHOLOGIES,
    collect_all_teeth,
)
from src.preprocess import crop_tooth_roi, apply_clahe, augment_roi
from src.graph_build import (
    _build_mesial_distal_edges,
    _build_bilateral_edges,
    _build_antagonist_edges,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_tooth(quad: int, enum: int, disease_ids: list[int] | None = None) -> ToothAnnotation:
    return ToothAnnotation(
        ann_id=quad * 10 + enum,
        image_id=1,
        quadrant_id=quad,
        enumeration_id=enum,
        disease_ids=disease_ids or [],
        bbox=[10.0, 10.0, 30.0, 40.0],
        segmentation=[],
    )


def make_full_arch() -> list[ToothAnnotation]:
    """All 32 teeth, no pathology."""
    teeth = []
    for q in range(1, 5):
        for e in range(1, 9):
            teeth.append(make_tooth(q, e))
    return teeth


# ---------------------------------------------------------------------------
# ToothAnnotation tests
# ---------------------------------------------------------------------------

class TestToothAnnotation:
    def test_global_index_range(self):
        for q in range(1, 5):
            for e in range(1, 9):
                t = make_tooth(q, e)
                assert 0 <= t.global_tooth_idx <= 31

    def test_global_index_unique(self):
        teeth = make_full_arch()
        indices = [t.global_tooth_idx for t in teeth]
        assert len(set(indices)) == 32

    def test_label_vector_shape(self):
        # DENTEX category_id_3 mapping: 0=Impacted→idx3, 1=Caries→idx0, 2=Periapical→idx2, 3=DeepCaries→idx1
        t = make_tooth(1, 1, disease_ids=[1, 2])  # Caries + Periapical
        vec = t.label_vector
        assert vec.shape == (N_PATHOLOGIES,)
        assert vec[0] == 1.0   # caries (disease_id=1 → index 0)
        assert vec[2] == 1.0   # periapical (disease_id=2 → index 2)
        assert vec[1] == 0.0   # deep_caries absent
        assert vec[3] == 0.0   # impacted absent

    def test_label_vector_healthy(self):
        t = make_tooth(2, 4)
        assert t.label_vector.sum() == 0.0

    def test_arch_id(self):
        for q in [1, 2]:
            assert make_tooth(q, 1).arch_id == 0  # maxilla
        for q in [3, 4]:
            assert make_tooth(q, 1).arch_id == 1  # mandible


# ---------------------------------------------------------------------------
# ROI extraction tests
# ---------------------------------------------------------------------------

class TestROIExtraction:
    def _make_image(self):
        return np.random.randint(0, 255, (600, 1200, 3), dtype=np.uint8)

    def test_crop_output_shape(self):
        img = self._make_image()
        roi = crop_tooth_roi(img, [100, 100, 80, 60], roi_size=64)
        assert roi.shape == (64, 64, 3)

    def test_crop_zero_size_bbox(self):
        img = self._make_image()
        roi = crop_tooth_roi(img, [0, 0, 0, 0], roi_size=32)
        assert roi.shape == (32, 32, 3)

    def test_crop_out_of_bounds(self):
        img = self._make_image()
        roi = crop_tooth_roi(img, [1100, 550, 200, 100], roi_size=64)
        assert roi.shape == (64, 64, 3)

    def test_clahe_preserves_shape(self):
        img = self._make_image()
        out = apply_clahe(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_augment_range(self):
        roi = np.full((64, 64, 3), 128, dtype=np.uint8)
        aug = augment_roi(roi, seed=0)
        assert aug.dtype == np.uint8
        assert aug.min() >= 0 and aug.max() <= 255


# ---------------------------------------------------------------------------
# Graph edge construction tests
# ---------------------------------------------------------------------------

class TestEdgeConstruction:
    def _tooth_map(self, teeth):
        return {t.global_tooth_idx: i for i, t in enumerate(teeth)}

    def test_mesial_distal_within_quadrant(self):
        teeth = [make_tooth(1, e) for e in range(1, 5)]
        tmap = self._tooth_map(teeth)
        edges = _build_mesial_distal_edges(teeth, tmap)
        assert edges.shape[0] == 2
        # 3 pairs × 2 directions = 6
        assert edges.shape[1] == 6

    def test_mesial_distal_empty(self):
        teeth = [make_tooth(1, 1)]
        tmap = self._tooth_map(teeth)
        edges = _build_mesial_distal_edges(teeth, tmap)
        assert edges.shape[1] == 0

    def test_bilateral_q1_q2(self):
        teeth = [make_tooth(1, 1), make_tooth(2, 1), make_tooth(1, 2), make_tooth(2, 2)]
        tmap = self._tooth_map(teeth)
        edges = _build_bilateral_edges(teeth, tmap)
        # 2 pairs × 2 directions = 4
        assert edges.shape[1] == 4

    def test_antagonist_q1_q4(self):
        teeth = [make_tooth(1, 3), make_tooth(4, 3)]
        tmap = self._tooth_map(teeth)
        edges = _build_antagonist_edges(teeth, tmap)
        assert edges.shape[1] == 2  # 1 pair × 2 directions

    def test_full_arch_edges(self):
        teeth = make_full_arch()
        tmap = self._tooth_map(teeth)
        md = _build_mesial_distal_edges(teeth, tmap)
        bl = _build_bilateral_edges(teeth, tmap)
        an = _build_antagonist_edges(teeth, tmap)
        # Sanity: all edge indices are within [0, 31]
        for edges in [md, bl, an]:
            assert edges.min() >= 0
            assert edges.max() <= 31

    def test_no_self_loops(self):
        teeth = make_full_arch()
        tmap = self._tooth_map(teeth)
        for fn in [_build_mesial_distal_edges, _build_bilateral_edges, _build_antagonist_edges]:
            edges = fn(teeth, tmap)
            if edges.shape[1] > 0:
                assert not np.any(edges[0] == edges[1]), f"{fn.__name__} has self-loops"


# ---------------------------------------------------------------------------
# Annotation parsing smoke test (with synthetic JSON)
# ---------------------------------------------------------------------------

class TestAnnotationParsing:
    def _write_coco_json(self, path: Path) -> None:
        data = {
            "images": [
                {"id": 1, "file_name": "img001.png", "width": 2880, "height": 1600},
                {"id": 2, "file_name": "img002.png", "width": 2880, "height": 1600},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "quadrant_id": 1, "enumeration_id": 1, "disease_id": 1,
                 "bbox": [10, 20, 50, 60], "segmentation": []},
                {"id": 2, "image_id": 1, "quadrant_id": 2, "enumeration_id": 3, "disease_id": 3,
                 "bbox": [100, 20, 50, 60], "segmentation": []},
                {"id": 3, "image_id": 2, "quadrant_id": 4, "enumeration_id": 8, "disease_id": None,
                 "bbox": [500, 100, 40, 50], "segmentation": []},
            ],
        }
        with open(path, "w") as fh:
            json.dump(data, fh)

    def test_load_images(self):
        from src.data_io import load_dentex_annotations
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "xrays.json"
            self._write_coco_json(p)
            images = load_dentex_annotations(p)
            assert len(images) == 2
            assert len(images[1].teeth) == 2
            assert len(images[2].teeth) == 1

    def test_disease_id_none_handled(self):
        from src.data_io import load_dentex_annotations
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "xrays.json"
            self._write_coco_json(p)
            images = load_dentex_annotations(p)
            tooth = images[2].teeth[0]
            assert tooth.disease_ids == []
            assert tooth.label_vector.sum() == 0.0

    def test_collect_all_teeth(self):
        from src.data_io import load_dentex_annotations
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "xrays.json"
            self._write_coco_json(p)
            images = load_dentex_annotations(p)
            teeth, iids = collect_all_teeth(images)
            assert len(teeth) == 3
            assert len(iids) == 3
