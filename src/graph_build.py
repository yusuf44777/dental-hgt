"""Heterogeneous dental graph construction.

Builds one HeteroData per panoramic image from detected tooth annotations
and their pre-extracted CNN feature vectors.

Node types:
    tooth    – each detected tooth, features from CNN backbone
    quadrant – 4 quadrants (UR, UL, LL, LR), learnable embedding initialised here
    arch     – 2 arches (maxilla=0, mandible=1), learnable embedding initialised here

Edge types (typed, directed; we add reverse automatically):
    tooth  --mesial_distal--> tooth     same quadrant, adjacent enumeration_id
    tooth  --bilateral------> tooth     Q1↔Q2, Q3↔Q4, same enumeration_id
    tooth  --antagonist-----> tooth     Q1↔Q4, Q2↔Q3, same enumeration_id
    tooth  --member_of------> quadrant
    quadrant --part_of------> arch
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .data_io import (
    PanoramicImage,
    ToothAnnotation,
    QUADRANT_TO_ARCH,
    BILATERAL_PAIRS,
    ANTAGONIST_PAIRS,
)

# Canonical edge type tuples for PyG HeteroData keys
EDGE_TYPE_MAP = {
    "mesial_distal": ("tooth", "mesial_distal", "tooth"),
    "bilateral":     ("tooth", "bilateral", "tooth"),
    "antagonist":    ("tooth", "antagonist", "tooth"),
    "member_of":     ("tooth", "member_of", "quadrant"),
    "part_of":       ("quadrant", "part_of", "arch"),
    # reverse edges for bidirectional message passing
    "rev_mesial_distal": ("tooth", "rev_mesial_distal", "tooth"),
    "rev_bilateral":     ("tooth", "rev_bilateral", "tooth"),
    "rev_antagonist":    ("tooth", "rev_antagonist", "tooth"),
    "rev_member_of":     ("quadrant", "rev_member_of", "tooth"),
    "rev_part_of":       ("arch", "rev_part_of", "quadrant"),
}


@dataclass
class DentalGraphArtifact:
    data: Any   # torch_geometric.data.HeteroData
    tooth_indices: list[int]           # global_tooth_idx for each node
    tooth_labels: np.ndarray           # [N_teeth, 4] float32 multi-label
    image_id: int
    n_teeth: int


def _build_mesial_distal_edges(
    teeth: list[ToothAnnotation],
    tooth_node_map: dict[int, int],   # global_tooth_idx → node_idx
) -> np.ndarray:
    """Connect adjacent teeth within each quadrant (mesial-distal relationship)."""
    srcs, dsts = [], []
    # Group by quadrant
    by_quadrant: dict[int, list[ToothAnnotation]] = {}
    for t in teeth:
        by_quadrant.setdefault(t.quadrant_id, []).append(t)
    for q_teeth in by_quadrant.values():
        sorted_q = sorted(q_teeth, key=lambda t: t.enumeration_id)
        for i in range(len(sorted_q) - 1):
            a = tooth_node_map.get(sorted_q[i].global_tooth_idx)
            b = tooth_node_map.get(sorted_q[i + 1].global_tooth_idx)
            if a is not None and b is not None:
                srcs.append(a); dsts.append(b)
                srcs.append(b); dsts.append(a)
    if not srcs:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array([srcs, dsts], dtype=np.int64)


def _build_bilateral_edges(
    teeth: list[ToothAnnotation],
    tooth_node_map: dict[int, int],
) -> np.ndarray:
    """Connect bilaterally symmetric teeth (Q1↔Q2, Q3↔Q4, same enumeration_id)."""
    srcs, dsts = [], []
    for q_a, q_b in BILATERAL_PAIRS:
        teeth_a = {t.enumeration_id: t for t in teeth if t.quadrant_id == q_a}
        teeth_b = {t.enumeration_id: t for t in teeth if t.quadrant_id == q_b}
        for enum_id in set(teeth_a) & set(teeth_b):
            a = tooth_node_map.get(teeth_a[enum_id].global_tooth_idx)
            b = tooth_node_map.get(teeth_b[enum_id].global_tooth_idx)
            if a is not None and b is not None:
                srcs.append(a); dsts.append(b)
                srcs.append(b); dsts.append(a)
    if not srcs:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array([srcs, dsts], dtype=np.int64)


def _build_antagonist_edges(
    teeth: list[ToothAnnotation],
    tooth_node_map: dict[int, int],
) -> np.ndarray:
    """Connect antagonist teeth across arches (Q1↔Q4, Q2↔Q3, same enumeration_id)."""
    srcs, dsts = [], []
    for q_a, q_b in ANTAGONIST_PAIRS:
        teeth_a = {t.enumeration_id: t for t in teeth if t.quadrant_id == q_a}
        teeth_b = {t.enumeration_id: t for t in teeth if t.quadrant_id == q_b}
        for enum_id in set(teeth_a) & set(teeth_b):
            a = tooth_node_map.get(teeth_a[enum_id].global_tooth_idx)
            b = tooth_node_map.get(teeth_b[enum_id].global_tooth_idx)
            if a is not None and b is not None:
                srcs.append(a); dsts.append(b)
                srcs.append(b); dsts.append(a)
    if not srcs:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array([srcs, dsts], dtype=np.int64)


def build_dental_graph(
    image: PanoramicImage,
    tooth_features: np.ndarray,   # [N_teeth, feature_dim] float32
    config: dict[str, Any],
    exclude_edge_types: set[str] | None = None,
) -> DentalGraphArtifact:
    """Build one HeteroData for a panoramic image.

    tooth_features must be ordered the same as image.teeth.
    exclude_edge_types: set of edge relation names (e.g. {"bilateral", "antagonist"})
      whose edge_index will be replaced with empty tensors (ablation experiments).
      Reverse edges are auto-excluded when their forward counterpart is excluded.
    """
    try:
        from torch_geometric.data import HeteroData
    except ImportError as exc:
        raise ImportError(
            "torch_geometric is required. Install: pip install torch-geometric"
        ) from exc

    teeth = image.teeth
    n_teeth = len(teeth)
    graph_cfg = config["graph"]
    q_dim = int(graph_cfg["quadrant_dim"])
    a_dim = int(graph_cfg["arch_dim"])

    # Map global_tooth_idx → local node index (position in tooth_features)
    tooth_node_map: dict[int, int] = {
        t.global_tooth_idx: i for i, t in enumerate(teeth)
    }

    # ---- Node features ----
    data = HeteroData()
    data["tooth"].x = torch.tensor(tooth_features, dtype=torch.float32)

    # Quadrant nodes: one-hot (4 dims) + learnable positional pattern
    # We use a fixed 4×q_dim embedding seeded by quadrant index
    q_feats = np.eye(4, q_dim, dtype=np.float32)
    data["quadrant"].x = torch.tensor(q_feats, dtype=torch.float32)

    # Arch nodes: one-hot (2 dims) padded to a_dim
    a_feats = np.eye(2, a_dim, dtype=np.float32)
    data["arch"].x = torch.tensor(a_feats, dtype=torch.float32)

    excl = exclude_edge_types or set()
    _empty = torch.zeros((2, 0), dtype=torch.long)

    # ---- Edges: tooth → quadrant (member_of) ----
    t_srcs = list(range(n_teeth))
    q_dsts = [teeth[i].quadrant_id - 1 for i in range(n_teeth)]
    data["tooth", "member_of", "quadrant"].edge_index = (
        _empty if "member_of" in excl
        else torch.tensor([t_srcs, q_dsts], dtype=torch.long)
    )
    data["quadrant", "rev_member_of", "tooth"].edge_index = (
        _empty if "member_of" in excl
        else torch.tensor([q_dsts, t_srcs], dtype=torch.long)
    )

    # ---- Edges: quadrant → arch (part_of) ----
    q_srcs = list(range(4))
    a_dsts = [QUADRANT_TO_ARCH[q + 1] for q in range(4)]
    data["quadrant", "part_of", "arch"].edge_index = (
        _empty if "part_of" in excl
        else torch.tensor([q_srcs, a_dsts], dtype=torch.long)
    )
    data["arch", "rev_part_of", "quadrant"].edge_index = (
        _empty if "part_of" in excl
        else torch.tensor([a_dsts, q_srcs], dtype=torch.long)
    )

    # ---- Tooth–tooth typed edges ----
    md_edges = _build_mesial_distal_edges(teeth, tooth_node_map)
    bl_edges = _build_bilateral_edges(teeth, tooth_node_map)
    an_edges = _build_antagonist_edges(teeth, tooth_node_map)

    for rel, edges in [
        ("mesial_distal", md_edges),
        ("bilateral", bl_edges),
        ("antagonist", an_edges),
    ]:
        data["tooth", rel, "tooth"].edge_index = (
            _empty if rel in excl
            else torch.tensor(edges, dtype=torch.long)
        )

    # ---- Labels ----
    labels = np.stack([t.label_vector for t in teeth], axis=0)  # [N, 4]

    tooth_indices = [t.global_tooth_idx for t in teeth]

    return DentalGraphArtifact(
        data=data,
        tooth_indices=tooth_indices,
        tooth_labels=labels,
        image_id=image.image_id,
        n_teeth=n_teeth,
    )


def graph_metadata(config: dict[str, Any]) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Return (node_types, edge_types) for HGTConv metadata argument."""
    node_types = ["tooth", "quadrant", "arch"]
    edge_types = [v for v in EDGE_TYPE_MAP.values()]
    return node_types, edge_types


def in_dims(config: dict[str, Any]) -> dict[str, int]:
    q_dim = int(config["graph"]["quadrant_dim"])
    a_dim = int(config["graph"]["arch_dim"])
    feat_dim = int(config["backbone"]["feature_dim"])
    return {"tooth": feat_dim, "quadrant": q_dim, "arch": a_dim}


if __name__ == "__main__":
    import json, sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.json"
    with open(cfg_path) as fh:
        config = json.load(fh)

    from src.data_io import load_dentex_annotations
    images = load_dentex_annotations(config["dataset"]["train_json"])
    img = next(iter(images.values()))
    feat_dim = config["backbone"]["feature_dim"]
    dummy_feats = np.random.randn(len(img.teeth), feat_dim).astype(np.float32)
    artifact = build_dental_graph(img, dummy_feats, config)
    print(f"Graph for image {img.image_id}: {artifact.n_teeth} teeth")
    print(f"  tooth.x shape: {artifact.data['tooth'].x.shape}")
    print(f"  mesial_distal edges: {artifact.data['tooth','mesial_distal','tooth'].edge_index.shape}")
    print(f"  bilateral edges:     {artifact.data['tooth','bilateral','tooth'].edge_index.shape}")
    print(f"  antagonist edges:    {artifact.data['tooth','antagonist','tooth'].edge_index.shape}")
