"""DentalHGT — Heterogeneous Graph Transformer for panoramic X-ray pathology detection.

Architecture mirrors HGTMultiTask from the scRNA-seq study (makale/):
  - Input projections per node type (Linear)
  - Stack of HGTConv layers (PyTorch Geometric)
  - Four independent binary classification heads (one per pathology)
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .graph_build import graph_metadata, in_dims, EDGE_TYPE_MAP

PATHOLOGY_NAMES = ["caries", "deep_caries", "periapical", "impacted"]


class DentalHGT(nn.Module):
    """Heterogeneous Graph Transformer for dental pathology multi-label classification.

    Predicts 4 binary labels (caries, deep_caries, periapical, impacted) for
    each tooth node using anatomically-typed graph edges.
    """

    def __init__(
        self,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import HGTConv
        except ImportError as exc:
            raise ImportError(
                "torch_geometric is required. Install: pip install torch-geometric"
            ) from exc

        hgt_cfg = config["hgt"]
        hidden_dim = int(hgt_cfg["hidden_dim"])
        heads = int(hgt_cfg["heads"])
        layers = int(hgt_cfg["layers"])
        dropout = float(hgt_cfg["dropout"])

        dims = in_dims(config)
        metadata = graph_metadata(config)

        # Input projection: heterogeneous input dims → shared hidden_dim
        self.proj = nn.ModuleDict(
            {node_type: nn.Linear(dim, hidden_dim) for node_type, dim in dims.items()}
        )

        # HGT message-passing layers (same design as scRNA-seq HGTMultiTask)
        self.convs = nn.ModuleList(
            [
                HGTConv(
                    in_channels={nt: hidden_dim for nt in dims},
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=heads,
                )
                for _ in range(layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # One binary head per pathology (tooth node only)
        self.caries_head = nn.Linear(hidden_dim, 1)
        self.deep_caries_head = nn.Linear(hidden_dim, 1)
        self.periapical_head = nn.Linear(hidden_dim, 1)
        self.impacted_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[Any, Tensor],
    ) -> dict[str, Tensor]:
        # Project all node types into shared space
        x = {ntype: F.relu(self.proj[ntype](feat)) for ntype, feat in x_dict.items() if ntype in self.proj}

        # Stack HGT layers
        for conv in self.convs:
            out = conv(x, edge_index_dict)
            # Skip-add + dropout where output exists, else keep input
            x = {
                ntype: self.dropout(self.layer_norm(F.relu(out[ntype]))) + x[ntype]
                if out.get(ntype) is not None else x[ntype]
                for ntype in x
            }

        tooth_emb = x["tooth"]  # [N_tooth, hidden_dim]
        return {
            "caries":      self.caries_head(tooth_emb),
            "deep_caries": self.deep_caries_head(tooth_emb),
            "periapical":  self.periapical_head(tooth_emb),
            "impacted":    self.impacted_head(tooth_emb),
            "tooth_embedding": tooth_emb,
        }

    def predict_proba(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[Any, Tensor],
    ) -> Tensor:
        """Return sigmoid probabilities [N_tooth, 4]."""
        out = self.forward(x_dict, edge_index_dict)
        return torch.cat(
            [torch.sigmoid(out[p]) for p in PATHOLOGY_NAMES], dim=1
        )


def build_model(config: dict[str, Any]) -> DentalHGT:
    return DentalHGT(config)


def compute_loss(
    outputs: dict[str, Tensor],
    labels: Tensor,                  # [N_tooth, 4] float32
    pos_weights: Tensor | None = None,  # [4] float32 for class imbalance
) -> Tensor:
    """Multi-label BCE loss summed over 4 pathologies."""
    total = torch.tensor(0.0, device=labels.device, requires_grad=True)
    for i, name in enumerate(PATHOLOGY_NAMES):
        pw = pos_weights[i].unsqueeze(0) if pos_weights is not None else None
        loss = F.binary_cross_entropy_with_logits(
            outputs[name].squeeze(1),
            labels[:, i],
            pos_weight=pw,
        )
        total = total + loss
    return total / len(PATHOLOGY_NAMES)


def compute_pos_weights(all_labels: "np.ndarray") -> Tensor:
    """Compute per-pathology positive class weights to handle imbalance.

    pos_weight = (n_negative / n_positive) clipped to [1, 20].
    """
    import numpy as np
    n_pos = all_labels.sum(axis=0).clip(min=1)
    n_neg = (1 - all_labels).sum(axis=0).clip(min=1)
    weights = (n_neg / n_pos).clip(1.0, 20.0)
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    import json, sys, numpy as np
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.json"
    with open(cfg_path) as fh:
        config = json.load(fh)

    from src.graph_build import in_dims as get_in_dims
    dims = get_in_dims(config)
    model = build_model(config)
    n_tooth = 12
    x_dict = {
        "tooth":    torch.randn(n_tooth, dims["tooth"]),
        "quadrant": torch.randn(4, dims["quadrant"]),
        "arch":     torch.randn(2, dims["arch"]),
    }
    # Minimal edge_index (empty) for a smoke test
    edge_index_dict = {}
    out = model(x_dict, edge_index_dict)
    print("DentalHGT smoke test")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
