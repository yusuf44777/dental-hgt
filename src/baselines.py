"""Baseline models for comparison against DentalHGT.

1. IndependentCNN  — classify each tooth independently (no graph)
2. HomogeneousGNN  — GATConv with all edges merged into one type (no typed edges)
3. YOLOv8Baseline  — end-to-end detection (optional, requires ultralytics)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .hgt_model import PATHOLOGY_NAMES, compute_loss


# ---------------------------------------------------------------------------
# 1. Independent CNN classifier (no graph context)
# ---------------------------------------------------------------------------

class IndependentCNNClassifier(nn.Module):
    """MLP on top of pre-extracted tooth features — no graph, no context."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(hidden_dim // 2, len(PATHOLOGY_NAMES))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.net(x))  # [N, 4] logits

    def predict_proba(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.forward(x))  # [N, 4]


def build_independent_cnn(config: dict[str, Any]) -> IndependentCNNClassifier:
    return IndependentCNNClassifier(
        feature_dim=int(config["backbone"]["feature_dim"]),
        hidden_dim=int(config["hgt"]["hidden_dim"]),
        dropout=float(config["hgt"]["dropout"]),
    )


# ---------------------------------------------------------------------------
# 2. Homogeneous GNN (GATConv, all edges merged, no type distinction)
# ---------------------------------------------------------------------------

class HomogeneousGNN(nn.Module):
    """Standard GATConv — treats all edge types as identical.

    Ablation: tests whether typed heterogeneous edges matter.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 128, heads: int = 4, layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import GATConv
        except ImportError as exc:
            raise ImportError("torch_geometric required for HomogeneousGNN") from exc

        self.proj = nn.Linear(feature_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for i in range(layers):
            in_ch = hidden_dim * heads if i > 0 else hidden_dim
            self.convs.append(GATConv(in_ch, hidden_dim, heads=heads, dropout=dropout, concat=True))
        self.out_proj = nn.Linear(hidden_dim * heads, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, len(PATHOLOGY_NAMES))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.proj(x))
        for conv in self.convs:
            x = self.dropout(F.relu(conv(x, edge_index)))
        x = F.relu(self.out_proj(x))
        return self.head(x)  # [N_tooth, 4]

    def predict_proba(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return torch.sigmoid(self.forward(x, edge_index))


def build_homogeneous_gnn(config: dict[str, Any]) -> HomogeneousGNN:
    return HomogeneousGNN(
        feature_dim=int(config["backbone"]["feature_dim"]),
        hidden_dim=int(config["hgt"]["hidden_dim"]),
        heads=int(config["hgt"]["heads"]),
        layers=int(config["hgt"]["layers"]),
        dropout=float(config["hgt"]["dropout"]),
    )


def merge_edges_for_homogeneous(
    data: Any,  # HeteroData
) -> tuple[Tensor, Tensor]:
    """Merge all tooth→tooth edges from a HeteroData into one edge_index.

    Used for HomogeneousGNN: only tooth node features + union of all tooth-tooth edges.
    """
    tooth_tooth_rels = ["mesial_distal", "bilateral", "antagonist"]
    all_edges = []
    for rel in tooth_tooth_rels:
        key = ("tooth", rel, "tooth")
        if key in data.edge_types:
            all_edges.append(data[key].edge_index)
    if not all_edges:
        n = data["tooth"].x.shape[0]
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.cat(all_edges, dim=1)
    return data["tooth"].x, edge_index


# ---------------------------------------------------------------------------
# 3. YOLOv8 wrapper (optional)
# ---------------------------------------------------------------------------

def run_yolo_baseline(
    image_paths: list[str],
    model_path: str = "yolov8n.pt",
    imgsz: int = 1280,
) -> list[dict]:
    """Run YOLOv8 on full panoramic images.

    Returns list of detection dicts per image.
    Requires: pip install ultralytics
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics required for YOLO baseline. pip install ultralytics") from exc

    model = YOLO(model_path)
    results = []
    for path in image_paths:
        preds = model(path, imgsz=imgsz, verbose=False)[0]
        results.append({
            "image_path": path,
            "boxes": preds.boxes.xyxy.cpu().numpy() if preds.boxes else np.zeros((0, 4)),
            "scores": preds.boxes.conf.cpu().numpy() if preds.boxes else np.zeros(0),
            "classes": preds.boxes.cls.cpu().numpy().astype(int) if preds.boxes else np.zeros(0, dtype=int),
        })
    return results


# ---------------------------------------------------------------------------
# Shared training loop for tooth-level classifiers
# ---------------------------------------------------------------------------

@dataclass
class BaselineResult:
    model_name: str
    train_loss_history: list[float]
    val_loss_history: list[float]
    best_epoch: int


def train_tooth_classifier(
    model: nn.Module,
    train_features: np.ndarray,      # [N_train_teeth, feature_dim]
    train_labels: np.ndarray,         # [N_train_teeth, 4]
    val_features: np.ndarray,         # [N_val_teeth, feature_dim]
    val_labels: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    model_name: str = "baseline",
    edge_index: Tensor | None = None,  # for HomogeneousGNN
) -> BaselineResult:
    """Generic training loop for IndependentCNN and HomogeneousGNN."""
    from .hgt_model import compute_pos_weights

    train_cfg = config["training"]
    epochs = int(train_cfg["epochs"])
    lr = float(train_cfg["lr"])
    wd = float(train_cfg["weight_decay"])
    patience = int(train_cfg["patience"])

    model = model.to(device)
    X_tr = torch.tensor(train_features, dtype=torch.float32, device=device)
    y_tr = torch.tensor(train_labels, dtype=torch.float32, device=device)
    X_va = torch.tensor(val_features, dtype=torch.float32, device=device)
    y_va = torch.tensor(val_labels, dtype=torch.float32, device=device)
    pos_w = compute_pos_weights(train_labels).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    is_gnn = edge_index is not None
    if is_gnn:
        edge_index = edge_index.to(device)

    train_hist, val_hist = [], []
    best_val = float("inf")
    best_epoch = 0
    patience_ctr = 0

    for epoch in range(epochs):
        model.train()
        optim.zero_grad()
        logits = model(X_tr, edge_index) if is_gnn else model(X_tr)
        losses = []
        for i, name in enumerate(PATHOLOGY_NAMES):
            losses.append(F.binary_cross_entropy_with_logits(
                logits[:, i], y_tr[:, i], pos_weight=pos_w[i].unsqueeze(0)
            ))
        loss = torch.stack(losses).mean()
        loss.backward()
        optim.step()
        scheduler.step()
        train_hist.append(loss.detach().item())

        model.eval()
        with torch.no_grad():
            v_logits = model(X_va, edge_index) if is_gnn else model(X_va)
            v_losses = [
                F.binary_cross_entropy_with_logits(v_logits[:, i], y_va[:, i])
                for i in range(len(PATHOLOGY_NAMES))
            ]
            v_loss = torch.stack(v_losses).mean().item()
        val_hist.append(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    return BaselineResult(
        model_name=model_name,
        train_loss_history=train_hist,
        val_loss_history=val_hist,
        best_epoch=best_epoch,
    )
