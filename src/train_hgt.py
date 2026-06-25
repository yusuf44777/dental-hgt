"""HGT training loop for dental pathology classification.

Architecture:
  - Pre-extract ALL backbone features once (frozen backbone)
  - Build one HeteroData graph per image (offline, at dataset start)
  - Mini-batch training with PyG Batch (batch_size images per step)
  - Train only HGT layers + classification heads (backbone proj optionally fine-tuned)

This mirrors the scRNA-seq study where node features were fixed PCA/HVG vectors
and only the HGT layers were learned.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .backbone import ToothFeatureExtractor, extract_features_batch
from .data_io import PanoramicImage, collect_all_teeth
from .evaluate import evaluate_predictions
from .graph_build import build_dental_graph, DentalGraphArtifact
from .hgt_model import DentalHGT, PATHOLOGY_NAMES, compute_loss, compute_pos_weights
from .preprocess import extract_all_rois


def build_all_graphs(
    images: dict[int, PanoramicImage],
    extractor: ToothFeatureExtractor,
    config: dict[str, Any],
    device: torch.device,
    image_dir: str,
    augment: bool = False,
    seed: int = 42,
    verbose: bool = True,
    exclude_edge_types: set[str] | None = None,
) -> list[DentalGraphArtifact]:
    """Pre-extract features and build one HeteroData graph per image.

    Returns list of DentalGraphArtifact (one per image, only images with teeth).
    exclude_edge_types: forwarded to build_dental_graph for ablation experiments.
    """
    extractor.eval()
    artifacts: list[DentalGraphArtifact] = []
    image_list = list(images.values())

    for i, img_obj in enumerate(image_list):
        if not img_obj.teeth:
            continue
        raw = img_obj.load_image(image_dir)
        rois = extract_all_rois(raw, img_obj.teeth, config, augment=augment, seed=seed + i)
        if not rois:
            continue
        with torch.no_grad():
            feats_np = extract_features_batch(extractor, rois, device, batch_size=64)
        artifact = build_dental_graph(img_obj, feats_np, config,
                                      exclude_edge_types=exclude_edge_types)
        artifacts.append(artifact)

    if verbose:
        total_teeth = sum(a.n_teeth for a in artifacts)
        print(f"    Built {len(artifacts)} graphs, {total_teeth} teeth total")
    return artifacts


def _batch_forward(
    model: DentalHGT,
    artifacts: list[DentalGraphArtifact],
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor]:
    """Stack a list of graphs into a PyG Batch and forward through the model.

    Returns:
        outputs : dict of {pathology_name: logit_tensor} for all teeth in batch
        labels  : [N_total_teeth, 4] float32
    """
    try:
        from torch_geometric.data import Batch
    except ImportError as exc:
        raise ImportError("torch_geometric required") from exc

    data_list = [a.data for a in artifacts]
    batch = Batch.from_data_list(data_list).to(device)
    labels = torch.tensor(
        np.concatenate([a.tooth_labels for a in artifacts]),
        dtype=torch.float32, device=device,
    )
    out = model(batch.x_dict, batch.edge_index_dict)
    return out, labels


def train_hgt(
    model: DentalHGT,
    extractor: ToothFeatureExtractor,
    train_images: dict[int, PanoramicImage],
    val_images: dict[int, PanoramicImage],
    config: dict[str, Any],
    device: torch.device,
    results_dir: str | Path,
    augment_train: bool = True,
    exclude_edge_types: set[str] | None = None,
    ckpt_name: str = "hgt_model.pt",
    hist_name: str = "hgt_training_history.csv",
) -> pd.DataFrame:
    """Full HGT training loop with graph batching.

    Saves best model to results_dir/hgt_model.pt.
    Returns per-epoch history DataFrame.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = config["training"]
    epochs      = int(train_cfg["epochs"])
    lr          = float(train_cfg["lr"])
    wd          = float(train_cfg["weight_decay"])
    patience    = int(train_cfg["patience"])
    batch_size  = int(train_cfg.get("graph_batch_size", train_cfg.get("batch_size", 16)))
    seed        = int(config["seed"])

    print("  Pre-extracting train features + building graphs...")
    train_artifacts = build_all_graphs(
        train_images, extractor, config, device,
        image_dir=config["dataset"]["train_image_dir"],
        augment=augment_train, seed=seed,
        exclude_edge_types=exclude_edge_types,
    )
    print("  Pre-extracting val features + building graphs...")
    val_artifacts = build_all_graphs(
        val_images, extractor, config, device,
        image_dir=config["dataset"]["val_image_dir"],
        augment=False, seed=seed,
        exclude_edge_types=exclude_edge_types,
    )

    # Class weights from training set
    all_labels = np.concatenate([a.tooth_labels for a in train_artifacts])
    pos_weights = compute_pos_weights(all_labels).to(device)

    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best_val_map  = -1.0
    patience_ctr  = 0
    history       = []
    rng           = np.random.default_rng(seed)

    for epoch in range(epochs):
        model.train()
        train_losses: list[float] = []
        indices = np.arange(len(train_artifacts))
        rng.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_arts = [train_artifacts[i] for i in indices[start : start + batch_size]]
            out, labels = _batch_forward(model, batch_arts, device)
            loss = compute_loss(out, labels, pos_weights)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            train_losses.append(loss.detach().item())

        scheduler.step()
        mean_train_loss = float(np.mean(train_losses))

        # Validation
        model.eval()
        val_losses: list[float] = []
        all_probs: list[np.ndarray] = []
        all_gt:    list[np.ndarray] = []

        with torch.no_grad():
            from torch_geometric.data import Batch as PyGBatch
            for start in range(0, len(val_artifacts), batch_size):
                batch_arts = val_artifacts[start : start + batch_size]
                out, labels = _batch_forward(model, batch_arts, device)
                val_losses.append(compute_loss(out, labels, pos_weights).item())
                b = PyGBatch.from_data_list([a.data for a in batch_arts]).to(device)
                probs = model.predict_proba(b.x_dict, b.edge_index_dict).cpu().numpy()
                all_probs.append(probs)
                all_gt.append(np.concatenate([a.tooth_labels for a in batch_arts]))

        mean_val_loss = float(np.mean(val_losses))
        y_prob = np.concatenate(all_probs)
        y_true = np.concatenate(all_gt)
        ev = evaluate_predictions(y_true, y_prob, "hgt_val")
        val_mAP = ev.map_score

        history.append({
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "val_loss": mean_val_loss,
            "val_mAP": val_mAP,
        })
        print(f"Epoch {epoch+1:3d}/{epochs} | train={mean_train_loss:.4f} val={mean_val_loss:.4f} mAP={val_mAP:.4f}")

        if val_mAP > best_val_map:
            best_val_map = val_mAP
            patience_ctr = 0
            torch.save(model.state_dict(), results_dir / ckpt_name)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch + 1} (best mAP={best_val_map:.4f})")
                break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(results_dir / hist_name, index=False)
    return hist_df
