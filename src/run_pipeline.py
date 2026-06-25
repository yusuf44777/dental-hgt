"""Main orchestration pipeline for DentalHGT.

Usage:
    python -m src.run_pipeline                        # full run
    python -m src.run_pipeline --config configs/default.json
    python -m src.run_pipeline --dry-run              # smoke test (1 image only)
    python -m src.run_pipeline --backbone dinov2      # override backbone
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from .backbone import build_extractor, extract_features_batch
from .baselines import (
    build_homogeneous_gnn,
    build_independent_cnn,
    train_tooth_classifier,
)
from .config import ensure_dirs, load_config
from .data_io import (
    dataset_summary,
    load_dentex_annotations,
)
from .evaluate import evaluate_predictions, save_metrics
from .hgt_model import build_model
from .preprocess import extract_all_rois
from .train_hgt import train_hgt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tooth_dataset(
    images: dict,
    extractor,
    config: dict,
    device: torch.device,
    image_dir: str,
    augment: bool = False,
    dry_run: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute features for all teeth across all images → (features, labels)."""
    all_feats, all_labels = [], []
    image_list = list(images.values())
    if dry_run:
        image_list = image_list[:1]
    for img_obj in image_list:
        if not img_obj.teeth:
            continue
        raw = img_obj.load_image(image_dir)
        rois = extract_all_rois(raw, img_obj.teeth, config, augment=augment, seed=0)
        if not rois:
            continue
        feats = extract_features_batch(extractor, rois, device)
        labels = np.stack([t.label_vector for t in img_obj.teeth])
        all_feats.append(feats)
        all_labels.append(labels)
    if not all_feats:
        raise RuntimeError("No tooth features extracted. Check dataset paths.")
    return np.concatenate(all_feats), np.concatenate(all_labels)


def run(config: dict, dry_run: bool = False) -> None:
    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    ensure_dirs(config)
    results_dir = Path(config["paths"]["results_dir"])

    # ------------------------------------------------------------------ #
    # 1. Load dataset
    # ------------------------------------------------------------------ #
    print("\n[1/6] Loading DENTEX annotations...")
    train_images = load_dentex_annotations(config["dataset"]["train_json"])
    val_images = load_dentex_annotations(config["dataset"]["val_json"])
    if dry_run:
        train_images = dict(list(train_images.items())[:2])
        val_images = dict(list(val_images.items())[:1])
    summary = dataset_summary(train_images, val_images)
    with open(results_dir / "dataset_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Train: {summary['train_images']} images, {summary['train_teeth']} teeth")
    print(f"  Val:   {summary['val_images']} images, {summary['val_teeth']} teeth")
    print(f"  Pathology counts (train): {summary['train_pathology_counts']}")

    # ------------------------------------------------------------------ #
    # 2. Build backbone + precompute flat features (for baselines)
    # ------------------------------------------------------------------ #
    print("\n[2/6] Extracting backbone features (ResNet50)...")
    extractor = build_extractor(config)
    extractor.eval()
    train_feats, train_labels = build_tooth_dataset(
        train_images, extractor, config, device,
        image_dir=config["dataset"]["train_image_dir"], augment=False, dry_run=dry_run)
    val_feats, val_labels = build_tooth_dataset(
        val_images, extractor, config, device,
        image_dir=config["dataset"]["val_image_dir"], dry_run=dry_run)
    np.savez_compressed(results_dir / "tooth_features.npz",
                        train_feats=train_feats, train_labels=train_labels,
                        val_feats=val_feats, val_labels=val_labels)
    print(f"  train teeth: {train_feats.shape}, val teeth: {val_feats.shape}")

    # ------------------------------------------------------------------ #
    # 3. Baseline: Independent CNN
    # ------------------------------------------------------------------ #
    print("\n[3/6] Training Independent CNN baseline...")
    cnn = build_independent_cnn(config)
    train_tooth_classifier(cnn, train_feats, train_labels, val_feats, val_labels,
                           config, device, model_name="independent_cnn")
    cnn.eval()
    with torch.no_grad():
        cnn_probs = cnn.predict_proba(torch.tensor(val_feats, device=device)).cpu().numpy()
    cnn_result = evaluate_predictions(val_labels, cnn_probs, "IndependentCNN")
    print(f"  mAP={cnn_result.map_score:.4f}, AUROC={cnn_result.mean_auroc:.4f}")
    torch.save(cnn.state_dict(), results_dir / "independent_cnn_model.pt")

    # ------------------------------------------------------------------ #
    # 4. Baseline: Homogeneous GNN
    # ------------------------------------------------------------------ #
    print("\n[4/6] Training Homogeneous GNN baseline...")
    hom_gnn = build_homogeneous_gnn(config)

    # Flat-feature ablation: all images concatenated, no graph edges across images.
    # Pass empty edge_index so GATConv still runs but with zero edges (= no propagation).
    empty_edge = torch.zeros((2, 0), dtype=torch.long)
    train_tooth_classifier(hom_gnn, train_feats, train_labels, val_feats, val_labels,
                           config, device, model_name="homogeneous_gnn", edge_index=empty_edge)
    hom_gnn.eval()
    with torch.no_grad():
        hom_probs = hom_gnn.predict_proba(
            torch.tensor(val_feats, device=device),
            torch.zeros((2, 0), dtype=torch.long, device=device),
        ).cpu().numpy()
    hom_result = evaluate_predictions(val_labels, hom_probs, "HomogeneousGNN")
    print(f"  mAP={hom_result.map_score:.4f}, AUROC={hom_result.mean_auroc:.4f}")
    torch.save(hom_gnn.state_dict(), results_dir / "homogeneous_gnn_model.pt")

    # ------------------------------------------------------------------ #
    # 5. Train DentalHGT
    # ------------------------------------------------------------------ #
    print("\n[5/6] Training DentalHGT...")
    hgt = build_model(config)
    # Reuse the same extractor used for CNN/GNN feature extraction so that all
    # models see identical 128-dim projected features. Saving the extractor
    # checkpoint enables post-hoc inference without re-training.
    torch.save(extractor.state_dict(), results_dir / "extractor.pt")
    train_hgt(hgt, extractor, train_images, val_images, config, device, results_dir,
              augment_train=bool(config["preprocess"]["augmentation"]))

    # Load best checkpoint and re-evaluate on val set
    hgt.load_state_dict(torch.load(results_dir / "hgt_model.pt", map_location=device))
    hgt.eval()

    from .train_hgt import build_all_graphs
    val_artifacts = build_all_graphs(
        val_images, extractor, config, device,
        image_dir=config["dataset"]["val_image_dir"],
        augment=False, verbose=False,
    )
    all_probs, all_gt = [], []
    with torch.no_grad():
        for art in val_artifacts:
            b = art.data.to(device)
            probs = hgt.predict_proba(b.x_dict, b.edge_index_dict).cpu().numpy()
            all_probs.append(probs)
            all_gt.append(art.tooth_labels)

    hgt_probs = np.concatenate(all_probs)
    hgt_gt = np.concatenate(all_gt)
    hgt_result = evaluate_predictions(hgt_gt, hgt_probs, "DentalHGT")
    print(f"  mAP={hgt_result.map_score:.4f}, AUROC={hgt_result.mean_auroc:.4f}")

    # ------------------------------------------------------------------ #
    # 6. Save all results
    # ------------------------------------------------------------------ #
    print("\n[6/6] Saving results...")
    backbone_tag = config["backbone"]["name"]
    metrics_fname = f"all_metrics_{backbone_tag}.csv"
    save_metrics([cnn_result, hom_result, hgt_result], str(results_dir / metrics_fname))
    # Also save/overwrite the generic file for backward compatibility
    save_metrics([cnn_result, hom_result, hgt_result], str(results_dir / "all_metrics.csv"))
    print("\nPipeline complete.")
    print(f"Results → {results_dir}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DentalHGT pipeline")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--dry-run", action="store_true", help="Run on 2 images only")
    parser.add_argument("--backbone", choices=["resnet50", "dinov2"], default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    if args.backbone:
        config["backbone"]["name"] = args.backbone
    run(config, dry_run=args.dry_run)
