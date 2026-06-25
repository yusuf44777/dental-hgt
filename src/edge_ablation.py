"""Edge-type ablation experiment for DentalHGT.

Trains DentalHGT 6 times:
  1. Full model (all edges)
  2. Remove mesial_distal (+ rev_mesial_distal)
  3. Remove bilateral (+ rev_bilateral)
  4. Remove antagonist (+ rev_antagonist)
  5. Remove member_of (tooth→quadrant hierarchy)
  6. Remove part_of (quadrant→arch hierarchy)

Results saved to results/edge_ablation.csv.

Usage:
    python -m src.edge_ablation [--config configs/default.json] [--backbone resnet50|dinov2]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .backbone import build_extractor
from .config import ensure_dirs, load_config
from .data_io import load_dentex_annotations
from .evaluate import evaluate_predictions
from .hgt_model import build_model
from .train_hgt import build_all_graphs, train_hgt

ABLATION_CONDITIONS = [
    ("Full model",       set()),
    ("-mesial_distal",   {"mesial_distal"}),
    ("-bilateral",       {"bilateral"}),
    ("-antagonist",      {"antagonist"}),
    ("-member_of",       {"member_of"}),
    ("-part_of",         {"part_of"}),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_ablation(config: dict, backbone_override: str | None = None) -> None:
    if backbone_override:
        config["backbone"]["name"] = backbone_override

    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dirs(config)
    results_dir = Path(config["paths"]["results_dir"])
    backbone_tag = config["backbone"]["name"]

    print(f"Device: {device} | Backbone: {backbone_tag}")
    print("\n[1] Loading annotations...")
    train_images = load_dentex_annotations(config["dataset"]["train_json"])
    val_images   = load_dentex_annotations(config["dataset"]["val_json"])
    print(f"  Train: {len(train_images)} images | Val: {len(val_images)} images")

    rows = []
    for label, excl in ABLATION_CONDITIONS:
        print(f"\n{'='*60}")
        print(f"Condition: {label}  (exclude: {excl or 'none'})")
        print(f"{'='*60}")

        set_seed(seed)
        model     = build_model(config)
        extractor = build_extractor(config)

        safe_label = label.replace(" ", "_").replace("-", "no_")
        ckpt_name  = f"hgt_ablation_{safe_label}_{backbone_tag}.pt"
        hist_name  = f"hgt_ablation_{safe_label}_{backbone_tag}_hist.csv"

        train_hgt(
            model, extractor,
            train_images, val_images,
            config, device, results_dir,
            augment_train=bool(config["preprocess"]["augmentation"]),
            exclude_edge_types=excl if excl else None,
            ckpt_name=ckpt_name,
            hist_name=hist_name,
        )

        # Evaluate best checkpoint on val set.
        # IMPORTANT: reuse the same extractor instance — its projection head
        # weights are what the HGT model was trained on; a fresh extractor would
        # have different random weights and produce incompatible features.
        model.load_state_dict(torch.load(results_dir / ckpt_name, map_location=device))
        model.eval()
        val_artifacts = build_all_graphs(
            val_images, extractor, config, device,
            image_dir=config["dataset"]["val_image_dir"],
            augment=False, verbose=False,
            exclude_edge_types=excl if excl else None,
        )
        all_probs, all_gt = [], []
        with torch.no_grad():
            for art in val_artifacts:
                b = art.data.to(device)
                probs = model.predict_proba(b.x_dict, b.edge_index_dict).cpu().numpy()
                all_probs.append(probs)
                all_gt.append(art.tooth_labels)

        ev = evaluate_predictions(
            np.concatenate(all_gt),
            np.concatenate(all_probs),
            label,
        )
        print(f"  => mAP={ev.map_score:.4f}, AUROC={ev.mean_auroc:.4f}")
        rows.append({
            "removed_edge": label,
            "backbone": backbone_tag,
            "mAP": ev.map_score,
            "auroc": ev.mean_auroc,
        })

    df = pd.DataFrame(rows)
    out_path = results_dir / f"edge_ablation_{backbone_tag}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nEdge ablation results → {out_path}")
    print(df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Edge-type ablation for DentalHGT")
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--backbone", choices=["resnet50", "dinov2"], default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    run_ablation(cfg, backbone_override=args.backbone)
