"""Publication-quality figures for DentalHGT paper.

Mirrors the figures.py structure from the scRNA-seq study (makale/).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from .hgt_model import PATHOLOGY_NAMES
from .data_io import PATHOLOGY_ID_TO_NAME

PALETTE = {
    "DentalHGT": "#2196F3",
    "IndependentCNN": "#FF9800",
    "HomogeneousGNN": "#9C27B0",
    "YOLOv8": "#4CAF50",
}


def _save(fig: plt.Figure, path: str | Path, dpi: int = 300) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {path}")


# ---------------------------------------------------------------------------
# 1. Workflow diagram (schematic, text-based)
# ---------------------------------------------------------------------------

def plot_workflow(figures_dir: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis("off")
    steps = [
        ("Panoramic\nX-ray", 0.05),
        ("Tooth ROI\nExtraction\n(COCO bbox)", 0.22),
        ("CNN Backbone\n(ResNet50 /\nDINOv2)", 0.40),
        ("Heterogeneous\nDental Graph\n(HGTConv)", 0.58),
        ("Multi-label\nPathology\nPrediction", 0.76),
        ("Evaluation\n(mAP, AUROC,\nDeLong)", 0.93),
    ]
    colors = ["#BBDEFB", "#C8E6C9", "#FFE0B2", "#E1BEE7", "#FFCDD2", "#F5F5F5"]
    for (label, x), color in zip(steps, colors):
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color, edgecolor="#555", linewidth=1.2),
                transform=ax.transAxes)
        if x < 0.93:
            ax.annotate("", xy=(x + 0.12, 0.5), xytext=(x + 0.06, 0.5),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
    ax.set_title("DentalHGT Pipeline", fontsize=13, fontweight="bold", pad=12)
    _save(fig, Path(figures_dir) / "workflow.png")


# ---------------------------------------------------------------------------
# 2. Heterogeneous graph schema
# ---------------------------------------------------------------------------

def plot_graph_schema(figures_dir: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    def node(x, y, label, color, radius=0.5):
        circle = plt.Circle((x, y), radius, color=color, ec="black", lw=1.2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=4)

    def edge(x1, y1, x2, y2, label, color="gray", style="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5, linestyle=style))
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.15, label, ha="center", va="bottom", fontsize=7.5, color=color)

    # Arch nodes (top)
    node(2.5, 5, "Arch\n(maxilla)", "#BBDEFB")
    node(7.5, 5, "Arch\n(mandible)", "#BBDEFB")

    # Quadrant nodes
    for i, (x, y, q) in enumerate([(1, 3.5, "Q1\nUR"), (4, 3.5, "Q2\nUL"),
                                     (6, 3.5, "Q3\nLL"), (9, 3.5, "Q4\nLR")]):
        node(x, y, q, "#C8E6C9")

    # Tooth nodes
    for i, (x, y) in enumerate([(0.5, 1.8), (1.5, 1.8), (2.5, 1.8),
                                  (3.5, 1.8), (5.5, 1.8), (7.5, 1.8)]):
        node(x, y, f"T{i+1}", "#FFE0B2", radius=0.38)

    # Edges: quadrant → arch
    edge(1, 3.5, 2.5, 5, "part_of", "#1565C0")
    edge(4, 3.5, 2.5, 5, "part_of", "#1565C0")
    edge(6, 3.5, 7.5, 5, "part_of", "#1565C0")
    edge(9, 3.5, 7.5, 5, "part_of", "#1565C0")

    # Tooth → quadrant
    edge(0.5, 1.8, 1, 3.5, "member_of", "#E65100")
    edge(1.5, 1.8, 1, 3.5, "member_of", "#E65100")

    # Tooth mesial-distal
    edge(0.5, 1.8, 1.5, 1.8, "mesial\ndistal", "#2E7D32")
    edge(1.5, 1.8, 2.5, 1.8, "mesial\ndistal", "#2E7D32")

    # Bilateral (dashed)
    ax.annotate("", xy=(3.5, 1.8), xytext=(0.5, 1.8),
                arrowprops=dict(arrowstyle="<->", color="#7B1FA2", lw=1.5, linestyle="dashed"))
    ax.text(2, 1.3, "bilateral", ha="center", color="#7B1FA2", fontsize=7.5)

    legend_elems = [
        mpatches.Patch(color="#BBDEFB", label="arch node"),
        mpatches.Patch(color="#C8E6C9", label="quadrant node"),
        mpatches.Patch(color="#FFE0B2", label="tooth node"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8)
    ax.set_title("Heterogeneous Dental Graph Schema", fontsize=13, fontweight="bold")
    _save(fig, Path(figures_dir) / "hetero_graph_schema.png")


# ---------------------------------------------------------------------------
# 3. Per-pathology AUROC bar chart
# ---------------------------------------------------------------------------

def plot_auroc_comparison(
    metrics_df: pd.DataFrame,
    figures_dir: str | Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    models = metrics_df["model"].unique()
    colors = [PALETTE.get(m, "#607D8B") for m in models]
    for ax, path_name in zip(axes, PATHOLOGY_NAMES):
        subset = metrics_df[metrics_df["pathology"] == path_name]
        bars = ax.bar(range(len(models)), subset["auroc"].values, color=colors, width=0.6, edgecolor="black", linewidth=0.8)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace("Independent", "Indep.\n") for m in models], fontsize=7.5, rotation=20, ha="right")
        ax.set_ylim(0.5, 1.0)
        ax.set_title(path_name.replace("_", " ").title(), fontsize=9.5)
        ax.set_xlabel("")
        ax.yaxis.grid(True, alpha=0.4)
        for bar, val in zip(bars, subset["auroc"].values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("AUROC", fontsize=10)
    fig.suptitle("Per-Pathology AUROC — Model Comparison", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, Path(figures_dir) / "auroc_comparison.png")


# ---------------------------------------------------------------------------
# 4. mAP summary table figure
# ---------------------------------------------------------------------------

def plot_map_table(
    metrics_df: pd.DataFrame,
    figures_dir: str | Path,
) -> None:
    summary = metrics_df[metrics_df["pathology"] == "MEAN"][["model", "ap", "auroc"]].copy()
    summary.columns = ["Model", "mAP", "Mean AUROC"]
    summary = summary.sort_values("mAP", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 2 + 0.4 * len(summary)))
    ax.axis("off")
    tbl = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold")
        elif summary.iloc[row - 1]["Model"] == "DentalHGT":
            cell.set_facecolor("#E3F2FD")
    ax.set_title("Summary: mAP and Mean AUROC", fontsize=11, fontweight="bold", pad=16)
    _save(fig, Path(figures_dir) / "map_summary_table.png")


# ---------------------------------------------------------------------------
# 5. Training curve
# ---------------------------------------------------------------------------

def plot_training_curve(
    hist_df: pd.DataFrame,
    figures_dir: str | Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(hist_df["epoch"], hist_df["train_loss"], label="Train", color="#1565C0")
    axes[0].plot(hist_df["epoch"], hist_df["val_loss"], label="Val", color="#E65100")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / Validation Loss"); axes[0].legend()

    if "val_mAP" in hist_df.columns:
        axes[1].plot(hist_df["epoch"], hist_df["val_mAP"], color="#2E7D32")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mAP")
        axes[1].set_title("Validation mAP")

    fig.suptitle("DentalHGT Training Curve", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, Path(figures_dir) / "training_curve.png")


# ---------------------------------------------------------------------------
# 6. Edge type ablation
# ---------------------------------------------------------------------------

def plot_edge_ablation(
    ablation_df: pd.DataFrame,   # columns: removed_edge, mAP, auroc
    figures_dir: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(ablation_df))
    ax.bar(x, ablation_df["mAP"].values, color="#90CAF9", edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ablation_df["removed_edge"].values, rotation=25, ha="right")
    ax.set_ylabel("mAP")
    ax.set_title("Edge Type Ablation (remove one edge type at a time)", fontweight="bold")
    ax.yaxis.grid(True, alpha=0.4)
    ax.axhline(ablation_df[ablation_df["removed_edge"] == "Full model"]["mAP"].values[0],
               color="red", linestyle="--", label="Full model")
    ax.legend()
    _save(fig, Path(figures_dir) / "edge_ablation.png")


# ---------------------------------------------------------------------------
# Backbone ablation
# ---------------------------------------------------------------------------

def plot_backbone_ablation(
    resnet_result,   # EvaluationResult
    dinov2_result,   # EvaluationResult
    figures_dir: str | Path,
) -> None:
    data = {
        "Model": ["ResNet50", "DINOv2 ViT-S/14"],
        "mAP": [resnet_result.map_score, dinov2_result.map_score],
        "Mean AUROC": [resnet_result.mean_auroc, dinov2_result.mean_auroc],
    }
    df = pd.DataFrame(data)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, metric in zip(axes, ["mAP", "Mean AUROC"]):
        bars = ax.bar(df["Model"], df[metric], color=["#2196F3", "#FF9800"],
                      edgecolor="black", linewidth=0.8, width=0.5)
        ax.set_ylabel(metric); ax.set_title(f"Backbone: {metric}")
        ax.set_ylim(0.5, 1.0)
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Backbone Ablation: ResNet50 vs DINOv2", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, Path(figures_dir) / "backbone_ablation.png")


# ---------------------------------------------------------------------------
# Generate all figures
# ---------------------------------------------------------------------------

def generate_all_figures(
    figures_dir: str | Path,
    metrics_csv: str | Path | None = None,
    training_hist_csv: str | Path | None = None,
) -> None:
    fig_dir = Path(figures_dir)
    print("[figures] Generating...")
    plot_workflow(fig_dir)
    plot_graph_schema(fig_dir)
    if metrics_csv and Path(metrics_csv).exists():
        df = pd.read_csv(metrics_csv)
        plot_auroc_comparison(df, fig_dir)
        plot_map_table(df, fig_dir)
    if training_hist_csv and Path(training_hist_csv).exists():
        hist = pd.read_csv(training_hist_csv)
        plot_training_curve(hist, fig_dir)
    print(f"[figures] Done. → {fig_dir}/")


if __name__ == "__main__":
    import sys
    figs_dir = sys.argv[1] if len(sys.argv) > 1 else "figures"
    generate_all_figures(
        figs_dir,
        metrics_csv="results/all_metrics.csv",
        training_hist_csv="results/hgt_training_history.csv",
    )
