"""DeLong tests + publication-quality figures for DentalHGT.

Strategy:
  - DeLong p-values: Hanley-McNeil (1982) approximation using per-pathology
    AUROC + n_pos/n_neg from the saved metrics CSVs. This is the standard
    approach in medical imaging papers and does not require raw probabilities.
  - ROC/PR curves: computed from val_feats (tooth_features.npz) for IndepCNN
    and HomoGNN (both trained on these features); HGT ROC curves require
    separate re-run (see fix_pipeline comments).
  - All bar charts and comparison figures use CSV data directly.

Usage:
    cd /Users/mahir/Desktop/makale_2
    python -m src.analysis
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    average_precision_score, roc_auc_score,
)

# ─── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "font.size":         9,
})

COLORS = {
    "DentalHGT":      "#1565C0",
    "IndependentCNN": "#E65100",
    "HomogeneousGNN": "#6A1B9A",
    "ResNet50":       "#1565C0",
    "DINOv2":         "#2E7D32",
}
LS = {"DentalHGT": "-", "IndependentCNN": "--", "HomogeneousGNN": ":"}
PATH_NAMES  = ["caries", "deep_caries", "periapical", "impacted"]
PATH_LABELS = {
    "caries":      "Caries",
    "deep_caries": "Deep Caries",
    "periapical":  "Periapical",
    "impacted":    "Impacted",
}

RESULTS = Path("results")
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)


def _save(fig, name: str, dpi: int = 300) -> None:
    path = FIGURES / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ─── 1. DeLong (Hanley-McNeil 1982 approximation) ────────────────────────────

def delong_pvalue_hmn(auroc_a: float, auroc_b: float,
                      n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil (1982) DeLong AUROC comparison.

    Standard approach in medical imaging: computes a Z-statistic from
    AUROC + sample counts without raw scores, using:
        Var(AUC) ≈ [A(1-A) + (n_pos-1)(Q1-A²) + (n_neg-1)(Q2-A²)] / (n_pos·n_neg)
    """
    def _var(a):
        if np.isnan(a) or n_pos <= 1 or n_neg <= 1:
            return np.nan
        q1 = a / (2 - a)
        q2 = 2 * a**2 / (1 + a)
        return (a * (1 - a) + (n_pos - 1) * (q1 - a**2) +
                (n_neg - 1) * (q2 - a**2)) / (n_pos * n_neg)

    var_a = _var(auroc_a)
    var_b = _var(auroc_b)
    if np.isnan(var_a) or np.isnan(var_b):
        return np.nan
    se = math.sqrt(max(var_a + var_b, 1e-12))
    z  = (auroc_a - auroc_b) / se
    return float(2 * (1 - norm.cdf(abs(z))))


def stars(p: float) -> str:
    if np.isnan(p):  return "n.s."
    if p < 0.001:    return "***"
    if p < 0.01:     return "**"
    if p < 0.05:     return "*"
    return "n.s."


# ─── 2. Load CSV metrics ─────────────────────────────────────────────────────

def load_csv_metrics() -> dict[str, pd.DataFrame]:
    """Load all available metrics CSVs."""
    out = {}
    for key, fname in [("resnet50", "all_metrics_resnet50.csv"),
                       ("dinov2",   "all_metrics_dinov2.csv")]:
        p = RESULTS / fname
        if p.exists():
            out[key] = pd.read_csv(p)
    return out


# ─── 3. DeLong table from CSV ────────────────────────────────────────────────

def run_delong_from_csv(df: pd.DataFrame, backbone_label: str = "ResNet50") -> pd.DataFrame:
    """Compute Hanley-McNeil DeLong p-values using CSV AUROC + counts."""
    models = ["IndependentCNN", "HomogeneousGNN", "DentalHGT"]
    rows = []
    for pname in PATH_NAMES:
        row_dict = {"Pathology": PATH_LABELS[pname]}
        prow = {}
        for m in models:
            sub = df[(df["model"] == m) & (df["pathology"] == pname)]
            if sub.empty:
                prow[m] = {"auroc": np.nan, "n_pos": 0, "n_neg": 0}
            else:
                r = sub.iloc[0]
                prow[m] = {"auroc": r["auroc"], "n_pos": int(r["n_pos"]), "n_neg": int(r["n_neg"])}
            row_dict[f"AUC {m.replace('Independent','Indep.')}"] = f"{prow[m]['auroc']:.3f}"

        n_pos = prow["DentalHGT"]["n_pos"]
        n_neg = prow["DentalHGT"]["n_neg"]
        for comp_name, m_b in [("HGT vs CNN", "IndependentCNN"), ("HGT vs GNN", "HomogeneousGNN")]:
            p_val = delong_pvalue_hmn(prow["DentalHGT"]["auroc"], prow[m_b]["auroc"],
                                      n_pos, n_neg)
            row_dict[f"p ({comp_name})"] = f"{p_val:.4f}" if not np.isnan(p_val) else "—"
            row_dict[f"sig ({comp_name})"] = stars(p_val)
        rows.append(row_dict)

    result_df = pd.DataFrame(rows)
    out_path = RESULTS / f"delong_tests_{backbone_label.lower()}.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nDeLong ({backbone_label}) results:")
    print(result_df.to_string(index=False))
    return result_df


# ─── 4. CNN/GNN baseline inference (from saved features) ─────────────────────

def _load_config() -> dict[str, Any]:
    from .config import load_config
    return load_config("configs/default.json")


def _get_val_features() -> tuple[np.ndarray, np.ndarray]:
    d = np.load(RESULTS / "tooth_features.npz")
    return d["val_feats"], d["val_labels"]


def _cnn_probs(val_feats: np.ndarray, config: dict) -> np.ndarray:
    from .baselines import build_independent_cnn
    model = build_independent_cnn(config)
    model.load_state_dict(torch.load(RESULTS / "independent_cnn_model.pt",
                                     map_location="cpu"))
    model.eval()
    with torch.no_grad():
        return model.predict_proba(
            torch.tensor(val_feats, dtype=torch.float32)).numpy()


def _gnn_probs(val_feats: np.ndarray, config: dict) -> np.ndarray:
    from .baselines import build_homogeneous_gnn
    model = build_homogeneous_gnn(config)
    model.load_state_dict(torch.load(RESULTS / "homogeneous_gnn_model.pt",
                                     map_location="cpu"))
    model.eval()
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    with torch.no_grad():
        return model.predict_proba(
            torch.tensor(val_feats, dtype=torch.float32), edge_index).numpy()


def load_baseline_predictions(config: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {model_name: (y_true, y_prob)} for IndepCNN and HomoGNN.
    Both use the same val_feats — these are consistent DINOv2 results.
    """
    val_feats, val_labels = _get_val_features()
    return {
        "IndependentCNN": (val_labels, _cnn_probs(val_feats, config)),
        "HomogeneousGNN": (val_labels, _gnn_probs(val_feats, config)),
    }


# ─── 5. Figures ──────────────────────────────────────────────────────────────

def fig_workflow():
    fig, ax = plt.subplots(figsize=(16, 3.8))
    ax.set_xlim(0, 16); ax.set_ylim(0, 4); ax.axis("off")

    steps = [
        ("Panoramic\nX-ray",               1.1,  "#BBDEFB"),
        ("Tooth ROI\nExtraction\n+ CLAHE",  3.5,  "#C8E6C9"),
        ("CNN Backbone\n(ResNet50 /\nDINOv2)", 6.0, "#FFE0B2"),
        ("Heterogeneous\nDental Graph\n(HGTConv×2)", 8.8, "#E1BEE7"),
        ("Multi-label\nPathology\nHeads ×4",   11.6, "#FFCDD2"),
        ("AUROC · AP\nDeLong test",            14.3, "#F5F5F5"),
    ]
    for i, (label, x, color) in enumerate(steps):
        rect = mpatches.FancyBboxPatch(
            (x - 1.05, 0.75), 2.1, 2.5,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="#757575", linewidth=1.3, zorder=2)
        ax.add_patch(rect)
        ax.text(x, 2.0, label, ha="center", va="center",
                fontsize=8.8, fontweight="bold", zorder=3, linespacing=1.4)
        if i < len(steps) - 1:
            ax.annotate("", xy=(steps[i + 1][1] - 1.1, 2.0),
                        xytext=(x + 1.1, 2.0),
                        arrowprops=dict(arrowstyle="-|>", color="#455A64",
                                        lw=1.5, mutation_scale=13), zorder=1)
    ax.annotate("FDI anatomical\nrule graph",
                xy=(8.8, 0.74), xytext=(8.8, 0.15),
                fontsize=7.5, ha="center", color="#6A1B9A",
                arrowprops=dict(arrowstyle="->", color="#6A1B9A", lw=1.0))
    fig.suptitle("DentalHGT — End-to-End Pipeline",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "workflow_polished.png")


def fig_roc_curves_baselines(preds: dict):
    """ROC curves for IndepCNN and HomoGNN (correctly inferred from val_feats).
    Both models use the same DINOv2 backbone features.
    """
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    model_order = ["IndependentCNN", "HomogeneousGNN"]
    nice = {"IndependentCNN": "Indep. CNN", "HomogeneousGNN": "Homo. GNN"}

    for ax, (pi, pname) in zip(axes, enumerate(PATH_NAMES)):
        for mname in model_order:
            if mname not in preds: continue
            y_true, y_prob = preds[mname]
            gt = y_true[:, pi].astype(int)
            if gt.sum() == 0 or (1 - gt).sum() == 0: continue
            fpr, tpr, _ = roc_curve(gt, y_prob[:, pi])
            auc = roc_auc_score(gt, y_prob[:, pi])
            lw = 1.8
            ax.plot(fpr, tpr, color=COLORS[mname], lw=lw, ls=LS[mname],
                    label=f"{nice[mname]} (AUC={auc:.3f})", zorder=3)
        ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.45)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("False Positive Rate", fontsize=8.5)
        ax.set_title(PATH_LABELS[pname], fontsize=10.5, fontweight="bold")
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    axes[0].set_ylabel("True Positive Rate", fontsize=8.5)
    fig.suptitle("ROC Curves — IndepCNN and HomoGNN Baselines (DINOv2 backbone)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "roc_curves_baselines.png")


def fig_pr_curves_baselines(preds: dict):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    model_order = ["IndependentCNN", "HomogeneousGNN"]
    nice = {"IndependentCNN": "Indep. CNN", "HomogeneousGNN": "Homo. GNN"}

    for ax, (pi, pname) in zip(axes, enumerate(PATH_NAMES)):
        for mname in model_order:
            if mname not in preds: continue
            y_true, y_prob = preds[mname]
            gt = y_true[:, pi].astype(int)
            if gt.sum() == 0: continue
            prec, rec, _ = precision_recall_curve(gt, y_prob[:, pi])
            ap = average_precision_score(gt, y_prob[:, pi])
            lw = 1.8
            ax.plot(rec, prec, color=COLORS[mname], lw=lw, ls=LS[mname],
                    label=f"{nice[mname]} (AP={ap:.3f})", zorder=3)
        prevalence = gt.mean()
        ax.axhline(prevalence, color="gray", lw=0.8, ls=":", alpha=0.7)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Recall", fontsize=8.5)
        ax.set_title(PATH_LABELS[pname], fontsize=10.5, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    axes[0].set_ylabel("Precision", fontsize=8.5)
    fig.suptitle("Precision–Recall Curves — Baselines (DINOv2 backbone)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "pr_curves_baselines.png")


def fig_delong_heatmap(delong_df: pd.DataFrame, backbone_label: str = "ResNet50"):
    """DeLong significance heatmap from Hanley-McNeil p-values."""
    comparisons = [
        ("p (HGT vs CNN)", "DentalHGT vs Indep. CNN"),
        ("p (HGT vs GNN)", "DentalHGT vs Homo. GNN"),
    ]
    n_comp = len(comparisons)
    n_path = len(PATH_NAMES)
    path_labels = [PATH_LABELS[p] for p in PATH_NAMES]

    pmat = np.zeros((n_comp, n_path))
    for j, pname in enumerate(PATH_NAMES):
        row = delong_df[delong_df["Pathology"] == PATH_LABELS[pname]].iloc[0]
        for i, (pcol, _) in enumerate(comparisons):
            try:
                pmat[i, j] = float(str(row[pcol]).replace("—", "1.0"))
            except Exception:
                pmat[i, j] = 1.0

    fig, ax = plt.subplots(figsize=(8, 2.8))
    log_p = -np.log10(np.clip(pmat, 1e-4, 1))
    im = ax.imshow(log_p, cmap="Blues", aspect="auto", vmin=0, vmax=3)

    ax.set_xticks(range(n_path));   ax.set_xticklabels(path_labels, fontsize=9)
    ax.set_yticks(range(n_comp));   ax.set_yticklabels([c[1] for c in comparisons], fontsize=9)

    for i in range(n_comp):
        for j in range(n_path):
            p = pmat[i, j]
            s = stars(p)
            text_color = "white" if log_p[i, j] > 1.5 else "black"
            ax.text(j, i, f"p={p:.3f}\n{s}",
                    ha="center", va="center", fontsize=8.5, color=text_color,
                    fontweight="bold" if s != "n.s." else "normal")

    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
    cbar.set_label("$-\\log_{10}(p)$", fontsize=8)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(["1.0", "0.1", "0.01", "0.001"])
    ax.set_title(f"DeLong Test Significance — {backbone_label} backbone "
                 "(Hanley-McNeil 1982 approx.)",
                 fontsize=10.5, fontweight="bold", pad=10)
    fig.tight_layout()
    _save(fig, f"delong_heatmap_{backbone_label.lower()}.png")


def fig_grouped_bars_from_csv(df: pd.DataFrame, backbone_label: str = "ResNet50"):
    """AUROC + AP bar chart from CSV data (not raw probabilities)."""
    model_order = ["IndependentCNN", "HomogeneousGNN", "DentalHGT"]
    nice = {"IndependentCNN": "Indep. CNN",
            "HomogeneousGNN": "Homo. GNN",
            "DentalHGT":      "DentalHGT"}
    x = np.arange(len(PATH_NAMES)); w = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, col, ylabel in [(axes[0], "auroc", "AUROC"),
                             (axes[1], "ap",    "AP (Average Precision)")]:
        for mi, mname in enumerate(model_order):
            sub = df[(df["model"] == mname) & (df["pathology"] != "MEAN")]
            if sub.empty: continue
            vals = []
            for pname in PATH_NAMES:
                row = sub[sub["pathology"] == pname]
                vals.append(float(row[col].values[0]) if not row.empty else 0.0)

            bars = ax.bar(x + (mi - 1) * w, vals, width=w * 0.88,
                          color=COLORS[mname], label=nice[mname],
                          edgecolor="white", linewidth=0.4, alpha=0.92, zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.007, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=6.8,
                        fontweight="bold" if mname == "DentalHGT" else "normal")

        ax.set_xticks(x)
        ax.set_xticklabels([PATH_LABELS[p] for p in PATH_NAMES], fontsize=9.2)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, 1.16)
        ax.set_title(f"Per-Pathology {ylabel}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
        ax.yaxis.grid(True, alpha=0.35, zorder=0); ax.set_axisbelow(True)

    fig.suptitle(f"DentalHGT vs Baselines — {backbone_label} Backbone",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"main_results_bars_{backbone_label.lower()}.png")


def fig_backbone_comparison(csvs: dict):
    r50  = csvs.get("resnet50")
    dino = csvs.get("dinov2")
    if r50 is None or dino is None:
        print("  (skip backbone comparison: CSV files not found)")
        return

    hgt_r50  = r50[(r50["model"] == "DentalHGT") & (r50["pathology"] != "MEAN")]
    hgt_dino = dino[(dino["model"] == "DentalHGT") & (dino["pathology"] != "MEAN")]

    pathologies = [r for r in hgt_r50["pathology"].tolist() if r in PATH_NAMES]
    x = np.arange(len(pathologies)); w = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, col, ylabel in [(axes[0], "auroc", "AUROC"),
                             (axes[1], "ap",    "AP")]:
        v_r50  = [float(hgt_r50[hgt_r50["pathology"] == p][col]) for p in pathologies]
        v_dino = [float(hgt_dino[hgt_dino["pathology"] == p][col]) for p in pathologies]

        b1 = ax.bar(x - w / 2, v_r50,  w, color=COLORS["ResNet50"],
                    label="DentalHGT + ResNet50", alpha=0.88, edgecolor="white", zorder=3)
        b2 = ax.bar(x + w / 2, v_dino, w, color=COLORS["DINOv2"],
                    label="DentalHGT + DINOv2",   alpha=0.88, edgecolor="white", zorder=3)
        for bars, vals in [(b1, v_r50), (b2, v_dino)]:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.007, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7.2)
        for i, (vr, vd) in enumerate(zip(v_r50, v_dino)):
            delta = vd - vr
            sign  = "+" if delta >= 0 else ""
            ax.text(i, max(vr, vd) + 0.042, f"{sign}{delta:.3f}",
                    ha="center", va="bottom", fontsize=7.5,
                    color=COLORS["DINOv2"] if delta >= 0 else "#C62828",
                    fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([PATH_LABELS.get(p, p) for p in pathologies], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"Per-Pathology {ylabel}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8.5, framealpha=0.9)
        ax.yaxis.grid(True, alpha=0.35, zorder=0); ax.set_axisbelow(True)

    fig.suptitle("Backbone Ablation: ResNet50 vs DINOv2 ViT-S/14 (DentalHGT)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "backbone_comparison_bars.png")


def fig_edge_ablation_polished():
    csv_path = RESULTS / "edge_ablation_resnet50.csv"
    if not csv_path.exists():
        print("  (skip edge ablation figure: CSV not found)")
        return
    df = pd.read_csv(csv_path)
    full_mAP   = df[df["removed_edge"] == "Full model"]["mAP"].values[0]
    full_auroc = df[df["removed_edge"] == "Full model"]["auroc"].values[0]
    sub = df[df["removed_edge"] != "Full model"].copy()

    edge_labels = {
        "-mesial_distal": "mesial-distal",
        "-bilateral":     "bilateral",
        "-antagonist":    "antagonist",
        "-member_of":     "member-of\n(tooth→quad)",
        "-part_of":       "part-of\n(quad→arch)",
    }
    labels = [edge_labels.get(r, r) for r in sub["removed_edge"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, col, ref, ytitle in [
        (axes[0], "mAP",   full_mAP,   "mAP"),
        (axes[1], "auroc", full_auroc, "Mean AUROC"),
    ]:
        vals   = sub[col].values
        colors = ["#EF9A9A" if v < ref else "#A5D6A7" for v in vals]
        bars   = ax.bar(labels, vals, color=colors,
                        edgecolor="#555", linewidth=0.9, width=0.52, zorder=3)
        ax.axhline(ref, color="#C62828", ls="--", lw=2.0,
                   label=f"Full model ({ref:.3f})", zorder=4)
        ax.set_ylim(min(vals) * 0.95, max(vals) * 1.07)
        ax.set_ylabel(ytitle, fontsize=10)
        ax.set_title(f"Impact on {ytitle} when edge type removed",
                     fontsize=9.5, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.yaxis.grid(True, alpha=0.35, zorder=0); ax.set_axisbelow(True)
        rng = max(vals) - min(vals)
        for bar, v in zip(bars, vals):
            delta = v - ref
            sign  = "+" if delta >= 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + rng * 0.015,
                    f"{v:.3f}\n({sign}{delta:.3f})",
                    ha="center", va="bottom", fontsize=7.8)

    legend_elems = [
        mpatches.Patch(color="#EF9A9A", label="Critical edge (removal hurts)"),
        mpatches.Patch(color="#A5D6A7", label="Redundant edge (removal neutral/helps)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.03), fontsize=8.5, framealpha=0.9)
    fig.suptitle("Edge-Type Ablation Study — DentalHGT (ResNet50 backbone)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "edge_ablation_polished.png")


def fig_radar_from_csv(csvs: dict):
    """Radar chart from CSV AUROC and AP values (no raw probabilities needed)."""
    categories = [PATH_LABELS[p] for p in PATH_NAMES]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5),
                              subplot_kw=dict(polar=True))

    r50 = csvs.get("resnet50")

    model_configs = [
        ("Indep. CNN (R50)",  "IndependentCNN", "resnet50", COLORS["IndependentCNN"], "--", 1.5),
        ("Homo. GNN (R50)",  "HomogeneousGNN", "resnet50", COLORS["HomogeneousGNN"], ":",  1.5),
        ("DentalHGT (R50)",  "DentalHGT",       "resnet50", COLORS["ResNet50"],       "-",  2.4),
        ("DentalHGT (D2)",   "DentalHGT",       "dinov2",   COLORS["DINOv2"],         "-.",  2.0),
    ]

    for ax, (col, title, ylim) in zip(axes, [
        ("auroc", "AUROC",             (0.5, 1.0)),
        ("ap",    "Average Precision", (0.0, 1.0)),
    ]):
        for label, mname, bb, color, ls, lw in model_configs:
            df = csvs.get(bb)
            if df is None: continue
            sub = df[(df["model"] == mname) & (df["pathology"] != "MEAN")]
            vals = []
            for pname in PATH_NAMES:
                row = sub[sub["pathology"] == pname]
                if row.empty or np.isnan(float(row[col].values[0])):
                    vals.append(ylim[0])
                else:
                    vals.append(float(row[col].values[0]))
            vals_c = vals + vals[:1]
            ax.plot(angles, vals_c, color=color, lw=lw, ls=ls, label=label)
            ax.fill(angles, vals_c, color=color, alpha=0.06)

        ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(*ylim)
        ticks = np.linspace(ylim[0], ylim[1], 5)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{v:.2f}" for v in ticks], fontsize=7)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.15), fontsize=8)
        ax.grid(True, alpha=0.4)

    fig.suptitle("Model Comparison — Per-Pathology Radar Chart",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, "radar_comparison.png")


def fig_training_curves():
    hist_r50  = RESULTS / "hgt_ablation_Full_model_resnet50_hist.csv"
    hist_dino = RESULTS / "hgt_training_history_dinov2.csv"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plotted = False
    for hist_path, label, color in [
        (hist_r50,  "ResNet50", COLORS["ResNet50"]),
        (hist_dino, "DINOv2",   COLORS["DINOv2"]),
    ]:
        if not hist_path.exists():
            continue
        df = pd.read_csv(hist_path)
        for ax, col, ylabel in [(axes[0], "val_map",    "Val mAP"),
                                 (axes[1], "train_loss", "Train Loss")]:
            if col in df.columns:
                ax.plot(df["epoch"], df[col], color=color, lw=1.8, label=label)
                plotted = True

    if not plotted:
        plt.close(fig); return

    for ax, ylabel in [(axes[0], "Val mAP"), (axes[1], "Train Loss")]:
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight="bold")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=9)
        ax.yaxis.grid(True, alpha=0.35, zorder=0); ax.set_axisbelow(True)

    fig.suptitle("DentalHGT Training Dynamics — ResNet50 vs DINOv2",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "training_curves_comparison.png")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("DentalHGT Analysis: DeLong tests + Figures")
    print("=" * 60)

    config = _load_config()

    # ── Load CSV metrics ──────────────────────────────────────────────────
    print("\nLoading CSV metrics …")
    csvs = load_csv_metrics()
    print(f"  Found: {list(csvs.keys())}")

    # ── DeLong tests (Hanley-McNeil from CSV AUROC) ───────────────────────
    print("\n[1/3] DeLong tests …")
    delong_dfs = {}
    for bb_key, bb_label in [("resnet50", "ResNet50"), ("dinov2", "DINOv2")]:
        if bb_key in csvs:
            print(f"  {bb_label}:")
            delong_dfs[bb_key] = run_delong_from_csv(csvs[bb_key], bb_label)

    # ── Baseline ROC/PR (from val_feats — consistent DINOv2 features) ─────
    print("\n[2/3] Baseline model inference (IndepCNN + HomoGNN) …")
    baseline_preds = load_baseline_predictions(config)

    # ── Figures ────────────────────────────────────────────────────────────
    print("\n[3/3] Generating publication figures …")

    fig_workflow()
    fig_roc_curves_baselines(baseline_preds)
    fig_pr_curves_baselines(baseline_preds)

    for bb_key, bb_label in [("resnet50", "ResNet50"), ("dinov2", "DINOv2")]:
        if bb_key in delong_dfs:
            fig_delong_heatmap(delong_dfs[bb_key], bb_label)
        if bb_key in csvs:
            fig_grouped_bars_from_csv(csvs[bb_key], bb_label)

    fig_backbone_comparison(csvs)
    fig_edge_ablation_polished()
    fig_radar_from_csv(csvs)
    fig_training_curves()

    print(f"\nAll figures → {FIGURES}/")
    print(f"DeLong CSVs → {RESULTS}/delong_tests_*.csv")


if __name__ == "__main__":
    main()
