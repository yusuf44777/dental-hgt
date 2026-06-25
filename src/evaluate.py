"""Evaluation metrics for multi-label dental pathology classification.

Per-pathology: AUROC, Average Precision (AP), F1, Sensitivity, Specificity
Aggregate:     mAP (mean AP over 4 pathologies)
Statistical:   DeLong test for AUROC comparison (pairwise)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from .hgt_model import PATHOLOGY_NAMES


@dataclass
class PathologyMetrics:
    name: str
    auroc: float
    ap: float
    f1: float
    sensitivity: float
    specificity: float
    n_pos: int
    n_neg: int


@dataclass
class EvaluationResult:
    model_name: str
    per_pathology: list[PathologyMetrics]
    map_score: float    # mean Average Precision
    mean_auroc: float

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for m in self.per_pathology:
            rows.append({
                "model": self.model_name,
                "pathology": m.name,
                "auroc": round(m.auroc, 4),
                "ap": round(m.ap, 4),
                "f1": round(m.f1, 4),
                "sensitivity": round(m.sensitivity, 4),
                "specificity": round(m.specificity, 4),
                "n_pos": m.n_pos,
                "n_neg": m.n_neg,
            })
        rows.append({
            "model": self.model_name,
            "pathology": "MEAN",
            "auroc": round(self.mean_auroc, 4),
            "ap": round(self.map_score, 4),
            "f1": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "n_pos": sum(m.n_pos for m in self.per_pathology),
            "n_neg": sum(m.n_neg for m in self.per_pathology),
        })
        return pd.DataFrame(rows)


def evaluate_predictions(
    y_true: np.ndarray,    # [N, 4] binary float
    y_prob: np.ndarray,    # [N, 4] sigmoid probabilities
    model_name: str,
    threshold: float = 0.5,
) -> EvaluationResult:
    per_path = []
    aps = []
    aurocs = []
    for i, name in enumerate(PATHOLOGY_NAMES):
        gt = y_true[:, i].astype(int)
        prob = y_prob[:, i]
        n_pos = int(gt.sum())
        n_neg = int((1 - gt).sum())

        if n_pos == 0 or n_neg == 0:
            auroc = float("nan")
            ap = float("nan")
        else:
            auroc = float(roc_auc_score(gt, prob))
            ap = float(average_precision_score(gt, prob))
            aurocs.append(auroc)
            aps.append(ap)

        pred = (prob >= threshold).astype(int)
        f1 = float(f1_score(gt, pred, zero_division=0))
        tp = int(((pred == 1) & (gt == 1)).sum())
        fn = int(((pred == 0) & (gt == 1)).sum())
        tn = int(((pred == 0) & (gt == 0)).sum())
        fp = int(((pred == 1) & (gt == 0)).sum())
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)

        per_path.append(PathologyMetrics(
            name=name,
            auroc=auroc,
            ap=ap,
            f1=f1,
            sensitivity=sens,
            specificity=spec,
            n_pos=n_pos,
            n_neg=n_neg,
        ))

    return EvaluationResult(
        model_name=model_name,
        per_pathology=per_path,
        map_score=float(np.mean(aps)) if aps else float("nan"),
        mean_auroc=float(np.mean(aurocs)) if aurocs else float("nan"),
    )


def delong_test(auroc_a: float, auroc_b: float, n_pos: int, n_neg: int) -> float:
    """Approximate DeLong test p-value (Hanley & McNeil 1982 approximation).

    This is a simplified variance estimate suitable for reporting.
    For precise results, use the full DeLong covariance estimator.
    """
    q1 = auroc_a / (2 - auroc_a)
    q2 = 2 * auroc_a ** 2 / (1 + auroc_a)
    var_a = (auroc_a * (1 - auroc_a) + (n_pos - 1) * (q1 - auroc_a ** 2) + (n_neg - 1) * (q2 - auroc_a ** 2)) / (n_pos * n_neg)
    q1 = auroc_b / (2 - auroc_b)
    q2 = 2 * auroc_b ** 2 / (1 + auroc_b)
    var_b = (auroc_b * (1 - auroc_b) + (n_pos - 1) * (q1 - auroc_b ** 2) + (n_neg - 1) * (q2 - auroc_b ** 2)) / (n_pos * n_neg)
    import math
    se = math.sqrt(max(var_a + var_b, 1e-12))
    z = (auroc_a - auroc_b) / se
    # Two-tailed p-value from standard normal
    from scipy.stats import norm
    return float(2 * (1 - norm.cdf(abs(z))))


def save_metrics(results: list[EvaluationResult], path: str) -> None:
    dfs = [r.to_dataframe() for r in results]
    pd.concat(dfs, ignore_index=True).to_csv(path, index=False)
    print(f"Metrics saved → {path}")
