# Copyright (c) 2026 Hoang Nhut Huynh and Tran Anh Tu
# SPDX-License-Identifier: MIT
# Authors: Hoang Nhut Huynh, Tran Anh Tu
"""Point-prediction and calibration metrics (incl. the ICR balanced log loss)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def balanced_log_loss(y: np.ndarray, p1: np.ndarray, eps: float = 1e-15) -> float:
    """ICR competition metric: class-averaged log loss (binary)."""
    p1 = np.clip(p1, eps, 1.0 - eps)
    y = np.asarray(y, dtype=int)
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    l0 = -np.sum((y == 0) * np.log(1.0 - p1)) / max(n0, 1)
    l1 = -np.sum((y == 1) * np.log(p1)) / max(n1, 1)
    return float((l0 + l1) / 2.0)


def worst_class_accuracy(y: np.ndarray, yhat: np.ndarray) -> float:
    accs = [float((yhat[y == c] == c).mean()) for c in np.unique(y)]
    return float(min(accs))


def expected_calibration_error(y: np.ndarray, p1: np.ndarray, n_bins: int = 10) -> float:
    """Binary ECE on the positive-class probability."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p1, bins) - 1, 0, n_bins - 1)
    ece, n = 0.0, len(y)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.sum() / n * abs(float(y[mask].mean()) - float(p1[mask].mean()))
    return float(ece)


def point_metrics(y: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Standard binary metrics at a given decision threshold."""
    p1 = probs[:, 1]
    yhat = (p1 >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "balanced_log_loss": balanced_log_loss(y, p1),
        "auroc": float(roc_auc_score(y, p1)) if len(np.unique(y)) > 1 else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "worst_class_accuracy": worst_class_accuracy(y, yhat),
        "f1": float(f1_score(y, yhat)),
        "mcc": float(matthews_corrcoef(y, yhat)),
        "brier": float(brier_score_loss(y, p1)),
        "ece": expected_calibration_error(y, p1),
        "sensitivity": float((yhat[y == 1] == 1).mean()) if (y == 1).any() else np.nan,
        "specificity": float((yhat[y == 0] == 0).mean()) if (y == 0).any() else np.nan,
    }
    return out


def mean_ci(values, confidence: float = 0.95) -> dict:
    """Mean and Student-t confidence interval over folds/repeats (ignoring NaNs)."""
    v = np.asarray([x for x in values if x == x], dtype=float)  # drop NaN
    if len(v) == 0:
        return {"mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "std": np.nan, "n": 0}
    from scipy import stats

    m, s = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
    if len(v) > 1:
        half = float(stats.t.ppf(0.5 + confidence / 2, len(v) - 1) * s / np.sqrt(len(v)))
    else:
        half = np.nan
    return {"mean": m, "ci_lo": m - half, "ci_hi": m + half, "std": s, "n": int(len(v))}
