# Copyright (c) 2026 Hoang Nhut Huynh and Tran Anh Tu
# SPDX-License-Identifier: MIT
# Authors: Hoang Nhut Huynh, Tran Anh Tu
"""Imbalance corrections for probabilistic classifiers (incl. in-context PFNs)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def tune_threshold(
    cal_probs: np.ndarray, cal_y: np.ndarray, metric: str = "balanced_accuracy"
) -> float:
    """Pick the binary decision threshold maximizing `metric` on the calibration split."""
    assert cal_probs.shape[1] == 2, "threshold tuning is for binary probabilities"
    p1 = cal_probs[:, 1]
    grid = np.unique(np.concatenate([p1, np.linspace(0.01, 0.99, 99)]))
    scorer = {
        "balanced_accuracy": balanced_accuracy_score,
        "f1": f1_score,
    }[metric]
    best_t, best_s = 0.5, -np.inf
    for t in grid:
        s = scorer(cal_y, (p1 >= t).astype(int))
        if s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t


def prior_shift(probs: np.ndarray, train_prior: np.ndarray, target_prior=None) -> np.ndarray:
    """Logit-adjustment / Bayes prior correction.

    Reweights p(y|x) by (target_prior / train_prior) and renormalizes - equivalent to
    logit adjustment and the standard correction for class-prior mismatch.
    Default target is the uniform prior (balanced operating point).
    """
    K = probs.shape[1]
    train_prior = np.asarray(train_prior, dtype=float)
    target = np.full(K, 1.0 / K) if target_prior is None else np.asarray(target_prior, float)
    w = target / np.clip(train_prior, 1e-12, None)
    adj = probs * w[None, :]
    return adj / np.clip(adj.sum(axis=1, keepdims=True), 1e-12, None)


def downsample_majority(X, y: np.ndarray, max_ratio: float, rng: np.random.Generator):
    """Subsample majority classes so that n_c <= max_ratio * n_min for every class c.

    For in-context models (TabPFN), this reshapes the context rather than the data
    distribution assumption. Returns positional indices into (X, y).
    """
    classes, counts = np.unique(y, return_counts=True)
    n_min = counts.min()
    keep_idx = []
    for c in classes:
        idx = np.flatnonzero(y == c)
        cap = int(np.ceil(max_ratio * n_min))
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep_idx.append(idx)
    keep = np.sort(np.concatenate(keep_idx))
    return keep
