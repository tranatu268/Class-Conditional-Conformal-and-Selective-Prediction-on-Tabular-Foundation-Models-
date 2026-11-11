# Copyright (c) 2026 Hoang Nhut Huynh and Tran Anh Tu
# SPDX-License-Identifier: MIT
# Authors: Hoang Nhut Huynh, Tran Anh Tu
"""Split conformal prediction (LAC / APS; marginal & Mondrian) + selective prediction.

Conventions
-----------
probs : (n, K) array of predicted class probabilities
y     : (n,) int labels in [0, K)
alpha : target miscoverage; prediction sets aim at P(y in set) >= 1 - alpha.
Mondrian = class-conditional calibration: one quantile per true class, giving
(approximate, finite-sample) per-class coverage guarantees - the imbalance-aware variant.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Nonconformity scores
# ---------------------------------------------------------------------------


def lac_score_matrix(probs: np.ndarray) -> np.ndarray:
    """LAC score for every candidate class: s(x, c) = 1 - p_c(x)."""
    return 1.0 - probs


def aps_score_matrix(probs: np.ndarray) -> np.ndarray:
    """APS (non-randomized) score for every candidate class.

    s(x, c) = sum of probabilities of classes ranked at or above c.
    """
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    scores = np.empty_like(probs)
    np.put_along_axis(scores, order, cum, axis=1)
    return scores


_SCORES = {"lac": lac_score_matrix, "aps": aps_score_matrix}


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Conformal quantile: ceil((n+1)(1-alpha))/n empirical quantile (method='higher')."""
    n = len(scores)
    if n == 0:
        return np.inf
    q = np.ceil((n + 1) * (1.0 - alpha)) / n
    if q > 1.0:
        return np.inf  # calibration set too small for this alpha
    return float(np.quantile(scores, q, method="higher"))


# ---------------------------------------------------------------------------
# Split conformal predictor
# ---------------------------------------------------------------------------


class SplitConformal:
    """Split conformal predictor with marginal or Mondrian (class-conditional) calibration."""

    def __init__(
        self,
        score: str = "lac",
        variant: str = "marginal",
        alpha: float = 0.1,
        groups: list[list[int]] | None = None,
    ):
        """groups (clustered variant only): partition of class indices; calibration
        scores are pooled within each group and one quantile is computed per group.
        This restores feasibility when some classes have fewer than ~1/alpha - 1
        calibration points (where per-class Mondrian quantiles become infinite),
        while keeping group-conditional validity (cf. clustered CP, Ding et al. 2023).
        """
        assert score in _SCORES, f"unknown score {score}"
        assert variant in ("marginal", "mondrian", "clustered"), f"unknown variant {variant}"
        if variant == "clustered":
            assert groups, "clustered variant requires `groups`"
        self.score = score
        self.variant = variant
        self.alpha = alpha
        self.groups = groups
        self.qhat_: float | None = None
        self.qhat_per_class_: np.ndarray | None = None
        self.n_classes_: int | None = None

    def fit(self, cal_probs: np.ndarray, cal_y: np.ndarray) -> "SplitConformal":
        K = cal_probs.shape[1]
        self.n_classes_ = K
        mat = _SCORES[self.score](cal_probs)
        true_scores = mat[np.arange(len(cal_y)), cal_y]
        if self.variant == "marginal":
            self.qhat_ = _finite_sample_quantile(true_scores, self.alpha)
        elif self.variant == "mondrian":
            self.qhat_per_class_ = np.array(
                [_finite_sample_quantile(true_scores[cal_y == c], self.alpha) for c in range(K)]
            )
        else:  # clustered: one pooled quantile per group, shared by its classes
            qhat = np.full(K, np.inf)
            for grp in self.groups:
                mask = np.isin(cal_y, grp)
                q = _finite_sample_quantile(true_scores[mask], self.alpha)
                for c in grp:
                    qhat[c] = q
            self.qhat_per_class_ = qhat
        return self

    def predict_sets(self, test_probs: np.ndarray) -> np.ndarray:
        """Boolean (n, K) matrix: True where class c is in the prediction set."""
        mat = _SCORES[self.score](test_probs)
        if self.variant == "marginal":
            return mat <= self.qhat_
        return mat <= self.qhat_per_class_[None, :]


# ---------------------------------------------------------------------------
# Evaluation of prediction sets & selective prediction
# ---------------------------------------------------------------------------


def coverage_metrics(sets: np.ndarray, y: np.ndarray) -> dict:
    """Marginal and per-class coverage, average and per-class set size."""
    n, K = sets.shape
    covered = sets[np.arange(n), y]
    out = {
        "coverage": float(covered.mean()),
        "avg_set_size": float(sets.sum(axis=1).mean()),
        "frac_empty": float((sets.sum(axis=1) == 0).mean()),
        "frac_singleton": float((sets.sum(axis=1) == 1).mean()),
    }
    for c in range(K):
        mask = y == c
        out[f"coverage_class{c}"] = float(covered[mask].mean()) if mask.any() else np.nan
    # imbalance-focused summary: worst-class coverage and coverage gap
    per_class = [out[f"coverage_class{c}"] for c in range(K)]
    out["worst_class_coverage"] = float(np.nanmin(per_class))
    out["coverage_gap"] = float(np.nanmax(per_class) - np.nanmin(per_class))
    return out


def selective_prediction(sets: np.ndarray, probs: np.ndarray, y: np.ndarray) -> dict:
    """Abstain unless the prediction set is a singleton; score the answered subset."""
    n, K = sets.shape
    size = sets.sum(axis=1)
    answered = size == 1
    out = {"abstention_rate": float(1.0 - answered.mean()),
           "frac_multilabel": float((size >= 2).mean())}
    if answered.any():
        yhat = sets[answered].argmax(axis=1)
        ytrue = y[answered]
        out["selective_accuracy"] = float((yhat == ytrue).mean())
        out["selective_risk"] = float((yhat != ytrue).mean())
        # balanced accuracy over classes present in the answered subset
        per_class = [
            float((yhat[ytrue == c] == c).mean()) for c in range(K) if (ytrue == c).any()
        ]
        out["selective_balanced_accuracy"] = float(np.mean(per_class))
        # per-class answered fraction (who gets abstained on?) and the fraction
        # of each class answered with a WRONG label (e.g. diseased -> healthy)
        yhat_all = np.where(answered, sets.argmax(axis=1), -1)
        for c in range(K):
            mask = y == c
            out[f"answered_frac_class{c}"] = float(answered[mask].mean()) if mask.any() else np.nan
            out[f"wrong_answer_frac_class{c}"] = (
                float(((yhat_all != -1) & (yhat_all != c))[mask].mean()) if mask.any() else np.nan
            )
    else:
        out.update(
            selective_accuracy=np.nan,
            selective_risk=np.nan,
            selective_balanced_accuracy=np.nan,
        )
    return out


def risk_coverage_curve(probs: np.ndarray, y: np.ndarray, n_grid: int = 101):
    """Confidence-threshold baseline: sort by max-prob confidence, sweep coverage.

    Returns (coverages, risks, aurc) where aurc is the area under the risk-coverage
    curve (lower is better). Used to compare conformal abstention against the
    standard softmax-confidence selective baseline.
    """
    conf = probs.max(axis=1)
    yhat = probs.argmax(axis=1)
    err = (yhat != y).astype(float)
    order = np.argsort(-conf)
    err_sorted = err[order]
    n = len(y)
    cum_err = np.cumsum(err_sorted)
    coverages = np.arange(1, n + 1) / n
    risks = cum_err / np.arange(1, n + 1)
    aurc = float(np.trapezoid(risks, coverages))
    # downsample to a grid for storage
    idx = np.linspace(0, n - 1, min(n_grid, n)).astype(int)
    return coverages[idx].tolist(), risks[idx].tolist(), aurc


# ---------------------------------------------------------------------------
# Multi-class -> binary mapping (auxiliary-label mode)
# ---------------------------------------------------------------------------


def aggregate_probs_to_binary(probs_multi: np.ndarray, positive_classes: list[int]) -> np.ndarray:
    """P(y=1) = sum of probabilities of the positive multi-class labels."""
    p1 = probs_multi[:, positive_classes].sum(axis=1)
    return np.column_stack([1.0 - p1, p1])


def sets_multi_to_binary(sets_multi: np.ndarray, positive_classes: list[int]) -> np.ndarray:
    """Map multi-class prediction sets to binary sets.

    Binary set contains 1 iff any positive multi-class label is in the set, and
    contains 0 iff any non-positive label is in the set.
    """
    n, K = sets_multi.shape
    pos = np.zeros(K, dtype=bool)
    pos[positive_classes] = True
    has_pos = sets_multi[:, pos].any(axis=1)
    has_neg = sets_multi[:, ~pos].any(axis=1)
    return np.column_stack([has_neg, has_pos])
