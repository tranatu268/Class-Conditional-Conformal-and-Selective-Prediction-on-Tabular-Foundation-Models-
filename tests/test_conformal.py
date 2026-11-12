# Copyright (c) 2026 Hoang Nhut Huynh and Tran Anh Tu
# SPDX-License-Identifier: MIT
# Authors: Hoang Nhut Huynh, Tran Anh Tu
"""Sanity checks for the conformal layer on synthetic data (no model needed).

Run:  python -m pytest tests/ -q      (or just: python tests/test_conformal.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icp.conformal import (  # noqa: E402
    SplitConformal,
    aggregate_probs_to_binary,
    coverage_metrics,
    selective_prediction,
    sets_multi_to_binary,
)


def _synthetic(n=4000, K=2, imbalance=0.15, noise=0.25, seed=0):
    """Well-specified probabilistic labels so true coverage is checkable."""
    rng = np.random.default_rng(seed)
    if K == 2:
        prior = np.array([1 - imbalance, imbalance])
    else:
        prior = np.full(K, 1.0 / K)
    logits = rng.normal(size=(n, K)) / max(noise, 1e-6)
    logits += np.log(prior)[None, :]
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(K, p=p) for p in probs])
    return probs, y


def test_marginal_coverage():
    probs, y = _synthetic()
    half = len(y) // 2
    for score in ("lac", "aps"):
        cp = SplitConformal(score=score, variant="marginal", alpha=0.1)
        cp.fit(probs[:half], y[:half])
        sets = cp.predict_sets(probs[half:])
        cov = coverage_metrics(sets, y[half:])["coverage"]
        assert cov >= 0.88, f"{score}: marginal coverage {cov:.3f} < 0.88"
        print(f"marginal {score}: coverage={cov:.3f}")


def test_mondrian_per_class_coverage():
    probs, y = _synthetic(n=8000, imbalance=0.15)
    half = len(y) // 2
    cp = SplitConformal(score="lac", variant="mondrian", alpha=0.1)
    cp.fit(probs[:half], y[:half])
    sets = cp.predict_sets(probs[half:])
    m = coverage_metrics(sets, y[half:])
    for c in (0, 1):
        assert m[f"coverage_class{c}"] >= 0.85, m
    print(f"mondrian per-class coverage: {m['coverage_class0']:.3f}, {m['coverage_class1']:.3f}")


def test_selective_and_multi_mapping():
    probs, y = _synthetic(n=4000, K=4)
    half = len(y) // 2
    cp = SplitConformal(score="aps", variant="mondrian", alpha=0.1).fit(probs[:half], y[:half])
    sets = cp.predict_sets(probs[half:])
    sel = selective_prediction(sets, probs[half:], y[half:])
    assert 0.0 <= sel["abstention_rate"] <= 1.0

    pos = [1, 2, 3]
    bsets = sets_multi_to_binary(sets, pos)
    ybin = (y[half:] != 0).astype(int)
    bcov = coverage_metrics(bsets, ybin)["coverage"]
    assert bcov >= 0.88, f"binary-mapped coverage {bcov:.3f}"
    bp = aggregate_probs_to_binary(probs[half:], pos)
    assert np.allclose(bp.sum(axis=1), 1.0)
    print(f"multi->binary coverage={bcov:.3f}, abstention={sel['abstention_rate']:.3f}")


if __name__ == "__main__":
    test_marginal_coverage()
    test_mondrian_per_class_coverage()
    test_selective_and_multi_mapping()
    print("all conformal sanity checks passed")


def test_threshold_is_kth_order_statistic():
    from icp.conformal import _finite_sample_quantile
    for n in (9, 20, 29, 136):
        s = np.arange(1, n + 1, dtype=float)
        k = int(np.ceil((n + 1) * 0.9))
        assert _finite_sample_quantile(s, 0.1) == k
    assert np.isinf(_finite_sample_quantile(np.arange(8, dtype=float), 0.1))
