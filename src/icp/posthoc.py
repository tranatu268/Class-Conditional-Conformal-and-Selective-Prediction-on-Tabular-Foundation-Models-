# Copyright (c) 2026 Hoang Nhut Huynh and Tran Anh Tu
# SPDX-License-Identifier: MIT
# Authors: Hoang Nhut Huynh, Tran Anh Tu
"""Post-hoc analyses from saved per-subject predictions (<stem>_preds.csv).

The predictions file holds, for every (repeat, fold), the class probabilities
of the calibration subjects and of the test subjects, so conformal sets can be
rebuilt exactly as in evaluate.py without refitting the base model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .conformal import SplitConformal


def load_preds(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.attrs["K"] = sum(c.startswith("p") and c[1:].isdigit() for c in df.columns)
    return df


def iter_folds(df: pd.DataFrame):
    """Yield (repeat, fold, P_cal, y_cal, P_test, y_test, test_rows)."""
    K = df.attrs.get("K") or sum(c.startswith("p") and c[1:].isdigit() for c in df.columns)
    pcols = [f"p{c}" for c in range(K)]
    for (r, f), g in df.groupby(["repeat", "fold"], sort=True):
        cal = g[g["split"] == "cal"]
        te = g[g["split"] == "test"]
        yield (int(r), int(f), cal[pcols].to_numpy(), cal["y"].to_numpy(),
               te[pcols].to_numpy(), te["y"].to_numpy(), te["row"].to_numpy())


def conformal_sets(P_cal, y_cal, P_test, variant="mondrian", score="lac", alpha=0.1):
    cp = SplitConformal(score=score, variant=variant, alpha=alpha).fit(P_cal, y_cal)
    return cp.predict_sets(P_test)


def selective_stats(answered: np.ndarray, yhat: np.ndarray, y: np.ndarray) -> dict:
    """Risk statistics of a selective rule (binary or multi-class)."""
    out = {"coverage": float(answered.mean())}
    if answered.any():
        ya, ha = y[answered], yhat[answered]
        out["risk"] = float((ha != ya).mean())
        per = [float((ha[ya == c] != c).mean()) for c in np.unique(ya)]
        out["balanced_risk"] = float(np.mean(per))
    else:
        out["risk"] = out["balanced_risk"] = np.nan
    for c in np.unique(y):
        m = y == c
        out[f"answered_frac_class{c}"] = float(answered[m].mean())
        out[f"wrong_answer_frac_class{c}"] = float((answered & (yhat != c))[m].mean())
    return out


def softmax_rule_at_coverage(P: np.ndarray, coverage: float):
    """Answer the ceil(coverage * n) most confident subjects (max probability)."""
    n = len(P)
    k = int(np.clip(np.round(coverage * n), 0, n))
    order = np.argsort(-P.max(axis=1), kind="stable")
    answered = np.zeros(n, dtype=bool)
    answered[order[:k]] = True
    return answered, P.argmax(axis=1)


def risk_coverage_grid(P: np.ndarray, y: np.ndarray, grid: np.ndarray):
    """Plain and balanced selective risk of the softmax rule on a coverage grid."""
    risks, brisks = [], []
    for c in grid:
        a, yhat = softmax_rule_at_coverage(P, c)
        s = selective_stats(a, yhat, y)
        risks.append(s["risk"]); brisks.append(s["balanced_risk"])
    return np.array(risks), np.array(brisks)
