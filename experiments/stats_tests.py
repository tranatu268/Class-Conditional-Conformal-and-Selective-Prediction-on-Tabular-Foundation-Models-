# Copyright (c) 2026 Hoang Nhut Huynh and Tran Anh Tu
# SPDX-License-Identifier: MIT
# Authors: Hoang Nhut Huynh, Tran Anh Tu
"""Paired statistical tests over per-fold results, formatted for the paper.

Usage:
    python experiments/stats_tests.py            # uses results/
    python experiments/stats_tests.py --results path/to/results

All comparisons are paired by (repeat, fold): the CV splitter is seeded by the
repeat index, so every model and calibration regime sees identical splits.
Because the 10 repeats reuse the same subjects, the 50 split-level pairs are
not independent; each test is therefore reported twice:
  split level   Wilcoxon signed-rank on the 50 paired splits
  repeat level  Wilcoxon on 10 pairs obtained by averaging the 5 folds of each
                repeat (the conservative figure to quote)
Holm correction is applied within each family, separately for each level.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("xgb", "lgbm", "logreg", "xgb_w", "lgbm_w", "xgb_cal", "lgbm_cal",
             "xgb_iso", "lgbm_iso", "catboost_cal")


def load(results: Path, stem: str) -> pd.DataFrame | None:
    p = results / f"{stem}.csv"
    return pd.read_csv(p) if p.exists() else None


def extract(df, method, col):
    if df is None:
        return None
    sub = df[df["method"] == method]
    if sub.empty or col not in sub or sub[col].isna().all():
        return None
    return sub.set_index(["repeat", "fold"])[col]


def _wil(a, b):
    d = a - b
    if len(d) < 5 or np.allclose(d, 0):
        return np.nan
    return float(wilcoxon(a, b).pvalue)


def paired_test(a: pd.Series, b: pd.Series):
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    r = j.groupby(level="repeat").mean()
    n_pos = int((j["a"] > j["b"]).sum())
    return {
        "n": len(j), "n_pos": n_pos,
        "med": float(np.median(j["a"] - j["b"])) if len(j) else np.nan,
        "p_split": _wil(j["a"], j["b"]),
        "n_rep": len(r), "rep_pos": int((r["a"] > r["b"]).sum()),
        "med_rep": float(np.median(r["a"] - r["b"])) if len(r) else np.nan,
        "p_rep": _wil(r["a"], r["b"]),
    }


def holm(pvals):
    idx = [i for i, p in enumerate(pvals) if p == p]
    m = len(idx)
    adj = [np.nan] * len(pvals)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda i: pvals[i])):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def report(title, rows):
    print(f"\n== {title} ==")
    lab, res = [], []
    for label, a, b in rows:
        if a is None or b is None:
            continue
        lab.append(label); res.append(paired_test(a, b))
    if not res:
        print("  (no data)"); return
    hs = holm([r["p_split"] for r in res]); hr = holm([r["p_rep"] for r in res])
    for label, r, ps, pr in zip(lab, res, hs, hr):
        print(f"  {label:44s} median diff {r['med']:+.4f} ({r['n_pos']}/{r['n']} splits a>b) "
              f"Holm p split {ps:.1e} | repeat-level median {r['med_rep']:+.4f} "
              f"({r['rep_pos']}/{r['n_rep']} repeats a>b) Holm p {pr:.3f}")
        print(f"    LaTeX: median $\\Delta={r['med']:+.3f}$, {r['n_pos']}/{r['n']} splits; "
              f"repeat level ($n={r['n_rep']}$) Holm-adjusted $p={pr:.3f}$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO_ROOT / "results"))
    R = Path(ap.parse_args().results)

    ds = ("icr", "hcv")
    models = ("tabpfn",) + BASELINES
    dfs = {(d, m): load(R, f"{d}_{m}_class") for d in ds for m in models}
    have = [(d, m) for (d, m), df in dfs.items() if df is not None]
    print("Found:", ", ".join(f"{d}/{m}" for d, m in have))

    # Family 1: validity claim
    report("Worst-class coverage: LAC Mondrian vs LAC marginal",
           [(f"{d}/{m}", extract(dfs[(d, m)], "cp_lac_mondrian", "worst_class_coverage"),
             extract(dfs[(d, m)], "cp_lac_marginal", "worst_class_coverage")) for d, m in have])

    # Family 2: abstention cost, baselines minus TabPFN (positive = TabPFN abstains less)
    report("Mondrian abstention: baseline vs TabPFN",
           [(f"{d}: {m} vs tabpfn", extract(dfs[(d, m)], "cp_lac_mondrian", "abstention_rate"),
             extract(dfs[(d, "tabpfn")], "cp_lac_mondrian", "abstention_rate"))
            for d in ds for m in BASELINES if dfs[(d, m)] is not None and dfs[(d, "tabpfn")] is not None])

    # Family 3: point prediction (baseline minus TabPFN, positive = TabPFN better)
    report("Balanced log loss (prior shift): baseline vs TabPFN",
           [(f"{d}: {m} vs tabpfn", extract(dfs[(d, m)], "point_prior_shift", "balanced_log_loss"),
             extract(dfs[(d, "tabpfn")], "point_prior_shift", "balanced_log_loss"))
            for d in ds for m in BASELINES if dfs[(d, m)] is not None and dfs[(d, "tabpfn")] is not None])

    # Family 4: effect of calibration on the tree ensembles themselves
    rows = []
    for d in ds:
        for raw, cal in (("xgb", "xgb_cal"), ("lgbm", "lgbm_cal"), ("xgb", "xgb_w"), ("lgbm", "lgbm_w")):
            if dfs[(d, raw)] is not None and dfs[(d, cal)] is not None:
                rows.append((f"{d}: {raw} vs {cal}, Mondrian abst.",
                             extract(dfs[(d, raw)], "cp_lac_mondrian", "abstention_rate"),
                             extract(dfs[(d, cal)], "cp_lac_mondrian", "abstention_rate")))
    report("Mondrian abstention, untuned vs weighted/calibrated GBDT", rows)

    # Family 5: auxiliary-label mode. Clustered vs Mondrian abstention is NOT
    # tested: Mondrian abstains on every subject by Proposition 1 (E46c).
    rows = []
    for d, target in (("icr", "alpha"), ("hcv", "category")):
        df = load(R, f"{d}_tabpfn_{target}")
        if df is None:
            continue
        rows += [
            (f"{d} ({target}): clustered vs marginal, WCC",
             extract(df, "cp_lac_clustered", "worst_class_coverage"),
             extract(df, "cp_lac_marginal", "worst_class_coverage")),
            (f"{d} ({target}): clustered vs marginal, bin WCC",
             extract(df, "cp_lac_clustered", "bin_worst_class_coverage"),
             extract(df, "cp_lac_marginal", "bin_worst_class_coverage")),
        ]
    report("Auxiliary-label mode: clustered vs marginal", rows)


if __name__ == "__main__":
    main()
