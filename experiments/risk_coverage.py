# Copyright (c) 2026 Hoang Nhut Huynh and Tran Anh Tu
# SPDX-License-Identifier: MIT
# Authors: Hoang Nhut Huynh, Tran Anh Tu
"""Conformal selective rule versus softmax-confidence rule at MATCHED coverage.

Usage (after a run with --save-preds, binary target):
    python experiments/risk_coverage.py results/icr_tabpfn_class_preds.csv
    python experiments/risk_coverage.py results/hcv_tabpfn_class_preds.csv

For every split, the Mondrian (and marginal) LAC rule is rebuilt from the saved
calibration probabilities. Its coverage (fraction answered) is then imposed on
the softmax rule, which answers the same number of most-confident subjects.
Both rules are compared on plain selective risk, balanced selective risk and
the per-class answered / wrongly answered fractions.

Outputs next to the input file:
    rc_<stem>_summary.json   mean and 95% t-interval of every quantity, plus a
                             paired Wilcoxon test (split level and repeat level)
    rc_<stem>_curve.csv      softmax risk-coverage curve averaged over splits,
                             and the conformal operating points (for plotting)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from icp.posthoc import (  # noqa: E402
    conformal_sets, iter_folds, load_preds, risk_coverage_grid,
    selective_stats, softmax_rule_at_coverage,
)


def mci(v):
    v = np.asarray([x for x in v if x == x], float)
    if len(v) < 2:
        return {"mean": float(v.mean()) if len(v) else np.nan, "ci": np.nan, "n": len(v)}
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return {"mean": float(v.mean()), "ci": float(h), "n": int(len(v))}


def paired(a, b, keys):
    """Wilcoxon at split level and at repeat level (folds averaged per repeat)."""
    d = pd.DataFrame({"a": a, "b": b, "rep": keys}).dropna()
    out = {"median_diff_split": float(np.median(d.a - d.b))}
    out["p_split"] = float(stats.wilcoxon(d.a, d.b).pvalue) if (d.a != d.b).any() else 1.0
    r = d.groupby("rep")[["a", "b"]].mean()
    out["median_diff_repeat"] = float(np.median(r.a - r.b))
    out["p_repeat"] = float(stats.wilcoxon(r.a, r.b).pvalue) if (r.a != r.b).any() else 1.0
    return out


def main():
    path = Path(sys.argv[1])
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    df = load_preds(path)
    if df.attrs["K"] != 2:
        raise SystemExit("risk_coverage.py expects a binary-target predictions file")
    stem = path.name.replace("_preds.csv", "")
    grid = np.linspace(0.5, 1.0, 51)

    rows, curves, curves_b = [], [], []
    for r, f, Pc, yc, Pt, yt, _ in iter_folds(df):
        rec = {"repeat": r, "fold": f}
        for var in ("marginal", "mondrian"):
            sets = conformal_sets(Pc, yc, Pt, variant=var, alpha=alpha)
            answered = sets.sum(axis=1) == 1
            cp = selective_stats(answered, sets.argmax(axis=1), yt)
            a_sm, yhat_sm = softmax_rule_at_coverage(Pt, cp["coverage"])
            sm = selective_stats(a_sm, yhat_sm, yt)
            rec.update({f"{var}_cp_{k}": v for k, v in cp.items()})
            rec.update({f"{var}_sm_{k}": v for k, v in sm.items()})
        # softmax at full coverage (plain arg-max, no abstention)
        a_full, yhat = softmax_rule_at_coverage(Pt, 1.0)
        rec.update({f"full_{k}": v for k, v in selective_stats(a_full, yhat, yt).items()})
        rows.append(rec)
        rk, brk = risk_coverage_grid(Pt, yt, grid)
        curves.append(rk); curves_b.append(brk)

    T = pd.DataFrame(rows)
    summary = {c: mci(T[c]) for c in T.columns if c not in ("repeat", "fold")}
    tests = {}
    for var in ("marginal", "mondrian"):
        for k in ("risk", "balanced_risk", "wrong_answer_frac_class1", "answered_frac_class1"):
            tests[f"{var}: conformal vs softmax, {k}"] = paired(
                T[f"{var}_cp_{k}"], T[f"{var}_sm_{k}"], T["repeat"])
    out = {"input": str(path), "alpha": alpha, "n_splits": len(T),
           "summary": summary, "paired_tests": tests}
    jpath = path.parent / f"rc_{stem}_summary.json"
    jpath.write_text(json.dumps(out, indent=2))

    C = np.vstack(curves); CB = np.vstack(curves_b)
    tcrit = stats.t.ppf(0.975, len(C) - 1)
    curve = pd.DataFrame({
        "coverage": grid,
        "risk_mean": np.nanmean(C, 0), "risk_ci": tcrit * np.nanstd(C, 0, ddof=1) / np.sqrt(len(C)),
        "brisk_mean": np.nanmean(CB, 0), "brisk_ci": tcrit * np.nanstd(CB, 0, ddof=1) / np.sqrt(len(CB)),
    })
    # conformal operating points appended as extra rows with a label
    pts = []
    for var in ("marginal", "mondrian"):
        pts.append({"coverage": summary[f"{var}_cp_coverage"]["mean"],
                    "risk_mean": summary[f"{var}_cp_risk"]["mean"],
                    "risk_ci": summary[f"{var}_cp_risk"]["ci"],
                    "brisk_mean": summary[f"{var}_cp_balanced_risk"]["mean"],
                    "brisk_ci": summary[f"{var}_cp_balanced_risk"]["ci"],
                    "point": var})
    curve["point"] = "softmax_curve"
    curve = pd.concat([curve, pd.DataFrame(pts)], ignore_index=True)
    cpath = path.parent / f"rc_{stem}_curve.csv"
    curve.to_csv(cpath, index=False)

    # ---- console digest, for the paper
    s = summary
    print(f"\n{stem}  ({len(T)} splits, alpha = {alpha})")
    for var in ("marginal", "mondrian"):
        cov = s[f"{var}_cp_coverage"]["mean"]
        print(f"\n[{var}] answered {100 * cov:.1f}% of subjects")
        for k, lab in (("risk", "selective risk"), ("balanced_risk", "balanced risk"),
                       ("answered_frac_class1", "diseased answered"),
                       ("wrong_answer_frac_class1", "diseased answered healthy")):
            a, b = s[f"{var}_cp_{k}"], s[f"{var}_sm_{k}"]
            t = tests[f"{var}: conformal vs softmax, {k}"]
            print(f"  {lab:28s} conformal {a['mean']:.4f} +/- {a['ci']:.4f} | "
                  f"softmax {b['mean']:.4f} +/- {b['ci']:.4f} | "
                  f"p_split {t['p_split']:.2g}, p_repeat {t['p_repeat']:.2g}")
    print(f"\n[arg-max, 100% answered] risk {s['full_risk']['mean']:.4f}, "
          f"balanced risk {s['full_balanced_risk']['mean']:.4f}, "
          f"diseased answered healthy {s['full_wrong_answer_frac_class1']['mean']:.4f}")
    print(f"softmax AURC-style mean risk over coverage 0.5-1: {np.nanmean(C):.4f}")
    print(f"\nwrote {jpath}\nwrote {cpath}")


if __name__ == "__main__":
    main()
