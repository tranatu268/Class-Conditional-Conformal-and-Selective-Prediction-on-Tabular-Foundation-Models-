# Copyright (c) 2026 Hoang Nhut Huynh and Tran Anh Tu
# SPDX-License-Identifier: MIT
# Authors: Hoang Nhut Huynh, Tran Anh Tu
"""Print LaTeX table bodies from results/*.json for every model that has been run.

Usage:
    python experiments/make_tables.py            # uses results/
    python experiments/make_tables.py --results path/to/results

Prints three blocks, ready to paste between \\midrule and \\bottomrule:
  Table II   point prediction (raw and prior-shift operating points)
  Table III  LAC conformal validity and selective efficiency (marginal, Mondrian)
  Table S1   the same for the APS score (supplementary table)
Columns that a results file does not contain (older runs) print as "--".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS = [  # (file key, display name)
    ("tabpfn", "TabPFN"),
    ("xgb", "XGBoost"),
    ("xgb_w", "XGBoost-w"),
    ("xgb_cal", "XGBoost-cal"),
    ("xgb_iso", "XGBoost-iso"),
    ("lgbm", "LightGBM"),
    ("lgbm_w", "LightGBM-w"),
    ("lgbm_cal", "LightGBM-cal"),
    ("lgbm_iso", "LightGBM-iso"),
    ("catboost_cal", "CatBoost-cal"),
    ("logreg", "LogReg"),
]
DATASETS = [("icr", "ICR"), ("hcv", "HCV")]


def get(S, method, key):
    v = S.get(method, {}).get(key)
    if not v or v.get("mean") is None or v["mean"] != v["mean"]:
        return None, None
    lo, hi = v.get("ci_lo"), v.get("ci_hi")
    half = (hi - lo) / 2 if lo is not None and hi is not None and hi == hi else None
    return v["mean"], half


def f(x, d=3, pm=None, scale=1.0):
    if x is None:
        return "--"
    s = f"{scale * x:.{d}f}"
    if pm is not None:
        s += f"$\\pm${scale * pm:.{d}f}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()
    R = Path(args.results)

    avail = {}
    for dk, dn in DATASETS:
        for mk, mn in MODELS:
            p = R / f"{dk}_{mk}_class.json"
            if p.exists():
                avail[(dk, mk)] = json.loads(p.read_text())["summary"]

    print("% ---------------- Table II: point prediction ----------------")
    print("% Cohort & Model & BLL raw & BLL PS & Brier raw & Brier PS & ECE raw & ECE PS & AUROC & Sens. PS & Spec. PS")
    for dk, dn in DATASETS:
        if not any((dk, mk) in avail for mk, _ in MODELS):
            continue
        for mk, mn in MODELS:
            S = avail.get((dk, mk))
            if S is None:
                continue
            bll_r, _ = get(S, "point_raw", "balanced_log_loss")
            bll_p, bll_h = get(S, "point_prior_shift", "balanced_log_loss")
            br_r, _ = get(S, "point_raw", "brier")
            br_p, _ = get(S, "point_prior_shift", "brier")
            ec_r, _ = get(S, "point_raw", "ece")
            ec_p, _ = get(S, "point_prior_shift", "ece")
            au, _ = get(S, "point_raw", "auroc")
            se, se_h = get(S, "point_prior_shift", "sensitivity")
            sp, sp_h = get(S, "point_prior_shift", "specificity")
            print(f"{dn} & {mn} & {f(bll_r)} & {f(bll_p, pm=bll_h)} & {f(br_r)} & {f(br_p)} & "
                  f"{f(ec_r)} & {f(ec_p)} & {f(au)} & {f(se, pm=se_h)} & {f(sp, pm=sp_h)} \\\\")
        print("\\midrule")

    for score, title in (("lac", "Table III: LAC conformal"), ("aps", "Table S1: APS conformal")):
        print(f"\n% ---------------- {title} ----------------")
        print("% Cohort / model & Calib. & Coverage & WCC & Gap & Abst. % (empty) & Sel. risk & "
              "Sel. BA & Dis. answ. % & Dis. missed/100")
        for dk, dn in DATASETS:
            if not any((dk, mk) in avail for mk, _ in MODELS):
                continue
            for mk, mn in MODELS:
                S = avail.get((dk, mk))
                if S is None:
                    continue
                for i, var in enumerate(("marginal", "mondrian")):
                    m = f"cp_{score}_{var}"
                    cov, _ = get(S, m, "coverage")
                    wcc, wcc_h = get(S, m, "worst_class_coverage")
                    gap, _ = get(S, m, "coverage_gap")
                    ab, ab_h = get(S, m, "abstention_rate")
                    em, _ = get(S, m, "frac_empty")
                    sr, _ = get(S, m, "selective_risk")
                    sba, _ = get(S, m, "selective_balanced_accuracy")
                    da, _ = get(S, m, "answered_frac_class1")
                    dm, _ = get(S, m, "wrong_answer_frac_class1")
                    lead = f"{dn} / {mn}" if i == 0 else ""
                    calib = "marginal" if var == "marginal" else "Mondrian"
                    print(f"{lead} & {calib} & {f(cov)} & {f(wcc, pm=wcc_h)} & {f(gap)} & "
                          f"{f(ab, 1, ab_h, 100)} ({f(em, 1, None, 100)}) & {f(sr)} & {f(sba)} & "
                          f"{f(da, 1, None, 100)} & {f(dm, 1, None, 100)} \\\\")
            print("\\midrule")

    missing = [f"{dk}_{mk}" for dk, _ in DATASETS for mk, _ in MODELS if (dk, mk) not in avail]
    print(f"\n% not found (not run): {', '.join(missing)}")


if __name__ == "__main__":
    main()
