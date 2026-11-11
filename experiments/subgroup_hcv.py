# Copyright (c) 2026 Hoang Nhut Huynh and Tran Anh Tu
# SPDX-License-Identifier: MIT
# Authors: Hoang Nhut Huynh, Tran Anh Tu
"""Coverage and abstention by sex and age tertile on HCV.

Usage (after `run_experiment.py --config configs/hcv.yaml --save-preds`):
    python experiments/subgroup_hcv.py results/hcv_tabpfn_class_preds.csv

Rebuilds the marginal and Mondrian LAC sets of every split from the saved
calibration probabilities and reports, per subgroup and class, the coverage,
the fraction answered and the fraction answered wrongly. Subgroups with fewer
than 5 test subjects of a class in a split contribute NaN for that split.
Writes results/subgroup_<stem>.csv and prints a LaTeX-ready table body.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from icp.data import load_dataset  # noqa: E402
from icp.posthoc import conformal_sets, iter_folds, load_preds  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "hcv.yaml"))
    a = ap.parse_args()
    path, alpha = Path(a.preds), a.alpha
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    for k, v in list(cfg["data"].items()):
        if isinstance(v, str) and ("/" in v or "\\" in v):
            cfg["data"][k] = str((REPO_ROOT / v).resolve())
    X, y_bin, *_ = load_dataset(cfg)
    age = X["Age"].to_numpy()
    sex = X["Sex"].to_numpy()                      # 1 = male, 0 = female
    cuts = np.nanquantile(age, [1 / 3, 2 / 3])
    tert = np.digitize(age, cuts)                  # 0, 1, 2
    groups = {
        "Female": sex == 0, "Male": sex == 1,
        f"Age <= {cuts[0]:.0f}": tert == 0,
        f"Age {cuts[0]:.0f}-{cuts[1]:.0f}": tert == 1,
        f"Age > {cuts[1]:.0f}": tert == 2,
    }
    print("Cohort counts (healthy / diseased):")
    for g, m in groups.items():
        print(f"  {g:14s} {int(((y_bin == 0) & m).sum()):4d} / {int(((y_bin == 1) & m).sum()):3d}")

    df = load_preds(path)
    rows = []
    for r, f, Pc, yc, Pt, yt, trow in iter_folds(df):
        for var in ("marginal", "mondrian"):
            sets = conformal_sets(Pc, yc, Pt, variant=var, alpha=alpha)
            covered = sets[np.arange(len(yt)), yt]
            answered = sets.sum(axis=1) == 1
            wrong = answered & (sets.argmax(axis=1) != yt)
            for g, gm in groups.items():
                m_g = gm[trow]
                for c in (0, 1):
                    m = m_g & (yt == c)
                    ok = m.sum() >= 5
                    rows.append({"repeat": r, "fold": f, "variant": var, "group": g,
                                 "cls": c, "n": int(m.sum()),
                                 "coverage": covered[m].mean() if ok else np.nan,
                                 "answered": answered[m].mean() if ok else np.nan,
                                 "wrong": wrong[m].mean() if ok else np.nan})
    T = pd.DataFrame(rows)

    def mci(v):
        v = v.dropna().to_numpy()
        if len(v) < 2:
            return np.nan, np.nan, len(v)
        return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)), len(v)

    out = []
    for (var, g, c), d in T.groupby(["variant", "group", "cls"], sort=False):
        cv, cvh, ns = mci(d["coverage"])
        an, _, _ = mci(d["answered"])
        wr, _, _ = mci(d["wrong"])
        out.append({"variant": var, "group": g, "class": "diseased" if c else "healthy",
                    "coverage": cv, "coverage_ci": cvh, "answered": an, "wrong": wr,
                    "splits_used": ns, "mean_n_per_split": d["n"].mean()})
    O = pd.DataFrame(out)
    stem = path.name.replace("_preds.csv", "")
    opath = path.parent / f"subgroup_{stem}.csv"
    O.to_csv(opath, index=False)

    print("\nLaTeX rows: group & class & marginal cov. & Mondrian cov. & Mondrian answered & splits")
    for g in groups:
        for cl in ("healthy", "diseased"):
            a = O[(O.group == g) & (O["class"] == cl) & (O.variant == "marginal")].iloc[0]
            b = O[(O.group == g) & (O["class"] == cl) & (O.variant == "mondrian")].iloc[0]
            f = lambda x, h=None: "--" if x != x else (f"{x:.3f}" if h is None or h != h else f"{x:.3f}$\\pm${h:.3f}")
            print(f"{g} & {cl} & {f(a.coverage, a.coverage_ci)} & {f(b.coverage, b.coverage_ci)} & "
                  f"{f(b.answered)} & {int(b.splits_used)} \\\\")
    print(f"\nwrote {opath}")


if __name__ == "__main__":
    main()
