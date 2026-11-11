# Copyright (c) 2026 Hoang Nhut Huynh and Tran Anh Tu
# SPDX-License-Identifier: MIT
# Authors: Hoang Nhut Huynh, Tran Anh Tu
"""CLI entry point.

Examples:
    python experiments/run_experiment.py --config configs/icr.yaml
    python experiments/run_experiment.py --config configs/icr.yaml --target alpha
    python experiments/run_experiment.py --config configs/hcv.yaml --model logreg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from icp.data import load_dataset  # noqa: E402
from icp.evaluate import run_cv  # noqa: E402
from icp.metrics import mean_ci  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None, help="override model from config")
    ap.add_argument("--target", default=None, help="override target (class|alpha|category)")
    ap.add_argument("--device", default=None, help="override device (cpu|cuda)")
    ap.add_argument("--repeats", type=int, default=None, help="override n_repeats")
    ap.add_argument("--alpha", type=float, default=None,
                    help="override conformal miscoverage level (e.g. 0.05, 0.1, 0.2)")
    ap.add_argument("--calib-fraction", type=float, default=None,
                    help="override calibration fraction of the training fold (e.g. 0.25, 0.5)")
    ap.add_argument("--downsample", type=float, default=None,
                    help="cap majority:minority ratio in the proper-training set "
                         "(TabPFN context), e.g. 2.0")
    ap.add_argument("--save-preds", action="store_true",
                    help="write per-subject calibration/test probabilities to "
                         "results/<stem>_preds.csv (needed by risk_coverage.py and subgroup_hcv.py)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.model:
        cfg["model"] = args.model
    if args.target:
        cfg["target"] = args.target
    if args.device:
        cfg["device"] = args.device
    if args.repeats:
        cfg["cv"]["n_repeats"] = args.repeats
    suffix = ""
    if args.alpha is not None:
        cfg["conformal"]["alpha"] = args.alpha
        suffix += f"_a{args.alpha:g}"
    if args.calib_fraction is not None:
        cfg["cv"]["calib_fraction"] = args.calib_fraction
        suffix += f"_cf{args.calib_fraction:g}"
    if args.downsample is not None:
        cfg.setdefault("imbalance", {})["downsample_majority"] = args.downsample
        suffix += f"_ds{args.downsample:g}"
    if args.save_preds:
        # keep the published metric files untouched; predictions go to
        # <stem>_preds.csv and metrics of this run to <stem>_sp.{csv,json}
        suffix_pred = suffix
        suffix += "_sp"

    # resolve data paths relative to the repo root; the results file keeps
    # the paths as written in the config, so no local path is stored
    data_as_given = dict(cfg.get("data", {}))
    for k, v in list(cfg.get("data", {}).items()):
        if isinstance(v, str) and ("/" in v or "\\" in v):
            cfg["data"][k] = str((REPO_ROOT / v).resolve())

    X, y_bin, y_multi, class_names, positive_multi = load_dataset(cfg)
    print(
        f"[{cfg['dataset']}] n={len(X)}  d={X.shape[1]}  "
        f"positives={int(np.sum(y_bin))} ({100 * float(np.mean(y_bin)):.1f}%)  "
        f"target={cfg.get('target', 'class')}  model={cfg.get('model')}"
    )

    t0 = time.time()
    preds = [] if args.save_preds else None
    records = run_cv(X, y_bin, y_multi, positive_multi, cfg, pred_sink=preds)
    elapsed = time.time() - t0
    print(f"done: {len(records)} fold-level records in {elapsed:.1f}s")

    df = pd.DataFrame.from_records(records)
    out_dir = REPO_ROOT / cfg.get("output_dir", "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg['dataset']}_{cfg.get('model')}_{cfg.get('target', 'class')}{suffix}"
    csv_path = out_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    if preds is not None:
        pred_stem = f"{cfg['dataset']}_{cfg.get('model')}_{cfg.get('target', 'class')}{suffix_pred}"
        pred_path = out_dir / f"{pred_stem}_preds.csv"
        pd.DataFrame.from_records(preds).to_csv(pred_path, index=False)
        print(f"wrote {pred_path}")

    # aggregate: mean / 95% CI per method x metric
    skip = {"repeat", "fold", "model", "method", "target", "alpha", "threshold"}
    summary: dict[str, dict] = {}
    for method, g in df.groupby("method"):
        summary[method] = {
            col: mean_ci(g[col].dropna().tolist())
            for col in g.columns
            if col not in skip and g[col].dtype.kind in "fc" and g[col].notna().any()
        }

    payload = {
        "config": {**cfg, "data": data_as_given},
        "class_names": class_names,
        "n_records": len(records),
        "elapsed_sec": elapsed,
        "summary": summary,
    }
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}\n")

    # console digest
    for method in sorted(summary):
        s = summary[method]
        keys = [
            k
            for k in (
                "balanced_log_loss", "balanced_accuracy", "auroc",
                "coverage", "worst_class_coverage", "coverage_gap",
                "avg_set_size", "abstention_rate", "selective_balanced_accuracy", "aurc",
            )
            if k in s
        ]
        line = "  ".join(f"{k}={s[k]['mean']:.4f}" for k in keys)
        print(f"{method:24s} {line}")


if __name__ == "__main__":
    main()
