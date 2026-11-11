# Copyright (c) 2026 Hoang Nhut Huynh and Tran Anh Tu
# SPDX-License-Identifier: MIT
# Authors: Hoang Nhut Huynh, Tran Anh Tu
"""Repeated stratified CV protocol with an inner calibration split.

Per fold:
  train fold -> stratified split into proper-train / calibration
  fit model on proper-train (binary target, or auxiliary multi-class target)
  on calibration: tune threshold, estimate prior-shift, fit conformal predictors
  on test fold: point metrics (raw / threshold / prior-shift) +
                conformal coverage/set-size/selective metrics for every
                (score x variant) combination.
The test fold is never used for any tuning or calibration.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .conformal import (
    SplitConformal,
    aggregate_probs_to_binary,
    coverage_metrics,
    risk_coverage_curve,
    selective_prediction,
    sets_multi_to_binary,
)
from .imbalance import downsample_majority, prior_shift, tune_threshold
from .metrics import point_metrics
from .models import get_model


def _fit_predict(model_name, device, seed, X_tr, y_tr, X_parts):
    """Fit a fresh model and return predict_proba for each X in X_parts."""
    model = get_model(model_name, seed=seed, device=device)
    model.fit(X_tr, y_tr)
    return [model.predict_proba(X) for X in X_parts]


def run_cv(X: pd.DataFrame, y_bin, y_multi, positive_multi, cfg: dict,
           pred_sink: list | None = None) -> list[dict]:
    """Run the full protocol; returns a list of per-fold flat metric dicts.

    If `pred_sink` is a list, one dict per calibration and test subject is
    appended to it (repeat, fold, split, row index, labels, class
    probabilities), which allows every post-hoc analysis (risk-coverage
    curves, subgroup coverage) to be recomputed without refitting.
    """
    y_bin = np.asarray(y_bin)
    y_multi = np.asarray(y_multi)
    use_multi = cfg.get("target", "class") not in ("class",)
    y_fit = y_multi if use_multi else y_bin

    model_name = cfg.get("model", "tabpfn")
    device = cfg.get("device", "cpu")
    alpha = float(cfg["conformal"]["alpha"])
    scores = cfg["conformal"].get("scores", ["lac"])
    variants = cfg["conformal"].get("variants", ["marginal", "mondrian"])
    imb = cfg.get("imbalance", {}) or {}
    calib_frac = float(cfg["cv"].get("calib_fraction", 1 / 3))

    Xv = X.to_numpy(dtype=float)
    records: list[dict] = []

    n_repeats = int(cfg["cv"].get("n_repeats", 1))
    n_splits = int(cfg["cv"].get("n_splits", 5))
    total = n_repeats * n_splits
    done = 0
    t_start = time.time()

    for repeat in range(n_repeats):
        seed = repeat
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        # stratify folds on the fitted label space so every fold sees all classes
        for fold, (tr_idx, te_idx) in enumerate(skf.split(Xv, y_fit)):
            rng = np.random.default_rng(1000 * seed + fold)
            X_trfull, y_trfull = Xv[tr_idx], y_fit[tr_idx]
            X_te = Xv[te_idx]
            ybin_te = y_bin[te_idx]

            pt_idx, cal_idx = train_test_split(
                np.arange(len(tr_idx)),
                test_size=calib_frac,
                stratify=y_trfull,
                random_state=seed,
            )
            X_pt, y_pt = X_trfull[pt_idx], y_trfull[pt_idx]
            X_cal, y_cal = X_trfull[cal_idx], y_trfull[cal_idx]
            ybin_cal = y_bin[tr_idx][cal_idx]

            # optional majority-downsampled context (in-context imbalance handling)
            ds_ratio = imb.get("downsample_majority")
            if ds_ratio:
                keep = downsample_majority(X_pt, y_pt, float(ds_ratio), rng)
                X_pt, y_pt = X_pt[keep], y_pt[keep]

            t0 = time.time()
            probs_cal, probs_te = _fit_predict(
                model_name, device, seed, X_pt, y_pt, [X_cal, X_te]
            )
            done += 1
            dt = time.time() - t0
            eta = (time.time() - t_start) / done * (total - done)
            print(
                f"  [{done}/{total}] repeat {repeat + 1}/{n_repeats} "
                f"fold {fold + 1}/{n_splits}: fit+predict {dt:.1f}s  ETA {eta / 60:.1f} min",
                flush=True,
            )

            if pred_sink is not None:
                for split, idx_rows, P in (("cal", tr_idx[cal_idx], probs_cal),
                                           ("test", te_idx, probs_te)):
                    for r, pr in zip(idx_rows, P):
                        row = {"repeat": repeat, "fold": fold, "split": split,
                               "row": int(r), "y": int(y_fit[r]), "y_bin": int(y_bin[r])}
                        row.update({f"p{c}": float(v) for c, v in enumerate(pr)})
                        pred_sink.append(row)

            # ----- binary probabilities (aggregate if trained on aux multi-class)
            if use_multi:
                bprobs_cal = aggregate_probs_to_binary(probs_cal, positive_multi)
                bprobs_te = aggregate_probs_to_binary(probs_te, positive_multi)
            else:
                bprobs_cal, bprobs_te = probs_cal, probs_te

            base = {"repeat": repeat, "fold": fold, "model": model_name,
                    "target": cfg.get("target", "class")}

            # ----- point predictions: raw 0.5 / tuned threshold / prior shift
            rec = dict(base, method="point_raw", **point_metrics(ybin_te, bprobs_te, 0.5))
            records.append(rec)

            t_star = tune_threshold(
                bprobs_cal, ybin_cal, imb.get("threshold_metric", "balanced_accuracy")
            )
            records.append(
                dict(base, method="point_threshold", **point_metrics(ybin_te, bprobs_te, t_star))
            )

            if imb.get("prior_shift", True):
                prior = np.bincount(y_pt, minlength=probs_te.shape[1]) / len(y_pt)
                shifted_te = prior_shift(probs_te, prior)
                sb_te = (
                    aggregate_probs_to_binary(shifted_te, positive_multi)
                    if use_multi
                    else shifted_te
                )
                records.append(
                    dict(base, method="point_prior_shift", **point_metrics(ybin_te, sb_te, 0.5))
                )

            # ----- softmax-confidence selective baseline (binary space)
            _, _, aurc = risk_coverage_curve(bprobs_te, ybin_te)
            records.append(dict(base, method="softmax_selective", aurc=aurc))

            # ----- conformal: calibrated in the fitted label space
            # clustered groups: semantic partition {no-condition} vs {condition subtypes}
            K = probs_cal.shape[1]
            neg = [c for c in range(K) if c not in positive_multi]
            groups = [neg, list(positive_multi)] if use_multi else [[0], [1]]
            for sc in scores:
                for var in variants:
                    if var == "clustered" and not use_multi:
                        continue  # binary: clustered == mondrian
                    cp = SplitConformal(
                        score=sc, variant=var, alpha=alpha,
                        groups=groups if var == "clustered" else None,
                    )
                    cp.fit(probs_cal, y_cal)
                    sets = cp.predict_sets(probs_te)

                    rec = dict(base, method=f"cp_{sc}_{var}", alpha=alpha)
                    rec.update(coverage_metrics(sets, y_fit[te_idx]))
                    rec.update(selective_prediction(sets, probs_te, y_fit[te_idx]))

                    if use_multi:
                        bsets = sets_multi_to_binary(sets, positive_multi)
                        bcov = coverage_metrics(bsets, ybin_te)
                        bsel = selective_prediction(bsets, bprobs_te, ybin_te)
                        rec.update({f"bin_{k}": v for k, v in bcov.items()})
                        rec.update({f"bin_{k}": v for k, v in bsel.items()})
                    records.append(rec)

    return records
