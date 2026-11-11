# Copyright (c) 2026 Hoang Nhut Huynh and Tran Anh Tu
# SPDX-License-Identifier: MIT
# Authors: Hoang Nhut Huynh, Tran Anh Tu
"""Model factory: TabPFN and classical baselines.

All models expose the sklearn API (fit / predict_proba).

Missing values: TabPFN, XGBoost and LightGBM handle NaN natively; logistic
regression and random forest are wrapped with median imputation.

Baseline families:
    logreg, rf               default sklearn models (imputation pipelines)
    xgb, lgbm                untuned GBDTs: 500 rounds, no class weights,
                             no early stopping, no post-hoc calibration
    xgb_w, lgbm_w            + balanced class weights + early stopping
    xgb_cal, lgbm_cal        + balanced class weights + early stopping
                             + cross-fitted Platt/temperature calibration
                             (unweighted, so probabilities follow the
                             empirical class prior like every other model)
    xgb_iso, lgbm_iso        as *_cal but isotonic (binary target only)
    catboost_cal             CatBoost (auto class weights) + same calibration
                             (requires `pip install catboost`)
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


# ---------------------------------------------------------------------------
# Gradient-boosted trees with class weights and early stopping
# ---------------------------------------------------------------------------


class WeightedGBDT(BaseEstimator, ClassifierMixin):
    """GBDT with balanced sample weights and early stopping on a stratified
    20% hold-out; the selected number of rounds is then refitted on all rows."""

    def __init__(self, kind="xgb", max_rounds=1000, learning_rate=0.05,
                 patience=50, class_weight=True, seed=0):
        self.kind = kind
        self.max_rounds = max_rounds
        self.learning_rate = learning_rate
        self.patience = patience
        self.class_weight = class_weight
        self.seed = seed

    def _make(self, n_rounds, n_classes):
        if self.kind == "xgb":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=n_rounds, learning_rate=self.learning_rate, max_depth=4,
                subsample=0.9, colsample_bytree=0.9, min_child_weight=1.0,
                eval_metric="mlogloss" if n_classes > 2 else "logloss",
                random_state=self.seed, n_jobs=-1,
            )
        if self.kind == "lgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=n_rounds, learning_rate=self.learning_rate, num_leaves=15,
                min_child_samples=10, subsample=0.9, subsample_freq=1,
                colsample_bytree=0.9, random_state=self.seed, n_jobs=-1, verbose=-1,
            )
        if self.kind == "catboost":
            from catboost import CatBoostClassifier

            return CatBoostClassifier(
                iterations=n_rounds, learning_rate=self.learning_rate, depth=4,
                random_seed=self.seed, verbose=False, allow_writing_files=False,
            )
        raise ValueError(self.kind)

    def _weights(self, y):
        return compute_sample_weight("balanced", y) if self.class_weight else None

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        # ---- stage 1: choose the number of rounds by early stopping
        tr, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y,
                                  random_state=self.seed)
        m = self._make(self.max_rounds, K)
        w_tr, w_va = self._weights(y[tr]), self._weights(y[va])
        if self.kind == "xgb":
            m.set_params(early_stopping_rounds=self.patience)
            m.fit(X[tr], y[tr], sample_weight=w_tr, eval_set=[(X[va], y[va])],
                  sample_weight_eval_set=[w_va] if w_va is not None else None, verbose=False)
            n_best = int(m.best_iteration) + 1
        elif self.kind == "lgbm":
            import lightgbm as lgb

            m.fit(X[tr], y[tr], sample_weight=w_tr, eval_set=[(X[va], y[va])],
                  eval_sample_weight=[w_va] if w_va is not None else None,
                  callbacks=[lgb.early_stopping(self.patience, verbose=False)])
            n_best = int(m.best_iteration_ or self.max_rounds)
        else:  # catboost
            m.set_params(auto_class_weights="Balanced" if self.class_weight else None)
            m.fit(X[tr], y[tr], eval_set=(X[va], y[va]),
                  early_stopping_rounds=self.patience, verbose=False)
            n_best = int(m.get_best_iteration()) + 1
        self.n_rounds_ = max(n_best, 10)
        # ---- stage 2: refit on all rows with the selected number of rounds
        self.model_ = self._make(self.n_rounds_, K)
        if self.kind == "catboost":
            self.model_.set_params(auto_class_weights="Balanced" if self.class_weight else None)
            self.model_.fit(X, y)
        else:
            self.model_.fit(X, y, sample_weight=self._weights(y))
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class CrossFitCalibrated(BaseEstimator, ClassifierMixin):
    """Post-hoc calibration fitted on out-of-fold predictions.

    The base model is fitted on k-1 folds and scored on the held-out fold;
    the calibration map is learned on these out-of-fold scores WITHOUT class
    weights, so a class-weighted base model is mapped back to the empirical
    prior. The final base model is then refitted on all rows.

    method="platt": multinomial logistic regression on log-probabilities
                    (Platt scaling for 2 classes, vector scaling for K > 2).
    method="isotonic": isotonic regression on P(class 1), binary only.
    """

    def __init__(self, base=None, method="platt", n_folds=5, seed=0):
        self.base = base
        self.method = method
        self.n_folds = n_folds
        self.seed = seed

    def _feat(self, P):
        return np.log(np.clip(P, 1e-6, 1.0))

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        if self.method == "isotonic" and K != 2:
            raise ValueError("isotonic calibration is implemented for binary targets only")
        oof = np.zeros((len(y), K))
        min_count = np.bincount(np.searchsorted(self.classes_, y)).min()
        k = int(min(self.n_folds, max(2, min_count)))
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.seed)
        for tr, va in skf.split(X, y):
            m = clone(self.base).fit(X[tr], y[tr])
            P = np.zeros((len(va), K))
            P[:, np.searchsorted(self.classes_, m.classes_)] = m.predict_proba(X[va])
            oof[va] = P
        if self.method == "platt":
            self.cal_ = LogisticRegression(C=1e4, max_iter=5000).fit(self._feat(oof), y)
        else:
            self.cal_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
            self.cal_.fit(oof[:, 1], (y == self.classes_[1]).astype(float))
        self.model_ = clone(self.base).fit(X, y)
        return self

    def predict_proba(self, X):
        P = self.model_.predict_proba(X)
        if self.method == "platt":
            return self.cal_.predict_proba(self._feat(P))
        p1 = self.cal_.predict(P[:, 1])
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# ---------------------------------------------------------------------------


def get_model(name: str, seed: int = 0, device: str = "cpu"):
    name = name.lower()

    if name == "tabpfn":
        from tabpfn import TabPFNClassifier

        try:
            return TabPFNClassifier(device=device, random_state=seed)
        except TypeError:  # older/newer API without random_state
            return TabPFNClassifier(device=device)

    if name == "logreg":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, random_state=seed)),
            ]
        )

    if name == "rf":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)),
            ]
        )

    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )

    if name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=15,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    # ---- control: weighted / early-stopped / calibrated tree ensembles
    for kind in ("xgb", "lgbm", "catboost"):
        if name == f"{kind}_w":
            return WeightedGBDT(kind=kind, seed=seed)
        if name == f"{kind}_cal":
            return CrossFitCalibrated(WeightedGBDT(kind=kind, seed=seed), "platt", seed=seed)
        if name == f"{kind}_iso":
            return CrossFitCalibrated(WeightedGBDT(kind=kind, seed=seed), "isotonic", seed=seed)

    raise ValueError(f"Unknown model: {name}")
