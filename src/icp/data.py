# Copyright (c) 2026 Hoang Nhut Huynh and Tran Anh Tu
# SPDX-License-Identifier: MIT
# Authors: Hoang Nhut Huynh, Tran Anh Tu
"""Dataset loaders for ICR (Kaggle) and UCI HCV (ARFF).

Each loader returns:
    X : pd.DataFrame  (numeric; categoricals already encoded)
    y_bin : np.ndarray of {0,1}        binary target
    y_multi : np.ndarray of int        auxiliary multi-class label (aligned with class_names)
    multi_class_names : list[str]      names of the multi-class labels
    positive_multi : list[int]         indices of multi-class labels that map to y_bin == 1
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ICR - Identifying Age-Related Conditions
# ---------------------------------------------------------------------------

ICR_ALPHA_ORDER = ["A", "B", "D", "G"]  # A = no condition; B/D/G = three conditions


def load_icr(train_csv: str | Path, greeks_csv: str | Path):
    """Load ICR train.csv merged with greeks.csv (Alpha as auxiliary label)."""
    train = pd.read_csv(train_csv)
    greeks = pd.read_csv(greeks_csv)

    # Some ICR column names carry trailing spaces ("BD ", "CD ", ...)
    train.columns = [c.strip() for c in train.columns]
    greeks.columns = [c.strip() for c in greeks.columns]

    df = train.merge(greeks[["Id", "Alpha"]], on="Id", how="left", validate="1:1")

    y_bin = df["Class"].to_numpy(dtype=int)
    alpha = df["Alpha"].astype(str).to_numpy()
    y_multi = np.array([ICR_ALPHA_ORDER.index(a) for a in alpha], dtype=int)

    # sanity: Alpha != 'A' <=> Class == 1
    derived = (y_multi != 0).astype(int)
    if not np.array_equal(derived, y_bin):
        n_bad = int((derived != y_bin).sum())
        raise ValueError(f"Alpha/Class mismatch on {n_bad} rows - check greeks merge.")

    X = df.drop(columns=["Id", "Class", "Alpha"]).copy()
    # EJ is the single categorical feature (values 'A'/'B')
    if not pd.api.types.is_numeric_dtype(X["EJ"]):
        X["EJ"] = (X["EJ"].astype(str).str.strip() == "B").astype(float)
    X = X.astype(float)

    positive_multi = [1, 2, 3]  # B, D, G
    return X, y_bin, y_multi, list(ICR_ALPHA_ORDER), positive_multi


# ---------------------------------------------------------------------------
# UCI HCV data (ID 571) - ARFF
# ---------------------------------------------------------------------------

HCV_CATEGORY_ORDER = [
    "0=Blood Donor",
    "1=Hepatitis",
    "2=Fibrosis",
    "3=Cirrhosis",
]


def _read_arff(path: str | Path) -> pd.DataFrame:
    """Minimal ARFF reader sufficient for the HCV file (quoted strings, CSV body)."""
    attrs: list[str] = []
    data_lines: list[str] = []
    in_data = False
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        low = line.lower()
        if low.startswith("@attribute"):
            m = re.match(r"@attribute\s+(\"[^\"]+\"|'[^']+'|\S+)", line, flags=re.I)
            name = m.group(1).strip("\"'")
            attrs.append(name)
        elif low.startswith("@data"):
            in_data = True
        elif in_data:
            data_lines.append(line)
    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        names=attrs,
        quotechar="'",
        na_values=["?", "NA", ""],
        skipinitialspace=True,
    )
    return df


def load_hcv(path: str | Path, suspect: str = "drop"):
    """Load UCI HCV data from the UCI CSV (hcvdat0.csv) or an ARFF export of it.

    suspect: how to treat '0s=suspect Blood Donor' rows: 'drop' | 'positive' | 'negative'
    """
    if str(path).lower().endswith(".arff"):
        df = _read_arff(path)
    else:
        df = pd.read_csv(path, na_values=["NA", "?", ""])
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")])

    cat = df["Category"].astype(str).str.strip()
    is_suspect = cat.str.startswith("0s")
    if suspect == "drop":
        df, cat = df[~is_suspect].reset_index(drop=True), cat[~is_suspect].reset_index(drop=True)
    elif suspect == "positive":
        cat = cat.where(~is_suspect, "1=Hepatitis")  # treat as disease
    elif suspect == "negative":
        cat = cat.where(~is_suspect, "0=Blood Donor")
    else:
        raise ValueError(f"Unknown suspect policy: {suspect}")

    y_multi = np.array([HCV_CATEGORY_ORDER.index(c) for c in cat], dtype=int)
    y_bin = (y_multi != 0).astype(int)

    X = df.drop(columns=["Category"]).copy()
    if not pd.api.types.is_numeric_dtype(X["Sex"]):
        X["Sex"] = (X["Sex"].astype(str).str.strip().str.lower() == "m").astype(float)
    X = X.astype(float)

    positive_multi = [1, 2, 3]
    return X, y_bin, y_multi, list(HCV_CATEGORY_ORDER), positive_multi


# ---------------------------------------------------------------------------

def load_dataset(cfg: dict):
    """Dispatch on cfg['dataset']; resolve paths relative to the repo root."""
    name = cfg["dataset"].lower()
    if name == "icr":
        return load_icr(cfg["data"]["train_csv"], cfg["data"]["greeks_csv"])
    if name == "hcv":
        d = cfg["data"]
        return load_hcv(d.get("path") or d["arff_path"], d.get("suspect", "drop"))
    raise ValueError(f"Unknown dataset: {name}")
