"""Data loading and preprocessing (paper Sections 4.2, 5).

Dataset.csv is the harmonized core-schema table produced by the data
harmonization stage: one row per customer, a `source` column (cc / bm / sa),
a fixed 70/15/15 `split` (seed 42) and reference labels kept out of the inputs.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_PATH = "Group16_DataScience (1)/Dataset.csv"

CORE_COLS = ["age", "tenure_months", "balance", "n_products", "is_active",
             "has_credit_card", "has_loan", "has_mortgage", "has_savings_deposit"]
# Skewed monetary / count variables that are log-transformed in P2.
LOG_COLS = ["balance", "tenure_months"]
SOURCE_NAMES = {"cc": "Credit card", "bm": "Bank Marketing", "sa": "Santander"}


@dataclass
class SourceData:
    name: str
    df: pd.DataFrame                 # rows of this source, original order
    features: list                   # model input columns
    X_raw: np.ndarray                # original (unscaled) feature values
    X: np.ndarray                    # P2: log + standardize (common space)
    X_p1: np.ndarray                 # P1: median impute, cap p95, scale, PCA 95%
    split: np.ndarray                # 'train' | 'val' | 'test'
    ref: np.ndarray = field(default=None)  # reference label (never used in training)

    def idx(self, s):
        return np.where(self.split == s)[0]


def load_all(path=DATA_PATH):
    return pd.read_csv(path)


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def prepare_source(full: pd.DataFrame, source: str) -> SourceData:
    df = full[full["source"] == source].reset_index(drop=True)
    train = df["split"] == "train"
    # Keep core columns observed in this source and not constant on train.
    feats = [c for c in CORE_COLS
             if df[c].notna().any() and df.loc[train, c].nunique() > 1]
    X_raw = df[feats].astype(float).copy()

    # Imputation statistics fitted on the training split only.
    med = X_raw[train].median()
    X_imp = X_raw.fillna(med)

    # P2: log transform of skewed columns, then standardization.
    X2 = X_imp.copy()
    for c in LOG_COLS:
        if c in X2:
            X2[c] = signed_log1p(X2[c])
    sc2 = StandardScaler().fit(X2[train])
    X = sc2.transform(X2)

    # P1 (base paper): median impute, cap at the 95th pct, standardize, PCA 95%.
    cap = X_imp[train].quantile(0.95)
    X1 = X_imp.clip(upper=cap, axis=1)
    # Binary columns are left uncapped (their p95 is 0 or 1 already).
    sc1 = StandardScaler().fit(X1[train])
    Z1 = sc1.transform(X1)
    Z1 = np.nan_to_num(Z1)  # a column can become constant after capping
    pca = PCA(n_components=0.95, svd_solver="full").fit(Z1[train.values])
    X_p1 = pca.transform(Z1)

    ref = df["ref_label"].values if df["ref_label"].notna().any() else None
    return SourceData(source, df, feats, X_raw.values, X.astype(np.float32),
                      X_p1.astype(np.float32), df["split"].values, ref)
