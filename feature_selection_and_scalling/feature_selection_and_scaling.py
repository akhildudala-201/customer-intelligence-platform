"""
feature_selection_and_scaling.py
==================================
PERSON F TASK (revised): Split-Dependent Fitting

Three reusable functions, importable by downstream teammates. NOTHING in
this file touches the database -- it operates purely on X/y DataFrames
already in memory, which the caller (Kalyan, Kuushalie, or you) is
responsible for loading and splitting.

    select_features(X_train, y_train)  -> list of column names to keep
    fit_scaler(X_train)                -> a FITTED scaler object
    apply_scaler(X, fitted)            -> scaled DataFrame (same columns)

WHY EVERYTHING HERE IS "TRAIN-ONLY"
------------------------------------
Any calculation that "learns" something from data (a correlation, a
chi-square statistic, a median/IQR for scaling) must only ever see the
training set. If you let it see the test set too, you leak information
about the "unseen future" into the model, and your test-set evaluation
later becomes falsely optimistic. This is why fit_scaler() only takes
X_train, while apply_scaler() can be called on X_train, X_test, or any
future data -- using the SAME already-learned parameters every time.

USAGE EXAMPLE (this part lives in someone ELSE's script, not here)
--------------------------------------------------------------------
    from feature_selection_and_scaling import select_features, fit_scaler, apply_scaler
    from sklearn.model_selection import train_test_split

    X = features_encoded.drop(columns=["customer_unique_id", "churned"])
    y = features_encoded["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    keep_cols = select_features(X_train, y_train)
    X_train, X_test = X_train[keep_cols], X_test[keep_cols]

    scaler = fit_scaler(X_train)
    X_train_scaled = apply_scaler(X_train, scaler)
    X_test_scaled = apply_scaler(X_test, scaler)   # SAME scaler, not refit
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.preprocessing import RobustScaler


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    corr_threshold: float = 0.90,
    chi2_alpha: float = 0.05,
) -> List[str]:
    """
    Decide which columns to keep, using ONLY X_train and y_train.

    Two checks are combined:
      1. Correlation pruning (numeric/continuous columns): if two features
         move almost identically (|correlation| > corr_threshold), one of
         them is redundant -- keep whichever correlates more strongly with
         the target, drop the other.
      2. Chi-square selection (binary/one-hot columns, values in {0,1}):
         tests whether each column's variation is statistically related to
         the target. Columns with p-value >= chi2_alpha look unrelated to
         the target and are dropped.

    Returns
    -------
    List[str] : column names from X_train to keep.
    """
    binary_cols = [
        c for c in X_train.columns
        if X_train[c].dropna().isin([0, 1]).all() and X_train[c].nunique() <= 2
    ]
    numeric_cols = [c for c in X_train.columns if c not in binary_cols]

    # --- 1. correlation pruning on continuous numeric columns ---
    # Greedy approach: process columns strongest-correlated-to-target first.
    # Keep a column only if it is NOT highly correlated with something
    # already kept. This handles clusters of 3+ mutually correlated
    # columns correctly (a naive pairwise comparison can get confused by
    # ties and accidentally keep a redundant column instead of dropping it).
    to_drop_corr = set()
    if len(numeric_cols) > 1:
        corr_matrix = X_train[numeric_cols].corr().abs()
        target_corr = X_train[numeric_cols].apply(lambda col: col.corr(y_train.astype(float)))
        # rank by |correlation to target|, descending; mergesort keeps ties
        # in a stable, reproducible order across runs
        ranked_cols = target_corr.abs().sort_values(ascending=False, kind="mergesort").index.tolist()

        kept_numeric = []
        for col in ranked_cols:
            is_redundant = any(corr_matrix.loc[col, kept] > corr_threshold for kept in kept_numeric)
            if is_redundant:
                to_drop_corr.add(col)
            else:
                kept_numeric.append(col)

    # --- 2. chi-square selection on binary/one-hot columns ---
    to_drop_chi2 = set()
    if binary_cols:
        chi2_stats, p_values = chi2(X_train[binary_cols].fillna(0), y_train)
        for col, p in zip(binary_cols, p_values):
            if p >= chi2_alpha:
                to_drop_chi2.add(col)

    keep_cols = [c for c in X_train.columns if c not in to_drop_corr and c not in to_drop_chi2]
    return keep_cols


def fit_scaler(X_train: pd.DataFrame) -> RobustScaler:
    """
    Fit a RobustScaler on X_train ONLY. Does not transform anything --
    just learns the median/IQR "recipe" from the training data and
    returns the fitted scaler object for later reuse.
    """
    scaler = RobustScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(X: pd.DataFrame, fitted: RobustScaler) -> pd.DataFrame:
    """
    Apply an ALREADY-FITTED scaler to X (train, test, or future data).
    Never re-fits -- reuses the exact parameters learned in fit_scaler(),
    so every dataset ends up on the same, consistent scale.
    """
    scaled_array = fitted.transform(X)
    return pd.DataFrame(scaled_array, columns=X.columns, index=X.index)
