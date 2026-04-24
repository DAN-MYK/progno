"""Walk-forward training: CatBoost + Platt calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression

BURN_IN_YEAR_ATP = 2004
BURN_IN_YEAR_WTA = 2011
CAL_YEAR = 2022
TEST_START_YEAR = 2023

CAT_FEATURES = ["surface", "tourney_level", "round"]

CATBOOST_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "early_stopping_rounds": 50,
    "random_seed": 42,
    "verbose": False,
}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"label", "tourney_date", "year"}
    return [c for c in df.columns if c not in exclude]


def walk_forward_splits(
    df: pd.DataFrame,
    burn_in_year: int = BURN_IN_YEAR_ATP,
    val_start: int = 2016,
    test_start: int = TEST_START_YEAR,
) -> list[tuple[pd.DataFrame, pd.DataFrame, str]]:
    """Expanding window walk-forward splits. No data before burn_in_year in training."""
    df = df[df["year"] > burn_in_year].copy()
    splits = []
    for year in sorted(df["year"].unique()):
        if year < val_start:
            continue
        train = df[df["year"] < year]
        if len(train) == 0:
            continue
        holdout = df[df["year"] == year]
        label = "val" if year < test_start else "test"
        splits.append((train, holdout, f"{label}_{year}"))
    return splits


def _train_catboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> CatBoostClassifier:
    cat_idx = [i for i, c in enumerate(feature_cols) if c in CAT_FEATURES]
    pool_tr = Pool(train_df[feature_cols].fillna(0), train_df["label"].values,
                   cat_features=cat_idx, feature_names=feature_cols)
    pool_val = Pool(val_df[feature_cols].fillna(0), val_df["label"].values,
                    cat_features=cat_idx, feature_names=feature_cols)
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(pool_tr, eval_set=pool_val)
    return model


def fit_platt(raw_probs: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Fit Platt scaling: P_cal = sigmoid(a * logit(P_raw) + b)."""
    eps = 1e-7
    clipped = np.clip(raw_probs, eps, 1 - eps)
    logits = np.log(clipped) - np.log(1 - clipped)
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(logits.reshape(-1, 1), y_true)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt(raw_probs: np.ndarray, a: float, b: float) -> np.ndarray:
    eps = 1e-7
    clipped = np.clip(raw_probs, eps, 1 - eps)
    logits = np.log(clipped) - np.log(1 - clipped)
    return 1.0 / (1.0 + np.exp(-(a * logits + b)))


def run_walk_forward(
    featurized_path: Path,
    burn_in_year: int = BURN_IN_YEAR_ATP,
    val_start: int = 2016,
) -> tuple[CatBoostClassifier, float, float, dict, list[str]]:
    """Run full walk-forward pipeline. Returns (model, platt_a, platt_b, metrics, feature_cols)."""
    df = pd.read_parquet(featurized_path)
    feature_cols = get_feature_cols(df)

    cal_df = df[df["year"] == CAL_YEAR]
    final_train = df[(df["year"] > burn_in_year) & (df["year"] < CAL_YEAR)]

    cat_idx = [i for i, c in enumerate(feature_cols) if c in CAT_FEATURES]
    model = _train_catboost(final_train, cal_df, feature_cols)

    cal_pool = Pool(cal_df[feature_cols].fillna(0), cat_features=cat_idx, feature_names=feature_cols)
    raw_cal = model.predict_proba(cal_pool)[:, 1]
    a, b = fit_platt(raw_cal, cal_df["label"].values)

    test_df = df[df["year"] >= TEST_START_YEAR]
    test_pool = Pool(test_df[feature_cols].fillna(0), cat_features=cat_idx, feature_names=feature_cols)
    raw_test = model.predict_proba(test_pool)[:, 1]
    cal_test = apply_platt(raw_test, a, b)

    from progno_train.validate import compute_log_loss, compute_ece
    metrics = {
        "logloss_catboost": compute_log_loss(test_df["label"].values, cal_test),
        "ece_catboost": compute_ece(test_df["label"].values, cal_test),
        "n_test": len(test_df),
        "platt_a": a,
        "platt_b": b,
        "cal_year": CAL_YEAR,
        "test_year": TEST_START_YEAR,
    }
    return model, a, b, metrics, feature_cols
