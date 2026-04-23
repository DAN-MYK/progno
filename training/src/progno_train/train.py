"""Walk-forward training: CatBoost + Platt calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression

CAT_FEATURES = ["surface", "tourney_level", "round"]
BURN_IN_YEAR = 2004  # data before this year used only for Elo warm-up

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


def walk_forward_splits(
    df: pd.DataFrame,
    val_start: int = 2016,
    test_start: int = 2023,
) -> list[tuple[pd.DataFrame, pd.DataFrame, str]]:
    """Yields (train, holdout, split_label) for each year from val_start onward."""
    df = df[df["year"] > BURN_IN_YEAR].copy()
    splits = []
    all_years = sorted(df["year"].unique())
    for year in all_years:
        if year < val_start:
            continue
        train = df[df["year"] < year]
        holdout = df[df["year"] == year]
        label = "val" if year < test_start else "test"
        splits.append((train, holdout, f"{label}_{year}"))
    return splits


def train_catboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> CatBoostClassifier:
    cat_idx = [i for i, c in enumerate(feature_cols) if c in CAT_FEATURES]
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df["label"].values

    pool_train = Pool(X_train, y_train, cat_features=cat_idx, feature_names=feature_cols)
    pool_val = Pool(X_val, y_val, cat_features=cat_idx, feature_names=feature_cols)

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(pool_train, eval_set=pool_val)
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


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"label", "tourney_date", "year"}
    return [c for c in df.columns if c not in exclude]


def run_walk_forward(featurized_path) -> tuple:
    """Run full walk-forward pipeline. Returns (model, a, b, metrics, feature_cols)."""
    df = pd.read_parquet(featurized_path)
    feature_cols = get_feature_cols(df)

    cal_df = df[df["year"] == 2022]
    final_train = df[df["year"] < 2023]

    model = train_catboost(final_train, cal_df, feature_cols)

    cal_pool = Pool(cal_df[feature_cols].fillna(0), feature_names=feature_cols)
    raw_cal = model.predict_proba(cal_pool)[:, 1]
    a, b = fit_platt(raw_cal, cal_df["label"].values)

    test_df = df[df["year"] >= 2023]
    test_pool = Pool(test_df[feature_cols].fillna(0), feature_names=feature_cols)
    raw_test = model.predict_proba(test_pool)[:, 1]
    cal_test = apply_platt(raw_test, a, b)

    from progno_train.validate import compute_log_loss, compute_ece
    metrics = {
        "logloss_catboost": compute_log_loss(test_df["label"].values, cal_test),
        "ece_catboost": compute_ece(test_df["label"].values, cal_test),
        "n_test": len(test_df),
        "platt_a": a,
        "platt_b": b,
    }
    return model, a, b, metrics, feature_cols
