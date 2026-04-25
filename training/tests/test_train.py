"""Tests: Platt correctness, walk-forward no-overlap, burn-in."""

import numpy as np
import pandas as pd

from progno_train.train import (
    fit_platt,
    apply_platt,
    get_feature_cols,
    walk_forward_splits,
    BURN_IN_YEAR_ATP,
    BURN_IN_YEAR_WTA,
)


def make_feature_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    years = np.repeat(np.arange(2016, 2026), n // 10)[:n]
    return pd.DataFrame({
        "elo_overall_diff": rng.normal(0, 100, n),
        "win_rate_diff": rng.normal(0, 0.2, n),
        "h2h_score": rng.uniform(0.3, 0.7, n),
        "h2h_sample_size": rng.integers(0, 20, n),
        "low_history_flag": rng.integers(0, 2, n),
        "age_diff": rng.normal(0, 3, n),
        "surface": rng.choice(["Hard", "Clay", "Grass"], n),
        "tourney_level": rng.choice(["G", "M", "A"], n),
        "round": rng.choice(["F", "SF", "QF", "R32"], n),
        "best_of_5": rng.integers(0, 2, n),
        "label": rng.integers(0, 2, n),
        "year": years,
        "tourney_date": pd.date_range("2016-01-01", periods=n, freq="3D"),
    })


def test_platt_output_in_range():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.1, 0.9, 100)
    y = (raw > 0.5).astype(int)
    a, b = fit_platt(raw, y)
    cal = apply_platt(raw, a, b)
    assert cal.min() >= 0.0
    assert cal.max() <= 1.0


def test_platt_deterministic():
    rng = np.random.default_rng(1)
    raw = rng.uniform(0.1, 0.9, 100)
    y = (raw > 0.5).astype(int)
    a1, b1 = fit_platt(raw, y)
    a2, b2 = fit_platt(raw, y)
    assert a1 == a2 and b1 == b2


def test_walk_forward_no_overlap():
    df = make_feature_df(200)
    splits = walk_forward_splits(df, burn_in_year=2015, val_start=2018)
    for train, holdout, _ in splits:
        assert set(train["year"]).isdisjoint(set(holdout["year"]))


def test_walk_forward_no_future_in_train():
    df = make_feature_df(200)
    splits = walk_forward_splits(df, burn_in_year=2015, val_start=2018)
    for train, holdout, _ in splits:
        assert train["year"].max() < holdout["year"].min()


def test_walk_forward_atp_burn_in():
    df = make_feature_df(200)
    splits = walk_forward_splits(df, burn_in_year=BURN_IN_YEAR_ATP, val_start=2016)
    for train, _, _ in splits:
        assert train["year"].min() > BURN_IN_YEAR_ATP


def test_walk_forward_wta_burn_in():
    import math
    rng = np.random.default_rng(42)
    n = 160
    years = np.repeat(np.arange(2012, 2026), math.ceil(n / 14))[:n]
    df = pd.DataFrame({
        "elo_overall_diff": rng.normal(0, 100, n),
        "win_rate_diff": rng.normal(0, 0.2, n),
        "h2h_score": rng.uniform(0.3, 0.7, n),
        "h2h_sample_size": rng.integers(0, 20, n),
        "low_history_flag": rng.integers(0, 2, n),
        "age_diff": rng.normal(0, 3, n),
        "surface": rng.choice(["Hard", "Clay", "Grass"], n),
        "tourney_level": rng.choice(["G", "M", "A"], n),
        "round": rng.choice(["F", "SF", "QF", "R32"], n),
        "best_of_5": rng.integers(0, 2, n),
        "label": rng.integers(0, 2, n),
        "year": years,
        "tourney_date": pd.date_range("2012-01-01", periods=n, freq="3D"),
    })
    splits = walk_forward_splits(df, burn_in_year=BURN_IN_YEAR_WTA, val_start=2019)
    for train, _, _ in splits:
        assert train["year"].min() > BURN_IN_YEAR_WTA


def test_get_feature_cols_excludes_metadata():
    df = make_feature_df(10)
    cols = get_feature_cols(df)
    assert "label" not in cols
    assert "year" not in cols
    assert "tourney_date" not in cols
    assert "elo_overall_diff" in cols


def test_wta_full_training_pipeline():
    """E2e: WTA synthetic data → _train_catboost → fit_platt → apply_platt → metrics.

    Verifies the full training stack produces finite, in-range outputs with no
    temporal leakage between train/cal/test splits.
    """
    import math
    from progno_train.train import (
        _train_catboost,
        fit_platt,
        apply_platt,
        get_feature_cols,
        CAL_YEAR,
        TEST_START_YEAR,
    )
    from progno_train.validate import compute_log_loss, compute_ece
    from catboost import Pool, CatBoostClassifier

    rng = np.random.default_rng(7)
    n = 560  # 40 rows × 14 years (2012–2025)
    years = np.repeat(np.arange(2012, 2026), math.ceil(n / 14))[:n]
    df = pd.DataFrame({
        "elo_overall_diff": rng.normal(0, 100, n),
        "win_rate_diff": rng.normal(0, 0.2, n),
        "h2h_score": rng.uniform(0.3, 0.7, n),
        "h2h_sample_size": rng.integers(0, 20, n),
        "low_history_flag": rng.integers(0, 2, n),
        "age_diff": rng.normal(0, 3, n),
        "surface": rng.choice(["Hard", "Clay", "Grass"], n),
        "tourney_level": rng.choice(["G", "M", "A"], n),
        "round": rng.choice(["F", "SF", "QF", "R32"], n),
        "best_of_5": rng.integers(0, 2, n),
        "label": rng.integers(0, 2, n),
        "year": years,
        "tourney_date": pd.date_range("2012-01-01", periods=n, freq="3D"),
    })

    feature_cols = get_feature_cols(df)
    train_df = df[(df["year"] > BURN_IN_YEAR_WTA) & (df["year"] < CAL_YEAR)]
    cal_df = df[df["year"] == CAL_YEAR]
    test_df = df[df["year"] >= TEST_START_YEAR]

    assert len(train_df) > 0
    assert len(cal_df) > 0
    assert len(test_df) > 0

    # No temporal leakage
    assert train_df["year"].max() < cal_df["year"].min()
    assert cal_df["year"].max() < test_df["year"].min()

    model: CatBoostClassifier = _train_catboost(train_df, cal_df, feature_cols)

    from progno_train.train import CAT_FEATURES
    cat_idx = [i for i, c in enumerate(feature_cols) if c in CAT_FEATURES]
    cal_pool = Pool(cal_df[feature_cols].fillna(0), cat_features=cat_idx, feature_names=feature_cols)
    raw_cal = model.predict_proba(cal_pool)[:, 1]
    a, b = fit_platt(raw_cal, cal_df["label"].values)

    test_pool = Pool(test_df[feature_cols].fillna(0), cat_features=cat_idx, feature_names=feature_cols)
    raw_test = model.predict_proba(test_pool)[:, 1]
    cal_probs = apply_platt(raw_test, a, b)

    assert cal_probs.min() >= 0.0
    assert cal_probs.max() <= 1.0
    assert np.isfinite(cal_probs).all()

    logloss = compute_log_loss(test_df["label"].values, cal_probs)
    ece = compute_ece(test_df["label"].values, cal_probs)

    assert np.isfinite(logloss) and logloss >= 0.0
    assert np.isfinite(ece) and 0.0 <= ece <= 1.0
