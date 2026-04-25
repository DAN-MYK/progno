"""Tests: Platt correctness, walk-forward no-overlap, burn-in."""

import numpy as np
import pandas as pd
import pytest

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


def test_wta_parameters_flow_through_splits():
    """Verify WTA burn-in year is respected in walk-forward splits."""
    import math
    rng = np.random.default_rng(42)
    n = 160
    years = np.repeat(np.arange(2008, 2026), math.ceil(n / 18))[:n]
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
        "tourney_date": pd.date_range("2008-01-01", periods=n, freq="3D"),
    })

    # WTA-specific parameters (per spec §6.2 and §2.5)
    splits = walk_forward_splits(
        df,
        burn_in_year=BURN_IN_YEAR_WTA,  # 2007 (serve stats start)
        val_start=2019,
        test_start=2023,
    )

    splits_list = list(splits)
    assert len(splits_list) > 0, "Should produce at least one split"

    for train_df, val_df, test_df in splits_list:
        # Burn-in is excluded from training
        assert train_df["year"].min() > BURN_IN_YEAR_WTA, \
            f"Train should exclude burn-in years (<= {BURN_IN_YEAR_WTA}), got {train_df['year'].min()}"

        # Validation and test years are disjoint from training
        if len(val_df) > 0:
            assert set(train_df["year"]).isdisjoint(set(val_df["year"])), \
                "Train and val should not overlap"
        if len(test_df) > 0:
            assert set(train_df["year"]).isdisjoint(set(test_df["year"])), \
                "Train and test should not overlap"
