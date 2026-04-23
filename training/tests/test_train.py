"""Tests for training pipeline: determinism, Platt correctness."""

import numpy as np
import pandas as pd
import pytest

from progno_train.train import fit_platt, apply_platt, get_feature_cols, walk_forward_splits


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
        "height_diff": rng.normal(0, 10, n),
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
    assert a1 == a2
    assert b1 == b2


def test_platt_identity_for_perfect_model():
    """If raw probs are already perfectly calibrated, a≈1 b≈0."""
    rng = np.random.default_rng(42)
    raw = rng.uniform(0.2, 0.8, 500)
    y = rng.binomial(1, raw)
    a, b = fit_platt(raw, y)
    cal = apply_platt(raw, a, b)
    assert np.mean((cal - raw) ** 2) < 0.02


def test_walk_forward_splits_no_overlap():
    df = make_feature_df(200)
    splits = walk_forward_splits(df, val_start=2018, test_start=2023)
    for train, holdout, label in splits:
        assert len(train) > 0
        assert len(holdout) > 0
        assert set(train["year"]).isdisjoint(set(holdout["year"]))


def test_walk_forward_no_future_in_train():
    df = make_feature_df(200)
    splits = walk_forward_splits(df, val_start=2018, test_start=2023)
    for train, holdout, label in splits:
        holdout_year = holdout["year"].min()
        assert train["year"].max() < holdout_year


def test_get_feature_cols_excludes_metadata():
    df = make_feature_df(10)
    cols = get_feature_cols(df)
    assert "label" not in cols
    assert "year" not in cols
    assert "tourney_date" not in cols
    assert "elo_overall_diff" in cols
