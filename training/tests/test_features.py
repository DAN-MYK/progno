"""Tests for feature engineering: no leakage, cold start, H2H."""

import pandas as pd
import pytest

from progno_train.features import (
    POPULATION_WIN_RATE,
    LOW_HISTORY_THRESHOLD,
    rolling_win_rate,
    fatigue_features,
    h2h_score,
    compute_match_features,
)


def make_history(n: int, player_a: int = 1, player_b: int = 2) -> pd.DataFrame:
    """Synthetic history: player_a wins all matches against player_b."""
    base = pd.Timestamp("2020-01-01")
    rows = []
    for i in range(n):
        rows.append({
            "winner_id": player_a, "loser_id": player_b,
            "tourney_date": base + pd.Timedelta(days=7 * i),
            "surface": "Hard", "tourney_level": "A", "round": "R32", "best_of": 3,
            "is_complete": True, "completed_sets": 2, "score": "6-3 6-4", "minutes": 90.0,
            "winner_rank": 10, "loser_rank": 50,
            "winner_age": 24.0, "loser_age": 26.0,
            "winner_ht": 185.0, "loser_ht": 180.0,
            "winner_hand": "R", "loser_hand": "R",
            "w_ace": 5.0, "w_df": 1.0, "w_svpt": 60.0, "w_1stIn": 40.0,
            "w_1stWon": 30.0, "w_2ndWon": 12.0, "w_bpSaved": 3.0, "w_bpFaced": 4.0,
            "l_ace": 2.0, "l_df": 3.0, "l_svpt": 58.0, "l_1stIn": 35.0,
            "l_1stWon": 22.0, "l_2ndWon": 10.0, "l_bpSaved": 2.0, "l_bpFaced": 5.0,
        })
    return pd.DataFrame(rows)


def test_rolling_win_rate_correct():
    hist = make_history(20)
    rate, low = rolling_win_rate(hist, 1, pd.Timestamp("2021-01-01"), 50)
    assert rate == 1.0
    assert low is False


def test_rolling_win_rate_cold_start():
    hist = make_history(3)
    rate, low = rolling_win_rate(hist, 1, pd.Timestamp("2021-01-01"), 50)
    assert rate == POPULATION_WIN_RATE
    assert low is True


def test_rolling_win_rate_no_future_leakage():
    hist = make_history(20)
    cutoff = hist["tourney_date"].iloc[9]
    rate, _ = rolling_win_rate(hist, 1, cutoff, 50)
    visible = hist[hist["tourney_date"] < cutoff]
    assert len(visible) == 9
    assert rate == 1.0


def test_no_leakage_on_match_date():
    hist = make_history(20)
    for i in range(5, 20):
        as_of = hist["tourney_date"].iloc[i]
        rate, _ = rolling_win_rate(hist, 1, as_of, 50)
        assert rate == 1.0
        visible = hist[hist["tourney_date"] < as_of]
        assert len(visible) == i


def test_h2h_shrinkage_no_history():
    hist = make_history(0)
    score, n = h2h_score(hist, 1, 2, pd.Timestamp("2021-01-01"))
    assert n == 0
    assert abs(score - 0.5) < 0.01


def test_h2h_shrinkage_with_history():
    hist = make_history(10)
    score, n = h2h_score(hist, 1, 2, pd.Timestamp("2021-01-01"))
    assert n == 10
    assert abs(score - 12.5 / 15.0) < 0.001


def test_h2h_no_future_leakage():
    hist = make_history(10)
    cutoff = hist["tourney_date"].iloc[5]
    _, n = h2h_score(hist, 1, 2, cutoff)
    assert n == 5


def test_fatigue_days_since():
    hist = make_history(10)
    last_date = hist["tourney_date"].max()
    as_of = last_date + pd.Timedelta(days=10)
    fat = fatigue_features(hist, 1, as_of, "Hard")
    assert fat["days_since_last_match"] == 10.0


def test_fatigue_capped_at_30():
    hist = make_history(1)
    as_of = hist["tourney_date"].iloc[0] + pd.Timedelta(days=60)
    fat = fatigue_features(hist, 1, as_of, "Hard")
    assert fat["days_since_last_match"] == 30.0


def test_fatigue_cold_start():
    hist = make_history(0)
    fat = fatigue_features(hist, 1, pd.Timestamp("2021-01-01"), "Hard")
    assert fat["days_since_last_match"] == 30.0
    assert fat["matches_last_30d"] == 0.0


def test_compute_match_features_returns_expected_keys():
    hist = make_history(20)
    elo_state = {"players": {
        "1": {"elo_overall": 1600, "elo_hard": 1620, "elo_clay": 1550, "elo_grass": 1580},
        "2": {"elo_overall": 1500, "elo_hard": 1520, "elo_clay": 1480, "elo_grass": 1500},
    }}
    feats = compute_match_features(
        history=hist, elo_state=elo_state,
        player_a_id=1, player_b_id=2,
        surface="Hard", tourney_level="A", round_="QF", best_of=3,
        tourney_date=pd.Timestamp("2021-01-01"),
    )
    assert isinstance(feats, dict)
    assert "elo_overall_diff" in feats
    assert "win_rate_diff" in feats
    assert "h2h_score" in feats
    assert feats["elo_overall_diff"] == 100.0


def test_elo_monotonicity():
    hist = make_history(20)
    strong = {"players": {"1": {"elo_overall": 1800, "elo_hard": 1800, "elo_clay": 1800, "elo_grass": 1800},
                          "2": {"elo_overall": 1500, "elo_hard": 1500, "elo_clay": 1500, "elo_grass": 1500}}}
    weak = {"players": {"1": {"elo_overall": 1510, "elo_hard": 1510, "elo_clay": 1510, "elo_grass": 1510},
                        "2": {"elo_overall": 1500, "elo_hard": 1500, "elo_clay": 1500, "elo_grass": 1500}}}
    f_strong = compute_match_features(hist, strong, 1, 2, "Hard", "A", "R32", 3, pd.Timestamp("2021-01-01"))
    f_weak = compute_match_features(hist, weak, 1, 2, "Hard", "A", "R32", 3, pd.Timestamp("2021-01-01"))
    assert f_strong["elo_overall_diff"] > f_weak["elo_overall_diff"]
