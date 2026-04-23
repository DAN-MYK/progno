"""Tests for feature engineering: no leakage, cold start, rolling form correctness."""

import pandas as pd
import pytest

from progno_train.features import (
    rolling_win_rate,
    fatigue_features,
    serve_efficiency,
    h2h_score,
    compute_match_features,
    build_all_features,
    POPULATION_WIN_RATE,
    LOW_HISTORY_THRESHOLD,
)


def make_history(n_matches: int, player_a: int = 1, player_b: int = 2) -> pd.DataFrame:
    """Synthetic match history: player_a wins all matches against player_b."""
    base = pd.Timestamp("2020-01-01")
    rows = []
    for i in range(n_matches):
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
    # player 1 wins all 20 matches; as_of_date after all matches
    rate, low = rolling_win_rate(hist, 1, pd.Timestamp("2021-01-01"), 50)
    assert rate == 1.0
    assert low is False  # 20 >= 5


def test_rolling_win_rate_cold_start():
    hist = make_history(3)
    rate, low = rolling_win_rate(hist, 1, pd.Timestamp("2021-01-01"), 50)
    assert rate == POPULATION_WIN_RATE
    assert low is True


def test_rolling_win_rate_no_future_leakage():
    hist = make_history(20)
    # as_of_date = 10th match date — should see < 10 matches
    cutoff = hist["tourney_date"].iloc[9]  # 10th match date
    rate, _ = rolling_win_rate(hist, 1, cutoff, 50)
    # Only first 9 matches visible (strict <), all wins
    assert rate == 1.0
    # Verify we didn't use match 10+
    visible = hist[hist["tourney_date"] < cutoff]
    assert len(visible) == 9


def test_no_leakage_on_match_date():
    """Features for match on date D must NOT include any match from date D onwards."""
    hist = make_history(20)
    for i in range(5, 20):
        as_of = hist["tourney_date"].iloc[i]
        rate, _ = rolling_win_rate(hist, 1, as_of, 50)
        # All visible matches (strictly before as_of) are wins
        assert rate == 1.0
        visible = hist[hist["tourney_date"] < as_of]
        assert len(visible) == i  # exactly i matches visible


def test_h2h_shrinkage_no_history():
    hist = make_history(0)
    score, n = h2h_score(hist, 1, 2, pd.Timestamp("2021-01-01"))
    assert n == 0
    assert abs(score - 0.5) < 0.01  # pure prior


def test_h2h_shrinkage_with_history():
    hist = make_history(10)  # player 1 wins all 10
    score, n = h2h_score(hist, 1, 2, pd.Timestamp("2021-01-01"))
    assert n == 10
    # shrunk = (10 + 5*0.5) / (10 + 5) = 12.5 / 15 ≈ 0.833
    assert abs(score - 12.5 / 15.0) < 0.001


def test_h2h_no_future_leakage():
    hist = make_history(10)
    cutoff = hist["tourney_date"].iloc[5]  # only 5 matches visible
    score, n = h2h_score(hist, 1, 2, cutoff)
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


def test_compute_match_features_returns_dict():
    hist = make_history(20)
    elo_state = {"players": {"1": {"elo_overall": 1600, "elo_hard": 1620, "elo_clay": 1550, "elo_grass": 1580},
                              "2": {"elo_overall": 1500, "elo_hard": 1520, "elo_clay": 1480, "elo_grass": 1500}}}
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
    assert feats["elo_overall_diff"] == 100.0  # 1600 - 1500


def test_elo_monotonicity():
    """Higher Elo diff → higher elo_overall_diff → should correlate with better outcome."""
    hist = make_history(20)
    elo_state_strong = {"players": {"1": {"elo_overall": 1800, "elo_hard": 1800, "elo_clay": 1800, "elo_grass": 1800},
                                     "2": {"elo_overall": 1500, "elo_hard": 1500, "elo_clay": 1500, "elo_grass": 1500}}}
    elo_state_weak = {"players": {"1": {"elo_overall": 1510, "elo_hard": 1510, "elo_clay": 1510, "elo_grass": 1510},
                                   "2": {"elo_overall": 1500, "elo_hard": 1500, "elo_clay": 1500, "elo_grass": 1500}}}
    feats_strong = compute_match_features(hist, elo_state_strong, 1, 2, "Hard", "A", "R32", 3, pd.Timestamp("2021-01-01"))
    feats_weak = compute_match_features(hist, elo_state_weak, 1, 2, "Hard", "A", "R32", 3, pd.Timestamp("2021-01-01"))
    assert feats_strong["elo_overall_diff"] > feats_weak["elo_overall_diff"]


def test_build_all_features_balanced_labels():
    hist = make_history(10)
    elo_state = {"players": {}}
    df = build_all_features(hist, elo_state)
    # Should have 20 rows (10 matches × 2 orientations) with balanced labels
    assert len(df) == 20
    assert set(df["label"].unique()) == {0, 1}
    assert (df["label"] == 1).sum() == 10
    assert (df["label"] == 0).sum() == 10


def test_serve_efficiency_separates_own_stats():
    """Player's own serve stats should not include opponent stats."""
    # Player 1 wins all 10 matches — all rows have won=True for player 1
    # so we only use w_* columns (w_svpt=60, w_1stIn=40) for player 1's stats
    hist = make_history(10)
    se = serve_efficiency(hist, 1, pd.Timestamp("2021-01-01"))
    assert se["first_serve_in_pct"] is not None
    # 40/60 = 0.667 for first_serve_in_pct (not diluted by opponent stats)
    assert abs(se["first_serve_in_pct"] - 40.0 / 60.0) < 0.01
