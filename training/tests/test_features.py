"""Tests for feature engineering: no leakage, cold start, H2H."""

import pandas as pd
import pytest

from progno_train.features import (
    LOW_HISTORY_THRESHOLD,
    POPULATION_WIN_RATE,
    _build_h2h_index,
    _build_player_index,
    _rolling_serve_stats,
    _slice_before,
    build_all_features,
    compute_match_features,
    fatigue_features,
    h2h_score,
    rolling_win_rate,
    serve_efficiency,
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


# --- _build_player_index ---

def test_build_player_index_groups_both_players():
    hist = make_history(6)
    idx = _build_player_index(hist)
    assert 1 in idx and 2 in idx
    assert len(idx[1]) == 6  # player 1 appears in all 6 (as winner)
    assert len(idx[2]) == 6  # player 2 appears in all 6 (as loser)


def test_build_player_index_excludes_incomplete():
    hist = make_history(4)
    hist.loc[1, "is_complete"] = False  # make row 1 incomplete
    idx = _build_player_index(hist)
    assert len(idx[1]) == 3
    assert len(idx[2]) == 3


def test_build_player_index_empty_history():
    idx = _build_player_index(pd.DataFrame())
    assert idx == {}


def test_build_player_index_sorted_by_date():
    hist = make_history(5)
    idx = _build_player_index(hist)
    dates = idx[1]["tourney_date"].tolist()
    assert dates == sorted(dates)


# --- _build_h2h_index ---

def test_build_h2h_index_correct_pair_key():
    hist = make_history(5)
    idx = _build_h2h_index(hist)
    assert (1, 2) in idx  # min_pid=1, max_pid=2


def test_build_h2h_index_cumsum_all_wins_by_min_pid():
    hist = make_history(5)  # player 1 wins all
    idx = _build_h2h_index(hist)
    dates, cumsum = idx[(1, 2)]
    assert len(dates) == 5
    assert list(cumsum) == [1, 2, 3, 4, 5]  # player 1 (min_pid) won all


def test_build_h2h_index_mixed_wins():
    base = pd.Timestamp("2020-01-01")
    rows = []
    for i in range(6):
        winner, loser = (1, 2) if i % 2 == 0 else (2, 1)
        rows.append({
            "winner_id": winner, "loser_id": loser,
            "tourney_date": base + pd.Timedelta(days=7 * i),
            "is_complete": True,
        })
    hist = pd.DataFrame(rows)
    idx = _build_h2h_index(hist)
    dates, cumsum = idx[(1, 2)]
    assert list(cumsum) == [1, 1, 2, 2, 3, 3]  # alternating wins for player 1


def test_build_h2h_index_empty_history():
    idx = _build_h2h_index(pd.DataFrame())
    assert idx == {}


# --- _slice_before ---

def test_slice_before_returns_strictly_before():
    hist = make_history(10)
    idx = _build_player_index(hist)
    frame = idx[1]
    cutoff = frame["tourney_date"].iloc[5]  # 6th date
    sliced = _slice_before(frame, cutoff)
    assert len(sliced) == 5
    assert all(sliced["tourney_date"] < cutoff)


def test_slice_before_cutoff_before_all_returns_empty():
    hist = make_history(5)
    idx = _build_player_index(hist)
    frame = idx[1]
    cutoff = frame["tourney_date"].iloc[0]  # first date → nothing before it
    sliced = _slice_before(frame, cutoff)
    assert len(sliced) == 0


def test_slice_before_cutoff_after_all_returns_all():
    hist = make_history(5)
    idx = _build_player_index(hist)
    frame = idx[1]
    cutoff = frame["tourney_date"].iloc[-1] + pd.Timedelta(days=1)
    sliced = _slice_before(frame, cutoff)
    assert len(sliced) == 5


def test_slice_before_empty_frame():
    empty = pd.DataFrame({"tourney_date": pd.Series([], dtype="datetime64[ns]")})
    result = _slice_before(empty, pd.Timestamp("2021-01-01"))
    assert len(result) == 0


# --- h2h_score index consistency ---

def test_h2h_index_matches_full_scan():
    hist = make_history(10)
    h2h_idx = _build_h2h_index(hist)
    as_of = pd.Timestamp("2021-06-01")  # after all matches

    score_idx, n_idx = h2h_score(hist, 1, 2, as_of, _h2h_index=h2h_idx)
    score_scan, n_scan = h2h_score(hist, 1, 2, as_of)

    assert n_idx == n_scan
    assert abs(score_idx - score_scan) < 1e-9


def test_h2h_index_respects_cutoff():
    hist = make_history(10)
    h2h_idx = _build_h2h_index(hist)
    cutoff = hist["tourney_date"].iloc[4]  # after 4 matches

    _, n_idx = h2h_score(hist, 1, 2, cutoff, _h2h_index=h2h_idx)
    _, n_scan = h2h_score(hist, 1, 2, cutoff)

    assert n_idx == n_scan == 4


# --- serve_efficiency ---

def test_serve_efficiency_returns_valid_fractions():
    hist = make_history(10)
    idx = _build_player_index(hist)
    frame = _slice_before(idx[1], pd.Timestamp("2021-01-01"))
    srv = serve_efficiency(pd.DataFrame(), 1, pd.Timestamp("2021-01-01"), _frame=frame)
    assert srv["first_serve_in_pct"] is not None
    assert 0.0 <= srv["first_serve_in_pct"] <= 1.0
    assert srv["ace_rate"] is not None
    assert srv["df_rate"] is not None


def test_serve_efficiency_empty_history():
    srv = serve_efficiency(pd.DataFrame(), 1, pd.Timestamp("2021-01-01"))
    assert srv["first_serve_in_pct"] is None
    assert srv["first_serve_won_pct"] is None
    assert srv["ace_rate"] is None
    assert srv["df_rate"] is None


def test_serve_efficiency_correct_first_serve_in():
    hist = make_history(10)
    idx = _build_player_index(hist)
    frame = _slice_before(idx[1], pd.Timestamp("2021-01-01"))
    srv = serve_efficiency(pd.DataFrame(), 1, pd.Timestamp("2021-01-01"), _frame=frame)
    # w_1stIn=40, w_svpt=60 (all as winner) → 40/60 ≈ 0.6667
    assert abs(srv["first_serve_in_pct"] - 40.0 / 60.0) < 1e-6


# --- build_all_features ---

def test_build_all_features_balanced_labels():
    hist = make_history(10)
    result = build_all_features(hist, {"players": {}})
    assert (result["label"] == 0).sum() == (result["label"] == 1).sum() == 10


def test_build_all_features_min_year_includes_matching():
    hist = make_history(10)  # all dates in 2020
    result = build_all_features(hist, {"players": {}}, min_year=2020)
    assert len(result) == 20  # 10 matches × 2 labels


def test_build_all_features_min_year_excludes_past():
    hist = make_history(10)  # all dates in 2020
    result = build_all_features(hist, {"players": {}}, min_year=2025)
    assert len(result) == 0


def test_build_all_features_has_year_column():
    hist = make_history(5)
    result = build_all_features(hist, {"players": {}})
    assert "year" in result.columns
    assert all(result["year"] == result["tourney_date"].dt.year)


def test_build_all_features_no_leakage_win_rate():
    """Each row's win_rate_diff must be computed from strictly prior matches."""
    hist = make_history(6)
    result = build_all_features(hist, {"players": {}}, min_year=2020)
    # First match: no prior history → low_history_flag=1 for both players
    first_match_rows = result[result["tourney_date"] == hist["tourney_date"].iloc[0]]
    assert all(first_match_rows["low_history_flag"] == 1)


def _make_serve_frame(n: int = 10, bp_faced: float = 4.0) -> pd.DataFrame:
    """Player wins all n matches; all serve columns populated."""
    base = pd.Timestamp("2020-01-01")
    rows = []
    for i in range(n):
        rows.append({
            "tourney_date": base + pd.Timedelta(days=7 * i),
            "surface": "Hard",
            "minutes": 90.0,
            "completed_sets": 2,
            "won": True,
            "opponent_rank": 50.0,
            "w_ace": 5.0, "w_df": 1.0,
            "w_svpt": 60.0, "w_1stIn": 40.0, "w_1stWon": 30.0, "w_2ndWon": 12.0,
            "w_bpSaved": 3.0, "w_bpFaced": bp_faced,
            "l_ace": 2.0, "l_df": 3.0,
            "l_svpt": 100.0, "l_1stIn": 60.0, "l_1stWon": 60.0, "l_2ndWon": 20.0,
            "l_bpSaved": 2.0, "l_bpFaced": 5.0,
        })
    return pd.DataFrame(rows)


def test_second_won_pct_correct_formula() -> None:
    frame = _make_serve_frame(10)
    stats = _rolling_serve_stats(pd.DataFrame(), 1, pd.Timestamp("2021-01-01"), _frame=frame)
    # Player wins all 10 matches: w_2ndWon=12, w_svpt=60, w_1stIn=40
    # second_won_pct = sum(w_2ndWon) / sum(w_svpt - w_1stIn) = 120 / 200 = 0.60
    assert stats["second_won_pct"] is not None
    assert abs(stats["second_won_pct"] - 120.0 / 200.0) < 1e-6


def test_bp_save_pct_zero_faced_returns_none() -> None:
    frame = _make_serve_frame(10, bp_faced=0.0)
    stats = _rolling_serve_stats(pd.DataFrame(), 1, pd.Timestamp("2021-01-01"), _frame=frame)
    # bpFaced=0 → denominator=0 → None (caller fills with population median)
    assert stats["bp_save_pct"] is None
    # Other stats still computed normally
    assert stats["second_won_pct"] is not None
    assert stats["return_pts_pct"] is not None


def test_return_pts_pct_correct_columns() -> None:
    frame = _make_serve_frame(10)
    stats = _rolling_serve_stats(pd.DataFrame(), 1, pd.Timestamp("2021-01-01"), _frame=frame)
    # return_pts_pct uses OPPONENT's serve columns (l_* when player won)
    # l_svpt=100, l_1stWon=60, l_2ndWon=20 → player won 100-60-20=20 of 100 opp serve points
    # return_pts_pct = 20/100 = 0.20
    assert stats["return_pts_pct"] is not None
    assert abs(stats["return_pts_pct"] - 20.0 / 100.0) < 1e-6


def test_new_features_no_future_leakage() -> None:
    hist = make_history(20)
    idx = _build_player_index(hist)
    cutoff = hist["tourney_date"].iloc[9]  # 10th match date
    frame_before = _slice_before(idx[1], cutoff)
    assert len(frame_before) == 9  # only 9 prior matches visible

    stats = _rolling_serve_stats(pd.DataFrame(), 1, cutoff, _frame=frame_before)
    # frame_before has 9 >= min_periods=5 → stats are computed (not None due to cold start)
    assert stats["second_won_pct"] is not None
    # All data in frame_before is strictly before cutoff
    assert all(frame_before["tourney_date"] < cutoff)
