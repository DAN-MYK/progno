"""Pre-match feature engineering — all features are time-gated (no leakage)."""

from __future__ import annotations

from typing import Any

import pandas as pd

POPULATION_WIN_RATE = 0.5
LOW_HISTORY_THRESHOLD = 5


def _player_matches_before(
    history: pd.DataFrame,
    player_id: int,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """All completed matches for player_id strictly before as_of_date."""
    if history.empty:
        return pd.DataFrame()

    won_mask = (
        (history["winner_id"] == player_id)
        & (history["tourney_date"] < as_of_date)
    )
    lost_mask = (
        (history["loser_id"] == player_id)
        & (history["tourney_date"] < as_of_date)
    )
    w = history[won_mask].assign(won=True, opponent_rank=history.loc[won_mask, "loser_rank"])
    l = history[lost_mask].assign(won=False, opponent_rank=history.loc[lost_mask, "winner_rank"])

    base_cols = [
        "tourney_date", "surface", "opponent_rank",
        "minutes", "completed_sets",
        "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced",
        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced",
    ]
    # Only keep columns that exist in the dataframe, plus "won" which was assigned
    keep_cols = [c for c in base_cols if c in w.columns] + ["won"]

    result = pd.concat([w[keep_cols], l[keep_cols]], ignore_index=True).sort_values("tourney_date")
    return result


def rolling_win_rate(
    history: pd.DataFrame,
    player_id: int,
    as_of_date: pd.Timestamp,
    n: int,
    surface: str | None = None,
    max_opponent_rank: int | None = None,
    since_date: pd.Timestamp | None = None,
) -> tuple[float, bool]:
    """Win rate over last n completed matches. Returns (win_rate, low_history_flag)."""
    df = _player_matches_before(history, player_id, as_of_date)
    if since_date is not None:
        df = df[df["tourney_date"] >= since_date]
    if surface:
        df = df[df["surface"] == surface]
    if max_opponent_rank is not None:
        df = df[df["opponent_rank"].fillna(9999) <= max_opponent_rank]
    df = df.tail(n)
    if len(df) < LOW_HISTORY_THRESHOLD:
        return POPULATION_WIN_RATE, True
    return float(df["won"].mean()), False


def fatigue_features(
    history: pd.DataFrame,
    player_id: int,
    as_of_date: pd.Timestamp,
    prev_surface: str | None,
) -> dict[str, float]:
    """Days since last match, sets played last 14d, matches last 30d, surface switch."""
    df = _player_matches_before(history, player_id, as_of_date)
    if df.empty:
        return {
            "days_since_last_match": 30.0,
            "sets_last_14d": 0.0,
            "matches_last_30d": 0.0,
            "surface_switch": 0.0,
        }
    last_match_date = df["tourney_date"].max()
    days_since = min((as_of_date - last_match_date).days, 30)

    cutoff_14d = as_of_date - pd.Timedelta(days=14)
    cutoff_30d = as_of_date - pd.Timedelta(days=30)
    sets_14d = float(df[df["tourney_date"] >= cutoff_14d]["completed_sets"].sum())
    matches_30d = float((df["tourney_date"] >= cutoff_30d).sum())

    last_surface = df.iloc[-1]["surface"] if "surface" in df.columns else None
    surface_switch = 1.0 if (prev_surface and last_surface and prev_surface != last_surface) else 0.0

    return {
        "days_since_last_match": float(days_since),
        "sets_last_14d": sets_14d,
        "matches_last_30d": matches_30d,
        "surface_switch": surface_switch,
    }


def serve_efficiency(
    history: pd.DataFrame,
    player_id: int,
    as_of_date: pd.Timestamp,
    n: int = 25,
) -> dict[str, float | None]:
    """Rolling serve/return stats over last n matches. Returns None for each stat if insufficient data."""
    df = _player_matches_before(history, player_id, as_of_date).tail(n)
    result: dict[str, float | None] = {}

    eps = 1e-6

    def _safe_div(a: float, b: float) -> float | None:
        return float(a / b) if b > eps else None

    if all(c in df.columns for c in ["w_svpt", "w_1stIn", "l_svpt", "l_1stIn"]):
        won_df = df[df["won"]]
        lost_df = df[~df["won"]]
        svpt      = won_df["w_svpt"].fillna(0).sum()  + lost_df["l_svpt"].fillna(0).sum()
        first_in  = won_df["w_1stIn"].fillna(0).sum() + lost_df["l_1stIn"].fillna(0).sum()
        first_won = won_df["w_1stWon"].fillna(0).sum()+ lost_df["l_1stWon"].fillna(0).sum()
        ace       = won_df["w_ace"].fillna(0).sum()   + lost_df["l_ace"].fillna(0).sum()
        df_count  = won_df["w_df"].fillna(0).sum()    + lost_df["l_df"].fillna(0).sum()

        result["first_serve_in_pct"]  = _safe_div(first_in, svpt)
        result["first_serve_won_pct"] = _safe_div(first_won, first_in)
        result["ace_rate"]            = _safe_div(ace, svpt)
        result["df_rate"]             = _safe_div(df_count, svpt)
    else:
        result = {"first_serve_in_pct": None, "first_serve_won_pct": None,
                  "ace_rate": None, "df_rate": None}

    return result


def h2h_score(
    history: pd.DataFrame,
    player_a_id: int,
    player_b_id: int,
    as_of_date: pd.Timestamp,
    prior: int = 5,
    prior_mean: float = 0.5,
) -> tuple[float, int]:
    """Shrinkage H2H win rate for A vs B. Returns (shrunk_win_rate, sample_size)."""
    if history.empty:
        shrunk = prior * prior_mean / prior
        return float(shrunk), 0

    mask = (
        (
            ((history["winner_id"] == player_a_id) & (history["loser_id"] == player_b_id))
            | ((history["winner_id"] == player_b_id) & (history["loser_id"] == player_a_id))
        )
        & (history["tourney_date"] < as_of_date)
        & history["is_complete"]
    )
    df = history[mask]
    n = len(df)
    wins_a = int((df["winner_id"] == player_a_id).sum())
    shrunk = (wins_a + prior * prior_mean) / (n + prior)
    return float(shrunk), int(n)


def compute_match_features(
    history: pd.DataFrame,
    elo_state: dict,
    player_a_id: int,
    player_b_id: int,
    surface: str,
    tourney_level: str,
    round_: str,
    best_of: int,
    tourney_date: pd.Timestamp,
    player_a_rank: int | None = None,
    player_b_rank: int | None = None,
    player_a_age: float | None = None,
    player_b_age: float | None = None,
    player_a_height: float | None = None,
    player_b_height: float | None = None,
    player_a_hand: str | None = None,
    player_b_hand: str | None = None,
) -> dict[str, Any]:
    """Compute all pre-match features for a single match. Time-gated at tourney_date."""
    feats: dict[str, Any] = {}

    # Rolling form
    wr_a_50, lhf_a = rolling_win_rate(history, player_a_id, tourney_date, 50)
    wr_b_50, lhf_b = rolling_win_rate(history, player_b_id, tourney_date, 50)
    wr_a_surf, _ = rolling_win_rate(history, player_a_id, tourney_date, 20, surface=surface)
    wr_b_surf, _ = rolling_win_rate(history, player_b_id, tourney_date, 20, surface=surface)
    since_12m = tourney_date - pd.DateOffset(months=12)
    wr_a_12m, _ = rolling_win_rate(history, player_a_id, tourney_date, 9999, since_date=since_12m)
    wr_b_12m, _ = rolling_win_rate(history, player_b_id, tourney_date, 9999, since_date=since_12m)
    wr_a_top20, _ = rolling_win_rate(history, player_a_id, tourney_date, 30, max_opponent_rank=20)
    wr_b_top20, _ = rolling_win_rate(history, player_b_id, tourney_date, 30, max_opponent_rank=20)

    feats["win_rate_diff"] = wr_a_50 - wr_b_50
    feats["win_rate_surface_diff"] = wr_a_surf - wr_b_surf
    feats["win_rate_12m_diff"] = wr_a_12m - wr_b_12m
    feats["win_rate_top20_diff"] = wr_a_top20 - wr_b_top20
    feats["low_history_flag"] = int(lhf_a or lhf_b)

    # Fatigue
    fat_a = fatigue_features(history, player_a_id, tourney_date, surface)
    fat_b = fatigue_features(history, player_b_id, tourney_date, surface)
    feats["days_since_last_diff"] = fat_a["days_since_last_match"] - fat_b["days_since_last_match"]
    feats["sets_last_14d_diff"] = fat_a["sets_last_14d"] - fat_b["sets_last_14d"]
    feats["matches_last_30d_diff"] = fat_a["matches_last_30d"] - fat_b["matches_last_30d"]
    feats["surface_switch_a"] = fat_a["surface_switch"]
    feats["surface_switch_b"] = fat_b["surface_switch"]

    # Serve efficiency diffs (None → 0 diff)
    srv_a = serve_efficiency(history, player_a_id, tourney_date)
    srv_b = serve_efficiency(history, player_b_id, tourney_date)
    for stat in ["first_serve_in_pct", "first_serve_won_pct", "ace_rate", "df_rate"]:
        va = srv_a.get(stat) or 0.0
        vb = srv_b.get(stat) or 0.0
        feats[f"{stat}_diff"] = va - vb

    # H2H
    h2h, h2h_n = h2h_score(history, player_a_id, player_b_id, tourney_date)
    feats["h2h_score"] = h2h
    feats["h2h_sample_size"] = h2h_n

    # Elo (from elo_state dict — keys are player ids as strings, values have elo fields)
    def _elo(pid: int, field: str) -> float:
        p = elo_state.get("players", {}).get(str(pid), {})
        return float(p.get(field, 1500))

    elo_a = _elo(player_a_id, "elo_overall")
    elo_b = _elo(player_b_id, "elo_overall")
    elo_a_surf = _elo(player_a_id, f"elo_{surface.lower()}")
    elo_b_surf = _elo(player_b_id, f"elo_{surface.lower()}")
    feats["elo_overall_diff"] = elo_a - elo_b
    feats["elo_surface_diff"] = elo_a_surf - elo_b_surf

    # Player meta diffs
    feats["age_diff"] = (player_a_age or 25.0) - (player_b_age or 25.0)
    feats["height_diff"] = (player_a_height or 185.0) - (player_b_height or 185.0)
    feats["lefty_vs_righty"] = int(
        (player_a_hand == "L") != (player_b_hand == "L")
    )

    # Match context (categorical, passed through for CatBoost native handling)
    feats["surface"] = surface
    feats["tourney_level"] = tourney_level
    feats["round"] = round_
    feats["best_of_5"] = int(best_of == 5)

    return feats


def build_all_features(
    history: pd.DataFrame,
    elo_state: dict,
) -> pd.DataFrame:
    """Compute features for every complete match in history. Returns balanced feature DataFrame."""
    import logging

    rows = []
    for _, row in history[history["is_complete"]].iterrows():
        common_kwargs = dict(
            history=history,
            elo_state=elo_state,
            surface=row["surface"],
            tourney_level=row.get("tourney_level", "A"),
            round_=row.get("round", "R32"),
            best_of=row.get("best_of", 3),
            tourney_date=row["tourney_date"],
        )

        # Winner as A (label=1)
        feats_pos = compute_match_features(
            player_a_id=row["winner_id"], player_b_id=row["loser_id"],
            player_a_rank=row.get("winner_rank"), player_b_rank=row.get("loser_rank"),
            player_a_age=row.get("winner_age"), player_b_age=row.get("loser_age"),
            player_a_height=row.get("winner_ht"), player_b_height=row.get("loser_ht"),
            player_a_hand=row.get("winner_hand"), player_b_hand=row.get("loser_hand"),
            **common_kwargs,
        )
        feats_pos["label"] = 1
        feats_pos["tourney_date"] = row["tourney_date"]
        feats_pos["year"] = row["tourney_date"].year
        rows.append(feats_pos)

        # Loser as A (label=0)
        feats_neg = compute_match_features(
            player_a_id=row["loser_id"], player_b_id=row["winner_id"],
            player_a_rank=row.get("loser_rank"), player_b_rank=row.get("winner_rank"),
            player_a_age=row.get("loser_age"), player_b_age=row.get("winner_age"),
            player_a_height=row.get("loser_ht"), player_b_height=row.get("winner_ht"),
            player_a_hand=row.get("loser_hand"), player_b_hand=row.get("winner_hand"),
            **common_kwargs,
        )
        feats_neg["label"] = 0
        feats_neg["tourney_date"] = row["tourney_date"]
        feats_neg["year"] = row["tourney_date"].year
        rows.append(feats_neg)

        if len(rows) % 20000 == 0 and len(rows) > 0:
            logging.getLogger(__name__).info("build_all_features: processed %d rows...", len(rows))

    return pd.DataFrame(rows)
