"""Pre-match feature engineering — all features are time-gated (no leakage)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

POPULATION_WIN_RATE = 0.5
LOW_HISTORY_THRESHOLD = 5
POPULATION_SECOND_WON_PCT = 0.50
POPULATION_BP_SAVE_PCT = 0.63
POPULATION_RETURN_PTS_PCT = 0.38

# Numeric constants — give them names so audit trail is clear
INITIAL_ELO = 1500.0              # default Elo for a player with no history
SURFACE_MIN_MATCHES = 20          # min surface matches before surface Elo is trusted (spec §2.3)
WIN_RATE_RECENT = 50              # rolling window: recent form
WIN_RATE_SURFACE = 20             # rolling window: surface-specific form (coincidentally same as SURFACE_MIN_MATCHES; independent concept)
WIN_RATE_TOP20 = 30               # rolling window: performance vs top-20
WIN_RATE_ALL_TIME = 9999          # sentinel: "all available history" (no window limit)
UNKNOWN_RANK_SENTINEL = 9999      # fillna substitute for missing opponent ranks (treated as unranked)
DEFAULT_PLAYER_AGE = 25.0         # fallback age when absent from match data
DEFAULT_PLAYER_HEIGHT_CM = 185.0  # fallback height (cm) when absent
EPS = 1e-7                        # numerical stability guard for division

_PLAYER_INDEX_BASE_COLS = [
    "tourney_date", "surface",
    "minutes", "completed_sets",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced",
]


def _build_player_index(history: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Pre-build sorted per-player match frames for O(n log n) feature computation."""
    if history.empty:
        return {}

    if "is_complete" in history.columns:
        history = history[history["is_complete"]]
    if history.empty:
        return {}

    keep = [c for c in _PLAYER_INDEX_BASE_COLS if c in history.columns]

    w = history[keep].copy()
    w["_pid"] = history["winner_id"].values
    w["won"] = True
    w["opponent_rank"] = history["loser_rank"].values if "loser_rank" in history.columns else float("nan")

    l = history[keep].copy()
    l["_pid"] = history["loser_id"].values
    l["won"] = False
    l["opponent_rank"] = history["winner_rank"].values if "winner_rank" in history.columns else float("nan")

    combined = (
        pd.concat([w, l], ignore_index=True)
        .sort_values("tourney_date", kind="mergesort")
        .reset_index(drop=True)
    )

    return {
        int(pid): grp.drop(columns=["_pid"]).reset_index(drop=True)
        for pid, grp in combined.groupby("_pid")
    }


def _build_h2h_index(history: pd.DataFrame) -> dict[tuple[int, int], np.ndarray]:
    """Pre-build pair → sorted date array for O(log k) H2H lookups.

    Key is (min_pid, max_pid). Value is sorted array of match dates where A won
    (A = the smaller pid), used to compute wins_a count via searchsorted.
    Stores tuple: (sorted_dates, wins_a_cumsum) where wins_a_cumsum[i] = wins by min_pid in first i+1 matches.
    """
    if history.empty:
        return {}

    complete = history[history["is_complete"]] if "is_complete" in history.columns else history
    if complete.empty:
        return {}

    pairs: dict[tuple[int, int], list[tuple]] = {}
    for row in complete[["winner_id", "loser_id", "tourney_date"]].itertuples(index=False):
        w, l, d = int(row.winner_id), int(row.loser_id), row.tourney_date
        key = (min(w, l), max(w, l))
        pairs.setdefault(key, []).append((d, w))

    result = {}
    for key, matches in pairs.items():
        matches.sort(key=lambda x: x[0])
        dates = np.array([m[0] for m in matches], dtype="datetime64[ns]")
        min_pid = key[0]
        wins_min = np.array([1 if m[1] == min_pid else 0 for m in matches], dtype=np.int32)
        result[key] = (dates, np.cumsum(wins_min))
    return result


def _slice_before(player_frame: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Return rows with tourney_date strictly before as_of_date (O(log k) via searchsorted)."""
    if player_frame.empty:
        return player_frame
    dates = player_frame["tourney_date"].to_numpy()
    idx = int(np.searchsorted(dates, as_of_date, side="left"))
    return player_frame.iloc[:idx]


def _player_matches_before(
    history: pd.DataFrame,
    player_id: int,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """All completed matches for player_id strictly before as_of_date (full-scan fallback)."""
    if history.empty or "winner_id" not in history.columns:
        return pd.DataFrame()
    complete = history["is_complete"] if "is_complete" in history.columns else True
    won_mask = (history["winner_id"] == player_id) & (history["tourney_date"] < as_of_date) & complete
    lost_mask = (history["loser_id"] == player_id) & (history["tourney_date"] < as_of_date) & complete

    cols_base = ["tourney_date", "surface", "minutes", "completed_sets",
                 "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced",
                 "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced"]
    available = [c for c in cols_base if c in history.columns]

    w = history.loc[won_mask, available + ["loser_rank"]].assign(won=True, opponent_rank=history.loc[won_mask, "loser_rank"])
    l = history.loc[lost_mask, available + ["winner_rank"]].assign(won=False, opponent_rank=history.loc[lost_mask, "winner_rank"])

    return pd.concat([w, l], ignore_index=True).sort_values("tourney_date")


def rolling_win_rate(
    history: pd.DataFrame,
    player_id: int,
    as_of_date: pd.Timestamp,
    n: int,
    surface: str | None = None,
    max_opponent_rank: int | None = None,
    since_date: pd.Timestamp | None = None,
    *,
    _frame: pd.DataFrame | None = None,
) -> tuple[float, bool]:
    """Win rate over last n completed matches. Returns (win_rate, low_history_flag)."""
    df = _frame if _frame is not None else _player_matches_before(history, player_id, as_of_date)
    if since_date is not None:
        df = df[df["tourney_date"] >= since_date]
    if surface:
        df = df[df["surface"] == surface]
    if max_opponent_rank is not None:
        df = df[df["opponent_rank"].fillna(UNKNOWN_RANK_SENTINEL) <= max_opponent_rank]
    df = df.tail(n)
    if len(df) < LOW_HISTORY_THRESHOLD:
        return POPULATION_WIN_RATE, True
    return float(df["won"].mean()), False


def fatigue_features(
    history: pd.DataFrame,
    player_id: int,
    as_of_date: pd.Timestamp,
    current_surface: str,
    *,
    _frame: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Days since last match, sets played last 14d, matches last 30d, surface switch."""
    df = _frame if _frame is not None else _player_matches_before(history, player_id, as_of_date)
    if df.empty:
        return {"days_since_last_match": 30.0, "sets_last_14d": 0.0,
                "matches_last_30d": 0.0, "surface_switch": 0.0}

    last_date = df["tourney_date"].max()
    days_since = min((as_of_date - last_date).days, 30)

    cutoff_14d = as_of_date - pd.Timedelta(days=14)
    cutoff_30d = as_of_date - pd.Timedelta(days=30)
    sets_14d = float(df.loc[df["tourney_date"] >= cutoff_14d, "completed_sets"].sum())
    matches_30d = float((df["tourney_date"] >= cutoff_30d).sum())

    last_surface = df.iloc[-1]["surface"] if "surface" in df.columns else None
    surface_switch = 1.0 if (last_surface and last_surface != current_surface) else 0.0

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
    *,
    _frame: pd.DataFrame | None = None,
) -> dict[str, float | None]:
    """Rolling serve stats over last n matches."""
    base = _frame if _frame is not None else _player_matches_before(history, player_id, as_of_date)
    df = base.tail(n)
    eps = EPS

    def _safe_div(a: float, b: float) -> float | None:
        return float(a / b) if b > eps else None

    if all(c in df.columns for c in ["w_svpt", "w_1stIn", "l_svpt", "l_1stIn"]):
        won_df  = df[df["won"]]  if "won" in df.columns else df
        lost_df = df[~df["won"]] if "won" in df.columns else pd.DataFrame()
        svpt      = won_df["w_svpt"].fillna(0).sum()   + lost_df["l_svpt"].fillna(0).sum()
        first_in  = won_df["w_1stIn"].fillna(0).sum()  + lost_df["l_1stIn"].fillna(0).sum()
        first_won = won_df["w_1stWon"].fillna(0).sum() + lost_df["l_1stWon"].fillna(0).sum()
        ace       = won_df["w_ace"].fillna(0).sum()    + lost_df["l_ace"].fillna(0).sum()
        df_count  = won_df["w_df"].fillna(0).sum()     + lost_df["l_df"].fillna(0).sum()
        return {
            "first_serve_in_pct":  _safe_div(first_in, svpt),
            "first_serve_won_pct": _safe_div(first_won, first_in),
            "ace_rate":            _safe_div(ace, svpt),
            "df_rate":             _safe_div(df_count, svpt),
        }

    return {"first_serve_in_pct": None, "first_serve_won_pct": None,
            "ace_rate": None, "df_rate": None}


def _rolling_serve_stats(
    history: pd.DataFrame,
    player_id: int,
    as_of_date: pd.Timestamp,
    n: int = 25,
    min_periods: int = 5,
    *,
    _frame: pd.DataFrame | None = None,
) -> dict[str, float | None]:
    """Rolling second_won_pct, bp_save_pct, return_pts_pct over last n matches."""
    base = _frame if _frame is not None else _player_matches_before(history, player_id, as_of_date)
    df = base.tail(n)

    if len(df) < min_periods:
        return {"second_won_pct": None, "bp_save_pct": None, "return_pts_pct": None}

    eps = EPS
    won_df = df[df["won"]] if "won" in df.columns else df
    lost_df = df[~df["won"]] if "won" in df.columns else pd.DataFrame()

    # second_won_pct: second-serve points won / second-serve points played
    w2w = won_df["w_2ndWon"].fillna(0).sum()
    l2l = lost_df["l_2ndWon"].fillna(0).sum()
    w_svpt = won_df["w_svpt"].fillna(0).sum()
    w_1stIn = won_df["w_1stIn"].fillna(0).sum()
    l_svpt_p = lost_df["l_svpt"].fillna(0).sum()
    l_1stIn_p = lost_df["l_1stIn"].fillna(0).sum()
    second_denom = (w_svpt - w_1stIn) + (l_svpt_p - l_1stIn_p)
    second_won_pct: float | None = float((w2w + l2l) / second_denom) if second_denom > eps else None

    # bp_save_pct: break points saved / break points faced (on own serve)
    bp_saved = won_df["w_bpSaved"].fillna(0).sum() + lost_df["l_bpSaved"].fillna(0).sum()
    bp_faced = won_df["w_bpFaced"].fillna(0).sum() + lost_df["l_bpFaced"].fillna(0).sum()
    bp_save_pct: float | None = float(bp_saved / bp_faced) if bp_faced > eps else None

    # return_pts_pct: return points won / opponent serve points
    # When player won: opponent is loser → use l_* columns
    # When player lost: opponent is winner → use w_* columns
    opp_svpt = won_df["l_svpt"].fillna(0).sum() + lost_df["w_svpt"].fillna(0).sum()
    opp_pts_won = (
        won_df["l_svpt"].fillna(0).sum()
        - won_df["l_1stWon"].fillna(0).sum()
        - won_df["l_2ndWon"].fillna(0).sum()
        + lost_df["w_svpt"].fillna(0).sum()
        - lost_df["w_1stWon"].fillna(0).sum()
        - lost_df["w_2ndWon"].fillna(0).sum()
    )
    return_pts_pct: float | None = float(opp_pts_won / opp_svpt) if opp_svpt > eps else None

    return {
        "second_won_pct": second_won_pct,
        "bp_save_pct": bp_save_pct,
        "return_pts_pct": return_pts_pct,
    }


def h2h_score(
    history: pd.DataFrame,
    player_a_id: int,
    player_b_id: int,
    as_of_date: pd.Timestamp,
    prior: int = 5,
    prior_mean: float = 0.5,
    *,
    _h2h_index: dict | None = None,
) -> tuple[float, int]:
    """Shrinkage H2H win rate for A vs B. Returns (shrunk_win_rate, sample_size).

    Pass _h2h_index (from _build_h2h_index) for O(log k) lookup instead of full scan.
    """
    if _h2h_index is not None:
        key = (min(player_a_id, player_b_id), max(player_a_id, player_b_id))
        entry = _h2h_index.get(key)
        if entry is None:
            return float(prior_mean), 0
        dates, wins_min_cumsum = entry
        idx = int(np.searchsorted(dates, np.datetime64(as_of_date), side="left"))
        if idx == 0:
            return float(prior_mean), 0
        n = idx
        wins_min = int(wins_min_cumsum[idx - 1])
        wins_a = wins_min if player_a_id == key[0] else (n - wins_min)
        shrunk = (wins_a + prior * prior_mean) / (n + prior)
        return float(shrunk), n

    # fallback: full scan
    if history.empty:
        return float(prior_mean), 0
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
    *,
    _frame_a: pd.DataFrame | None = None,
    _frame_b: pd.DataFrame | None = None,
    _h2h_index: dict | None = None,
) -> dict[str, Any]:
    """Compute all pre-match features. Time-gated at tourney_date."""
    feats: dict[str, Any] = {}

    since_12m = tourney_date - pd.DateOffset(months=12)
    wr_a_50,    lhf_a = rolling_win_rate(history, player_a_id, tourney_date, WIN_RATE_RECENT, _frame=_frame_a)
    wr_b_50,    lhf_b = rolling_win_rate(history, player_b_id, tourney_date, WIN_RATE_RECENT, _frame=_frame_b)
    wr_a_surf,  _     = rolling_win_rate(history, player_a_id, tourney_date, WIN_RATE_SURFACE, surface=surface, _frame=_frame_a)
    wr_b_surf,  _     = rolling_win_rate(history, player_b_id, tourney_date, WIN_RATE_SURFACE, surface=surface, _frame=_frame_b)
    wr_a_12m,   _     = rolling_win_rate(history, player_a_id, tourney_date, WIN_RATE_ALL_TIME, since_date=since_12m, _frame=_frame_a)
    wr_b_12m,   _     = rolling_win_rate(history, player_b_id, tourney_date, WIN_RATE_ALL_TIME, since_date=since_12m, _frame=_frame_b)
    wr_a_top20, _     = rolling_win_rate(history, player_a_id, tourney_date, WIN_RATE_TOP20, max_opponent_rank=20, _frame=_frame_a)
    wr_b_top20, _     = rolling_win_rate(history, player_b_id, tourney_date, WIN_RATE_TOP20, max_opponent_rank=20, _frame=_frame_b)

    feats["win_rate_diff"]         = wr_a_50   - wr_b_50
    feats["win_rate_surface_diff"] = wr_a_surf  - wr_b_surf
    feats["win_rate_12m_diff"]     = wr_a_12m   - wr_b_12m
    feats["win_rate_top20_diff"]   = wr_a_top20 - wr_b_top20
    feats["low_history_flag"]      = int(lhf_a or lhf_b)

    fat_a = fatigue_features(history, player_a_id, tourney_date, surface, _frame=_frame_a)
    fat_b = fatigue_features(history, player_b_id, tourney_date, surface, _frame=_frame_b)
    feats["days_since_last_diff"]  = fat_a["days_since_last_match"] - fat_b["days_since_last_match"]
    feats["sets_last_14d_diff"]    = fat_a["sets_last_14d"]         - fat_b["sets_last_14d"]
    feats["matches_last_30d_diff"] = fat_a["matches_last_30d"]      - fat_b["matches_last_30d"]
    feats["surface_switch_a"]      = fat_a["surface_switch"]
    feats["surface_switch_b"]      = fat_b["surface_switch"]

    srv_a = serve_efficiency(history, player_a_id, tourney_date, _frame=_frame_a)
    srv_b = serve_efficiency(history, player_b_id, tourney_date, _frame=_frame_b)
    for stat in ["first_serve_in_pct", "first_serve_won_pct", "ace_rate", "df_rate"]:
        feats[f"{stat}_diff"] = (srv_a.get(stat) or 0.0) - (srv_b.get(stat) or 0.0)

    h2h, h2h_n = h2h_score(history, player_a_id, player_b_id, tourney_date, _h2h_index=_h2h_index)
    feats["h2h_score"]       = h2h
    feats["h2h_sample_size"] = h2h_n

    def _elo(pid: int, field: str) -> float:
        return float(elo_state.get("players", {}).get(str(pid), {}).get(field, INITIAL_ELO))

    surf_key = surface.lower() if isinstance(surface, str) else "hard"
    feats["elo_overall_diff"] = _elo(player_a_id, "elo_overall") - _elo(player_b_id, "elo_overall")
    feats["elo_surface_diff"] = _elo(player_a_id, f"elo_{surf_key}") - _elo(player_b_id, f"elo_{surf_key}")

    feats["welo_overall_diff"] = _elo(player_a_id, "welo_overall") - _elo(player_b_id, "welo_overall")

    def _welo_surf(pid: int) -> float:
        # Mirrors surface_elo() in app/src-tauri/src/elo.rs — keep in sync.
        # If n_surf < SURFACE_MIN_MATCHES, fall back to overall WElo (not enough surface data).
        # Blend 50/50 once SURFACE_MIN_MATCHES is reached.
        n_surf = int(_elo(pid, f"matches_played_{surf_key}") or 0)
        welo_surf = _elo(pid, f"welo_{surf_key}")
        welo_ovrl = _elo(pid, "welo_overall")
        if n_surf >= SURFACE_MIN_MATCHES:
            return 0.5 * welo_surf + 0.5 * welo_ovrl
        return welo_ovrl

    feats["welo_surface_diff"] = _welo_surf(player_a_id) - _welo_surf(player_b_id)

    new_srv_a = _rolling_serve_stats(history, player_a_id, tourney_date, _frame=_frame_a)
    new_srv_b = _rolling_serve_stats(history, player_b_id, tourney_date, _frame=_frame_b)
    _srv_medians = {
        "second_won_pct": POPULATION_SECOND_WON_PCT,
        "bp_save_pct": POPULATION_BP_SAVE_PCT,
        "return_pts_pct": POPULATION_RETURN_PTS_PCT,
    }
    for _stat, _median in _srv_medians.items():
        _a = new_srv_a[_stat] if new_srv_a[_stat] is not None else _median
        _b = new_srv_b[_stat] if new_srv_b[_stat] is not None else _median
        feats[f"{_stat}_diff"] = _a - _b

    feats["age_diff"]    = (player_a_age    or DEFAULT_PLAYER_AGE)        - (player_b_age    or DEFAULT_PLAYER_AGE)
    feats["height_diff"] = (player_a_height or DEFAULT_PLAYER_HEIGHT_CM)  - (player_b_height or DEFAULT_PLAYER_HEIGHT_CM)
    feats["lefty_vs_righty"] = int((player_a_hand == "L") != (player_b_hand == "L"))

    feats["surface"]       = surface
    feats["tourney_level"] = tourney_level
    feats["round"]         = round_
    feats["best_of_5"]     = int(best_of == 5)

    return feats


def _build_feature_row(
    row: pd.Series,
    a_prefix: str,
    b_prefix: str,
    label: int,
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    common: dict,
    h2h_index: dict,
) -> dict:
    """Build one feature dict for a match, treating `a_prefix` player as player_a."""
    feats = compute_match_features(
        player_a_id=int(row[f"{a_prefix}_id"]),
        player_b_id=int(row[f"{b_prefix}_id"]),
        player_a_rank=row.get(f"{a_prefix}_rank"),
        player_b_rank=row.get(f"{b_prefix}_rank"),
        player_a_age=row.get(f"{a_prefix}_age"),
        player_b_age=row.get(f"{b_prefix}_age"),
        player_a_height=row.get(f"{a_prefix}_ht"),
        player_b_height=row.get(f"{b_prefix}_ht"),
        player_a_hand=row.get(f"{a_prefix}_hand"),
        player_b_hand=row.get(f"{b_prefix}_hand"),
        _frame_a=frame_a,
        _frame_b=frame_b,
        _h2h_index=h2h_index,
        **common,
    )
    feats["label"] = label
    feats["tourney_date"] = common["tourney_date"]
    feats["year"] = common["tourney_date"].year
    return feats


_CHUNK_SIZE = 100_000  # flush Python dicts → DataFrame every N rows to bound memory


def build_all_features(
    history: pd.DataFrame,
    elo_state: dict,
    min_year: int = 0,
) -> pd.DataFrame:
    """Compute features for every complete match. Returns balanced DataFrame (label 0 and 1).

    min_year: skip matches before this year in the outer loop (player index still uses all history).
    """
    import logging
    log = logging.getLogger(__name__)

    player_index = _build_player_index(history)
    h2h_index = _build_h2h_index(history)
    _empty: pd.DataFrame = pd.DataFrame()
    rows: list[dict] = []
    chunks: list[pd.DataFrame] = []

    complete = history[history["is_complete"]]
    if min_year > 0:
        complete = complete[complete["tourney_date"].dt.year >= min_year]
    log.info("build_all_features: iterating %d complete matches (min_year=%d)...", len(complete), min_year)

    for _, row in complete.iterrows():
        date  = row["tourney_date"]
        pid_w = int(row["winner_id"])
        pid_l = int(row["loser_id"])

        fa = _slice_before(player_index.get(pid_w, _empty), date)
        fb = _slice_before(player_index.get(pid_l, _empty), date)

        common = dict(
            history=history, elo_state=elo_state,
            surface=row["surface"],
            tourney_level=row.get("tourney_level", "A"),
            round_=row.get("round", "R32"),
            best_of=row.get("best_of", 3),
            tourney_date=date,
        )

        fp = _build_feature_row(row, "winner", "loser", 1, fa, fb, common, h2h_index)
        fn = _build_feature_row(row, "loser", "winner", 0, fb, fa, common, h2h_index)
        # odds_a_winner: PSW when A=winner, PSL when A=loser (closing odds never used as feature)
        fp["odds_a_winner"] = row.get("PSW")
        fn["odds_a_winner"] = row.get("PSL")
        rows.extend([fp, fn])

        if len(rows) % 20000 == 0:
            log.info("build_all_features: %d rows processed...", len(rows))

        if len(rows) >= _CHUNK_SIZE:
            chunks.append(pd.DataFrame(rows))
            rows = []

    if rows:
        chunks.append(pd.DataFrame(rows))

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
