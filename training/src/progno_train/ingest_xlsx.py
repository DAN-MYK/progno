"""Ingest tennis-data.co.uk XLSX files into a tidy odds DataFrame."""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import pandas as pd

from progno_train.odds_join import normalize_name as _norm_name

log = logging.getLogger(__name__)

_ODDS_COLS = ["PSW", "PSL", "B365W", "B365L"]

# Synthetic player IDs for players not in the Sackmann players table.
# Sackmann IDs top out around 212970; 300000+ is safe.
_SYNTHETIC_ID_START = 300_000

_ATP_SERIES_TO_LEVEL: dict[str, str] = {
    "Grand Slam": "G",
    "Masters 1000": "M",
    "Masters Cup": "F",
    "ATP500": "A",
    "ATP250": "A",
}

_WTA_TIER_TO_LEVEL: dict[str, str] = {
    "Grand Slam": "G",
    "WTA1000": "M",
    "Tour Championships": "F",
    "WTA500": "A",
    "WTA250": "A",
}

# tennis-data.co.uk round names → Sackmann round codes.
# All early-round factors are identical in elo.py (0.85), so collapsing
# 1st/2nd/3rd Round to R32 is ELO-neutral and avoids ambiguity.
_ROUND_MAP: dict[str, str] = {
    "The Final": "F",
    "Semifinals": "SF",
    "Quarterfinals": "QF",
    "4th Round": "R16",
    "3rd Round": "R32",
    "2nd Round": "R32",
    "1st Round": "R32",
    "Round Robin": "RR",
}


def _parse_xlsx_date(s: object) -> pd.Timestamp:
    """Parse DD/MM/YYYY string, pd.Timestamp, or datetime; return NaT on failure."""
    if isinstance(s, (pd.Timestamp, datetime.datetime, datetime.date)):
        return pd.Timestamp(s)
    try:
        return pd.to_datetime(str(s), format="%d/%m/%Y", errors="raise")
    except Exception:
        return pd.NaT


def _monday_of_week(ts: pd.Timestamp) -> pd.Timestamp:
    """Return the Monday of the ISO week containing ts."""
    return ts - pd.Timedelta(days=ts.weekday())


def _build_score(row: pd.Series) -> str:
    """Reconstruct score string from W1/L1 … W5/L5 set-score columns."""
    sets = []
    for i in range(1, 6):
        w, l = row.get(f"W{i}"), row.get(f"L{i}")
        try:
            wi, li = int(w), int(l)  # type: ignore[arg-type]
            sets.append(f"{wi}-{li}")
        except (TypeError, ValueError):
            break
    return " ".join(sets)


def _build_player_lookup(players: pd.DataFrame | None) -> dict[str, int]:
    """Return normalized-name → player_id mapping from players table."""
    if players is None or players.empty:
        return {}
    lookup: dict[str, int] = {}
    for row in players.itertuples(index=False):
        key = _norm_name(str(row.name))
        if key:
            lookup[key] = int(row.player_id)
    return lookup


def ingest_tennis_data_xlsx_dir(
    xlsx_dir: Path,
    players: pd.DataFrame | None,
    tour: str,
) -> pd.DataFrame:
    """Convert tennis-data.co.uk XLSXs to Sackmann-compatible match rows.

    The caller is responsible for filtering to rows after the latest Sackmann
    date before appending. Returns an empty DataFrame if no files match.
    """
    prefix = "atp_" if tour == "atp" else "wta_"
    paths = sorted(xlsx_dir.glob(f"{prefix}*.xlsx"))
    if not paths:
        return pd.DataFrame()

    frames = []
    for p in paths:
        try:
            frames.append(pd.read_excel(p, engine="openpyxl"))
        except Exception as exc:
            log.warning("could not read %s: %s", p, exc)
    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)

    # --- dates ---
    raw["_date"] = raw["Date"].apply(_parse_xlsx_date)
    raw = raw.dropna(subset=["_date", "Winner", "Loser"]).copy()
    raw = raw[raw["Winner"].astype(str).str.strip() != ""]
    raw = raw[raw["Loser"].astype(str).str.strip() != ""]
    raw = raw.reset_index(drop=True)

    # --- level ---
    if tour == "atp":
        tier_col = raw.get("Series", pd.Series(["ATP250"] * len(raw)))
        level_map = _ATP_SERIES_TO_LEVEL
    else:
        tier_col = raw.get("Tier", pd.Series(["WTA250"] * len(raw)))
        level_map = _WTA_TIER_TO_LEVEL

    # --- player IDs ---
    name_to_id = _build_player_lookup(players)
    new_id_counter = _SYNTHETIC_ID_START
    new_players: dict[str, int] = {}

    def _resolve_id(name: str) -> int:
        nonlocal new_id_counter
        key = _norm_name(name)
        pid = name_to_id.get(key) or new_players.get(key)
        if pid is None:
            pid = new_id_counter
            new_players[key] = pid
            new_id_counter += 1
        return pid

    winner_ids = raw["Winner"].astype(str).map(_resolve_id)
    loser_ids = raw["Loser"].astype(str).map(_resolve_id)

    if new_players:
        log.info("assigned %d synthetic player IDs for XLSX-only players", len(new_players))

    # --- assemble output ---
    out = pd.DataFrame()
    out["tourney_date"] = raw["_date"]
    out["tourney_id"] = (
        raw["Tournament"].astype(str).str.lower().str.replace(r"\W+", "_", regex=True)
        + "_"
        + raw["_date"].dt.strftime("%Y%m%d")
    )
    out["tourney_name"] = raw["Tournament"].astype(str)
    out["surface"] = raw["Surface"].astype(str)
    out["tourney_level"] = tier_col.astype(str).map(level_map).fillna("A")
    out["draw_size"] = float("nan")
    out["match_num"] = raw.index  # stable sort key within the concat'd frame
    out["winner_id"] = winner_ids
    out["winner_name"] = raw["Winner"].astype(str)
    out["winner_seed"] = float("nan")
    out["winner_entry"] = float("nan")  # NaN matches Sackmann float dtype when entry col is all-NaN
    out["winner_hand"] = ""
    out["winner_ht"] = float("nan")
    out["winner_ioc"] = ""
    out["winner_age"] = float("nan")
    out["loser_id"] = loser_ids
    out["loser_name"] = raw["Loser"].astype(str)
    out["loser_seed"] = float("nan")
    out["loser_entry"] = float("nan")
    out["loser_hand"] = ""
    out["loser_ht"] = float("nan")
    out["loser_ioc"] = ""
    out["loser_age"] = float("nan")
    out["score"] = raw.apply(_build_score, axis=1)
    out["best_of"] = pd.to_numeric(raw.get("Best of", pd.Series([3] * len(raw))), errors="coerce").fillna(3).astype(int)
    out["round"] = raw["Round"].astype(str).map(_ROUND_MAP).fillna("R32")
    out["minutes"] = float("nan")
    out["winner_rank"] = pd.to_numeric(raw.get("WRank"), errors="coerce")
    out["winner_rank_points"] = pd.to_numeric(raw.get("WPts"), errors="coerce")
    out["loser_rank"] = pd.to_numeric(raw.get("LRank"), errors="coerce")
    out["loser_rank_points"] = pd.to_numeric(raw.get("LPts"), errors="coerce")

    # Serve stats not available in tennis-data.co.uk; feature engineering handles NaN.
    for col in ("w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
                "w_SvGms", "w_bpSaved", "w_bpFaced", "l_ace", "l_df", "l_svpt",
                "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced"):
        out[col] = float("nan")

    # Derive set counts for ELO margin-of-victory (welo)
    out["w_sets"] = pd.to_numeric(raw.get("Wsets"), errors="coerce")
    out["l_sets"] = pd.to_numeric(raw.get("Lsets"), errors="coerce")
    out["is_complete"] = True
    out["completed_sets"] = (out["w_sets"].fillna(0) + out["l_sets"].fillna(0)).astype(int)

    return out.reset_index(drop=True)


def ingest_tennis_data_xlsx(paths: list[Path]) -> pd.DataFrame:
    """Parse tennis-data.co.uk XLSX files into a tidy DataFrame.

    Returns a DataFrame with columns:
        date_week    -- Monday of the match week (pd.Timestamp)
        winner_norm  -- normalized winner name ('lastname initial')
        loser_norm   -- normalized loser name
        PSW          -- Pinnacle winner odds (float, NaN if absent)
        PSL          -- Pinnacle loser odds (float, NaN if absent)
        B365W        -- Bet365 winner odds (float, NaN if absent)
        B365L        -- Bet365 loser odds (float, NaN if absent)
    """
    frames = []
    for p in paths:
        df = pd.read_excel(p, engine="openpyxl")
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["date_week", "winner_norm", "loser_norm"] + _ODDS_COLS)

    raw = pd.concat(frames, ignore_index=True)

    raw["_date"] = raw["Date"].apply(_parse_xlsx_date)
    raw = raw.dropna(subset=["_date"])

    out = pd.DataFrame()
    out["date_week"] = raw["_date"].apply(_monday_of_week)
    out["winner_norm"] = raw["Winner"].apply(_norm_name)
    out["loser_norm"] = raw["Loser"].apply(_norm_name)

    for col in _ODDS_COLS:
        if col in raw.columns:
            out[col] = pd.to_numeric(raw[col], errors="coerce")
        else:
            out[col] = float("nan")

    return out.reset_index(drop=True)
