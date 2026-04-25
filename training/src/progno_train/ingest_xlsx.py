"""
Ingest tennis-data.co.uk XLSX match files into a Sackmann-compatible DataFrame.

tennis-data.co.uk column map (ATP):
  Date, Tournament, Series, Court, Surface, Round, Best of,
  Winner, Loser, WRank, LRank, WPts, LPts,
  W1..W5, L1..L5, Wsets, Lsets, Comment, [odds columns...]

WTA uses Tier instead of Series and has up to 3 sets.

Serve stats are unavailable → NaN.
Player IDs are resolved from players.parquet; unmatched names get
a deterministic negative synthetic ID to avoid Sackmann ID collision.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("progno_train")

# Column name variants across file versions
_SERIES_COLS = ("Series", "Tier")

_DATE_FMTS = ("%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%Y%m%d")

_LEVEL_MAP: list[tuple[str, str]] = [
    ("grand slam",            "G"),
    ("masters cup",           "F"),
    ("year-end championship", "F"),
    ("wta finals",            "F"),
    ("nitto atp finals",      "F"),
    ("masters 1000",          "M"),
    ("atp masters series",    "M"),
    ("premier mandatory",     "M"),
    ("premier 5",             "M"),
    ("atp500",                "A"),
    ("atp 500",               "A"),
    ("international gold",    "A"),
    ("premier",               "A"),
    ("atp250",                "A"),
    ("atp 250",               "A"),
    ("international",         "A"),
]

_ROUND_MAP: dict[str, str] = {
    "the final":      "F",
    "final":          "F",
    "semifinals":     "SF",
    "semifinal":      "SF",
    "semi-finals":    "SF",
    "semi-final":     "SF",
    "quarterfinals":  "QF",
    "quarterfinal":   "QF",
    "quarter-finals": "QF",
    "quarter-final":  "QF",
    "round of 16":    "R16",
    "round of 32":    "R32",
    "round of 64":    "R64",
    "round of 128":   "R128",
    "round robin":    "RR",
}

# When only "1st Round" / "2nd Round" style is available, use best_of to distinguish GS
_ROUND_ORDINAL_GS  = {"1st round": "R128", "2nd round": "R64",  "3rd round": "R32", "4th round": "R16"}
_ROUND_ORDINAL_STD = {"1st round": "R32",  "2nd round": "R16",  "3rd round": "QF"}


def _map_level(series_str: str) -> str:
    s = str(series_str).lower().strip()
    for key, code in _LEVEL_MAP:
        if key in s:
            return code
    return "A"


def _map_round(round_str: str, is_gs: bool = False) -> str:
    r = str(round_str).lower().strip()
    if r in _ROUND_MAP:
        return _ROUND_MAP[r]
    ordinal = _ROUND_ORDINAL_GS if is_gs else _ROUND_ORDINAL_STD
    if r in ordinal:
        return ordinal[r]
    return r.upper()[:4] or "R32"


def _parse_date(val) -> pd.Timestamp | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, pd.Timestamp):
        return val
    s = str(val).strip()
    for fmt in _DATE_FMTS:
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            pass
    try:
        return pd.to_datetime(s)
    except Exception:
        return None


def _synthetic_player_id(name: str) -> int:
    """Deterministic negative ID from name hash — never collides with Sackmann IDs (positive)."""
    h = int(hashlib.sha256(name.encode()).hexdigest(), 16)
    return -(h % 10_000_000 + 1)


def build_name_lookup(players: pd.DataFrame) -> dict[str, int]:
    """name (lowercase) → player_id, also indexes by last-name if unique."""
    lookup: dict[str, int] = {}
    last_name_seen: dict[str, int] = {}

    for row in players.itertuples(index=False):
        name = str(row.name).strip()
        pid = int(row.player_id)
        lookup[name.lower()] = pid

        parts = name.split()
        if parts:
            last = parts[-1].lower()
            if last in last_name_seen and last_name_seen[last] != pid:
                # ambiguous last name — remove to prevent wrong matches
                lookup.pop(last, None)
            else:
                last_name_seen[last] = pid
                lookup[last] = pid

    return lookup


def _resolve_id(name: str, lookup: dict[str, int]) -> int:
    key = name.lower()
    if key in lookup:
        return lookup[key]
    parts = name.split()
    if parts:
        # "First Last" format → try last word
        last = parts[-1].lower().rstrip(".")
        if last in lookup:
            return lookup[last]
        # "Last F." format → try first word
        first = parts[0].lower().rstrip(".")
        if first in lookup:
            return lookup[first]
    return _synthetic_player_id(name)


def _reconstruct_score(row: dict, max_sets: int) -> tuple[str, bool, int]:
    """Return (score_string, is_complete, completed_sets)."""
    parts: list[str] = []
    for i in range(1, max_sets + 1):
        w = row.get(f"W{i}")
        l = row.get(f"L{i}")
        if w is None or l is None:
            break
        try:
            if pd.isna(w) or pd.isna(l):
                break
            parts.append(f"{int(w)}-{int(l)}")
        except (ValueError, TypeError):
            break

    comment = str(row.get("Comment", "") or "").strip()
    retired = any(kw in comment for kw in ("Retired", "Ret.", "W/O", "Walkover", "Def."))
    is_complete = not retired
    completed_sets = len(parts)

    score = " ".join(parts)
    if retired and parts:
        score += " RET"

    return score, is_complete, completed_sets


def ingest_tennis_data_xlsx(
    path: Path,
    players: pd.DataFrame | None = None,
    tour: str = "atp",
) -> pd.DataFrame:
    """
    Read one tennis-data.co.uk XLSX and return a Sackmann-compatible DataFrame.
    """
    try:
        xl = pd.read_excel(path, engine="openpyxl")
    except Exception as e:
        log.warning("failed to read %s: %s", path, e)
        return pd.DataFrame()

    # Drop obviously empty rows
    xl = xl.dropna(subset=["Winner", "Loser"], how="any")
    xl = xl[xl["Winner"].astype(str).str.strip() != ""]
    xl = xl[xl["Loser"].astype(str).str.strip() != ""]

    series_col = next((c for c in _SERIES_COLS if c in xl.columns), None)

    lookup: dict[str, int] = {}
    if players is not None and not players.empty:
        lookup = build_name_lookup(players)

    # Player metadata index for enriching hand/height/country
    player_meta: dict[int, dict] = {}
    if players is not None and not players.empty:
        for prow in players.itertuples(index=False):
            player_meta[int(prow.player_id)] = {
                "hand": str(prow.hand) if hasattr(prow, "hand") else "",
                "height_cm": float(prow.height_cm) if hasattr(prow, "height_cm") and not pd.isna(prow.height_cm) else float("nan"),
                "country": str(prow.country) if hasattr(prow, "country") else "",
            }

    rows: list[dict] = []
    match_counter: dict[str, int] = {}

    for _, xl_row in xl.iterrows():
        row = xl_row.to_dict()

        date = _parse_date(row.get("Date"))
        if date is None:
            continue

        tournament = str(row.get("Tournament", "")).strip()
        if not tournament:
            continue

        surface_raw = str(row.get("Surface", "Hard")).strip()
        surface = surface_raw.capitalize() if surface_raw.capitalize() in ("Hard", "Clay", "Grass", "Carpet") else "Hard"

        series_raw = str(row.get(series_col, "")) if series_col else ""
        tourney_level = _map_level(series_raw)
        is_gs = tourney_level == "G"

        round_raw = str(row.get("Round", "R32")).strip()
        round_norm = _map_round(round_raw, is_gs=is_gs)

        try:
            best_of = int(row.get("Best of", 3) or 3)
        except (ValueError, TypeError):
            best_of = 5 if is_gs else 3

        winner_name = str(row.get("Winner", "")).strip()
        loser_name  = str(row.get("Loser", "")).strip()
        if not winner_name or not loser_name:
            continue

        winner_id = _resolve_id(winner_name, lookup)
        loser_id  = _resolve_id(loser_name, lookup)

        def _rank(key: str):
            v = row.get(key)
            try:
                return float(v) if v is not None and not pd.isna(v) else float("nan")
            except (TypeError, ValueError):
                return float("nan")

        winner_rank = _rank("WRank")
        loser_rank  = _rank("LRank")
        winner_pts  = _rank("WPts")
        loser_pts   = _rank("LPts")

        score, is_complete, completed_sets = _reconstruct_score(row, max_sets=best_of)

        year = date.year
        slug = tournament.lower().replace(" ", "_")[:20]
        tourney_id = f"{year}-td-{slug}"

        match_counter[tourney_id] = match_counter.get(tourney_id, 0) + 1
        match_num = match_counter[tourney_id]

        w_meta = player_meta.get(winner_id, {})
        l_meta = player_meta.get(loser_id, {})

        nan = float("nan")
        rows.append({
            "tourney_id":           tourney_id,
            "tourney_name":         tournament,
            "surface":              surface,
            "draw_size":            nan,
            "tourney_level":        tourney_level,
            "tourney_date":         date,
            "match_num":            match_num,
            "winner_id":            winner_id,
            "winner_seed":          nan,
            "winner_entry":         "",
            "winner_name":          winner_name,
            "winner_hand":          w_meta.get("hand", ""),
            "winner_ht":            w_meta.get("height_cm", nan),
            "winner_ioc":           w_meta.get("country", ""),
            "winner_age":           nan,
            "loser_id":             loser_id,
            "loser_seed":           nan,
            "loser_entry":          "",
            "loser_name":           loser_name,
            "loser_hand":           l_meta.get("hand", ""),
            "loser_ht":             l_meta.get("height_cm", nan),
            "loser_ioc":            l_meta.get("country", ""),
            "loser_age":            nan,
            "score":                score,
            "best_of":              best_of,
            "round":                round_norm,
            "minutes":              nan,
            "w_ace":  nan, "w_df":  nan, "w_svpt": nan,
            "w_1stIn":nan, "w_1stWon":nan,"w_2ndWon":nan,
            "w_SvGms":nan, "w_bpSaved":nan,"w_bpFaced":nan,
            "l_ace":  nan, "l_df":  nan, "l_svpt": nan,
            "l_1stIn":nan, "l_1stWon":nan,"l_2ndWon":nan,
            "l_SvGms":nan, "l_bpSaved":nan,"l_bpFaced":nan,
            "winner_rank":          winner_rank,
            "winner_rank_points":   winner_pts,
            "loser_rank":           loser_rank,
            "loser_rank_points":    loser_pts,
            "is_complete":          is_complete,
            "completed_sets":       completed_sets,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    log.info("ingest_xlsx: %d matches from %s", len(df), path.name)
    return df


def ingest_tennis_data_xlsx_dir(
    directory: Path,
    players: pd.DataFrame | None = None,
    tour: str = "atp",
) -> pd.DataFrame:
    """Concatenate all XLSX files in directory matching the tour prefix."""
    prefix = "atp_" if tour == "atp" else "wta_"
    paths = sorted(directory.glob(f"{prefix}*.xlsx"))
    if not paths:
        paths = sorted(directory.glob("*.xlsx"))
    if not paths:
        return pd.DataFrame()

    frames = [ingest_tennis_data_xlsx(p, players=players, tour=tour) for p in paths]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    return combined
