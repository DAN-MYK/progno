"""Join tennis-data.co.uk odds to Sackmann match records."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz
from unidecode import unidecode

log = logging.getLogger(__name__)

_ODDS_COLS = ["PSW", "PSL", "B365W", "B365L"]
_TOLERANCE_DAYS = [0, 7, 14]   # weeks to search: exact → +1 week → +2 weeks
_FUZZY_THRESHOLD = 90           # rapidfuzz score threshold (0–100)
_FUZZY_WINDOW_DAYS = 21         # max date diff for fuzzy name search


def normalize_name(name: object) -> str:
    """Normalize player name to 'lastname initial' lowercase ASCII.

    'Carlos Alcaraz' → 'alcaraz c'
    'Kévin Krawietz' → 'krawietz k'
    'Sinner' → 'sinner'
    """
    s = unidecode(str(name)).strip()
    parts = s.split()
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0].lower()
    last = parts[-1].lower()
    initial = parts[0][0].lower()
    return f"{last} {initial}"


def _monday(ts: pd.Timestamp) -> pd.Timestamp:
    return ts - pd.Timedelta(days=ts.weekday())


def _load_name_map(path: Path | None) -> dict[str, str]:
    """Load sackmann_name → odds_name overrides from CSV.

    Keys are normalize_name(sackmann_name); values are odds_name literals.
    Applied to remap odds normalized names before building the lookup index.
    When an odds winner_norm or loser_norm equals a map key, it is replaced
    with the corresponding odds_name value for the purpose of pair matching.
    """
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    # Key: normalize_name(sackmann_name) → Value: odds_name literal (already normalized)
    return dict(zip(df["sackmann_name"].apply(normalize_name),
                    df["odds_name"].str.strip()))


def _players_match(a: str, b: str) -> bool:
    """Return True if two normalized player names are close enough to be the same player."""
    if a == b:
        return True
    return fuzz.partial_ratio(a, b) >= _FUZZY_THRESHOLD


def join_odds(
    sackmann: pd.DataFrame,
    odds_df: pd.DataFrame,
    name_map_path: Path | None,
) -> pd.DataFrame:
    """Join Pinnacle/Bet365 odds to Sackmann matches.

    Adds PSW, PSL, B365W, B365L columns to sackmann (NaN where unmatched).
    Logs join yield.

    Join strategy (spec §2.6):
      1. Exact: (tourney_week, sorted pair) — covers 1-week tournaments
      2. ±7 days, ±14 days — covers Grand Slam weeks
      3. rapidfuzz per-player partial_ratio on names within ±21 days — catches
         spelling discrepancies and compound-name orderings
      4. Manual name_map.csv override remaps odds normalized names before (1)–(3)
    """
    name_map = _load_name_map(name_map_path)

    result = sackmann.copy()
    for col in _ODDS_COLS:
        result[col] = float("nan")

    if odds_df.empty:
        return result

    # Apply name_map to remap odds names: any odds norm matching a map key is
    # replaced with the corresponding canonical value, enabling pair-level lookup.
    def _remap(norm: str) -> str:
        return name_map.get(norm, norm)

    odds_remapped = odds_df.copy()
    odds_remapped["winner_norm"] = odds_df["winner_norm"].apply(_remap)
    odds_remapped["loser_norm"] = odds_df["loser_norm"].apply(_remap)

    # Pre-compute index: {(date_week, sorted pair) → original odds_row_index}
    odds_index: dict[tuple, int] = {}
    for i, row in odds_remapped.iterrows():
        pair = tuple(sorted([row["winner_norm"], row["loser_norm"]]))
        odds_index[(row["date_week"], pair)] = int(i)

    # Only iterate over matches that can possibly appear in tennis-data.co.uk XLSX:
    #   • within the XLSX date range (2013+)
    #   • main-tour levels only (G/F/M/A/D) — challengers and quals are never in XLSX
    # This cuts 941k rows → ~40k, keeping the loop fast.
    _XLSX_LEVELS = {"G", "F", "M", "A", "D"}
    if not odds_df.empty:
        odds_min_date = odds_df["date_week"].min() - pd.Timedelta(days=_FUZZY_WINDOW_DAYS)
        odds_max_date = odds_df["date_week"].max() + pd.Timedelta(days=_FUZZY_WINDOW_DAYS)
        candidate_mask = (
            (result["tourney_date"] >= odds_min_date)
            & (result["tourney_date"] <= odds_max_date)
            & result["winner_name"].notna()
            & result["loser_name"].notna()
        )
        if "tourney_level" in result.columns:
            candidate_mask &= result["tourney_level"].isin(_XLSX_LEVELS)
    else:
        candidate_mask = pd.Series(False, index=result.index)

    matched = 0
    for sack_i in result.index[candidate_mask]:
        sack_row = result.loc[sack_i]
        tourney_week = _monday(sack_row["tourney_date"])
        w_norm = normalize_name(sack_row["winner_name"])
        l_norm = normalize_name(sack_row["loser_name"])
        pair = tuple(sorted([w_norm, l_norm]))

        # Step 1 + 2: exact and tolerance match (using remapped odds index)
        odds_i = None
        for delta in _TOLERANCE_DAYS:
            week = tourney_week + pd.Timedelta(days=delta)
            odds_i = odds_index.get((week, pair))
            if odds_i is not None:
                break

        # Step 3: fuzzy per-player match within ±21 days if still unmatched
        if odds_i is None:
            cutoff_lo = tourney_week - pd.Timedelta(days=_FUZZY_WINDOW_DAYS)
            cutoff_hi = tourney_week + pd.Timedelta(days=_FUZZY_WINDOW_DAYS)
            candidates = odds_remapped[
                (odds_remapped["date_week"] >= cutoff_lo)
                & (odds_remapped["date_week"] <= cutoff_hi)
            ]
            best_score = 0.0
            for _, cand in candidates.iterrows():
                # Match player A to either odds player; player B to the other
                s1 = (fuzz.partial_ratio(w_norm, cand["winner_norm"])
                      + fuzz.partial_ratio(l_norm, cand["loser_norm"])) / 2
                s2 = (fuzz.partial_ratio(w_norm, cand["loser_norm"])
                      + fuzz.partial_ratio(l_norm, cand["winner_norm"])) / 2
                score = max(s1, s2)
                if score > best_score and score >= _FUZZY_THRESHOLD:
                    best_score = score
                    odds_i = cand.name

        if odds_i is None:
            continue

        odds_row = odds_df.loc[odds_i]
        odds_row_remapped = odds_remapped.loc[odds_i]

        # Determine orientation using remapped names
        xlsx_winner_is_sack_winner = _players_match(
            odds_row_remapped["winner_norm"], w_norm
        )

        if xlsx_winner_is_sack_winner:
            result.at[sack_i, "PSW"] = odds_row["PSW"]
            result.at[sack_i, "PSL"] = odds_row["PSL"]
            result.at[sack_i, "B365W"] = odds_row["B365W"]
            result.at[sack_i, "B365L"] = odds_row["B365L"]
        else:
            # XLSX has B as winner; swap W↔L
            result.at[sack_i, "PSW"] = odds_row["PSL"]
            result.at[sack_i, "PSL"] = odds_row["PSW"]
            result.at[sack_i, "B365W"] = odds_row["B365L"]
            result.at[sack_i, "B365L"] = odds_row["B365W"]
        matched += 1

    n = len(sackmann)
    yield_pct = 100.0 * matched / n if n > 0 else 0.0
    log.info("odds join yield: %d / %d (%.1f%%)", matched, n, yield_pct)
    if yield_pct < 95.0 and n > 100:
        log.warning("odds join yield below 95%% — ROI backtest may be limited (spec §2.9)")

    return result
