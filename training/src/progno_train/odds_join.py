"""Join tennis-data.co.uk odds to Sackmann match records."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from unidecode import unidecode

log = logging.getLogger(__name__)

_ODDS_COLS = ["PSW", "PSL", "B365W", "B365L"]
_TOLERANCE_DAYS = [0, 7, 14]   # weeks to search: exact → +1 week → +2 weeks
_FUZZY_THRESHOLD = 90           # rapidfuzz score threshold (0–100)
_FUZZY_WINDOW_DAYS = 21         # max date diff for fuzzy name search
# tennis-data.co.uk only covers main-tour levels
_XLSX_LEVELS = {"G", "F", "M", "A", "D"}


def normalize_name(name: object) -> str:
    """Normalize player name to 'lastname initial' lowercase ASCII.

    Handles both Sackmann 'First Last' and tennis-data.co.uk 'Last F.' formats:
      'Carlos Alcaraz'  → 'alcaraz c'
      'Alcaraz C.'      → 'alcaraz c'
      'Kévin Krawietz'  → 'krawietz k'
      'Krawietz K.'     → 'krawietz k'
      'Sinner'          → 'sinner'
    """
    s = unidecode(str(name)).strip()
    parts = s.split()
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0].lower()
    # Detect XLSX 'Surname Initial.' format: last token is 1-2 chars after stripping punctuation
    last_token_clean = parts[-1].replace(".", "").replace("-", "")
    if 1 <= len(last_token_clean) <= 2 and len(parts) >= 2:
        initial = last_token_clean[0].lower()
        last = parts[-2].lower()
        return f"{last} {initial}"
    last = parts[-1].lower()
    initial = parts[0][0].lower()
    return f"{last} {initial}"


def _monday(ts: pd.Timestamp) -> pd.Timestamp:
    return ts - pd.Timedelta(days=ts.weekday())


def _load_name_map(path: Path | None) -> dict[str, str]:
    """Load sackmann_name → odds_name overrides from CSV."""
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["sackmann_name"].apply(normalize_name),
                    df["odds_name"].str.strip()))


def _make_pair_key(a: pd.Series, b: pd.Series) -> pd.Series:
    """Sorted pair key: 'min_norm||max_norm' per row (vectorized via numpy)."""
    a_arr = a.to_numpy(dtype=object)
    b_arr = b.to_numpy(dtype=object)
    mask = a_arr <= b_arr
    lo = np.where(mask, a_arr, b_arr)
    hi = np.where(mask, b_arr, a_arr)
    return pd.Series(
        [f"{l}||{h}" for l, h in zip(lo, hi)],
        index=a.index,
    )


def join_odds(
    sackmann: pd.DataFrame,
    odds_df: pd.DataFrame,
    name_map_path: Path | None,
) -> pd.DataFrame:
    """Join Pinnacle/Bet365 odds to Sackmann matches.

    Adds PSW, PSL, B365W, B365L columns to sackmann (NaN where unmatched).
    Logs join yield.

    Join strategy (spec §2.6):
      1. Vectorized merge on (tourney_week, sorted pair) — exact match
      2. Repeat with ±7 days, ±14 days for Grand Slam multi-week windows
      3. rapidfuzz fuzzy search on remaining unmatched rows (typically <5%)
      4. Manual name_map.csv override remaps odds names before steps 1–3
    """
    name_map = _load_name_map(name_map_path)

    result = sackmann.copy()
    for col in _ODDS_COLS:
        result[col] = float("nan")

    if odds_df.empty:
        return result

    # Apply name_map to odds names
    def _remap(norm: str) -> str:
        return name_map.get(norm, norm)

    odds_remapped = odds_df.copy()
    odds_remapped["winner_norm"] = odds_df["winner_norm"].apply(_remap)
    odds_remapped["loser_norm"] = odds_df["loser_norm"].apply(_remap)

    # --- Build odds lookup frame with pair key ---
    odds_remapped["_pair_key"] = _make_pair_key(
        odds_remapped["winner_norm"], odds_remapped["loser_norm"]
    )
    odds_remapped["_week_str"] = odds_remapped["date_week"].dt.strftime("%Y-%m-%d")
    odds_remapped["_join_key"] = odds_remapped["_week_str"] + "|" + odds_remapped["_pair_key"]
    # Dedup: keep first occurrence per key (handles rare duplicate rows)
    odds_lookup = odds_remapped.drop_duplicates("_join_key").set_index("_join_key")

    # --- Restrict sackmann rows to XLSX-coverable candidates ---
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

    cands = result[candidate_mask].copy()
    if cands.empty:
        return result

    # Normalize sackmann names
    cands["_w_norm"] = cands["winner_name"].apply(normalize_name)
    cands["_l_norm"] = cands["loser_name"].apply(normalize_name)
    cands["_pair_key"] = _make_pair_key(cands["_w_norm"], cands["_l_norm"])
    cands["_week"] = cands["tourney_date"].apply(_monday)

    # --- Steps 1 & 2: vectorized merge with date tolerances ---
    unmatched_idx = cands.index.tolist()

    for delta in _TOLERANCE_DAYS:
        if not unmatched_idx:
            break
        sub = cands.loc[unmatched_idx].copy()
        sub["_week_str"] = (sub["_week"] + pd.Timedelta(days=delta)).dt.strftime("%Y-%m-%d")
        sub["_join_key"] = sub["_week_str"] + "|" + sub["_pair_key"]

        # Map join key → odds row
        matched_mask = sub["_join_key"].isin(odds_lookup.index)
        matched_sub = sub[matched_mask]

        for sack_i, srow in matched_sub.iterrows():
            orow = odds_lookup.loc[srow["_join_key"]]
            xlsx_winner_is_sack_winner = (orow["winner_norm"] == srow["_w_norm"]) or (
                orow["loser_norm"] == srow["_l_norm"]
            )
            _assign_odds(result, sack_i, orow, xlsx_winner_is_sack_winner)

        newly_matched = set(matched_sub.index)
        unmatched_idx = [i for i in unmatched_idx if i not in newly_matched]

    # --- Step 3: fuzzy search for remaining unmatched (typically <5%) ---
    if unmatched_idx:
        # Pre-build per-week bucket: week_str → list of odds row positions
        week_buckets: dict[str, list[int]] = {}
        for pos, (idx, wk) in enumerate(odds_remapped["_week_str"].items()):
            week_buckets.setdefault(wk, []).append(pos)

        # Pre-extract name columns as numpy arrays for O(1) positional access
        winner_norms = odds_remapped["winner_norm"].to_numpy(dtype=object)
        loser_norms = odds_remapped["loser_norm"].to_numpy(dtype=object)
        # Map positional index → DataFrame index for later .iloc lookups
        odds_positions = list(range(len(odds_remapped)))

        sub = cands.loc[unmatched_idx]
        for sack_i, srow in sub.iterrows():
            w_norm = srow["_w_norm"]
            l_norm = srow["_l_norm"]
            tourney_week = srow["_week"]

            # Collect candidate positions within ±21 days
            candidate_pos: list[int] = []
            for delta_d in range(-_FUZZY_WINDOW_DAYS, _FUZZY_WINDOW_DAYS + 1, 7):
                wk_str = (tourney_week + pd.Timedelta(days=delta_d)).strftime("%Y-%m-%d")
                candidate_pos.extend(week_buckets.get(wk_str, []))

            best_score = 0.0
            best_pos: int | None = None
            for pos in candidate_pos:
                wn = winner_norms[pos]
                ln = loser_norms[pos]
                s1 = (fuzz.partial_ratio(w_norm, wn) + fuzz.partial_ratio(l_norm, ln)) / 2
                s2 = (fuzz.partial_ratio(w_norm, ln) + fuzz.partial_ratio(l_norm, wn)) / 2
                score = max(s1, s2)
                if score > best_score and score >= _FUZZY_THRESHOLD:
                    best_score = score
                    best_pos = pos

            if best_pos is not None:
                orow = odds_remapped.iloc[best_pos]
                xlsx_winner_is_sack_winner = (orow["winner_norm"] == w_norm) or (
                    orow["loser_norm"] == l_norm
                )
                _assign_odds(result, sack_i, orow, xlsx_winner_is_sack_winner)

    matched = int(candidate_mask.sum()) - len(unmatched_idx)
    n = len(sackmann)
    yield_pct = 100.0 * matched / n if n > 0 else 0.0
    log.info("odds join: %d / %d matched (%.1f%% of all, %.1f%% of candidates)",
             matched, n, yield_pct, 100.0 * matched / max(candidate_mask.sum(), 1))
    if matched / max(candidate_mask.sum(), 1) < 0.90:
        log.warning("odds join candidate hit rate below 90%% — check name_map.csv")

    return result


def _assign_odds(
    result: pd.DataFrame,
    sack_i: int,
    odds_row: pd.Series,
    xlsx_winner_is_sack_winner: bool,
) -> None:
    if xlsx_winner_is_sack_winner:
        result.at[sack_i, "PSW"]   = odds_row["PSW"]
        result.at[sack_i, "PSL"]   = odds_row["PSL"]
        result.at[sack_i, "B365W"] = odds_row["B365W"]
        result.at[sack_i, "B365L"] = odds_row["B365L"]
    else:
        result.at[sack_i, "PSW"]   = odds_row["PSL"]
        result.at[sack_i, "PSL"]   = odds_row["PSW"]
        result.at[sack_i, "B365W"] = odds_row["B365L"]
        result.at[sack_i, "B365L"] = odds_row["B365W"]
