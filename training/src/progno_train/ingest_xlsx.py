"""Ingest tennis-data.co.uk XLSX files into a tidy odds DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from unidecode import unidecode

_ODDS_COLS = ["PSW", "PSL", "B365W", "B365L"]


def _parse_xlsx_date(s: object) -> pd.Timestamp:
    """Parse DD/MM/YYYY string or return NaT on failure."""
    try:
        return pd.to_datetime(str(s), format="%d/%m/%Y", errors="raise")
    except Exception:
        return pd.NaT


def _monday_of_week(ts: pd.Timestamp) -> pd.Timestamp:
    """Return the Monday of the ISO week containing ts."""
    return ts - pd.Timedelta(days=ts.weekday())


def _norm_name(name: object) -> str:
    """Normalize player name to 'lastname initial' lowercase ASCII.

    'Carlos Alcaraz' -> 'alcaraz c'
    'Kevin Krawietz' -> 'krawietz k'
    """
    s = unidecode(str(name)).strip()
    parts = s.split()
    if len(parts) == 0:
        return ""
    last = parts[-1].lower()
    initial = parts[0][0].lower() if parts[0] else ""
    return f"{last} {initial}" if initial else last


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

    # Parse dates
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
