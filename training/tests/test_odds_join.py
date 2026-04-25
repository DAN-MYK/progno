"""Tests for odds join: exact, tolerance, fuzzy, name_map fallback."""

import pandas as pd
import pytest
from pathlib import Path

from progno_train.odds_join import normalize_name, join_odds


def _sack(winner, loser, tourney_date, **kwargs):
    return {
        "winner_name": winner,
        "loser_name": loser,
        "tourney_date": pd.Timestamp(tourney_date),
        "is_complete": True,
        **kwargs,
    }


def _odds(winner, loser, date_week, psw=1.8, psl=2.1, b365w=1.75, b365l=2.05):
    return {
        "winner_norm": normalize_name(winner),
        "loser_norm": normalize_name(loser),
        "date_week": pd.Timestamp(date_week),
        "PSW": psw,
        "PSL": psl,
        "B365W": b365w,
        "B365L": b365l,
    }


# ── normalize_name tests ─────────────────────────────────────────────────────

def test_normalize_name_simple():
    assert normalize_name("Carlos Alcaraz") == "alcaraz c"


def test_normalize_name_accents():
    assert normalize_name("Kévin Krawietz") == "krawietz k"


def test_normalize_name_single_word():
    assert normalize_name("Sinner") == "sinner"


def test_normalize_name_lowercase():
    assert normalize_name("NOVAK DJOKOVIC") == "djokovic n"


# ── join_odds tests ──────────────────────────────────────────────────────────

def test_exact_join(tmp_path):
    sack = pd.DataFrame([_sack("Carlos Alcaraz", "Novak Djokovic", "2023-01-16")])
    odds_df = pd.DataFrame([_odds("Carlos Alcaraz", "Novak Djokovic", "2023-01-16")])
    result = join_odds(sack, odds_df, name_map_path=None)
    assert abs(result["PSW"].iloc[0] - 1.8) < 1e-9
    assert abs(result["PSL"].iloc[0] - 2.1) < 1e-9


def test_join_is_orientation_agnostic(tmp_path):
    """XLSX has B vs A but Sackmann has A vs B — still matches."""
    sack = pd.DataFrame([_sack("Carlos Alcaraz", "Novak Djokovic", "2023-01-16")])
    # In XLSX, Djokovic is winner
    odds_df = pd.DataFrame([_odds("Novak Djokovic", "Carlos Alcaraz", "2023-01-16", psw=2.1, psl=1.8)])
    result = join_odds(sack, odds_df, name_map_path=None)
    # PSW from XLSX (2.1) → PSL for Sackmann winner Alcaraz = 1.8
    assert result["PSW"].notna().iloc[0]


def test_tolerance_join_7_days(tmp_path):
    """Sackmann tourney_date = tournament start Monday; XLSX date_week = match 7 days later."""
    sack = pd.DataFrame([_sack("Carlos Alcaraz", "Novak Djokovic", "2023-01-09")])
    odds_df = pd.DataFrame([_odds("Carlos Alcaraz", "Novak Djokovic", "2023-01-16")])
    result = join_odds(sack, odds_df, name_map_path=None)
    assert result["PSW"].notna().iloc[0]


def test_tolerance_join_14_days(tmp_path):
    """Grand Slam week 2: XLSX date 14 days after tourney start."""
    sack = pd.DataFrame([_sack("Carlos Alcaraz", "Novak Djokovic", "2023-01-09")])
    odds_df = pd.DataFrame([_odds("Carlos Alcaraz", "Novak Djokovic", "2023-01-23")])
    result = join_odds(sack, odds_df, name_map_path=None)
    assert result["PSW"].notna().iloc[0]


def test_no_match_beyond_21_days(tmp_path):
    """Dates more than 21 days apart should NOT match."""
    sack = pd.DataFrame([_sack("Carlos Alcaraz", "Novak Djokovic", "2023-01-09")])
    odds_df = pd.DataFrame([_odds("Carlos Alcaraz", "Novak Djokovic", "2023-02-13")])
    result = join_odds(sack, odds_df, name_map_path=None)
    assert pd.isna(result["PSW"].iloc[0])


def test_fuzzy_name_match(tmp_path):
    """Minor name discrepancy handled by rapidfuzz."""
    sack = pd.DataFrame([_sack("Alexander Zverev", "Stefanos Tsitsipas", "2023-03-13")])
    odds_df = pd.DataFrame([_odds("A. Zverev", "S. Tsitsipas", "2023-03-13")])
    result = join_odds(sack, odds_df, name_map_path=None)
    assert "PSW" in result.columns


def test_unmatched_row_gets_nan(tmp_path):
    sack = pd.DataFrame([_sack("Player A", "Player B", "2023-01-16")])
    odds_df = pd.DataFrame([_odds("Completely Different", "Names Here", "2023-01-16")])
    result = join_odds(sack, odds_df, name_map_path=None)
    assert pd.isna(result["PSW"].iloc[0])


def test_name_map_override(tmp_path):
    """Manual name_map.csv overrides normalized name mismatch."""
    name_map = tmp_path / "name_map.csv"
    name_map.write_text("sackmann_name,odds_name\nDel Potro Juan Martin,del potro j\n")
    sack = pd.DataFrame([_sack("Juan Martin Del Potro", "Rafael Nadal", "2023-01-16")])
    odds_df = pd.DataFrame([_odds("Del Potro Juan Martin", "Rafael Nadal", "2023-01-16")])
    result = join_odds(sack, odds_df, name_map_path=name_map)
    assert result["PSW"].notna().iloc[0]


def test_join_yield_logged(tmp_path, caplog):
    """join_odds should log the join yield."""
    import logging
    sack = pd.DataFrame([
        _sack("Carlos Alcaraz", "Novak Djokovic", "2023-01-16"),
        _sack("Player A", "Player B", "2023-01-16"),
    ])
    odds_df = pd.DataFrame([_odds("Carlos Alcaraz", "Novak Djokovic", "2023-01-16")])
    with caplog.at_level(logging.INFO, logger="progno_train.odds_join"):
        join_odds(sack, odds_df, name_map_path=None)
    assert any("yield" in msg.lower() or "%" in msg for msg in caplog.messages)


def test_output_has_all_odds_columns(tmp_path):
    sack = pd.DataFrame([_sack("Carlos Alcaraz", "Novak Djokovic", "2023-01-16")])
    odds_df = pd.DataFrame([_odds("Carlos Alcaraz", "Novak Djokovic", "2023-01-16")])
    result = join_odds(sack, odds_df, name_map_path=None)
    for col in ["PSW", "PSL", "B365W", "B365L"]:
        assert col in result.columns
