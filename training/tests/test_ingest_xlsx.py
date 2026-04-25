"""Tests for tennis-data.co.uk XLSX ingestion."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from progno_train.ingest_xlsx import (
    _map_level,
    _map_round,
    _parse_date,
    _reconstruct_score,
    _resolve_id,
    _synthetic_player_id,
    build_name_lookup,
    ingest_tennis_data_xlsx,
)


# ── unit tests ────────────────────────────────────────────────────────────────

def test_map_level_grand_slam():
    assert _map_level("Grand Slam") == "G"

def test_map_level_masters():
    assert _map_level("Masters 1000") == "M"
    assert _map_level("Premier Mandatory") == "M"
    assert _map_level("Premier 5") == "M"

def test_map_level_regular():
    assert _map_level("ATP250") == "A"
    assert _map_level("International") == "A"
    assert _map_level("Premier") == "A"

def test_map_level_unknown():
    assert _map_level("Something Unknown") == "A"


def test_map_round_named():
    assert _map_round("The Final") == "F"
    assert _map_round("Semifinals") == "SF"
    assert _map_round("Quarterfinals") == "QF"
    assert _map_round("Round of 16") == "R16"
    assert _map_round("Round of 64") == "R64"

def test_map_round_ordinal_gs():
    assert _map_round("1st Round", is_gs=True) == "R128"
    assert _map_round("4th Round", is_gs=True) == "R16"

def test_map_round_ordinal_std():
    assert _map_round("1st Round", is_gs=False) == "R32"
    assert _map_round("2nd Round", is_gs=False) == "R16"


def test_parse_date_slashed():
    ts = _parse_date("14/01/25")
    assert ts is not None
    assert ts.year == 2025
    assert ts.month == 1
    assert ts.day == 14

def test_parse_date_iso():
    ts = _parse_date("2025-06-15")
    assert ts is not None
    assert ts.year == 2025

def test_parse_date_none():
    assert _parse_date(None) is None
    assert _parse_date(float("nan")) is None


def test_reconstruct_score_complete():
    row = {"W1": 6, "L1": 3, "W2": 7, "L2": 5, "Comment": "Completed"}
    score, is_complete, n_sets = _reconstruct_score(row, max_sets=3)
    assert score == "6-3 7-5"
    assert is_complete is True
    assert n_sets == 2

def test_reconstruct_score_retired():
    row = {"W1": 6, "L1": 2, "W2": 3, "L2": 1, "Comment": "Retired"}
    score, is_complete, n_sets = _reconstruct_score(row, max_sets=3)
    assert "RET" in score
    assert is_complete is False
    assert n_sets == 2

def test_reconstruct_score_walkover():
    row = {"W1": float("nan"), "L1": float("nan"), "Comment": "W/O"}
    score, is_complete, n_sets = _reconstruct_score(row, max_sets=3)
    assert is_complete is False
    assert n_sets == 0


def test_synthetic_player_id_negative():
    pid = _synthetic_player_id("John Doe")
    assert pid < 0

def test_synthetic_player_id_deterministic():
    assert _synthetic_player_id("Roger Federer") == _synthetic_player_id("Roger Federer")

def test_synthetic_player_id_unique():
    assert _synthetic_player_id("Roger Federer") != _synthetic_player_id("Rafael Nadal")


def test_build_name_lookup():
    players = pd.DataFrame([
        {"player_id": 101, "name": "Roger Federer", "hand": "R", "height_cm": 185.0, "country": "SUI"},
        {"player_id": 102, "name": "Rafael Nadal",  "hand": "L", "height_cm": 185.0, "country": "ESP"},
    ])
    lookup = build_name_lookup(players)
    assert lookup["roger federer"] == 101
    assert lookup["nadal"] == 102

def test_resolve_id_exact_match():
    players = pd.DataFrame([
        {"player_id": 101, "name": "Roger Federer", "hand": "R", "height_cm": 185.0, "country": "SUI"},
    ])
    lookup = build_name_lookup(players)
    assert _resolve_id("Roger Federer", lookup) == 101

def test_resolve_id_last_name():
    players = pd.DataFrame([
        {"player_id": 102, "name": "Rafael Nadal", "hand": "L", "height_cm": 185.0, "country": "ESP"},
    ])
    lookup = build_name_lookup(players)
    assert _resolve_id("Nadal R.", lookup) == 102

def test_resolve_id_unmatched_synthetic():
    pid = _resolve_id("Unknown Player XYZ", {})
    assert pid < 0


# ── integration test with a synthetic XLSX ───────────────────────────────────

def _make_xlsx(tmp_path: Path) -> Path:
    """Create a minimal tennis-data.co.uk style XLSX for testing."""
    data = {
        "Date": ["14/01/25", "15/01/25", "15/01/25"],
        "Tournament": ["Australian Open"] * 3,
        "Series": ["Grand Slam"] * 3,
        "Court": ["Outdoor"] * 3,
        "Surface": ["Hard"] * 3,
        "Round": ["Round of 128", "Round of 64", "Semifinals"],
        "Best of": [5, 5, 5],
        "Winner": ["Roger Federer", "Rafael Nadal", "Novak Djokovic"],
        "Loser":  ["Unknown Player", "Some Opponent", "Rafael Nadal"],
        "WRank": [1, 2, 3],
        "LRank": [50, 30, 2],
        "WPts":  [10000, 9000, 8500],
        "LPts":  [500, 700, 9000],
        "W1": [6, 6, 7], "L1": [3, 4, 6],
        "W2": [6, 7, 6], "L2": [2, 5, 4],
        "W3": [float("nan"), float("nan"), 6], "L3": [float("nan"), float("nan"), 2],
        "Wsets": [2, 2, 3], "Lsets": [0, 0, 0],
        "Comment": ["Completed", "Completed", "Completed"],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "atp_2025.xlsx"
    df.to_excel(path, index=False, engine="openpyxl")
    return path


def test_ingest_xlsx_basic(tmp_path):
    path = _make_xlsx(tmp_path)
    result = ingest_tennis_data_xlsx(path, players=None, tour="atp")

    assert len(result) == 3
    assert set(result["surface"]) == {"Hard"}
    assert set(result["tourney_level"]) == {"G"}
    assert "F" in result["round"].values or "SF" in result["round"].values
    assert result["winner_rank"].iloc[0] == 1.0
    assert result["is_complete"].all()
    assert result["completed_sets"].iloc[2] == 3  # Djokovic match went 3 sets

def test_ingest_xlsx_retired(tmp_path):
    data = {
        "Date": ["20/01/25"],
        "Tournament": ["Some Open"],
        "Series": ["ATP250"],
        "Surface": ["Clay"],
        "Round": ["The Final"],
        "Best of": [3],
        "Winner": ["Player A"], "Loser": ["Player B"],
        "WRank": [10], "LRank": [20],
        "WPts": [500], "LPts": [300],
        "W1": [6], "L1": [3],
        "W2": [3], "L2": [2],
        "Comment": ["Retired"],
    }
    path = tmp_path / "atp_retired.xlsx"
    pd.DataFrame(data).to_excel(path, index=False, engine="openpyxl")
    result = ingest_tennis_data_xlsx(path)
    assert len(result) == 1
    assert not result["is_complete"].iloc[0]
    assert "RET" in result["score"].iloc[0]

def test_ingest_xlsx_with_player_lookup(tmp_path):
    players = pd.DataFrame([
        {"player_id": 999, "name": "Roger Federer", "hand": "R", "height_cm": 185.0, "country": "SUI"},
    ])
    path = _make_xlsx(tmp_path)
    result = ingest_tennis_data_xlsx(path, players=players, tour="atp")
    federer_row = result[result["winner_name"] == "Roger Federer"]
    assert not federer_row.empty
    assert federer_row["winner_id"].iloc[0] == 999
    assert federer_row["winner_hand"].iloc[0] == "R"

def test_ingest_xlsx_bad_file(tmp_path):
    bad = tmp_path / "bad.xlsx"
    bad.write_bytes(b"not an xlsx")
    result = ingest_tennis_data_xlsx(bad)
    assert result.empty
