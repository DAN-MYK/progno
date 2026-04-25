from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from progno_train.cli import run_elo, run_ingest
from progno_train.config import Paths

FIXTURE = Path(__file__).parent / "fixtures" / "mini_atp_matches.csv"


@pytest.fixture()
def paths(tmp_path: Path) -> Paths:
    raw = tmp_path / "raw"
    atp_dir = raw / "tennis_atp"
    atp_dir.mkdir(parents=True)
    shutil.copy(FIXTURE, atp_dir / "atp_matches_2024.csv")
    return Paths(
        data_raw=raw,
        data_staging=tmp_path / "staging",
        artifacts=tmp_path / "artifacts",
    )


def test_run_ingest_writes_staging_parquet(paths: Paths) -> None:
    rc = run_ingest(paths, "atp")
    assert rc == 0
    out = paths.matches_clean
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) > 0
    assert "is_complete" in df.columns


def test_run_elo_writes_artifacts(paths: Paths) -> None:
    run_ingest(paths, "atp")
    rc = run_elo(paths)
    assert rc == 0
    assert (paths.artifacts / "elo_state.json").exists()
    assert (paths.artifacts / "players.parquet").exists()
    assert (paths.artifacts / "match_history.parquet").exists()

    state = json.loads((paths.artifacts / "elo_state.json").read_text())
    assert "players" in state
    assert "data_as_of" in state
    # Alpha Alpha (id 100001) won two matches in the fixture, should be > 1500
    assert state["players"]["alpha"]["elo_overall"] > 1500.0


def test_run_elo_joins_odds_when_xlsx_present(tmp_path):
    """run_elo should join odds from XLSX files if present in odds_xlsx_dir."""
    import openpyxl
    from progno_train.cli import run_elo
    from progno_train.config import Paths

    paths = Paths.for_tour(tmp_path, "atp")
    paths.data_staging.mkdir(parents=True, exist_ok=True)
    paths.data_raw.mkdir(parents=True, exist_ok=True)

    matches = pd.DataFrame([{
        "tourney_id": "2023-001",
        "tourney_date": pd.Timestamp("2023-01-16"),
        "match_num": 1,
        "surface": "Hard",
        "tourney_level": "A",
        "round": "R32",
        "best_of": 3,
        "winner_id": 1,
        "winner_name": "Carlos Alcaraz",
        "winner_hand": "R",
        "winner_ht": 185.0,
        "winner_age": 19.5,
        "winner_ioc": "ESP",
        "winner_rank": 1,
        "loser_id": 2,
        "loser_name": "Novak Djokovic",
        "loser_hand": "R",
        "loser_ht": 188.0,
        "loser_age": 35.0,
        "loser_ioc": "SRB",
        "loser_rank": 5,
        "is_complete": True,
        "completed_sets": 3,
        "score": "6-3 4-6 6-4",
        "minutes": 120.0,
        "w_ace": 5.0, "w_df": 2.0, "w_svpt": 60.0, "w_1stIn": 40.0,
        "w_1stWon": 28.0, "w_2ndWon": 10.0, "w_bpSaved": 3.0, "w_bpFaced": 5.0,
        "l_ace": 3.0, "l_df": 3.0, "l_svpt": 58.0, "l_1stIn": 38.0,
        "l_1stWon": 25.0, "l_2ndWon": 9.0, "l_bpSaved": 2.0, "l_bpFaced": 4.0,
        "winner_rank_points": 8000, "loser_rank_points": 7000,
    }])
    matches.to_parquet(paths.matches_clean, index=False)

    xlsx_dir = paths.odds_xlsx_dir
    xlsx_dir.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Date", "Winner", "Loser", "PSW", "PSL", "B365W", "B365L"])
    ws.append(["16/01/2023", "Carlos Alcaraz", "Novak Djokovic", 1.85, 2.10, 1.83, 2.05])
    wb.save(xlsx_dir / "atp_2023.xlsx")

    rc = run_elo(paths)
    assert rc == 0

    mh = pd.read_parquet(paths.match_history)
    assert "PSW" in mh.columns
    assert mh["PSW"].notna().any(), "Expected at least one PSW value after odds join"


def test_run_elo_skips_odds_gracefully_when_no_xlsx(tmp_path, caplog):
    """run_elo should log a warning and proceed without odds if no XLSX files."""
    import logging
    from progno_train.cli import run_elo
    from progno_train.config import Paths

    paths = Paths.for_tour(tmp_path, "atp")
    paths.data_staging.mkdir(parents=True, exist_ok=True)
    paths.data_raw.mkdir(parents=True, exist_ok=True)

    matches = pd.DataFrame([{
        "tourney_id": "2023-001", "tourney_date": pd.Timestamp("2023-01-16"),
        "match_num": 1, "surface": "Hard", "tourney_level": "A",
        "round": "R32", "best_of": 3,
        "winner_id": 1, "winner_name": "Alcaraz", "winner_hand": "R",
        "winner_ht": 185.0, "winner_age": 19.5, "winner_ioc": "ESP", "winner_rank": 1,
        "loser_id": 2, "loser_name": "Djokovic", "loser_hand": "R",
        "loser_ht": 188.0, "loser_age": 35.0, "loser_ioc": "SRB", "loser_rank": 5,
        "is_complete": True, "completed_sets": 2, "score": "6-3 6-4",
        "minutes": 95.0,
        "w_ace": 5.0, "w_df": 2.0, "w_svpt": 60.0, "w_1stIn": 40.0,
        "w_1stWon": 28.0, "w_2ndWon": 10.0, "w_bpSaved": 3.0, "w_bpFaced": 5.0,
        "l_ace": 3.0, "l_df": 3.0, "l_svpt": 58.0, "l_1stIn": 38.0,
        "l_1stWon": 25.0, "l_2ndWon": 9.0, "l_bpSaved": 2.0, "l_bpFaced": 4.0,
        "winner_rank_points": 8000, "loser_rank_points": 7000,
    }])
    matches.to_parquet(paths.matches_clean, index=False)

    with caplog.at_level(logging.WARNING, logger="progno_train"):
        rc = run_elo(paths)
    assert rc == 0
    assert any("xlsx" in msg.lower() or "roi" in msg.lower() for msg in caplog.messages)
