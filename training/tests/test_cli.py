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
