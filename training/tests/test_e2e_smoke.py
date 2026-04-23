"""End-to-end smoke test on real Sackmann data.

Skipped unless the Sackmann CSVs are present locally. To prepare:

    cd training
    bash scripts/fetch_sackmann.sh
    uv run pytest tests/test_e2e_smoke.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from progno_train.cli import run_elo, run_ingest
from progno_train.config import Paths

TRAINING_ROOT = Path(__file__).parent.parent
RAW = TRAINING_ROOT / "data" / "raw"

KNOWN_TOP_PLAYERS_SINCE_2020 = {
    "Novak Djokovic",
    "Carlos Alcaraz",
    "Jannik Sinner",
    "Daniil Medvedev",
    "Alexander Zverev",
}


def _have_real_data() -> bool:
    return any(RAW.glob("atp_matches_20*.csv"))


pytestmark = pytest.mark.skipif(
    not _have_real_data(),
    reason="Real Sackmann CSVs not present; run scripts/fetch_sackmann.sh",
)


def test_e2e_pipeline_on_real_data(tmp_path: Path) -> None:
    paths = Paths(
        data_raw=RAW,
        data_staging=tmp_path / "staging",
        artifacts=tmp_path / "artifacts",
    )

    assert run_ingest(paths) == 0
    assert run_elo(paths) == 0

    elo_state_path = paths.artifacts / "elo_state.json"
    assert elo_state_path.exists()
    state = json.loads(elo_state_path.read_text())
    players = state["players"]
    assert len(players) > 1000

    # elo_state.json keys are last-name strings (e.g. "djokovic"); extract top-20 by elo_overall
    elo_by_key = {key: p["elo_overall"] for key, p in players.items()}
    top20_keys = pd.Series(elo_by_key).sort_values(ascending=False).head(20).index.tolist()

    # Load players.parquet to map last-name keys back to full names
    players_df = pd.read_parquet(paths.artifacts / "players.parquet")
    # Build a lowercase last-name → full name lookup
    players_df["last_name_key"] = players_df["name"].str.split().str[-1].str.lower()
    last_name_to_full = players_df.set_index("last_name_key")["name"].to_dict()
    top20_names = {last_name_to_full[k] for k in top20_keys if k in last_name_to_full}

    overlap = top20_names & KNOWN_TOP_PLAYERS_SINCE_2020
    assert len(overlap) >= 2, (
        f"Expected at least 2 known top players in top-20 Elo, got overlap={overlap}, "
        f"top20_names={top20_names}"
    )
