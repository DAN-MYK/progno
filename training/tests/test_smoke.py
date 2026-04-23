from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from progno_train.artifacts import write_elo_state, write_players
from progno_train.elo import INITIAL_RATING
from progno_train.rollup import rollup_elo


def test_package_imports() -> None:
    import progno_train
    import progno_train.artifacts
    import progno_train.cli
    import progno_train.config
    import progno_train.elo
    import progno_train.ingest
    import progno_train.rollup
    import progno_train.score

    assert progno_train.cli.main is not None


def test_synthetic_pipeline(tmp_path: Path) -> None:
    matches = pd.DataFrame(
        {
            "tourney_date": [pd.Timestamp("2024-01-15")] * 3,
            "match_num": [1, 2, 3],
            "winner_id": [1, 1, 2],
            "loser_id": [2, 2, 1],
            "surface": ["Hard", "Clay", "Hard"],
            "tourney_level": ["G", "G", "G"],
            "round": ["F", "SF", "F"],
            "best_of": [5, 5, 5],
            "is_complete": [True, True, True],
            "winner_name": ["Alice", "Alice", "Bob"],
            "loser_name": ["Bob", "Bob", "Alice"],
            "winner_hand": ["R", "R", "L"],
            "loser_hand": ["L", "L", "R"],
            "winner_ht": [180.0, 180.0, 175.0],
            "loser_ht": [175.0, 175.0, 180.0],
            "winner_ioc": ["USA", "USA", "GBR"],
            "loser_ioc": ["GBR", "GBR", "USA"],
        }
    )

    state = rollup_elo(matches)

    assert set(state.keys()) == {1, 2}
    assert state[1].elo_overall != INITIAL_RATING
    assert state[2].elo_overall != INITIAL_RATING
    assert state[1].matches_played == 3
    assert state[2].matches_played == 3

    elo_path = tmp_path / "elo_state.json"
    players_path = tmp_path / "players.parquet"

    player_names = {1: "alice", 2: "bob"}
    write_elo_state(state, elo_path, pd.Timestamp("2024-01-16"), player_names=player_names)
    write_players(matches, players_path)

    payload = json.loads(elo_path.read_text())
    assert "data_as_of" in payload
    assert "players" in payload
    assert len(payload["players"]) == 2
    assert "alice" in payload["players"]
    assert "bob" in payload["players"]

    players_df = pd.read_parquet(players_path)
    assert set(players_df.columns) >= {"player_id", "name"}
    assert len(players_df) == 2
