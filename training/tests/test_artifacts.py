from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from progno_train.artifacts import (
    write_calibration,
    write_elo_state,
    write_match_history,
    write_model_card,
    write_players,
)
from progno_train.rollup import PlayerElo


def test_write_elo_state_produces_expected_json(tmp_path: Path) -> None:
    state = {
        1: PlayerElo(player_id=1, elo_overall=1600.0, elo_hard=1650.0, matches_played=10),
        2: PlayerElo(player_id=2, elo_overall=1400.0, elo_clay=1450.0, matches_played=8),
    }
    out = tmp_path / "elo_state.json"
    write_elo_state(state, out, data_as_of=pd.Timestamp("2026-04-22"), player_names={1: "alpha", 2: "beta"})

    content = json.loads(out.read_text())
    assert content["data_as_of"] == "2026-04-22"
    assert "players" in content
    assert "alpha" in content["players"]
    p1 = content["players"]["alpha"]
    assert p1["elo_overall"] == 1600.0
    assert p1["elo_hard"] == 1650.0
    assert p1["matches_played"] == 10


def test_write_elo_state_is_deterministic(tmp_path: Path) -> None:
    state = {
        2: PlayerElo(player_id=2, elo_overall=1400.0),
        1: PlayerElo(player_id=1, elo_overall=1600.0),
    }
    names = {1: "one", 2: "two"}
    out1 = tmp_path / "a.json"
    out2 = tmp_path / "b.json"
    write_elo_state(state, out1, data_as_of=pd.Timestamp("2026-04-22"), player_names=names)
    write_elo_state(state, out2, data_as_of=pd.Timestamp("2026-04-22"), player_names=names)
    assert out1.read_text() == out2.read_text()


def test_write_players_creates_parquet(tmp_path: Path) -> None:
    matches = pd.DataFrame(
        [
            {
                "winner_id": 1,
                "winner_name": "Alpha Alpha",
                "winner_hand": "R",
                "winner_ht": 185,
                "winner_ioc": "USA",
                "loser_id": 2,
                "loser_name": "Beta Beta",
                "loser_hand": "L",
                "loser_ht": 180,
                "loser_ioc": "ESP",
            },
        ]
    )
    out = tmp_path / "players.parquet"
    write_players(matches, out)
    df = pd.read_parquet(out)
    assert set(df.columns) == {"player_id", "name", "hand", "height_cm", "country"}
    assert len(df) == 2
    row1 = df.set_index("player_id").loc[1]
    assert row1["name"] == "Alpha Alpha"
    assert row1["hand"] == "R"


def test_write_players_deduplicates_across_matches(tmp_path: Path) -> None:
    matches = pd.DataFrame(
        [
            {
                "winner_id": 1,
                "winner_name": "Alpha A",
                "winner_hand": "R",
                "winner_ht": 185,
                "winner_ioc": "USA",
                "loser_id": 2,
                "loser_name": "Beta B",
                "loser_hand": "L",
                "loser_ht": 180,
                "loser_ioc": "ESP",
            },
            {
                "winner_id": 2,
                "winner_name": "Beta B",
                "winner_hand": "L",
                "winner_ht": 180,
                "winner_ioc": "ESP",
                "loser_id": 1,
                "loser_name": "Alpha A",
                "loser_hand": "R",
                "loser_ht": 185,
                "loser_ioc": "USA",
            },
        ]
    )
    out = tmp_path / "players.parquet"
    write_players(matches, out)
    df = pd.read_parquet(out)
    assert len(df) == 2


def test_write_calibration_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "calibration.json"
    write_calibration(1.23, -0.45, out)
    content = json.loads(out.read_text())
    assert content["a"] == pytest.approx(1.23)
    assert content["b"] == pytest.approx(-0.45)


def test_write_calibration_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir" / "calibration.json"
    write_calibration(0.0, 0.0, out)
    assert out.exists()


def test_write_model_card_structure(tmp_path: Path) -> None:
    out = tmp_path / "model_card.json"
    write_model_card(
        train_years=(2005, 2022),
        test_year=2023,
        metrics={"log_loss": 0.58, "ece": 0.03},
        feature_names=["elo_overall_diff", "win_rate_diff"],
        git_sha="abc123",
        out_path=out,
    )
    card = json.loads(out.read_text())
    assert card["train_years"] == [2005, 2022]
    assert card["test_year"] == 2023
    assert card["metrics"]["log_loss"] == pytest.approx(0.58)
    assert card["feature_names"] == ["elo_overall_diff", "win_rate_diff"]
    assert card["git_sha"] == "abc123"
    assert "generated_at" in card


def test_write_model_card_features_roundtrip(tmp_path: Path) -> None:
    features = [f"feat_{i}" for i in range(20)]
    write_model_card(
        train_years=(2010, 2022),
        test_year=2023,
        metrics={},
        feature_names=features,
        git_sha="deadbeef",
        out_path=tmp_path / "card.json",
    )
    card = json.loads((tmp_path / "card.json").read_text())
    assert card["feature_names"] == features


def test_write_match_history_projects_expected_columns(tmp_path: Path) -> None:
    matches = pd.DataFrame(
        [
            {
                "tourney_id": "2024-01",
                "tourney_date": pd.Timestamp("2024-01-08"),
                "match_num": 1,
                "surface": "Hard",
                "tourney_level": "A",
                "round": "R32",
                "best_of": 3,
                "winner_id": 1,
                "loser_id": 2,
                "is_complete": True,
                "completed_sets": 2,
                "score": "6-4 6-3",
                "minutes": 90,
                "extra_column_ignored": "x",
            }
        ]
    )
    out = tmp_path / "match_history.parquet"
    write_match_history(matches, out)
    df = pd.read_parquet(out)
    expected = {
        "tourney_id",
        "tourney_date",
        "match_num",
        "surface",
        "tourney_level",
        "round",
        "best_of",
        "winner_id",
        "loser_id",
        "is_complete",
        "completed_sets",
        "score",
        "minutes",
    }
    assert set(df.columns) == expected
