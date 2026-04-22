"""Write training artifacts consumed by the Tauri app."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from progno_train.rollup import PlayerElo

MATCH_HISTORY_COLUMNS = [
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
]


def write_elo_state(
    state: dict[int, PlayerElo],
    out_path: Path,
    data_as_of: pd.Timestamp,
) -> None:
    players_out: dict[str, dict] = {}
    for pid in sorted(state.keys()):
        d = asdict(state[pid])
        d.pop("player_id")
        players_out[str(pid)] = d

    payload = {
        "data_as_of": data_as_of.strftime("%Y-%m-%d"),
        "players": players_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_players(matches: pd.DataFrame, out_path: Path) -> None:
    winners = matches[
        ["winner_id", "winner_name", "winner_hand", "winner_ht", "winner_ioc"]
    ].rename(
        columns={
            "winner_id": "player_id",
            "winner_name": "name",
            "winner_hand": "hand",
            "winner_ht": "height_cm",
            "winner_ioc": "country",
        }
    )
    losers = matches[
        ["loser_id", "loser_name", "loser_hand", "loser_ht", "loser_ioc"]
    ].rename(
        columns={
            "loser_id": "player_id",
            "loser_name": "name",
            "loser_hand": "hand",
            "loser_ht": "height_cm",
            "loser_ioc": "country",
        }
    )
    players = (
        pd.concat([winners, losers], ignore_index=True)
        .drop_duplicates(subset=["player_id"], keep="last")
        .sort_values("player_id")
        .reset_index(drop=True)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    players.to_parquet(out_path, index=False)


def write_match_history(matches: pd.DataFrame, out_path: Path) -> None:
    projected = matches[MATCH_HISTORY_COLUMNS].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    projected.to_parquet(out_path, index=False)
