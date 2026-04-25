"""Write training artifacts consumed by the Tauri app."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from progno_train.rollup import PlayerElo

MATCH_HISTORY_COLUMNS = [
    # identifiers and context
    "tourney_id", "tourney_date", "match_num",
    "surface", "tourney_level", "round", "best_of",
    # players
    "winner_id", "winner_name", "winner_hand", "winner_ht", "winner_age", "winner_rank",
    "loser_id", "loser_name", "loser_hand", "loser_ht", "loser_age", "loser_rank",
    # outcome
    "is_complete", "completed_sets", "score", "minutes",
    # serve stats (available 1991+ ATP / 2007+ WTA, null before)
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced",
    # closing odds (tennis-data.co.uk, Phase 3.5 — NaN before odds ingest)
    "PSW", "PSL", "B365W", "B365L",
]


def write_elo_state(
    state: dict[int, PlayerElo],
    out_path: Path,
    data_as_of: pd.Timestamp,
    player_names: dict[int, str] | None = None,
) -> None:
    players_out: dict[str, dict] = {}
    for pid in sorted(state.keys()):
        key = player_names[pid] if player_names and pid in player_names else str(pid)
        d = asdict(state[pid])
        d.pop("player_id")
        players_out[key] = d

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
    losers = matches[["loser_id", "loser_name", "loser_hand", "loser_ht", "loser_ioc"]].rename(
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
    available = [c for c in MATCH_HISTORY_COLUMNS if c in matches.columns]
    projected = matches[available].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    projected.to_parquet(out_path, index=False)


def write_calibration(a: float, b: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"a": a, "b": b}) + "\n")


def write_model_card(
    train_years: tuple[int, int],
    test_year: int,
    metrics: dict,
    feature_names: list[str],
    git_sha: str,
    out_path: Path,
) -> None:
    card = {
        "train_years": list(train_years),
        "test_year": test_year,
        "metrics": metrics,
        "feature_names": feature_names,
        "git_sha": git_sha,
        "generated_at": pd.Timestamp.now().isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(card, indent=2) + "\n")
