"""Ingest Sackmann ATP CSV files into a cleaned DataFrame."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from progno_train.score import parse_score

SACKMANN_COLUMNS = [
    "tourney_id",
    "tourney_name",
    "surface",
    "draw_size",
    "tourney_level",
    "tourney_date",
    "match_num",
    "winner_id",
    "winner_seed",
    "winner_entry",
    "winner_name",
    "winner_hand",
    "winner_ht",
    "winner_ioc",
    "winner_age",
    "loser_id",
    "loser_seed",
    "loser_entry",
    "loser_name",
    "loser_hand",
    "loser_ht",
    "loser_ioc",
    "loser_age",
    "score",
    "best_of",
    "round",
    "minutes",
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
    "winner_rank",
    "winner_rank_points",
    "loser_rank",
    "loser_rank_points",
]


def ingest_sackmann_csv(paths: Iterable[Path]) -> pd.DataFrame:
    paths = list(paths)
    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Sackmann CSV not found: {p}")

    frames = [pd.read_csv(p, low_memory=False) for p in paths]
    df = pd.concat(frames, ignore_index=True)

    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["tourney_date"])
    df["score"] = df["score"].fillna("").astype(str)

    parsed = df["score"].apply(parse_score)
    df["is_complete"] = parsed.apply(lambda p: p.is_complete)
    df["completed_sets"] = parsed.apply(lambda p: p.completed_sets)

    # Fix mixed-type object columns (e.g. draw_size has ints + NaN across old/new CSVs)
    for col in df.select_dtypes(include="object").columns:
        if col in ("score", "tourney_id", "tourney_name", "surface", "tourney_level",
                   "round", "winner_name", "loser_name", "winner_hand", "loser_hand",
                   "winner_ioc", "loser_ioc"):
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    return df
