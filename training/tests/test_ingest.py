from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from progno_train.ingest import ingest_sackmann_csv

FIXTURE = Path(__file__).parent / "fixtures" / "mini_atp_matches.csv"


def test_ingest_returns_dataframe_with_expected_columns() -> None:
    df = ingest_sackmann_csv([FIXTURE])
    expected_cols = {
        "tourney_id",
        "tourney_name",
        "tourney_date",
        "surface",
        "tourney_level",
        "best_of",
        "round",
        "winner_id",
        "loser_id",
        "score",
        "is_complete",
        "completed_sets",
    }
    assert expected_cols.issubset(df.columns)


def test_ingest_parses_dates() -> None:
    df = ingest_sackmann_csv([FIXTURE])
    assert pd.api.types.is_datetime64_any_dtype(df["tourney_date"])
    assert df["tourney_date"].iloc[0] == pd.Timestamp("2024-01-08")


def test_ingest_flags_completed_matches() -> None:
    df = ingest_sackmann_csv([FIXTURE])
    scores = df.set_index("match_num")["is_complete"].to_dict()
    assert scores[1] is True  # "6-4 6-3"
    assert scores[2] is False  # retirement
    assert scores[3] is False  # walkover
    assert scores[4] is True  # "7-6(4) 6-4"


def test_ingest_sorts_by_tourney_date_then_match_num() -> None:
    df = ingest_sackmann_csv([FIXTURE])
    assert list(df["match_num"]) == sorted(df["match_num"])


def test_ingest_raises_on_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        ingest_sackmann_csv([missing])
