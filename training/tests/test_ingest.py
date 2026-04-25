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


_CSV_HEADER = (
    "tourney_id,tourney_name,surface,draw_size,tourney_level,tourney_date,match_num,"
    "winner_id,winner_seed,winner_entry,winner_name,winner_hand,winner_ht,winner_ioc,winner_age,"
    "loser_id,loser_seed,loser_entry,loser_name,loser_hand,loser_ht,loser_ioc,loser_age,"
    "score,best_of,round,minutes,"
    "w_ace,w_df,w_svpt,w_1stIn,w_1stWon,w_2ndWon,w_SvGms,w_bpSaved,w_bpFaced,"
    "l_ace,l_df,l_svpt,l_1stIn,l_1stWon,l_2ndWon,l_SvGms,l_bpSaved,l_bpFaced,"
    "winner_rank,winner_rank_points,loser_rank,loser_rank_points\n"
)

_CSV_ROW = (
    "2024-001,Test,Hard,{draw_size},A,{date},1,1,,,A,R,185,USA,24.0,"
    "2,,,B,R,180,ESP,26.0,6-3 6-4,3,R32,90,"
    "5,1,60,40,30,12,8,3,4,2,3,58,35,22,10,8,2,5,10,5000,50,2000\n"
)


def test_ingest_drops_invalid_dates(tmp_path: Path) -> None:
    content = (
        _CSV_HEADER
        + _CSV_ROW.format(draw_size=32, date="20240101")
        + _CSV_ROW.replace("2024-001", "2024-002").format(draw_size=32, date="300")
    )
    csv_file = tmp_path / "test_invalid_date.csv"
    csv_file.write_text(content)
    df = ingest_sackmann_csv([csv_file])
    assert len(df) == 1
    assert df["tourney_date"].iloc[0] == pd.Timestamp("2024-01-01")


def test_ingest_coerces_mixed_type_numeric_columns(tmp_path: Path) -> None:
    content = (
        _CSV_HEADER
        + _CSV_ROW.format(draw_size=32, date="20240101")
        + _CSV_ROW.replace("2024-001", "2024-002").format(draw_size="", date="20240108")
    )
    csv_file = tmp_path / "test_mixed_type.csv"
    csv_file.write_text(content)
    df = ingest_sackmann_csv([csv_file])
    assert len(df) == 2
    assert not pd.api.types.is_object_dtype(df["draw_size"])


def test_ingest_extracts_set_counts() -> None:
    df = ingest_sackmann_csv([FIXTURE])
    assert "w_sets" in df.columns
    assert "l_sets" in df.columns
    # match_num=1 has score "6-4 6-3" → winner 2 sets, loser 0 sets
    completed_row = df[df["match_num"] == 1].iloc[0]
    assert int(completed_row["w_sets"]) == 2
    assert int(completed_row["l_sets"]) == 0
