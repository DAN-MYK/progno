"""Tests for tennis-data.co.uk XLSX ingestion."""

import io
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from progno_train.ingest_xlsx import ingest_tennis_data_xlsx, _parse_xlsx_date


def _make_xlsx(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a minimal tennis-data.co.uk XLSX to tmp_path."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    if rows:
        ws.append(list(rows[0].keys()))
        for row in rows:
            ws.append(list(row.values()))
    p = tmp_path / "test.xlsx"
    wb.save(p)
    return p


def _sample_rows():
    return [
        {
            "Date": "15/01/2023",
            "Winner": "Carlos Alcaraz",
            "Loser": "Novak Djokovic",
            "PSW": 1.85,
            "PSL": 2.10,
            "B365W": 1.83,
            "B365L": 2.00,
        },
        {
            "Date": "16/01/2023",
            "Winner": "Rafael Nadal",
            "Loser": "Daniil Medvedev",
            "PSW": 1.45,
            "PSL": 2.95,
            "B365W": 1.44,
            "B365L": 2.87,
        },
    ]


def test_ingest_returns_dataframe(tmp_path):
    p = _make_xlsx(tmp_path, _sample_rows())
    df = ingest_tennis_data_xlsx([p])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_ingest_output_columns(tmp_path):
    p = _make_xlsx(tmp_path, _sample_rows())
    df = ingest_tennis_data_xlsx([p])
    for col in ["date_week", "winner_norm", "loser_norm", "PSW", "PSL", "B365W", "B365L"]:
        assert col in df.columns, f"missing column: {col}"


def test_date_parsed_to_monday(tmp_path):
    p = _make_xlsx(tmp_path, _sample_rows())
    df = ingest_tennis_data_xlsx([p])
    # 15/01/2023 is Sunday → monday = 09/01/2023
    assert df["date_week"].iloc[0] == pd.Timestamp("2023-01-09")


def test_names_normalized(tmp_path):
    p = _make_xlsx(tmp_path, _sample_rows())
    df = ingest_tennis_data_xlsx([p])
    assert df["winner_norm"].iloc[0] == "alcaraz c"
    assert df["loser_norm"].iloc[0] == "djokovic n"


def test_odds_preserved(tmp_path):
    p = _make_xlsx(tmp_path, _sample_rows())
    df = ingest_tennis_data_xlsx([p])
    assert abs(df["PSW"].iloc[0] - 1.85) < 1e-9
    assert abs(df["PSL"].iloc[0] - 2.10) < 1e-9


def test_missing_odds_columns_filled_with_nan(tmp_path):
    """XLSX without PSW/PSL should produce NaN columns."""
    rows = [{"Date": "15/01/2023", "Winner": "Alcaraz", "Loser": "Djokovic"}]
    p = _make_xlsx(tmp_path, rows)
    df = ingest_tennis_data_xlsx([p])
    assert pd.isna(df["PSW"].iloc[0])


def test_multiple_files_concatenated(tmp_path):
    p1 = tmp_path / "a.xlsx"
    p2 = tmp_path / "b.xlsx"
    import openpyxl
    for rows, out in [([_sample_rows()[0]], p1), ([_sample_rows()[1]], p2)]:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(list(rows[0].keys()))
        for row in rows:
            ws.append(list(row.values()))
        wb.save(out)
    df = ingest_tennis_data_xlsx([p1, p2])
    assert len(df) == 2


def test_rows_with_invalid_date_dropped(tmp_path):
    rows = _sample_rows() + [{"Date": "bad", "Winner": "A", "Loser": "B", "PSW": 2.0, "PSL": 1.8, "B365W": 2.0, "B365L": 1.8}]
    p = _make_xlsx(tmp_path, rows)
    df = ingest_tennis_data_xlsx([p])
    assert len(df) == 2  # bad row dropped


def test_parse_xlsx_date():
    assert _parse_xlsx_date("15/01/2023") == pd.Timestamp("2023-01-15")
    assert _parse_xlsx_date("01/07/2022") == pd.Timestamp("2022-07-01")
    assert pd.isna(_parse_xlsx_date("invalid"))
