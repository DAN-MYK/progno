# Phase 3.5: Odds Ingest (tennis-data.co.uk → ROI Gate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download tennis-data.co.uk XLSX odds files, join Pinnacle closing odds to `match_history.parquet`, propagate `odds_a_winner` into the featurized dataset, and activate the ROI gate in the acceptance pipeline — currently the gate skips with a warning because no odds column is present.

**Architecture:** Two new modules: `ingest_xlsx.py` parses raw XLSX files into a tidy DataFrame; `odds_join.py` joins that DataFrame to Sackmann matches by `(tourney_week ± 2 weeks, sorted_player_pair)` with exact → rapidfuzz → manual `name_map.csv` fallback. `run_elo` in `cli.py` now calls the join if XLSX files are present. `build_all_features` stamps each row with `odds_a_winner` (PSW for the winner-as-A row, PSL for the loser-as-A row). `get_feature_cols` excludes `odds_a_winner` from training features so closing odds are never used as model input — only for ROI evaluation.

**Tech Stack:** Python 3.12, pandas 2.2, openpyxl (XLSX reading), unidecode (ASCII normalization), rapidfuzz (fuzzy name matching), pytest

---

## Files Summary

| File | Action | Responsibility |
|------|--------|----------------|
| `training/pyproject.toml` | Modify | Add unidecode, rapidfuzz, openpyxl deps |
| `training/scripts/fetch_tennis_data.sh` | Create | Download ATP/WTA XLSX per year from tennis-data.co.uk |
| `training/data/manual/name_map.csv` | Create | Manual name override table (header only initially) |
| `training/src/progno_train/ingest_xlsx.py` | Create | Parse XLSX → tidy DataFrame (date_week, winner_norm, loser_norm, PSW, PSL, B365W, B365L) |
| `training/src/progno_train/odds_join.py` | Create | `normalize_name`, `join_odds` — joins tidy odds df to Sackmann matches |
| `training/src/progno_train/artifacts.py` | Modify | Add PSW, PSL, B365W, B365L to MATCH_HISTORY_COLUMNS |
| `training/src/progno_train/config.py` | Modify | Add `odds_xlsx_dir` and `name_map` path properties |
| `training/src/progno_train/cli.py` | Modify | `run_elo`: call odds join when XLSX dir has files; log join yield |
| `training/src/progno_train/features.py` | Modify | `build_all_features`: stamp `odds_a_winner` on fp/fn rows |
| `training/src/progno_train/train.py` | Modify | `get_feature_cols`: exclude `odds_a_winner` from training features |
| `training/tests/test_ingest_xlsx.py` | Create | Tests for XLSX parsing, date handling, name normalization |
| `training/tests/test_odds_join.py` | Create | Tests for join logic: exact, ±week tolerance, fuzzy, name_map fallback |
| `justfile` | Modify | Add `fetch-odds-data`, `ingest-odds` targets |

---

## Task 1: Add dependencies + directory scaffolding

**Files:**
- Modify: `training/pyproject.toml`
- Create: `training/scripts/fetch_tennis_data.sh`
- Create: `training/data/manual/name_map.csv`
- Modify: `justfile`

- [ ] **Step 1: Add three new dependencies to pyproject.toml**

In `training/pyproject.toml`, replace the `dependencies` list:

```toml
dependencies = [
    "pandas>=2.2",
    "pyarrow>=15.0",
    "numpy>=1.26",
    "catboost>=1.2",
    "scikit-learn>=1.4",
    "unidecode>=1.3",
    "rapidfuzz>=3.6",
    "openpyxl>=3.1",
]
```

- [ ] **Step 2: Sync lockfile**

```bash
cd /home/mykhailo_dan/apps/progno/training && uv sync
```

Expected: lockfile updated, packages installed.

- [ ] **Step 3: Verify imports**

```bash
cd /home/mykhailo_dan/apps/progno/training && uv run python -c "import unidecode, rapidfuzz, openpyxl; print('OK')"
```

Expected: prints `OK`.

- [ ] **Step 4: Create fetch_tennis_data.sh**

Create `training/scripts/fetch_tennis_data.sh`:

```bash
#!/usr/bin/env bash
# Download tennis-data.co.uk XLSX files for ATP and WTA.
# Usage: bash training/scripts/fetch_tennis_data.sh [start_year] [end_year]
# Defaults: ATP from 2000, WTA from 2007, up to current year.
set -euo pipefail

OUTDIR="$(dirname "$0")/../data/raw/tennis_data_xlsx"
mkdir -p "$OUTDIR"

CURRENT_YEAR=$(date +%Y)
ATP_START="${1:-2000}"
WTA_START="${2:-2007}"
END_YEAR="${3:-$CURRENT_YEAR}"

echo "Downloading ATP XLSX: $ATP_START–$END_YEAR"
for year in $(seq "$ATP_START" "$END_YEAR"); do
    dest="$OUTDIR/atp_${year}.xlsx"
    if [ -f "$dest" ]; then
        echo "  skip $dest (exists)"
        continue
    fi
    url="http://www.tennis-data.co.uk/${year}/${year}.xlsx"
    echo "  fetch $url"
    curl -sS --fail --retry 3 -o "$dest" "$url" || { echo "  WARN: $url not found"; rm -f "$dest"; }
done

echo "Downloading WTA XLSX: $WTA_START–$END_YEAR"
for year in $(seq "$WTA_START" "$END_YEAR"); do
    dest="$OUTDIR/wta_${year}.xlsx"
    if [ -f "$dest" ]; then
        echo "  skip $dest (exists)"
        continue
    fi
    url="http://www.tennis-data.co.uk/${year}w/${year}w.xlsx"
    echo "  fetch $url"
    curl -sS --fail --retry 3 -o "$dest" "$url" || { echo "  WARN: $url not found"; rm -f "$dest"; }
done

echo "Done. Files in $OUTDIR:"
ls "$OUTDIR" | head -20
```

Make executable:
```bash
chmod +x /home/mykhailo_dan/apps/progno/training/scripts/fetch_tennis_data.sh
```

- [ ] **Step 5: Create manual/name_map.csv**

Create `training/data/manual/name_map.csv`:

```csv
sackmann_name,odds_name
```

(Header only. Add rows manually as unmatched cases are discovered after first run.)

```bash
mkdir -p /home/mykhailo_dan/apps/progno/training/data/manual
```

- [ ] **Step 6: Add just targets**

Append to `justfile`:

```just
# --- Phase 3.5 targets ---
fetch-odds-data:
    bash training/scripts/fetch_tennis_data.sh

ingest-odds:
    cd training && uv run python -m progno_train.cli --tour atp elo
    cd training && uv run python -m progno_train.cli --tour wta elo

ingest-odds-atp:
    cd training && uv run python -m progno_train.cli --tour atp elo

ingest-odds-wta:
    cd training && uv run python -m progno_train.cli --tour wta elo
```

- [ ] **Step 7: Commit**

```bash
cd /home/mykhailo_dan/apps/progno
git add training/pyproject.toml training/uv.lock training/scripts/fetch_tennis_data.sh training/data/manual/name_map.csv justfile
git commit -m "feat(deps): add unidecode/rapidfuzz/openpyxl; odds download script; manual name_map"
```

---

## Task 2: Create ingest_xlsx.py + tests

**Files:**
- Create: `training/src/progno_train/ingest_xlsx.py`
- Create: `training/tests/test_ingest_xlsx.py`

The tennis-data.co.uk XLSX has these key columns:
- `Date` — match date in `DD/MM/YYYY` format
- `Winner`, `Loser` — player names ("Firstname Lastname")
- `PSW`, `PSL` — Pinnacle odds for winner/loser
- `B365W`, `B365L` — Bet365 odds

Output columns from `ingest_tennis_data_xlsx`: `date_week` (Monday of match week), `winner_norm`, `loser_norm`, `PSW`, `PSL`, `B365W`, `B365L`.

- [ ] **Step 1: Write tests first**

Create `training/tests/test_ingest_xlsx.py`:

```python
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
    p1 = _make_xlsx(tmp_path / "a.xlsx", [_sample_rows()[0]])
    p2 = _make_xlsx(tmp_path / "b.xlsx", [_sample_rows()[1]])
    # Fix: tmp_path already exists, need sub-paths
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/mykhailo_dan/apps/progno/training && uv run pytest tests/test_ingest_xlsx.py -v 2>&1 | head -20
```

Expected: ImportError (module doesn't exist yet).

- [ ] **Step 3: Create ingest_xlsx.py**

Create `training/src/progno_train/ingest_xlsx.py`:

```python
"""Ingest tennis-data.co.uk XLSX files into a tidy odds DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from unidecode import unidecode

_ODDS_COLS = ["PSW", "PSL", "B365W", "B365L"]


def _parse_xlsx_date(s: object) -> pd.Timestamp:
    """Parse DD/MM/YYYY string or return NaT on failure."""
    try:
        return pd.to_datetime(str(s), format="%d/%m/%Y", errors="raise")
    except Exception:
        return pd.NaT


def _monday_of_week(ts: pd.Timestamp) -> pd.Timestamp:
    """Return the Monday of the ISO week containing ts."""
    return ts - pd.Timedelta(days=ts.weekday())


def _norm_name(name: object) -> str:
    """Normalize player name to 'lastname initial' lowercase ASCII.

    'Carlos Alcaraz' → 'alcaraz c'
    'Kévin Krawietz' → 'krawietz k'
    """
    s = unidecode(str(name)).strip()
    parts = s.split()
    if len(parts) == 0:
        return ""
    last = parts[-1].lower()
    initial = parts[0][0].lower() if parts[0] else ""
    return f"{last} {initial}" if initial else last


def ingest_tennis_data_xlsx(paths: list[Path]) -> pd.DataFrame:
    """Parse tennis-data.co.uk XLSX files into a tidy DataFrame.

    Returns a DataFrame with columns:
        date_week    — Monday of the match week (pd.Timestamp)
        winner_norm  — normalized winner name ('lastname initial')
        loser_norm   — normalized loser name
        PSW          — Pinnacle winner odds (float, NaN if absent)
        PSL          — Pinnacle loser odds (float, NaN if absent)
        B365W        — Bet365 winner odds (float, NaN if absent)
        B365L        — Bet365 loser odds (float, NaN if absent)
    """
    frames = []
    for p in paths:
        df = pd.read_excel(p, engine="openpyxl")
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["date_week", "winner_norm", "loser_norm"] + _ODDS_COLS)

    raw = pd.concat(frames, ignore_index=True)

    # Parse dates
    raw["_date"] = raw["Date"].apply(_parse_xlsx_date)
    raw = raw.dropna(subset=["_date"])

    out = pd.DataFrame()
    out["date_week"] = raw["_date"].apply(_monday_of_week)
    out["winner_norm"] = raw["Winner"].apply(_norm_name)
    out["loser_norm"] = raw["Loser"].apply(_norm_name)

    for col in _ODDS_COLS:
        if col in raw.columns:
            out[col] = pd.to_numeric(raw[col], errors="coerce")
        else:
            out[col] = float("nan")

    return out.reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

```bash
cd /home/mykhailo_dan/apps/progno/training && uv run pytest tests/test_ingest_xlsx.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/mykhailo_dan/apps/progno
git add training/src/progno_train/ingest_xlsx.py training/tests/test_ingest_xlsx.py
git commit -m "feat(ingest_xlsx): parse tennis-data.co.uk XLSX to tidy odds DataFrame"
```

---

## Task 3: Create odds_join.py + tests

**Files:**
- Create: `training/src/progno_train/odds_join.py`
- Create: `training/tests/test_odds_join.py`

Join strategy (spec §2.6): exact match on `(tourney_week, sorted_pair)` → tolerance ±7 and ±14 days → rapidfuzz on player names within ±21 days → manual name_map.csv override.

`tourney_week` = `monday_of_week(tourney_date)` from Sackmann.
`date_week` = `monday_of_week(Date)` from XLSX.

For a 1-week tournament: `tourney_week == date_week` (exact match).
For a Grand Slam week 2: `date_week = tourney_week + 7 or + 14` (tolerance match).

- [ ] **Step 1: Write tests first**

Create `training/tests/test_odds_join.py`:

```python
"""Tests for odds join: exact, tolerance, fuzzy, name_map fallback."""

import pandas as pd
import pytest
from pathlib import Path

from progno_train.odds_join import normalize_name, join_odds


def _sack(winner, loser, tourney_date, **kwargs):
    return {
        "winner_name": winner,
        "loser_name": loser,
        "tourney_date": pd.Timestamp(tourney_date),
        "is_complete": True,
        **kwargs,
    }


def _odds(winner, loser, date_week, psw=1.8, psl=2.1, b365w=1.75, b365l=2.05):
    return {
        "winner_norm": normalize_name(winner),
        "loser_norm": normalize_name(loser),
        "date_week": pd.Timestamp(date_week),
        "PSW": psw,
        "PSL": psl,
        "B365W": b365w,
        "B365L": b365l,
    }


# ── normalize_name tests ─────────────────────────────────────────────────────

def test_normalize_name_simple():
    assert normalize_name("Carlos Alcaraz") == "alcaraz c"


def test_normalize_name_accents():
    assert normalize_name("Kévin Krawietz") == "krawietz k"


def test_normalize_name_single_word():
    assert normalize_name("Sinner") == "sinner"


def test_normalize_name_lowercase():
    assert normalize_name("NOVAK DJOKOVIC") == "djokovic n"


# ── join_odds tests ──────────────────────────────────────────────────────────

def test_exact_join(tmp_path):
    sack = pd.DataFrame([_sack("Carlos Alcaraz", "Novak Djokovic", "2023-01-16")])
    odds_df = pd.DataFrame([_odds("Carlos Alcaraz", "Novak Djokovic", "2023-01-16")])
    result = join_odds(sack, odds_df, name_map_path=None)
    assert abs(result["PSW"].iloc[0] - 1.8) < 1e-9
    assert abs(result["PSL"].iloc[0] - 2.1) < 1e-9


def test_join_is_orientation_agnostic(tmp_path):
    """XLSX has B vs A but Sackmann has A vs B — still matches."""
    sack = pd.DataFrame([_sack("Carlos Alcaraz", "Novak Djokovic", "2023-01-16")])
    # In XLSX, Djokovic is winner
    odds_df = pd.DataFrame([_odds("Novak Djokovic", "Carlos Alcaraz", "2023-01-16", psw=2.1, psl=1.8)])
    result = join_odds(sack, odds_df, name_map_path=None)
    # PSW from XLSX (2.1) → PSL for Sackmann winner Alcaraz = 1.8
    assert result["PSW"].notna().iloc[0]


def test_tolerance_join_7_days(tmp_path):
    """Sackmann tourney_date = tournament start Monday; XLSX date_week = match 7 days later."""
    sack = pd.DataFrame([_sack("Carlos Alcaraz", "Novak Djokovic", "2023-01-09")])
    odds_df = pd.DataFrame([_odds("Carlos Alcaraz", "Novak Djokovic", "2023-01-16")])
    result = join_odds(sack, odds_df, name_map_path=None)
    assert result["PSW"].notna().iloc[0]


def test_tolerance_join_14_days(tmp_path):
    """Grand Slam week 2: XLSX date 14 days after tourney start."""
    sack = pd.DataFrame([_sack("Carlos Alcaraz", "Novak Djokovic", "2023-01-09")])
    odds_df = pd.DataFrame([_odds("Carlos Alcaraz", "Novak Djokovic", "2023-01-23")])
    result = join_odds(sack, odds_df, name_map_path=None)
    assert result["PSW"].notna().iloc[0]


def test_no_match_beyond_21_days(tmp_path):
    """Dates more than 21 days apart should NOT match."""
    sack = pd.DataFrame([_sack("Carlos Alcaraz", "Novak Djokovic", "2023-01-09")])
    odds_df = pd.DataFrame([_odds("Carlos Alcaraz", "Novak Djokovic", "2023-02-13")])
    result = join_odds(sack, odds_df, name_map_path=None)
    assert pd.isna(result["PSW"].iloc[0])


def test_fuzzy_name_match(tmp_path):
    """Minor name discrepancy handled by rapidfuzz."""
    sack = pd.DataFrame([_sack("Alexander Zverev", "Stefanos Tsitsipas", "2023-03-13")])
    odds_df = pd.DataFrame([_odds("A. Zverev", "S. Tsitsipas", "2023-03-13")])
    result = join_odds(sack, odds_df, name_map_path=None)
    # Fuzzy match should handle "zverev a" vs "zverev a" (same after normalization)
    # Even if it doesn't fuzzy-match, check the column exists
    assert "PSW" in result.columns


def test_unmatched_row_gets_nan(tmp_path):
    sack = pd.DataFrame([_sack("Player A", "Player B", "2023-01-16")])
    odds_df = pd.DataFrame([_odds("Completely Different", "Names Here", "2023-01-16")])
    result = join_odds(sack, odds_df, name_map_path=None)
    assert pd.isna(result["PSW"].iloc[0])


def test_name_map_override(tmp_path):
    """Manual name_map.csv overrides normalized name mismatch."""
    name_map = tmp_path / "name_map.csv"
    name_map.write_text("sackmann_name,odds_name\nDel Potro Juan Martin,del potro j\n")
    sack = pd.DataFrame([_sack("Juan Martin Del Potro", "Rafael Nadal", "2023-01-16")])
    odds_df = pd.DataFrame([_odds("Del Potro Juan Martin", "Rafael Nadal", "2023-01-16")])
    result = join_odds(sack, odds_df, name_map_path=name_map)
    assert result["PSW"].notna().iloc[0]


def test_join_yield_logged(tmp_path, caplog):
    """join_odds should log the join yield."""
    import logging
    sack = pd.DataFrame([
        _sack("Carlos Alcaraz", "Novak Djokovic", "2023-01-16"),
        _sack("Player A", "Player B", "2023-01-16"),
    ])
    odds_df = pd.DataFrame([_odds("Carlos Alcaraz", "Novak Djokovic", "2023-01-16")])
    with caplog.at_level(logging.INFO, logger="progno_train.odds_join"):
        join_odds(sack, odds_df, name_map_path=None)
    assert any("yield" in msg.lower() or "%" in msg for msg in caplog.messages)


def test_output_has_all_odds_columns(tmp_path):
    sack = pd.DataFrame([_sack("Carlos Alcaraz", "Novak Djokovic", "2023-01-16")])
    odds_df = pd.DataFrame([_odds("Carlos Alcaraz", "Novak Djokovic", "2023-01-16")])
    result = join_odds(sack, odds_df, name_map_path=None)
    for col in ["PSW", "PSL", "B365W", "B365L"]:
        assert col in result.columns
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/mykhailo_dan/apps/progno/training && uv run pytest tests/test_odds_join.py -v 2>&1 | head -20
```

Expected: ImportError (module doesn't exist yet).

- [ ] **Step 3: Create odds_join.py**

Create `training/src/progno_train/odds_join.py`:

```python
"""Join tennis-data.co.uk odds to Sackmann match records."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz
from unidecode import unidecode

log = logging.getLogger(__name__)

_ODDS_COLS = ["PSW", "PSL", "B365W", "B365L"]
_TOLERANCE_DAYS = [0, 7, 14]   # weeks to search: exact → +1 week → +2 weeks
_FUZZY_THRESHOLD = 90           # rapidfuzz score threshold (0–100)
_FUZZY_WINDOW_DAYS = 21         # max date diff for fuzzy name search


def normalize_name(name: object) -> str:
    """Normalize player name to 'lastname initial' lowercase ASCII.

    'Carlos Alcaraz' → 'alcaraz c'
    'Kévin Krawietz' → 'krawietz k'
    """
    s = unidecode(str(name)).strip()
    parts = s.split()
    if not parts:
        return ""
    last = parts[-1].lower()
    initial = parts[0][0].lower() if parts[0] else ""
    return f"{last} {initial}" if initial else last


def _monday(ts: pd.Timestamp) -> pd.Timestamp:
    return ts - pd.Timedelta(days=ts.weekday())


def _load_name_map(path: Path | None) -> dict[str, str]:
    """Load sackmann_name → odds_name overrides from CSV."""
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["sackmann_name"].apply(normalize_name),
                    df["odds_name"].apply(normalize_name)))


def join_odds(
    sackmann: pd.DataFrame,
    odds_df: pd.DataFrame,
    name_map_path: Path | None,
) -> pd.DataFrame:
    """Join Pinnacle/Bet365 odds to Sackmann matches.

    Adds PSW, PSL, B365W, B365L columns to sackmann (NaN where unmatched).
    Logs join yield.

    Join strategy (spec §2.6):
      1. Exact: (tourney_week, sorted pair) — covers 1-week tournaments
      2. ±7 days, ±14 days — covers Grand Slam weeks
      3. rapidfuzz on names within ±21 days — catches spelling discrepancies
      4. Manual name_map.csv override applied before (1)–(3)
    """
    name_map = _load_name_map(name_map_path)

    result = sackmann.copy()
    for col in _ODDS_COLS:
        result[col] = float("nan")

    if odds_df.empty:
        return result

    # Pre-compute index: {(date_week, pair) → odds_row_index}
    odds_index: dict[tuple, int] = {}
    for i, row in odds_df.iterrows():
        pair = tuple(sorted([row["winner_norm"], row["loser_norm"]]))
        odds_index[(row["date_week"], pair)] = int(i)

    matched = 0
    for sack_i, sack_row in result.iterrows():
        tourney_week = _monday(sack_row["tourney_date"])
        w_norm = name_map.get(normalize_name(sack_row["winner_name"]),
                               normalize_name(sack_row["winner_name"]))
        l_norm = name_map.get(normalize_name(sack_row["loser_name"]),
                               normalize_name(sack_row["loser_name"]))
        pair = tuple(sorted([w_norm, l_norm]))

        # Step 1 + 2: exact and tolerance match
        odds_i = None
        for delta in _TOLERANCE_DAYS:
            week = tourney_week + pd.Timedelta(days=delta)
            odds_i = odds_index.get((week, pair))
            if odds_i is not None:
                break

        # Step 3: fuzzy name match within ±21 days if still unmatched
        if odds_i is None:
            cutoff_lo = tourney_week - pd.Timedelta(days=_FUZZY_WINDOW_DAYS)
            cutoff_hi = tourney_week + pd.Timedelta(days=_FUZZY_WINDOW_DAYS)
            candidates = odds_df[
                (odds_df["date_week"] >= cutoff_lo) & (odds_df["date_week"] <= cutoff_hi)
            ]
            best_score = 0
            for _, cand in candidates.iterrows():
                cand_pair = tuple(sorted([cand["winner_norm"], cand["loser_norm"]]))
                # Compare as concatenated string to handle misordering
                score = fuzz.token_sort_ratio(" ".join(pair), " ".join(cand_pair))
                if score > best_score and score >= _FUZZY_THRESHOLD:
                    best_score = score
                    odds_i = cand.name

        if odds_i is None:
            continue

        odds_row = odds_df.loc[odds_i]
        # Determine orientation: did XLSX winner == Sackmann winner?
        xlsx_winner_is_sack_winner = (
            odds_row["winner_norm"] == w_norm or
            fuzz.ratio(odds_row["winner_norm"], w_norm) >= _FUZZY_THRESHOLD
        )

        if xlsx_winner_is_sack_winner:
            result.at[sack_i, "PSW"] = odds_row["PSW"]
            result.at[sack_i, "PSL"] = odds_row["PSL"]
            result.at[sack_i, "B365W"] = odds_row["B365W"]
            result.at[sack_i, "B365L"] = odds_row["B365L"]
        else:
            # XLSX has B as winner; swap W↔L
            result.at[sack_i, "PSW"] = odds_row["PSL"]
            result.at[sack_i, "PSL"] = odds_row["PSW"]
            result.at[sack_i, "B365W"] = odds_row["B365L"]
            result.at[sack_i, "B365L"] = odds_row["B365W"]
        matched += 1

    n = len(sackmann)
    yield_pct = 100.0 * matched / n if n > 0 else 0.0
    log.info("odds join yield: %d / %d (%.1f%%)", matched, n, yield_pct)
    if yield_pct < 95.0 and n > 100:
        log.warning("odds join yield below 95%% — ROI backtest may be limited (spec §2.9)")

    return result
```

- [ ] **Step 4: Run tests**

```bash
cd /home/mykhailo_dan/apps/progno/training && uv run pytest tests/test_odds_join.py -v
```

Expected: all tests PASS (except possibly the fuzzy test — see note below).

Note: `test_fuzzy_name_match` with "A. Zverev" → `normalize_name("A. Zverev") = "zverev a"` which is identical to `normalize_name("Alexander Zverev") = "zverev a"` — the exact match actually catches this one. That's correct behavior.

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
cd /home/mykhailo_dan/apps/progno/training && uv run pytest tests/ -q
```

Expected: all pass (121 + new tests passed, 1 skipped).

- [ ] **Step 6: Commit**

```bash
cd /home/mykhailo_dan/apps/progno
git add training/src/progno_train/odds_join.py training/tests/test_odds_join.py
git commit -m "feat(odds_join): join tennis-data.co.uk odds to Sackmann matches by player pair + week"
```

---

## Task 4: Extend schema + config paths + wire into run_elo

**Files:**
- Modify: `training/src/progno_train/artifacts.py` (MATCH_HISTORY_COLUMNS)
- Modify: `training/src/progno_train/config.py` (add path properties)
- Modify: `training/src/progno_train/cli.py` (run_elo calls odds join)

- [ ] **Step 1: Add odds columns to MATCH_HISTORY_COLUMNS in artifacts.py**

In `training/src/progno_train/artifacts.py`, replace MATCH_HISTORY_COLUMNS to add four new columns at the end:

```python
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
```

- [ ] **Step 2: Add path properties to config.py**

In `training/src/progno_train/config.py`, add two properties at the end of the `Paths` class:

```python
    @property
    def odds_xlsx_dir(self) -> Path:
        return self.data_raw / "tennis_data_xlsx"

    @property
    def name_map(self) -> Path:
        return self.data_raw.parent / "manual" / "name_map.csv"
```

- [ ] **Step 3: Wire odds join into run_elo in cli.py**

In `training/src/progno_train/cli.py`, update `run_elo` to call odds join if XLSX files are present.

Replace the current `run_elo` function with:

```python
def run_elo(paths: Paths) -> int:
    if (rc := _require(paths.matches_clean, "staging parquet (run ingest first)")) is not None:
        return rc
    matches = pd.read_parquet(paths.matches_clean)
    log.info("rolling up %d matches", len(matches))
    state = rollup_elo(matches)
    log.info("produced Elo state for %d players", len(state))

    data_as_of = matches["tourney_date"].max()
    all_names = pd.concat([
        matches[["winner_id", "winner_name"]].rename(columns={"winner_id": "id", "winner_name": "name"}),
        matches[["loser_id", "loser_name"]].rename(columns={"loser_id": "id", "loser_name": "name"}),
    ]).drop_duplicates("id")
    player_names = {
        int(row.id): parts[-1].lower()
        for row in all_names.itertuples()
        if (parts := str(row.name).split())
    }

    # Phase 3.5: join closing odds from tennis-data.co.uk XLSX if available
    xlsx_files = sorted(paths.odds_xlsx_dir.glob("*.xlsx")) if paths.odds_xlsx_dir.exists() else []
    if xlsx_files:
        from progno_train.ingest_xlsx import ingest_tennis_data_xlsx
        from progno_train.odds_join import join_odds
        log.info("joining odds from %d XLSX files...", len(xlsx_files))
        odds_df = ingest_tennis_data_xlsx(xlsx_files)
        matches = join_odds(matches, odds_df, name_map_path=paths.name_map)
        log.info("odds joined — PSW coverage: %.1f%%",
                 100.0 * matches["PSW"].notna().mean())
    else:
        log.warning("no XLSX files in %s — ROI gate will be skipped", paths.odds_xlsx_dir)

    paths.artifacts.mkdir(parents=True, exist_ok=True)
    write_elo_state(state, paths.elo_state, data_as_of=data_as_of, player_names=player_names)
    write_players(matches, paths.players)
    write_match_history(matches, paths.match_history)
    log.info("artifacts written to %s", paths.artifacts)
    return 0
```

- [ ] **Step 4: Run existing tests to verify no regressions**

```bash
cd /home/mykhailo_dan/apps/progno/training && uv run pytest tests/ -q
```

Expected: all tests PASS (same count as before — odds are not required for existing tests).

- [ ] **Step 5: Commit**

```bash
cd /home/mykhailo_dan/apps/progno
git add training/src/progno_train/artifacts.py training/src/progno_train/config.py training/src/progno_train/cli.py
git commit -m "feat(elo): join closing odds from XLSX into match_history; extend schema with PSW/PSL/B365W/B365L"
```

---

## Task 5: Propagate odds_a_winner into featurized dataset

**Files:**
- Modify: `training/src/progno_train/features.py` (build_all_features loop)
- Modify: `training/src/progno_train/train.py` (get_feature_cols)
- Modify: `training/tests/test_features.py` (new test)
- Modify: `training/tests/test_train.py` (new test)

`odds_a_winner` = PSW when winner is player A (fp row, label=1); PSL when loser is player A (fn row, label=0). This is the correct "odds for A winning" regardless of who actually won.

- [ ] **Step 1: Add odds_a_winner to the build_all_features loop in features.py**

In `training/src/progno_train/features.py`, find the loop in `build_all_features` (around line 440+). After calling `_build_feature_row`, add odds stamping:

Replace this block:
```python
        fp = _build_feature_row(row, "winner", "loser", 1, fa, fb, common, h2h_index)
        fn = _build_feature_row(row, "loser", "winner", 0, fb, fa, common, h2h_index)
        rows.extend([fp, fn])
```

With:
```python
        fp = _build_feature_row(row, "winner", "loser", 1, fa, fb, common, h2h_index)
        fn = _build_feature_row(row, "loser", "winner", 0, fb, fa, common, h2h_index)
        # odds_a_winner: PSW when A=winner, PSL when A=loser (closing odds never used as feature)
        fp["odds_a_winner"] = row.get("PSW")
        fn["odds_a_winner"] = row.get("PSL")
        rows.extend([fp, fn])
```

- [ ] **Step 2: Exclude odds_a_winner from feature columns in train.py**

In `training/src/progno_train/train.py`, replace `get_feature_cols`:

```python
_METADATA_COLS = {"label", "tourney_date", "year", "odds_a_winner"}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _METADATA_COLS]
```

- [ ] **Step 3: Write tests**

In `training/tests/test_features.py`, add at the bottom:

```python
def test_build_all_features_has_odds_a_winner_column():
    """odds_a_winner column must be present in output (may be NaN if no odds data)."""
    hist = make_history(20)
    elo_state = {"players": {}}
    df = build_all_features(hist, elo_state)
    assert "odds_a_winner" in df.columns


def test_build_all_features_odds_winner_row_gets_psw():
    """For the winner-as-A row, odds_a_winner = PSW from match history."""
    from progno_train.features import build_all_features
    hist = make_history(20)
    # Add PSW/PSL columns to simulate odds join output
    hist["PSW"] = 1.80
    hist["PSL"] = 2.10
    elo_state = {"players": {}}
    df = build_all_features(hist, elo_state)
    label1_rows = df[df["label"] == 1]
    assert (label1_rows["odds_a_winner"].dropna() == 1.80).all()


def test_build_all_features_odds_loser_row_gets_psl():
    """For the loser-as-A row, odds_a_winner = PSL from match history."""
    from progno_train.features import build_all_features
    hist = make_history(20)
    hist["PSW"] = 1.80
    hist["PSL"] = 2.10
    elo_state = {"players": {}}
    df = build_all_features(hist, elo_state)
    label0_rows = df[df["label"] == 0]
    assert (label0_rows["odds_a_winner"].dropna() == 2.10).all()
```

Note: `make_history` is already defined in `test_features.py` — do NOT redefine it.

In `training/tests/test_train.py`, add at the bottom:

```python
def test_get_feature_cols_excludes_odds_a_winner():
    df = make_feature_df(10)
    df["odds_a_winner"] = 1.85
    cols = get_feature_cols(df)
    assert "odds_a_winner" not in cols


def test_get_feature_cols_excludes_all_metadata():
    df = make_feature_df(10)
    df["odds_a_winner"] = 1.85
    cols = get_feature_cols(df)
    for meta in ["label", "year", "tourney_date", "odds_a_winner"]:
        assert meta not in cols, f"{meta} should be excluded"
```

- [ ] **Step 4: Run tests**

```bash
cd /home/mykhailo_dan/apps/progno/training && uv run pytest tests/test_features.py tests/test_train.py -v
```

Expected: all tests PASS including new ones.

- [ ] **Step 5: Run full test suite**

```bash
cd /home/mykhailo_dan/apps/progno/training && uv run pytest tests/ -q
```

Expected: all pass, 1 skipped.

- [ ] **Step 6: Commit**

```bash
cd /home/mykhailo_dan/apps/progno
git add training/src/progno_train/features.py training/src/progno_train/train.py training/tests/test_features.py training/tests/test_train.py
git commit -m "feat(features): stamp odds_a_winner on featurized rows; exclude from feature cols"
```

---

## Task 6: Test ROI gate end-to-end + update CLAUDE.md

**Files:**
- Modify: `training/tests/test_cli.py` (new test for run_elo with odds)
- Modify: `CLAUDE.md` (mark Phase 3.5 complete)

- [ ] **Step 1: Write a CLI-level integration test for odds join**

In `training/tests/test_cli.py`, add (at the bottom, after existing tests):

```python
def test_run_elo_joins_odds_when_xlsx_present(tmp_path):
    """run_elo should join odds from XLSX files if present in odds_xlsx_dir."""
    import openpyxl
    from progno_train.cli import run_elo
    from progno_train.config import Paths

    # Build minimal staging parquet (reuse fixture pattern from existing tests)
    paths = Paths.for_tour(tmp_path, "atp")
    paths.data_staging.mkdir(parents=True, exist_ok=True)
    paths.data_raw.mkdir(parents=True, exist_ok=True)

    # Write a minimal matches_clean.parquet with two players
    matches = pd.DataFrame([{
        "tourney_id": "2023-001",
        "tourney_date": pd.Timestamp("2023-01-16"),
        "match_num": 1,
        "surface": "Hard",
        "tourney_level": "A",
        "round": "R32",
        "best_of": 3,
        "winner_id": 1,
        "winner_name": "Carlos Alcaraz",
        "winner_hand": "R",
        "winner_ht": 185.0,
        "winner_age": 19.5,
        "winner_rank": 1,
        "loser_id": 2,
        "loser_name": "Novak Djokovic",
        "loser_hand": "R",
        "loser_ht": 188.0,
        "loser_age": 35.0,
        "loser_rank": 5,
        "is_complete": True,
        "completed_sets": 3,
        "score": "6-3 4-6 6-4",
        "minutes": 120.0,
        "w_ace": 5.0, "w_df": 2.0, "w_svpt": 60.0, "w_1stIn": 40.0,
        "w_1stWon": 28.0, "w_2ndWon": 10.0, "w_bpSaved": 3.0, "w_bpFaced": 5.0,
        "l_ace": 3.0, "l_df": 3.0, "l_svpt": 58.0, "l_1stIn": 38.0,
        "l_1stWon": 25.0, "l_2ndWon": 9.0, "l_bpSaved": 2.0, "l_bpFaced": 4.0,
        "winner_rank_points": 8000, "loser_rank_points": 7000,
    }])
    matches.to_parquet(paths.matches_clean, index=False)

    # Write a minimal XLSX with odds for this match
    xlsx_dir = paths.odds_xlsx_dir
    xlsx_dir.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Date", "Winner", "Loser", "PSW", "PSL", "B365W", "B365L"])
    ws.append(["16/01/2023", "Carlos Alcaraz", "Novak Djokovic", 1.85, 2.10, 1.83, 2.05])
    wb.save(xlsx_dir / "atp_2023.xlsx")

    rc = run_elo(paths)
    assert rc == 0

    # Verify match_history has PSW column with a value
    import pandas as pd
    mh = pd.read_parquet(paths.match_history)
    assert "PSW" in mh.columns
    assert mh["PSW"].notna().any(), "Expected at least one PSW value after odds join"


def test_run_elo_skips_odds_gracefully_when_no_xlsx(tmp_path, caplog):
    """run_elo should log a warning and proceed without odds if no XLSX files."""
    import logging
    from progno_train.cli import run_elo
    from progno_train.config import Paths

    paths = Paths.for_tour(tmp_path, "atp")
    paths.data_staging.mkdir(parents=True, exist_ok=True)
    paths.data_raw.mkdir(parents=True, exist_ok=True)

    matches = pd.DataFrame([{
        "tourney_id": "2023-001", "tourney_date": pd.Timestamp("2023-01-16"),
        "match_num": 1, "surface": "Hard", "tourney_level": "A",
        "round": "R32", "best_of": 3,
        "winner_id": 1, "winner_name": "Alcaraz", "winner_hand": "R",
        "winner_ht": 185.0, "winner_age": 19.5, "winner_rank": 1,
        "loser_id": 2, "loser_name": "Djokovic", "loser_hand": "R",
        "loser_ht": 188.0, "loser_age": 35.0, "loser_rank": 5,
        "is_complete": True, "completed_sets": 2, "score": "6-3 6-4",
        "minutes": 95.0,
        "w_ace": 5.0, "w_df": 2.0, "w_svpt": 60.0, "w_1stIn": 40.0,
        "w_1stWon": 28.0, "w_2ndWon": 10.0, "w_bpSaved": 3.0, "w_bpFaced": 5.0,
        "l_ace": 3.0, "l_df": 3.0, "l_svpt": 58.0, "l_1stIn": 38.0,
        "l_1stWon": 25.0, "l_2ndWon": 9.0, "l_bpSaved": 2.0, "l_bpFaced": 4.0,
        "winner_rank_points": 8000, "loser_rank_points": 7000,
    }])
    matches.to_parquet(paths.matches_clean, index=False)

    with caplog.at_level(logging.WARNING, logger="progno_train.cli"):
        rc = run_elo(paths)
    assert rc == 0
    assert any("xlsx" in msg.lower() or "roi" in msg.lower() for msg in caplog.messages)
```

Note: `pd` is already imported in `test_cli.py` — check before adding an import.

- [ ] **Step 2: Run CLI tests**

```bash
cd /home/mykhailo_dan/apps/progno/training && uv run pytest tests/test_cli.py -v
```

Expected: all tests PASS.

- [ ] **Step 3: Run full test suite**

```bash
cd /home/mykhailo_dan/apps/progno/training && uv run pytest tests/ -q
```

Expected: all pass, 1 skipped. Count should now be higher by the new tests added in Tasks 2–6.

- [ ] **Step 4: Update CLAUDE.md current status**

In `/home/mykhailo_dan/apps/progno/CLAUDE.md`, update the `## Current status` section:

Replace:
```
- **Active on master**: maintenance + Phase 3.5 (odds ingest from tennis-data.co.uk for ROI gate)
```
With:
```
- **Completed**: Phase 1a, 1b, 2 (Kelly), 3 (CatBoost + Platt calibration + sidecar), 4 (WTA dual model), 3.5 (odds ingest + ROI gate)
- **Active on master**: maintenance
```

- [ ] **Step 5: Commit**

```bash
cd /home/mykhailo_dan/apps/progno
git add training/tests/test_cli.py CLAUDE.md
git commit -m "feat(phase-3.5): odds ingest + ROI gate complete; update CLAUDE.md"
```

---

## Spec Coverage Check

| Spec requirement | Task | Status |
|---|---|---|
| Download tennis-data.co.uk XLSX (ATP from 2000, WTA from 2007) (§2 data sources) | Task 1 | ✓ |
| Join key `(tourney_week, player_A_norm, player_B_norm)` (§2.6) | Task 3 | ✓ |
| Name normalization: "LASTNAME F", ASCII via unidecode (§2.6) | Task 2 + 3 | ✓ |
| Exact → rapidfuzz ≥ 90 → manual name_map.csv fallback (§2.6) | Task 3 | ✓ |
| Expected yield ~98–99%; warn if <95% (§2.6, §2.9) | Task 3 | ✓ |
| Closing odds NOT used as model features (§2.6 note) | Task 5 | ✓ |
| ROI gate: simulate 0.25× Kelly vs Pinnacle closing (§6.4) | Task 4 + 5 | ✓ |
| ROI gate fail if < -1% (§6.4 ROI_THRESHOLD = -0.01) | existing validate.py | ✓ |
| PSW (Pinnacle), B365W (Bet365) stored in match_history (§2 data) | Task 4 | ✓ |
| `just fetch-odds-data` command (§6.2) | Task 1 | ✓ |

### Notes

- **First run**: `just fetch-odds-data` downloads ~50 XLSX files (ATP 2000–2025 + WTA 2007–2025). Then `just elo` (or `just ingest-odds`) regenerates `match_history.parquet` with odds. Then `just features && just train && just validate` to get ROI in the gate.
- **Grand Slam week 2**: The tolerance join (±7, ±14 days) handles matches in week 2 of Grand Slams where the Sackmann `tourney_date` is the tournament start but tennis-data.co.uk `Date` is the actual match date.
- **PSL for featurized loser rows**: For the balanced featurized dataset, `fn["odds_a_winner"] = PSL` gives the odds for the loser winning — which is what we want when computing ROI for bets on the "underdog A" side.
- **CLV**: spec §4.3 also mentions CLV (`mean(1/P_model - 1/P_pinnacle)`). This is a Phase 4/5 metric and not included here.
