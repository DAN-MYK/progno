# Phase 1a — Python Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Python pipeline that ingests Jeff Sackmann's ATP match CSVs, computes surface-specific Elo ratings using the FiveThirtyEight K-factor formula, and writes the artifact files (`elo_state.json`, `players.parquet`, `match_history.parquet`) that the Tauri app will consume in Phase 1b.

**Architecture:** Pure Python 3.12 package under `training/`, managed with `uv`. Data flows: Sackmann CSVs → staging parquet (cleaned) → artifacts (Elo snapshot + players dir + match history for later rolling-form lookup). A `justfile` at repo root orchestrates commands. No Tauri code in this phase; Phase 1b consumes the artifacts.

**Tech Stack:** Python 3.12, `uv` for deps, `pandas`, `pyarrow`, `pytest`, `ruff`. `just` for commands. Git for version control.

**Spec reference:** `docs/superpowers/specs/2026-04-22-tennis-prediction-app-design.md` — sections 2.1–2.3, 2.7, 3.1, 6.2, 6.5.1–6.5.2.

**Non-negotiables from spec:**
- Retirements (score contains `RET`/`W/O`/`DEF`) excluded from training labels but kept in history.
- `random_seed=42` anywhere randomness is used.
- Elo snapshot must be deterministic: same input CSV → bit-identical `elo_state.json`.
- Surface-specific ratings are **separate** (not blended): `elo_overall`, `elo_hard`, `elo_clay`, `elo_grass`. Carpet updates only `elo_overall`.
- K-factor formula: `K(n) = 250 / (n + 5)^0.4` with multiplicative context factors (tourney level, round, best_of).

---

### Task 0: Initialise repository and root tooling

**Files:**
- Create: `.gitignore`
- Create: `justfile`
- Create: `README.md` (minimal)

- [ ] **Step 1: Initialise git**

Run:
```bash
cd /home/mykhailo_dan/apps/progno
git init
git config user.name "Progno Developer"
git config user.email "dunmisha@gmail.com"
```

Expected: `Initialized empty Git repository in /home/mykhailo_dan/apps/progno/.git/`.

- [ ] **Step 2: Write `.gitignore`**

Create `/home/mykhailo_dan/apps/progno/.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.pytest_cache/
.ruff_cache/

# Data — raw and staging are derived, don't commit
training/data/raw/
training/data/staging/
training/data/manual/name_map.csv

# Artifacts — versioned on disk, not in git
training/artifacts/v*/
training/artifacts/current

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db

# Rust (for Phase 1b, safe to include now)
app/src-tauri/target/
app/node_modules/
app/dist/
```

- [ ] **Step 3: Write `justfile`**

Create `/home/mykhailo_dan/apps/progno/justfile`:

```just
default:
    @just --list

# --- Phase 1a targets ---
update-data:
    cd training && uv run python -m progno_train.cli update_data

ingest:
    cd training && uv run python -m progno_train.cli ingest

elo:
    cd training && uv run python -m progno_train.cli elo

publish version:
    cd training && uv run python -m progno_train.cli publish {{version}}

# --- Dev helpers ---
test:
    cd training && uv run pytest -v

fmt:
    cd training && uv run ruff format .
    cd training && uv run ruff check --fix .

check:
    cd training && uv run ruff check .
    cd training && uv run pytest -v
```

- [ ] **Step 4: Write `README.md`**

Create `/home/mykhailo_dan/apps/progno/README.md`:

```markdown
# Progno

Personal desktop app for pre-match ATP tennis winner prediction.

See:
- Architecture spec: `docs/superpowers/specs/2026-04-22-tennis-prediction-app-design.md`
- AI workflow: `CLAUDE.md`
- Implementation plans: `docs/superpowers/plans/`

Phase 1a (this phase): Python training pipeline. Produces Elo artifacts consumed by the Tauri app in Phase 1b.
```

- [ ] **Step 5: Verify `just` is installed, else instruct user**

Run: `just --version`

Expected: version ≥ 1.16.0. If missing, stop and tell the user to install via `cargo install just` or their package manager.

- [ ] **Step 6: Commit**

```bash
cd /home/mykhailo_dan/apps/progno
git add .gitignore justfile README.md CLAUDE.md agents/ docs/
git commit -m "chore: initialise repo, add root tooling (justfile, gitignore)"
```

Expected: commit created, clean working tree.

---

### Task 1: Python project scaffold with `uv`

**Files:**
- Create: `training/pyproject.toml`
- Create: `training/src/progno_train/__init__.py`
- Create: `training/src/progno_train/cli.py`
- Create: `training/tests/__init__.py`
- Create: `training/tests/test_smoke.py`
- Create: `training/ruff.toml`

- [ ] **Step 1: Create `pyproject.toml`**

Create `/home/mykhailo_dan/apps/progno/training/pyproject.toml`:

```toml
[project]
name = "progno-train"
version = "0.1.0"
description = "Training pipeline for Progno tennis prediction"
requires-python = ">=3.12"
dependencies = [
    "pandas>=2.2",
    "pyarrow>=15.0",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.4",
]

[project.scripts]
progno-train = "progno_train.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/progno_train"]
```

- [ ] **Step 2: Create `ruff.toml`**

Create `/home/mykhailo_dan/apps/progno/training/ruff.toml`:

```toml
line-length = 100
target-version = "py312"

[lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = []

[format]
quote-style = "double"
```

- [ ] **Step 3: Create package files**

Create `/home/mykhailo_dan/apps/progno/training/src/progno_train/__init__.py` (empty):

```python
```

Create `/home/mykhailo_dan/apps/progno/training/src/progno_train/cli.py`:

```python
"""CLI entry points for the training pipeline."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="progno-train")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("update_data", help="Fetch latest Sackmann data")
    sub.add_parser("ingest", help="Ingest CSVs to staging parquet")
    sub.add_parser("elo", help="Compute Elo state snapshot")
    publish = sub.add_parser("publish", help="Publish artifacts to app-data")
    publish.add_argument("version")

    args = parser.parse_args()
    if args.command == "update_data":
        print("not implemented yet")
        return 1
    if args.command == "ingest":
        print("not implemented yet")
        return 1
    if args.command == "elo":
        print("not implemented yet")
        return 1
    if args.command == "publish":
        print(f"not implemented yet (version={args.version})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Create `/home/mykhailo_dan/apps/progno/training/tests/__init__.py` (empty).

- [ ] **Step 4: Write smoke test**

Create `/home/mykhailo_dan/apps/progno/training/tests/test_smoke.py`:

```python
def test_package_imports() -> None:
    import progno_train
    import progno_train.cli

    assert progno_train is not None
    assert progno_train.cli.main is not None
```

- [ ] **Step 5: Install dependencies with uv**

Run:
```bash
cd /home/mykhailo_dan/apps/progno/training
uv sync --extra dev
```

Expected: `.venv/` created, `uv.lock` generated, no errors.

If `uv` is not installed, stop and tell the user to install per https://docs.astral.sh/uv/.

- [ ] **Step 6: Run the smoke test**

Run:
```bash
cd /home/mykhailo_dan/apps/progno/training
uv run pytest -v
```

Expected: `test_smoke.py::test_package_imports PASSED`.

- [ ] **Step 7: Verify ruff passes**

Run:
```bash
cd /home/mykhailo_dan/apps/progno/training
uv run ruff check .
uv run ruff format --check .
```

Expected: no issues.

- [ ] **Step 8: Commit**

```bash
cd /home/mykhailo_dan/apps/progno
git add training/
git commit -m "feat(training): scaffold Python package with uv + ruff + pytest"
```

(This is acceptable because `training/` only contains new files at this point. The `.gitignore` from Task 0 excludes `__pycache__`, `.venv/`, and data directories.)

---

### Task 2: Score parser — detect retirements and parse sets

**Files:**
- Create: `training/src/progno_train/score.py`
- Create: `training/tests/test_score.py`

- [ ] **Step 1: Write failing tests**

Create `/home/mykhailo_dan/apps/progno/training/tests/test_score.py`:

```python
from __future__ import annotations

import pytest

from progno_train.score import ParsedScore, parse_score


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "6-4 3-6 7-5",
            ParsedScore(is_complete=True, completed_sets=3, winner_set_count=2, loser_set_count=1),
        ),
        (
            "6-4 6-3",
            ParsedScore(is_complete=True, completed_sets=2, winner_set_count=2, loser_set_count=0),
        ),
        (
            "7-6(4) 6-7(5) 6-4",
            ParsedScore(is_complete=True, completed_sets=3, winner_set_count=2, loser_set_count=1),
        ),
        (
            "6-4 3-6 RET",
            ParsedScore(is_complete=False, completed_sets=2, winner_set_count=1, loser_set_count=1),
        ),
        (
            "W/O",
            ParsedScore(is_complete=False, completed_sets=0, winner_set_count=0, loser_set_count=0),
        ),
        (
            "6-4 2-3 RET",
            ParsedScore(is_complete=False, completed_sets=1, winner_set_count=1, loser_set_count=0),
        ),
        (
            "6-3 3-6 6-2 4-6 8-6",
            ParsedScore(is_complete=True, completed_sets=5, winner_set_count=3, loser_set_count=2),
        ),
    ],
)
def test_parse_score(raw: str, expected: ParsedScore) -> None:
    assert parse_score(raw) == expected


def test_parse_score_none_or_empty() -> None:
    assert parse_score("").is_complete is False
    assert parse_score("").completed_sets == 0


def test_parse_score_def_treated_as_walkover() -> None:
    result = parse_score("DEF")
    assert result.is_complete is False
    assert result.completed_sets == 0
```

- [ ] **Step 2: Run test to confirm failure**

Run:
```bash
cd /home/mykhailo_dan/apps/progno/training
uv run pytest tests/test_score.py -v
```

Expected: all tests FAIL with `ModuleNotFoundError: No module named 'progno_train.score'`.

- [ ] **Step 3: Implement `score.py`**

Create `/home/mykhailo_dan/apps/progno/training/src/progno_train/score.py`:

```python
"""Parse Sackmann score strings into structured data."""

from __future__ import annotations

import re
from dataclasses import dataclass

RETIREMENT_MARKERS = ("RET", "W/O", "DEF", "Def.", "ABN")
SET_PATTERN = re.compile(r"^(\d+)-(\d+)(?:\([^)]*\))?$")


@dataclass(frozen=True)
class ParsedScore:
    is_complete: bool
    completed_sets: int
    winner_set_count: int
    loser_set_count: int


def parse_score(raw: str) -> ParsedScore:
    if not raw or not raw.strip():
        return ParsedScore(False, 0, 0, 0)

    tokens = raw.strip().split()
    tokens_upper = [t.upper() for t in tokens]

    is_retirement = any(t in RETIREMENT_MARKERS for t in tokens_upper)
    set_tokens = [t for t in tokens if SET_PATTERN.match(t)]

    winner_sets = 0
    loser_sets = 0
    completed = 0
    for tok in set_tokens:
        m = SET_PATTERN.match(tok)
        assert m is not None
        w, l = int(m.group(1)), int(m.group(2))
        if w > l:
            winner_sets += 1
        elif l > w:
            loser_sets += 1
        completed += 1

    if is_retirement:
        return ParsedScore(False, completed, winner_sets, loser_sets)

    if completed == 0:
        return ParsedScore(False, 0, 0, 0)

    return ParsedScore(True, completed, winner_sets, loser_sets)
```

- [ ] **Step 4: Run tests to confirm pass**

Run:
```bash
cd /home/mykhailo_dan/apps/progno/training
uv run pytest tests/test_score.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/mykhailo_dan/apps/progno
git add training/src/progno_train/score.py training/tests/test_score.py
git commit -m "feat(training): score parser with retirement/walkover detection"
```

---

### Task 3: Data ingestion — Sackmann CSV → cleaned parquet

**Files:**
- Create: `training/src/progno_train/ingest.py`
- Create: `training/tests/test_ingest.py`
- Create: `training/tests/fixtures/mini_atp_matches.csv`

- [ ] **Step 1: Create test fixture**

Create `/home/mykhailo_dan/apps/progno/training/tests/fixtures/mini_atp_matches.csv`:

```csv
tourney_id,tourney_name,surface,draw_size,tourney_level,tourney_date,match_num,winner_id,winner_seed,winner_entry,winner_name,winner_hand,winner_ht,winner_ioc,winner_age,loser_id,loser_seed,loser_entry,loser_name,loser_hand,loser_ht,loser_ioc,loser_age,score,best_of,round,minutes,w_ace,w_df,w_svpt,w_1stIn,w_1stWon,w_2ndWon,w_SvGms,w_bpSaved,w_bpFaced,l_ace,l_df,l_svpt,l_1stIn,l_1stWon,l_2ndWon,l_SvGms,l_bpSaved,l_bpFaced,winner_rank,winner_rank_points,loser_rank,loser_rank_points
2024-0001,Test Open,Hard,32,A,20240108,1,100001,1,,Alpha Alpha,R,185,USA,25.5,100002,,,Beta Beta,L,180,ESP,24.0,6-4 6-3,3,R32,90,5,2,60,40,28,12,10,2,3,2,1,55,35,20,10,9,3,5,10,500,25,200
2024-0001,Test Open,Hard,32,A,20240108,2,100003,,,Gamma Gamma,R,190,FRA,28.0,100004,,,Delta Delta,R,178,GER,22.0,6-4 3-6 RET,3,R32,75,4,1,50,30,22,10,8,2,3,1,2,48,28,18,8,7,4,6,30,300,40,250
2024-0001,Test Open,Hard,32,A,20240108,3,100001,1,,Alpha Alpha,R,185,USA,25.5,100003,,,Gamma Gamma,R,190,FRA,28.0,W/O,3,R16,,,,,,,,,,,,,,,,,,,10,500,30,300
2024-0001,Test Open,Hard,32,A,20240108,4,100002,,,Beta Beta,L,180,ESP,24.0,100004,,,Delta Delta,R,178,GER,22.0,7-6(4) 6-4,3,R16,98,8,3,70,45,35,15,11,3,4,5,2,65,40,30,12,10,4,7,25,200,40,250
```

- [ ] **Step 2: Write failing tests**

Create `/home/mykhailo_dan/apps/progno/training/tests/test_ingest.py`:

```python
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
```

- [ ] **Step 3: Run test to confirm failure**

Run:
```bash
cd /home/mykhailo_dan/apps/progno/training
uv run pytest tests/test_ingest.py -v
```

Expected: all fail with `ModuleNotFoundError: No module named 'progno_train.ingest'`.

- [ ] **Step 4: Implement `ingest.py`**

Create `/home/mykhailo_dan/apps/progno/training/src/progno_train/ingest.py`:

```python
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

    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
    df["score"] = df["score"].fillna("").astype(str)

    parsed = df["score"].apply(parse_score)
    df["is_complete"] = parsed.apply(lambda p: p.is_complete)
    df["completed_sets"] = parsed.apply(lambda p: p.completed_sets)

    df = df.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    return df
```

- [ ] **Step 5: Run tests to confirm pass**

Run:
```bash
cd /home/mykhailo_dan/apps/progno/training
uv run pytest tests/test_ingest.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/mykhailo_dan/apps/progno
git add training/src/progno_train/ingest.py training/tests/test_ingest.py training/tests/fixtures
git commit -m "feat(training): ingest Sackmann CSVs with completion flags"
```

---

### Task 4: Elo — single-match update function

**Files:**
- Create: `training/src/progno_train/elo.py`
- Create: `training/tests/test_elo.py`

- [ ] **Step 1: Write failing tests**

Create `/home/mykhailo_dan/apps/progno/training/tests/test_elo.py`:

```python
from __future__ import annotations

import math

import pytest

from progno_train.elo import (
    apply_elo_update,
    expected_probability,
    k_factor,
)


def test_expected_prob_equal_ratings() -> None:
    assert expected_probability(1500, 1500) == pytest.approx(0.5)


def test_expected_prob_higher_rating_favoured() -> None:
    p = expected_probability(1700, 1500)
    assert 0.74 < p < 0.77  # classical 200 gap ≈ 0.76


def test_expected_prob_symmetric() -> None:
    p_ab = expected_probability(1600, 1500)
    p_ba = expected_probability(1500, 1600)
    assert p_ab + p_ba == pytest.approx(1.0)


def test_k_factor_formula_538() -> None:
    # K(n) = 250 / (n + 5)^0.4
    assert k_factor(0) == pytest.approx(250 / 5**0.4)
    assert k_factor(100) == pytest.approx(250 / 105**0.4)


def test_k_factor_decreases_with_experience() -> None:
    assert k_factor(0) > k_factor(10) > k_factor(100) > k_factor(1000)


def test_apply_elo_update_equal_ratings_winner_gains_k_half() -> None:
    new_winner, new_loser = apply_elo_update(
        winner_rating=1500, loser_rating=1500, k=32
    )
    assert new_winner == pytest.approx(1516.0)
    assert new_loser == pytest.approx(1484.0)


def test_apply_elo_update_upset_winner_gains_more() -> None:
    # Lower-rated player wins → gains more than they would at parity
    new_winner, new_loser = apply_elo_update(
        winner_rating=1400, loser_rating=1700, k=32
    )
    gain = new_winner - 1400
    assert gain > 16.0


def test_apply_elo_update_preserves_total() -> None:
    # Elo is zero-sum when K is identical
    for wr, lr in [(1500, 1500), (1800, 1400), (1200, 1700)]:
        nw, nl = apply_elo_update(wr, lr, k=32)
        assert (nw + nl) == pytest.approx(wr + lr)


def test_apply_elo_update_k_zero_no_change() -> None:
    nw, nl = apply_elo_update(1500, 1600, k=0)
    assert nw == 1500
    assert nl == 1600


def test_apply_elo_update_returns_floats() -> None:
    nw, nl = apply_elo_update(1500, 1500, k=32)
    assert isinstance(nw, float)
    assert isinstance(nl, float)


def test_k_factor_rejects_negative_n() -> None:
    with pytest.raises(ValueError):
        k_factor(-1)
```

- [ ] **Step 2: Run tests to confirm failure**

Run:
```bash
cd /home/mykhailo_dan/apps/progno/training
uv run pytest tests/test_elo.py -v
```

Expected: all fail with `ModuleNotFoundError: No module named 'progno_train.elo'`.

- [ ] **Step 3: Implement `elo.py` — core functions only**

Create `/home/mykhailo_dan/apps/progno/training/src/progno_train/elo.py`:

```python
"""Elo rating updates (FiveThirtyEight tennis variant).

K-factor formula: K(n) = 250 / (n + 5)^0.4  where n = player's prior match count.

Reference: FiveThirtyEight tennis Elo methodology. The formula is multiplied by
context factors (tournament level, round, best_of) at callsites in `rollup.py`.
"""

from __future__ import annotations


INITIAL_RATING = 1500.0
K_BASE = 250.0
K_OFFSET = 5.0
K_SHAPE = 0.4


def expected_probability(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def k_factor(prior_matches: int) -> float:
    if prior_matches < 0:
        raise ValueError(f"prior_matches must be >= 0, got {prior_matches}")
    return K_BASE / (prior_matches + K_OFFSET) ** K_SHAPE


def apply_elo_update(
    winner_rating: float,
    loser_rating: float,
    k: float,
) -> tuple[float, float]:
    expected_w = expected_probability(winner_rating, loser_rating)
    delta = k * (1.0 - expected_w)
    return winner_rating + delta, loser_rating - delta
```

- [ ] **Step 4: Run tests to confirm pass**

Run:
```bash
cd /home/mykhailo_dan/apps/progno/training
uv run pytest tests/test_elo.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/mykhailo_dan/apps/progno
git add training/src/progno_train/elo.py training/tests/test_elo.py
git commit -m "feat(training): Elo update with 538 K-factor formula"
```

---

### Task 5: Context K-modifiers (tournament level / round / best_of)

**Files:**
- Modify: `training/src/progno_train/elo.py` (add `context_multiplier`)
- Modify: `training/tests/test_elo.py` (add new tests at end)

- [ ] **Step 1: Add failing tests**

Append to `/home/mykhailo_dan/apps/progno/training/tests/test_elo.py`:

```python
from progno_train.elo import context_multiplier  # noqa: E402


def test_context_multiplier_grand_slam_bo5_final() -> None:
    # G (1.0) * F (1.0) * BO5 (1.0) = 1.0
    assert context_multiplier("G", "F", 5) == pytest.approx(1.0)


def test_context_multiplier_masters_early_round_bo3() -> None:
    # M (0.85) * R32 (0.85) * BO3 (0.90) = 0.65025
    assert context_multiplier("M", "R32", 3) == pytest.approx(0.85 * 0.85 * 0.90)


def test_context_multiplier_challenger_qualifier_bo3() -> None:
    # C (0.50) * Q (0.85) * BO3 (0.90)
    assert context_multiplier("C", "Q1", 3) == pytest.approx(0.50 * 0.85 * 0.90)


def test_context_multiplier_unknown_level_falls_back_to_atp() -> None:
    # Unknown level defaults to A (0.75)
    assert context_multiplier("UNKNOWN", "R16", 3) == pytest.approx(0.75 * 0.85 * 0.90)


@pytest.mark.parametrize(
    ("level", "round_", "best_of"),
    [
        ("G", "F", 5),
        ("M", "SF", 3),
        ("A", "QF", 3),
    ],
)
def test_context_multiplier_positive(level: str, round_: str, best_of: int) -> None:
    assert context_multiplier(level, round_, best_of) > 0.0
```

- [ ] **Step 2: Run tests to confirm failure**

Run:
```bash
cd /home/mykhailo_dan/apps/progno/training
uv run pytest tests/test_elo.py::test_context_multiplier_grand_slam_bo5_final -v
```

Expected: FAIL (ImportError on `context_multiplier`).

- [ ] **Step 3: Implement `context_multiplier`**

Append to `/home/mykhailo_dan/apps/progno/training/src/progno_train/elo.py`:

```python


LEVEL_FACTORS = {
    "G": 1.00,   # Grand Slam
    "M": 0.85,   # Masters 1000
    "A": 0.75,   # ATP 250 / 500 (default)
    "F": 0.90,   # Tour Finals
    "D": 0.70,   # Davis Cup
    "C": 0.50,   # Challenger
    "S": 0.40,   # ITF Satellite
}

ROUND_FACTORS = {
    "F": 1.00,
    "SF": 0.95,
    "QF": 0.90,
    "R16": 0.85,
    "R32": 0.85,
    "R64": 0.85,
    "R128": 0.85,
    "RR": 0.90,   # round robin
    "BR": 0.85,   # bronze
    "ER": 0.85,   # early round
}


def _round_factor(round_: str) -> float:
    r = (round_ or "").strip().upper()
    if r in ROUND_FACTORS:
        return ROUND_FACTORS[r]
    if r.startswith("Q"):
        return 0.85
    return 0.85


def context_multiplier(tourney_level: str, round_: str, best_of: int) -> float:
    lf = LEVEL_FACTORS.get((tourney_level or "").strip().upper(), LEVEL_FACTORS["A"])
    rf = _round_factor(round_)
    bo5 = 1.0 if best_of == 5 else 0.90
    return lf * rf * bo5
```

- [ ] **Step 4: Run the new tests**

Run:
```bash
cd /home/mykhailo_dan/apps/progno/training
uv run pytest tests/test_elo.py -v
```

Expected: all tests PASS (existing 11 + new 7 = 18).

- [ ] **Step 5: Commit**

```bash
cd /home/mykhailo_dan/apps/progno
git add training/src/progno_train/elo.py training/tests/test_elo.py
git commit -m "feat(training): Elo context multiplier for level/round/best_of"
```

---

### Task 6: Elo rollup — iterate matches, maintain per-player state

**Files:**
- Create: `training/src/progno_train/rollup.py`
- Create: `training/tests/test_rollup.py`

Surface rule (from spec §3.1): maintain `elo_overall`, `elo_hard`, `elo_clay`, `elo_grass`. Carpet updates only `elo_overall`. Each match updates (a) the overall rating always, (b) the surface-specific rating if surface ∈ {Hard, Clay, Grass}.

- [ ] **Step 1: Write failing tests**

Create `/home/mykhailo_dan/apps/progno/training/tests/test_rollup.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from progno_train.elo import INITIAL_RATING
from progno_train.rollup import PlayerElo, rollup_elo


def _mk_match(
    winner: int,
    loser: int,
    date: str = "2020-01-01",
    surface: str = "Hard",
    level: str = "A",
    round_: str = "R32",
    best_of: int = 3,
    is_complete: bool = True,
    match_num: int = 1,
) -> dict:
    return {
        "tourney_id": f"{date}-T",
        "tourney_date": pd.Timestamp(date),
        "match_num": match_num,
        "surface": surface,
        "tourney_level": level,
        "round": round_,
        "best_of": best_of,
        "winner_id": winner,
        "loser_id": loser,
        "is_complete": is_complete,
    }


def test_rollup_empty_returns_empty_state() -> None:
    df = pd.DataFrame(columns=["tourney_date", "match_num"])
    state = rollup_elo(df)
    assert state == {}


def test_rollup_single_match_updates_both_players_overall_and_surface() -> None:
    df = pd.DataFrame([_mk_match(1, 2, surface="Clay")])
    state = rollup_elo(df)
    assert 1 in state
    assert 2 in state
    assert state[1].elo_overall > INITIAL_RATING
    assert state[2].elo_overall < INITIAL_RATING
    assert state[1].elo_clay > INITIAL_RATING
    assert state[1].elo_hard == INITIAL_RATING
    assert state[1].elo_grass == INITIAL_RATING
    assert state[1].matches_played == 1
    assert state[2].matches_played == 1


def test_rollup_carpet_updates_only_overall() -> None:
    df = pd.DataFrame([_mk_match(1, 2, surface="Carpet")])
    state = rollup_elo(df)
    assert state[1].elo_overall > INITIAL_RATING
    assert state[1].elo_hard == INITIAL_RATING
    assert state[1].elo_clay == INITIAL_RATING
    assert state[1].elo_grass == INITIAL_RATING


def test_rollup_skips_incomplete_matches() -> None:
    df = pd.DataFrame(
        [
            _mk_match(1, 2, is_complete=False, match_num=1),
            _mk_match(3, 4, is_complete=True, match_num=2),
        ]
    )
    state = rollup_elo(df)
    assert 1 not in state
    assert 2 not in state
    assert 3 in state
    assert 4 in state


def test_rollup_unknown_surface_updates_only_overall() -> None:
    df = pd.DataFrame([_mk_match(1, 2, surface="")])
    state = rollup_elo(df)
    assert state[1].elo_overall > INITIAL_RATING
    assert state[1].elo_hard == INITIAL_RATING


def test_rollup_order_matters_matches_sorted_by_date_then_match_num() -> None:
    # If rollup is processing out of order we'd see different ratings
    df = pd.DataFrame(
        [
            _mk_match(1, 2, date="2020-01-01", match_num=1),
            _mk_match(1, 2, date="2020-01-01", match_num=2),
            _mk_match(1, 2, date="2020-01-01", match_num=3),
        ]
    )
    state = rollup_elo(df)
    assert state[1].matches_played == 3
    assert state[2].matches_played == 3


def test_rollup_deterministic() -> None:
    df = pd.DataFrame(
        [
            _mk_match(1, 2, match_num=1),
            _mk_match(3, 4, match_num=2),
            _mk_match(1, 3, match_num=3),
        ]
    )
    a = rollup_elo(df)
    b = rollup_elo(df)
    for pid in a:
        assert a[pid].elo_overall == pytest.approx(b[pid].elo_overall)
        assert a[pid].matches_played == b[pid].matches_played
```

- [ ] **Step 2: Run tests to confirm failure**

Run:
```bash
cd /home/mykhailo_dan/apps/progno/training
uv run pytest tests/test_rollup.py -v
```

Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement `rollup.py`**

Create `/home/mykhailo_dan/apps/progno/training/src/progno_train/rollup.py`:

```python
"""Roll up matches into per-player Elo state (overall + surface-specific)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from progno_train.elo import (
    INITIAL_RATING,
    apply_elo_update,
    context_multiplier,
    k_factor,
)

TRACKED_SURFACES = {"Hard", "Clay", "Grass"}


@dataclass
class PlayerElo:
    player_id: int
    elo_overall: float = INITIAL_RATING
    elo_hard: float = INITIAL_RATING
    elo_clay: float = INITIAL_RATING
    elo_grass: float = INITIAL_RATING
    matches_played: int = 0
    matches_played_hard: int = 0
    matches_played_clay: int = 0
    matches_played_grass: int = 0


def _get_or_init(state: dict[int, PlayerElo], pid: int) -> PlayerElo:
    if pid not in state:
        state[pid] = PlayerElo(player_id=pid)
    return state[pid]


def _update_surface(
    winner: PlayerElo,
    loser: PlayerElo,
    surface: str,
    k_with_context: float,
) -> None:
    if surface not in TRACKED_SURFACES:
        return
    attr = f"elo_{surface.lower()}"
    played_attr = f"matches_played_{surface.lower()}"
    new_w, new_l = apply_elo_update(
        getattr(winner, attr),
        getattr(loser, attr),
        k=k_with_context,
    )
    setattr(winner, attr, new_w)
    setattr(loser, attr, new_l)
    setattr(winner, played_attr, getattr(winner, played_attr) + 1)
    setattr(loser, played_attr, getattr(loser, played_attr) + 1)


def rollup_elo(matches: pd.DataFrame) -> dict[int, PlayerElo]:
    if matches.empty:
        return {}

    required = {
        "tourney_date",
        "match_num",
        "winner_id",
        "loser_id",
        "surface",
        "tourney_level",
        "round",
        "best_of",
        "is_complete",
    }
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches DataFrame missing columns: {missing}")

    df = matches.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    state: dict[int, PlayerElo] = {}

    for row in df.itertuples(index=False):
        if not row.is_complete:
            continue

        winner = _get_or_init(state, int(row.winner_id))
        loser = _get_or_init(state, int(row.loser_id))

        ctx = context_multiplier(row.tourney_level, row.round, int(row.best_of))
        # K uses overall match count (538 convention); min of winner/loser is conservative
        k = min(k_factor(winner.matches_played), k_factor(loser.matches_played)) * ctx

        new_w, new_l = apply_elo_update(winner.elo_overall, loser.elo_overall, k=k)
        winner.elo_overall = new_w
        loser.elo_overall = new_l
        winner.matches_played += 1
        loser.matches_played += 1

        _update_surface(winner, loser, row.surface, k_with_context=k)

    return state
```

- [ ] **Step 4: Run tests to confirm pass**

Run:
```bash
cd /home/mykhailo_dan/apps/progno/training
uv run pytest tests/test_rollup.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/mykhailo_dan/apps/progno
git add training/src/progno_train/rollup.py training/tests/test_rollup.py
git commit -m "feat(training): roll up matches into per-player Elo state"
```

---

### Task 7: Artifact writer — Elo snapshot + players + history parquet

**Files:**
- Create: `training/src/progno_train/artifacts.py`
- Create: `training/tests/test_artifacts.py`

- [ ] **Step 1: Write failing tests**

Create `/home/mykhailo_dan/apps/progno/training/tests/test_artifacts.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from progno_train.artifacts import (
    write_elo_state,
    write_match_history,
    write_players,
)
from progno_train.rollup import PlayerElo


def test_write_elo_state_produces_expected_json(tmp_path: Path) -> None:
    state = {
        1: PlayerElo(player_id=1, elo_overall=1600.0, elo_hard=1650.0, matches_played=10),
        2: PlayerElo(player_id=2, elo_overall=1400.0, elo_clay=1450.0, matches_played=8),
    }
    out = tmp_path / "elo_state.json"
    write_elo_state(state, out, data_as_of=pd.Timestamp("2026-04-22"))

    content = json.loads(out.read_text())
    assert content["data_as_of"] == "2026-04-22"
    assert "players" in content
    assert str(1) in content["players"]
    p1 = content["players"]["1"]
    assert p1["elo_overall"] == 1600.0
    assert p1["elo_hard"] == 1650.0
    assert p1["matches_played"] == 10


def test_write_elo_state_is_deterministic(tmp_path: Path) -> None:
    state = {
        2: PlayerElo(player_id=2, elo_overall=1400.0),
        1: PlayerElo(player_id=1, elo_overall=1600.0),
    }
    out1 = tmp_path / "a.json"
    out2 = tmp_path / "b.json"
    write_elo_state(state, out1, data_as_of=pd.Timestamp("2026-04-22"))
    write_elo_state(state, out2, data_as_of=pd.Timestamp("2026-04-22"))
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
```

- [ ] **Step 2: Run tests to confirm failure**

Run:
```bash
cd /home/mykhailo_dan/apps/progno/training
uv run pytest tests/test_artifacts.py -v
```

Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement `artifacts.py`**

Create `/home/mykhailo_dan/apps/progno/training/src/progno_train/artifacts.py`:

```python
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
```

- [ ] **Step 4: Run tests to confirm pass**

Run:
```bash
cd /home/mykhailo_dan/apps/progno/training
uv run pytest tests/test_artifacts.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/mykhailo_dan/apps/progno
git add training/src/progno_train/artifacts.py training/tests/test_artifacts.py
git commit -m "feat(training): artifact writers (elo_state.json, players, history)"
```

---

### Task 8: CLI wiring — `ingest` and `elo` commands

**Files:**
- Modify: `training/src/progno_train/cli.py` (replace stubs)
- Create: `training/src/progno_train/config.py`
- Create: `training/tests/test_cli.py`

- [ ] **Step 1: Write failing tests for CLI**

Create `/home/mykhailo_dan/apps/progno/training/tests/test_cli.py`:

```python
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from progno_train.cli import run_elo, run_ingest
from progno_train.config import Paths

FIXTURE = Path(__file__).parent / "fixtures" / "mini_atp_matches.csv"


@pytest.fixture()
def paths(tmp_path: Path) -> Paths:
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(FIXTURE, raw / "atp_matches_2024.csv")
    return Paths(
        data_raw=raw,
        data_staging=tmp_path / "staging",
        artifacts=tmp_path / "artifacts",
    )


def test_run_ingest_writes_staging_parquet(paths: Paths) -> None:
    rc = run_ingest(paths)
    assert rc == 0
    out = paths.data_staging / "matches_clean.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) > 0
    assert "is_complete" in df.columns


def test_run_elo_writes_artifacts(paths: Paths) -> None:
    run_ingest(paths)
    rc = run_elo(paths)
    assert rc == 0
    assert (paths.artifacts / "elo_state.json").exists()
    assert (paths.artifacts / "players.parquet").exists()
    assert (paths.artifacts / "match_history.parquet").exists()

    state = json.loads((paths.artifacts / "elo_state.json").read_text())
    assert "players" in state
    assert "data_as_of" in state
    # Alpha Alpha (id 100001) won two matches in the fixture, should be > 1500
    assert state["players"]["100001"]["elo_overall"] > 1500.0
```

- [ ] **Step 2: Run tests to confirm failure**

Run:
```bash
cd /home/mykhailo_dan/apps/progno/training
uv run pytest tests/test_cli.py -v
```

Expected: FAIL — `ImportError` on `Paths`, `run_ingest`, `run_elo`.

- [ ] **Step 3: Implement `config.py`**

Create `/home/mykhailo_dan/apps/progno/training/src/progno_train/config.py`:

```python
"""Path configuration for the training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    data_raw: Path
    data_staging: Path
    artifacts: Path

    @classmethod
    def default(cls, root: Path) -> Paths:
        return cls(
            data_raw=root / "data" / "raw",
            data_staging=root / "data" / "staging",
            artifacts=root / "artifacts",
        )
```

- [ ] **Step 4: Rewrite `cli.py`**

Replace `/home/mykhailo_dan/apps/progno/training/src/progno_train/cli.py` with:

```python
"""CLI entry points for the training pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from progno_train.artifacts import write_elo_state, write_match_history, write_players
from progno_train.config import Paths
from progno_train.ingest import ingest_sackmann_csv
from progno_train.rollup import rollup_elo

log = logging.getLogger("progno_train")


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def run_update_data(paths: Paths) -> int:
    log.info("update_data is a stub in Phase 1a; expected to be implemented later")
    log.info("expected input: Sackmann CSVs placed at %s", paths.data_raw)
    if not paths.data_raw.exists():
        paths.data_raw.mkdir(parents=True, exist_ok=True)
    return 0


def run_ingest(paths: Paths) -> int:
    csvs = sorted(paths.data_raw.glob("atp_matches_*.csv"))
    if not csvs:
        log.error("no Sackmann CSVs found in %s", paths.data_raw)
        return 2
    log.info("ingesting %d CSV files", len(csvs))
    df = ingest_sackmann_csv(csvs)
    out = paths.data_staging / "matches_clean.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("wrote %d rows to %s", len(df), out)
    return 0


def run_elo(paths: Paths) -> int:
    staging = paths.data_staging / "matches_clean.parquet"
    if not staging.exists():
        log.error("no staging parquet at %s; run ingest first", staging)
        return 2
    matches = pd.read_parquet(staging)
    log.info("rolling up %d matches", len(matches))
    state = rollup_elo(matches)
    log.info("produced Elo state for %d players", len(state))

    data_as_of = matches["tourney_date"].max()
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    write_elo_state(state, paths.artifacts / "elo_state.json", data_as_of=data_as_of)
    write_players(matches, paths.artifacts / "players.parquet")
    write_match_history(matches, paths.artifacts / "match_history.parquet")
    log.info("artifacts written to %s", paths.artifacts)
    return 0


def run_publish(paths: Paths, version: str) -> int:
    log.warning("publish is a stub in Phase 1a; will copy artifacts to app-data in Phase 1b")
    _ = version
    _ = paths
    return 0


def main() -> int:
    _setup_logging()
    parser = argparse.ArgumentParser(prog="progno-train")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("update_data")
    sub.add_parser("ingest")
    sub.add_parser("elo")
    publish = sub.add_parser("publish")
    publish.add_argument("version")
    args = parser.parse_args()

    paths = Paths.default(Path.cwd())

    if args.command == "update_data":
        return run_update_data(paths)
    if args.command == "ingest":
        return run_ingest(paths)
    if args.command == "elo":
        return run_elo(paths)
    if args.command == "publish":
        return run_publish(paths, args.version)
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run CLI tests to confirm pass**

Run:
```bash
cd /home/mykhailo_dan/apps/progno/training
uv run pytest tests/test_cli.py -v
```

Expected: both tests PASS.

- [ ] **Step 6: Run full test suite**

Run:
```bash
cd /home/mykhailo_dan/apps/progno/training
uv run pytest -v
```

Expected: all tests in `test_smoke.py`, `test_score.py`, `test_ingest.py`, `test_elo.py`, `test_rollup.py`, `test_artifacts.py`, `test_cli.py` PASS.

- [ ] **Step 7: Commit**

```bash
cd /home/mykhailo_dan/apps/progno
git add training/src/progno_train/cli.py training/src/progno_train/config.py training/tests/test_cli.py
git commit -m "feat(training): wire CLI ingest+elo pipeline"
```

---

### Task 9: End-to-end smoke — pull real Sackmann, run pipeline, sanity check output

**Files:**
- Create: `training/scripts/fetch_sackmann.sh`
- Create: `training/tests/test_e2e_smoke.py`

This task uses real data. It's a one-off local validation.

- [ ] **Step 1: Write fetch script**

Create `/home/mykhailo_dan/apps/progno/training/scripts/fetch_sackmann.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Clone or update Sackmann's tennis_atp repo into data/raw/
# Usage: bash scripts/fetch_sackmann.sh

TARGET="data/raw/tennis_atp"

if [ ! -d "$TARGET" ]; then
    mkdir -p "$(dirname "$TARGET")"
    git clone https://github.com/JeffSackmann/tennis_atp.git "$TARGET"
else
    (cd "$TARGET" && git pull --ff-only)
fi

# Symlink match CSVs to the flat location our ingester expects
cd data/raw
rm -f atp_matches_*.csv
for f in tennis_atp/atp_matches_[0-9]*.csv; do
    ln -sf "$f" "$(basename "$f")"
done

echo "Sackmann data available in $TARGET"
```

Make it executable:
```bash
chmod +x /home/mykhailo_dan/apps/progno/training/scripts/fetch_sackmann.sh
```

- [ ] **Step 2: Write sanity E2E test**

Create `/home/mykhailo_dan/apps/progno/training/tests/test_e2e_smoke.py`:

```python
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

    # Load names via players.parquet for top-20 overall
    players_df = pd.read_parquet(paths.artifacts / "players.parquet")
    elo_by_id = {int(pid): p["elo_overall"] for pid, p in players.items()}
    top20 = (
        pd.Series(elo_by_id)
        .sort_values(ascending=False)
        .head(20)
        .index.tolist()
    )
    top20_names = set(players_df.set_index("player_id").loc[top20]["name"].tolist())

    overlap = top20_names & KNOWN_TOP_PLAYERS_SINCE_2020
    assert len(overlap) >= 2, (
        f"Expected at least 2 known top players in top-20 Elo, got overlap={overlap}, "
        f"top20_names={top20_names}"
    )
```

- [ ] **Step 3: Run the smoke test**

Run:
```bash
cd /home/mykhailo_dan/apps/progno/training
bash scripts/fetch_sackmann.sh
uv run pytest tests/test_e2e_smoke.py -v
```

Expected:
- Fetch script clones `tennis_atp` (≈50MB) and symlinks CSVs.
- Test PASSES with at least 2 known top players in the top-20.

If it fails with low overlap: investigate in the notebook (is Elo rollup correct? Are matches being skipped?). Do not weaken the threshold — it's the smoke gate.

- [ ] **Step 4: Commit**

```bash
cd /home/mykhailo_dan/apps/progno
git add training/scripts/fetch_sackmann.sh training/tests/test_e2e_smoke.py
git commit -m "feat(training): E2E smoke test on real Sackmann data"
```

---

### Task 10: Full-suite green run + lint + format

**Files:** none

- [ ] **Step 1: Run full test suite**

Run:
```bash
cd /home/mykhailo_dan/apps/progno
just check
```

Expected: ruff clean + all pytest tests pass.

- [ ] **Step 2: If anything fails, fix inline and re-run**

Rule: do not commit failing code. Do not skip or xfail tests without recording why in the commit message.

- [ ] **Step 3: Final commit if anything changed**

```bash
cd /home/mykhailo_dan/apps/progno
git status
# only if there are changes:
git add <files>
git commit -m "chore(training): format + lint pass"
```

- [ ] **Step 4: Verify phase 1a acceptance criteria**

Run:
```bash
cd /home/mykhailo_dan/apps/progno/training
ls -la artifacts/ 2>/dev/null || echo "run: cd training && uv run python -m progno_train.cli ingest && uv run python -m progno_train.cli elo"
```

Phase 1a is done when:
- [ ] `just check` is green.
- [ ] `scripts/fetch_sackmann.sh` succeeds.
- [ ] `just ingest && just elo` produces `training/artifacts/elo_state.json`, `players.parquet`, `match_history.parquet`.
- [ ] The E2E smoke test passes with ≥2 known top players in the top-20.

---

## Execution notes

- **TDD discipline**: every task except scaffolding writes the failing test first. If you skip this, you lose the guarantee that your implementation is testing what the test claims.
- **Keep the working tree clean between tasks**. Commit after each task, no half-finished state.
- **When Gemma writes code for a task**, Claude reviews against the brief *and* against the spec section listed at the top of the task. The spec is the invariant.
- **If a test fails unexpectedly on real data**, investigate before weakening — most "flaky" Elo results are a leakage or sorting bug.
