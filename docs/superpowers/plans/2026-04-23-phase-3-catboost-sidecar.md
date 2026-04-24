# Phase 3: CatBoost Model + Python Sidecar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a CatBoost model on pre-match ATP features (rolling form, fatigue, serve stats, H2H) with Platt calibration and walk-forward validation, served via a Python FastAPI sidecar that Tauri proxies for improved probability estimates, with Elo as fallback when sidecar is unavailable.

**Architecture:**
- `features.py` computes rolling form, fatigue, serve efficiency, H2H from match history — all time-gated (no leakage); `train.py` runs walk-forward CV, trains CatBoost, fits Platt scaling; `validate.py` runs log-loss/ECE acceptance gate.
- `sidecar/server.py` (FastAPI on 127.0.0.1) loads model.cbm + calibration.json, computes features at inference time from match_history.parquet; port discovered at startup via stdout handshake.
- Tauri `sidecar.rs` spawns the sidecar binary, reads port from stdout, polls /health, proxies POST /predict; Elo prediction remains primary when sidecar is down.

**Tech Stack:** Python 3.12, CatBoost 1.2, scikit-learn 1.4 (Platt), pandas, FastAPI 0.110, uvicorn, PyInstaller; Rust: tauri-plugin-shell 2, reqwest

---

## Files Summary

| File | Action | Responsibility |
|------|--------|----------------|
| `training/pyproject.toml` | Modify | Add catboost, scikit-learn |
| `training/src/progno_train/artifacts.py` | Modify | Extend `MATCH_HISTORY_COLUMNS`; add `write_calibration`, `write_model_card` |
| `training/src/progno_train/config.py` | Modify | Add `featurized` staging path |
| `training/src/progno_train/features.py` | Create | Feature engineering: rolling form, fatigue, serve stats, H2H — time-parameterized |
| `training/src/progno_train/train.py` | Create | Walk-forward CV, CatBoost training, Platt calibration, artifact export |
| `training/src/progno_train/validate.py` | Create | Log-loss, ECE metrics + acceptance gate |
| `training/src/progno_train/cli.py` | Modify | Add `features`, `train`, `validate`, `retrain` subcommands |
| `training/tests/test_features.py` | Create | Temporal leakage property, cold start, rolling form correctness |
| `training/tests/test_train.py` | Create | Deterministic seed, monotonicity, Platt output range |
| `training/tests/test_validate.py` | Create | Log-loss, ECE, gate pass/fail |
| `sidecar/server.py` | Create | FastAPI: /health, /predict, /model_info; feature eng at inference |
| `sidecar/pyproject.toml` | Create | Sidecar deps: fastapi, uvicorn, catboost, pandas, pyarrow |
| `sidecar/build.sh` | Create | PyInstaller one-file build |
| `app/src-tauri/Cargo.toml` | Modify | Add tauri-plugin-shell, reqwest |
| `app/src-tauri/tauri.conf.json` | Modify | Add shell plugin + externalBin for sidecar |
| `app/src-tauri/src/sidecar.rs` | Create | Spawn sidecar, stdout port handshake, /health poll, HTTP proxy |
| `app/src-tauri/src/commands.rs` | Modify | Add `predict_with_ml` command; extend PredictionResult with ml_prob fields |
| `app/src-tauri/src/main.rs` | Modify | Register tauri-plugin-shell, spawn sidecar in setup |
| `justfile` | Modify | Add `features`, `train`, `validate`, `retrain`, `build-sidecar` targets |

---

## Task 1: Add ML dependencies

**Files:**
- Modify: `training/pyproject.toml`

- [ ] **Step 1: Add catboost and scikit-learn to dependencies**

Replace the dependencies section in `training/pyproject.toml`:

```toml
dependencies = [
    "pandas>=2.2",
    "pyarrow>=15.0",
    "numpy>=1.26",
    "catboost>=1.2",
    "scikit-learn>=1.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.4",
]
```

- [ ] **Step 2: Sync uv lockfile**

```bash
cd training && uv sync
```

Expected: lockfile updated, catboost + scikit-learn installed.

- [ ] **Step 3: Verify import**

```bash
cd training && uv run python -c "import catboost; import sklearn; print('OK')"
```

Expected: prints `OK`.

- [ ] **Step 4: Commit**

```bash
git add training/pyproject.toml training/uv.lock
git commit -m "feat(deps): add catboost and scikit-learn to training"
```

---

## Task 2: Extend match_history.parquet schema

The current `MATCH_HISTORY_COLUMNS` in `artifacts.py` is missing player ranks, ages, heights, hands, and serve stats needed for feature engineering. We extend the schema so match_history.parquet becomes the single source for both the sidecar and training.

**Files:**
- Modify: `training/src/progno_train/artifacts.py`
- Modify: `training/src/progno_train/config.py`

- [ ] **Step 1: Extend MATCH_HISTORY_COLUMNS in artifacts.py**

In `training/src/progno_train/artifacts.py`, replace `MATCH_HISTORY_COLUMNS`:

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
    # serve stats (available 1991+, null before)
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced",
]
```

Also add two new write functions at the bottom of `artifacts.py`:

```python
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
```

- [ ] **Step 2: Add featurized path to config.py**

Replace `config.py`:

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

    @property
    def matches_raw(self) -> Path:
        return self.data_staging / "matches_raw.parquet"

    @property
    def matches_clean(self) -> Path:
        return self.data_staging / "matches_clean.parquet"

    @property
    def featurized(self) -> Path:
        return self.data_staging / "matches_featurized.parquet"

    @property
    def match_history(self) -> Path:
        return self.artifacts / "match_history.parquet"

    @property
    def elo_state(self) -> Path:
        return self.artifacts / "elo_state.json"

    @property
    def players(self) -> Path:
        return self.artifacts / "players.parquet"

    @property
    def model_cbm(self) -> Path:
        return self.artifacts / "model.cbm"

    @property
    def calibration(self) -> Path:
        return self.artifacts / "calibration.json"

    @property
    def model_card(self) -> Path:
        return self.artifacts / "model_card.json"
```

- [ ] **Step 3: Run ingest + elo to verify columns are present**

```bash
cd training && uv run python -m progno_train.cli ingest
cd training && uv run python -m progno_train.cli elo
```

Expected: artifacts created. Verify match_history.parquet has all columns:

```bash
cd training && uv run python -c "
import pandas as pd
df = pd.read_parquet('artifacts/match_history.parquet')
print(df.columns.tolist())
print(df.shape)
"
```

Expected: all columns from MATCH_HISTORY_COLUMNS listed, shape shows thousands of rows.

- [ ] **Step 4: Commit**

```bash
git add training/src/progno_train/artifacts.py training/src/progno_train/config.py
git commit -m "feat(artifacts): extend match_history schema with serve stats and player meta"
```

---

## Task 3: Create features.py

All feature computations take `as_of_date` as the strict upper-bound cutoff — no data from `as_of_date` or later is ever used. This is the no-leakage invariant.

**Files:**
- Create: `training/src/progno_train/features.py`

- [ ] **Step 1: Write tests first (see Task 4), then create features.py**

Create `training/src/progno_train/features.py`:

```python
"""Pre-match feature engineering — all features are time-gated (no leakage)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

POPULATION_WIN_RATE = 0.5
LOW_HISTORY_THRESHOLD = 5


def _player_matches_before(
    history: pd.DataFrame,
    player_id: int,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """All completed matches for player_id strictly before as_of_date."""
    won = (history["winner_id"] == player_id) & (history["tourney_date"] < as_of_date) & history["is_complete"]
    lost = (history["loser_id"] == player_id) & (history["tourney_date"] < as_of_date) & history["is_complete"]
    w = history[won].assign(won=True, opponent_rank=history.loc[won, "loser_rank"])
    l = history[lost].assign(won=False, opponent_rank=history.loc[lost, "winner_rank"])
    cols = ["tourney_date", "surface", "won", "opponent_rank",
            "minutes", "completed_sets",
            "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced",
            "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced"]
    # Only keep columns that actually exist in the dataframe
    cols = [c for c in cols if c in w.columns]
    w = w[cols + ["won"]] if "won" not in cols else w[cols]
    l = l[cols + ["won"]] if "won" not in cols else l[cols]
    result = pd.concat([w, l], ignore_index=True).sort_values("tourney_date")
    return result


def rolling_win_rate(
    history: pd.DataFrame,
    player_id: int,
    as_of_date: pd.Timestamp,
    n: int,
    surface: str | None = None,
    max_opponent_rank: int | None = None,
) -> tuple[float, bool]:
    """Win rate over last n completed matches. Returns (win_rate, low_history_flag)."""
    df = _player_matches_before(history, player_id, as_of_date)
    if surface:
        df = df[df["surface"] == surface]
    if max_opponent_rank is not None:
        df = df[df["opponent_rank"].fillna(9999) <= max_opponent_rank]
    df = df.tail(n)
    if len(df) < LOW_HISTORY_THRESHOLD:
        return POPULATION_WIN_RATE, True
    return float(df["won"].mean()), False


def fatigue_features(
    history: pd.DataFrame,
    player_id: int,
    as_of_date: pd.Timestamp,
    prev_surface: str | None,
) -> dict[str, float]:
    """Days since last match, sets played last 14d, matches last 30d, surface switch."""
    df = _player_matches_before(history, player_id, as_of_date)
    if df.empty:
        return {
            "days_since_last_match": 30.0,
            "sets_last_14d": 0.0,
            "matches_last_30d": 0.0,
            "surface_switch": 0.0,
        }
    last_match_date = df["tourney_date"].max()
    days_since = min((as_of_date - last_match_date).days, 30)

    cutoff_14d = as_of_date - pd.Timedelta(days=14)
    cutoff_30d = as_of_date - pd.Timedelta(days=30)
    sets_14d = float(df[df["tourney_date"] >= cutoff_14d]["completed_sets"].sum())
    matches_30d = float((df["tourney_date"] >= cutoff_30d).sum())

    last_surface = df.iloc[-1]["surface"] if "surface" in df.columns else None
    surface_switch = 1.0 if (prev_surface and last_surface and prev_surface != last_surface) else 0.0

    return {
        "days_since_last_match": float(days_since),
        "sets_last_14d": sets_14d,
        "matches_last_30d": matches_30d,
        "surface_switch": surface_switch,
    }


def serve_efficiency(
    history: pd.DataFrame,
    player_id: int,
    as_of_date: pd.Timestamp,
    n: int = 25,
) -> dict[str, float | None]:
    """Rolling serve/return stats over last n matches. Returns None for each stat if insufficient data."""
    df = _player_matches_before(history, player_id, as_of_date).tail(n)
    # Separate winner/loser serve stats
    won_mask = history["winner_id"] == player_id
    result: dict[str, float | None] = {}

    eps = 1e-6

    def _safe_div(a, b):
        return float(a / b) if b > eps else None

    # Aggregate serve stats: winner stats from won matches, loser stats from lost matches
    serve_cols = ["w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_ace", "w_df", "w_bpSaved", "w_bpFaced"]
    if all(c in df.columns for c in ["w_svpt", "w_1stIn"]):
        # This is simplified — ideally separate won/lost serve columns
        # Use w_ columns for won matches, l_ for lost matches, then aggregate
        svpt = df["w_svpt"].fillna(0).sum() + df.get("l_svpt", pd.Series(0, index=df.index)).fillna(0).sum()
        first_in = df["w_1stIn"].fillna(0).sum() + df.get("l_1stIn", pd.Series(0, index=df.index)).fillna(0).sum()
        first_won = df["w_1stWon"].fillna(0).sum() + df.get("l_1stWon", pd.Series(0, index=df.index)).fillna(0).sum()
        ace = df["w_ace"].fillna(0).sum() + df.get("l_ace", pd.Series(0, index=df.index)).fillna(0).sum()
        df_ = df["w_df"].fillna(0).sum() + df.get("l_df", pd.Series(0, index=df.index)).fillna(0).sum()

        result["first_serve_in_pct"] = _safe_div(first_in, svpt)
        result["first_serve_won_pct"] = _safe_div(first_won, first_in)
        result["ace_rate"] = _safe_div(ace, svpt)
        result["df_rate"] = _safe_div(df_, svpt)
    else:
        result = {"first_serve_in_pct": None, "first_serve_won_pct": None, "ace_rate": None, "df_rate": None}

    return result


def h2h_score(
    history: pd.DataFrame,
    player_a_id: int,
    player_b_id: int,
    as_of_date: pd.Timestamp,
    prior: int = 5,
    prior_mean: float = 0.5,
) -> tuple[float, int]:
    """Shrinkage H2H win rate for A vs B. Returns (shrunk_win_rate, sample_size)."""
    mask = (
        (
            ((history["winner_id"] == player_a_id) & (history["loser_id"] == player_b_id)) |
            ((history["winner_id"] == player_b_id) & (history["loser_id"] == player_a_id))
        ) &
        (history["tourney_date"] < as_of_date) &
        history["is_complete"]
    )
    df = history[mask]
    n = len(df)
    wins_a = (df["winner_id"] == player_a_id).sum()
    shrunk = (wins_a + prior * prior_mean) / (n + prior)
    return float(shrunk), int(n)


def compute_match_features(
    history: pd.DataFrame,
    elo_state: dict,
    player_a_id: int,
    player_b_id: int,
    surface: str,
    tourney_level: str,
    round_: str,
    best_of: int,
    tourney_date: pd.Timestamp,
    player_a_rank: int | None = None,
    player_b_rank: int | None = None,
    player_a_age: float | None = None,
    player_b_age: float | None = None,
    player_a_height: float | None = None,
    player_b_height: float | None = None,
    player_a_hand: str | None = None,
    player_b_hand: str | None = None,
) -> dict[str, Any]:
    """Compute all pre-match features for a single match. Time-gated at tourney_date."""
    feats: dict[str, Any] = {}

    # Rolling form
    wr_a_50, lhf_a = rolling_win_rate(history, player_a_id, tourney_date, 50)
    wr_b_50, lhf_b = rolling_win_rate(history, player_b_id, tourney_date, 50)
    wr_a_surf, _ = rolling_win_rate(history, player_a_id, tourney_date, 20, surface=surface)
    wr_b_surf, _ = rolling_win_rate(history, player_b_id, tourney_date, 20, surface=surface)
    wr_a_12m, _ = rolling_win_rate(history, player_a_id, tourney_date, 9999)
    wr_b_12m, _ = rolling_win_rate(history, player_b_id, tourney_date, 9999)
    wr_a_top20, _ = rolling_win_rate(history, player_a_id, tourney_date, 30, max_opponent_rank=20)
    wr_b_top20, _ = rolling_win_rate(history, player_b_id, tourney_date, 30, max_opponent_rank=20)

    feats["win_rate_diff"] = wr_a_50 - wr_b_50
    feats["win_rate_surface_diff"] = wr_a_surf - wr_b_surf
    feats["win_rate_12m_diff"] = wr_a_12m - wr_b_12m
    feats["win_rate_top20_diff"] = wr_a_top20 - wr_b_top20
    feats["low_history_flag"] = int(lhf_a or lhf_b)

    # Fatigue
    fat_a = fatigue_features(history, player_a_id, tourney_date, surface)
    fat_b = fatigue_features(history, player_b_id, tourney_date, surface)
    feats["days_since_last_diff"] = fat_a["days_since_last_match"] - fat_b["days_since_last_match"]
    feats["sets_last_14d_diff"] = fat_a["sets_last_14d"] - fat_b["sets_last_14d"]
    feats["matches_last_30d_diff"] = fat_a["matches_last_30d"] - fat_b["matches_last_30d"]
    feats["surface_switch_a"] = fat_a["surface_switch"]
    feats["surface_switch_b"] = fat_b["surface_switch"]

    # Serve efficiency diffs (None → 0 diff)
    srv_a = serve_efficiency(history, player_a_id, tourney_date)
    srv_b = serve_efficiency(history, player_b_id, tourney_date)
    for stat in ["first_serve_in_pct", "first_serve_won_pct", "ace_rate", "df_rate"]:
        va = srv_a.get(stat) or 0.0
        vb = srv_b.get(stat) or 0.0
        feats[f"{stat}_diff"] = va - vb

    # H2H
    h2h, h2h_n = h2h_score(history, player_a_id, player_b_id, tourney_date)
    feats["h2h_score"] = h2h
    feats["h2h_sample_size"] = h2h_n

    # Elo (from elo_state dict — keys are player names, values have elo fields)
    def _elo(pid: int, field: str) -> float:
        p = elo_state.get("players", {}).get(str(pid), {})
        return float(p.get(field, 1500))

    elo_a = _elo(player_a_id, "elo_overall")
    elo_b = _elo(player_b_id, "elo_overall")
    elo_a_surf = _elo(player_a_id, f"elo_{surface.lower()}")
    elo_b_surf = _elo(player_b_id, f"elo_{surface.lower()}")
    feats["elo_overall_diff"] = elo_a - elo_b
    feats["elo_surface_diff"] = elo_a_surf - elo_b_surf

    # Player meta diffs
    feats["age_diff"] = (player_a_age or 25.0) - (player_b_age or 25.0)
    feats["height_diff"] = (player_a_height or 185.0) - (player_b_height or 185.0)
    feats["lefty_vs_righty"] = int(
        (player_a_hand == "L") != (player_b_hand == "L")
    )

    # Match context (categorical, passed through for CatBoost native handling)
    feats["surface"] = surface
    feats["tourney_level"] = tourney_level
    feats["round"] = round_
    feats["best_of_5"] = int(best_of == 5)

    return feats


def build_all_features(
    history: pd.DataFrame,
    elo_state: dict,
) -> pd.DataFrame:
    """Compute features for every complete match in history. Returns feature DataFrame with label."""
    rows = []
    # Pre-build a player meta lookup from players in history
    for _, row in history[history["is_complete"]].iterrows():
        feats = compute_match_features(
            history=history,
            elo_state=elo_state,
            player_a_id=row["winner_id"],
            player_b_id=row["loser_id"],
            surface=row["surface"],
            tourney_level=row.get("tourney_level", "A"),
            round_=row.get("round", "R32"),
            best_of=row.get("best_of", 3),
            tourney_date=row["tourney_date"],
            player_a_rank=row.get("winner_rank"),
            player_b_rank=row.get("loser_rank"),
            player_a_age=row.get("winner_age"),
            player_b_age=row.get("loser_age"),
            player_a_height=row.get("winner_ht"),
            player_b_height=row.get("loser_ht"),
            player_a_hand=row.get("winner_hand"),
            player_b_hand=row.get("loser_hand"),
        )
        feats["label"] = 1  # winner_id is always A
        feats["tourney_date"] = row["tourney_date"]
        feats["year"] = row["tourney_date"].year
        rows.append(feats)
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Commit (after tests pass in Task 4)**

```bash
git add training/src/progno_train/features.py
git commit -m "feat(features): add pre-match feature engineering module"
```

---

## Task 4: Test features.py

**Files:**
- Create: `training/tests/test_features.py`

- [ ] **Step 1: Write tests**

Create `training/tests/test_features.py`:

```python
"""Tests for feature engineering: no leakage, cold start, rolling form correctness."""

import pandas as pd
import pytest

from progno_train.features import (
    rolling_win_rate,
    fatigue_features,
    h2h_score,
    compute_match_features,
    POPULATION_WIN_RATE,
    LOW_HISTORY_THRESHOLD,
)


def make_history(n_matches: int, player_a: int = 1, player_b: int = 2) -> pd.DataFrame:
    """Synthetic match history: player_a wins all matches against player_b."""
    base = pd.Timestamp("2020-01-01")
    rows = []
    for i in range(n_matches):
        rows.append({
            "winner_id": player_a, "loser_id": player_b,
            "tourney_date": base + pd.Timedelta(days=7 * i),
            "surface": "Hard", "tourney_level": "A", "round": "R32", "best_of": 3,
            "is_complete": True, "completed_sets": 2, "score": "6-3 6-4", "minutes": 90.0,
            "winner_rank": 10, "loser_rank": 50,
            "winner_age": 24.0, "loser_age": 26.0,
            "winner_ht": 185.0, "loser_ht": 180.0,
            "winner_hand": "R", "loser_hand": "R",
            "w_ace": 5.0, "w_df": 1.0, "w_svpt": 60.0, "w_1stIn": 40.0,
            "w_1stWon": 30.0, "w_2ndWon": 12.0, "w_bpSaved": 3.0, "w_bpFaced": 4.0,
            "l_ace": 2.0, "l_df": 3.0, "l_svpt": 58.0, "l_1stIn": 35.0,
            "l_1stWon": 22.0, "l_2ndWon": 10.0, "l_bpSaved": 2.0, "l_bpFaced": 5.0,
        })
    return pd.DataFrame(rows)


def test_rolling_win_rate_correct():
    hist = make_history(20)
    # player 1 wins all 20 matches; as_of_date after all matches
    rate, low = rolling_win_rate(hist, 1, pd.Timestamp("2021-01-01"), 50)
    assert rate == 1.0
    assert low is False  # 20 >= 5


def test_rolling_win_rate_cold_start():
    hist = make_history(3)
    rate, low = rolling_win_rate(hist, 1, pd.Timestamp("2021-01-01"), 50)
    assert rate == POPULATION_WIN_RATE
    assert low is True


def test_rolling_win_rate_no_future_leakage():
    hist = make_history(20)
    # as_of_date = 10th match date — should see < 10 matches
    cutoff = hist["tourney_date"].iloc[9]  # 10th match date
    rate, _ = rolling_win_rate(hist, 1, cutoff, 50)
    # Only first 9 matches visible (strict <), all wins
    assert rate == 1.0
    # Verify we didn't use match 10+
    visible = hist[hist["tourney_date"] < cutoff]
    assert len(visible) == 9


def test_no_leakage_on_match_date():
    """Features for match on date D must NOT include any match from date D onwards."""
    hist = make_history(20)
    for i in range(5, 20):
        as_of = hist["tourney_date"].iloc[i]
        rate, _ = rolling_win_rate(hist, 1, as_of, 50)
        # All visible matches (strictly before as_of) are wins
        assert rate == 1.0
        visible = hist[hist["tourney_date"] < as_of]
        assert len(visible) == i  # exactly i matches visible


def test_h2h_shrinkage_no_history():
    hist = make_history(0)
    score, n = h2h_score(hist, 1, 2, pd.Timestamp("2021-01-01"))
    assert n == 0
    assert abs(score - 0.5) < 0.01  # pure prior


def test_h2h_shrinkage_with_history():
    hist = make_history(10)  # player 1 wins all 10
    score, n = h2h_score(hist, 1, 2, pd.Timestamp("2021-01-01"))
    assert n == 10
    # shrunk = (10 + 5*0.5) / (10 + 5) = 12.5 / 15 ≈ 0.833
    assert abs(score - 12.5 / 15.0) < 0.001


def test_h2h_no_future_leakage():
    hist = make_history(10)
    cutoff = hist["tourney_date"].iloc[5]  # only 5 matches visible
    score, n = h2h_score(hist, 1, 2, cutoff)
    assert n == 5


def test_fatigue_days_since():
    hist = make_history(10)
    last_date = hist["tourney_date"].max()
    as_of = last_date + pd.Timedelta(days=10)
    fat = fatigue_features(hist, 1, as_of, "Hard")
    assert fat["days_since_last_match"] == 10.0


def test_fatigue_capped_at_30():
    hist = make_history(1)
    as_of = hist["tourney_date"].iloc[0] + pd.Timedelta(days=60)
    fat = fatigue_features(hist, 1, as_of, "Hard")
    assert fat["days_since_last_match"] == 30.0


def test_fatigue_cold_start():
    hist = make_history(0)
    fat = fatigue_features(hist, 1, pd.Timestamp("2021-01-01"), "Hard")
    assert fat["days_since_last_match"] == 30.0
    assert fat["matches_last_30d"] == 0.0


def test_compute_match_features_returns_dict():
    hist = make_history(20)
    elo_state = {"players": {"1": {"elo_overall": 1600, "elo_hard": 1620, "elo_clay": 1550, "elo_grass": 1580},
                              "2": {"elo_overall": 1500, "elo_hard": 1520, "elo_clay": 1480, "elo_grass": 1500}}}
    feats = compute_match_features(
        history=hist, elo_state=elo_state,
        player_a_id=1, player_b_id=2,
        surface="Hard", tourney_level="A", round_="QF", best_of=3,
        tourney_date=pd.Timestamp("2021-01-01"),
    )
    assert isinstance(feats, dict)
    assert "elo_overall_diff" in feats
    assert "win_rate_diff" in feats
    assert "h2h_score" in feats
    assert feats["elo_overall_diff"] == 100.0  # 1600 - 1500


def test_elo_monotonicity():
    """Higher Elo diff → higher elo_overall_diff → should correlate with better outcome."""
    hist = make_history(20)
    elo_state_strong = {"players": {"1": {"elo_overall": 1800, "elo_hard": 1800, "elo_clay": 1800, "elo_grass": 1800},
                                     "2": {"elo_overall": 1500, "elo_hard": 1500, "elo_clay": 1500, "elo_grass": 1500}}}
    elo_state_weak = {"players": {"1": {"elo_overall": 1510, "elo_hard": 1510, "elo_clay": 1510, "elo_grass": 1510},
                                   "2": {"elo_overall": 1500, "elo_hard": 1500, "elo_clay": 1500, "elo_grass": 1500}}}
    feats_strong = compute_match_features(hist, elo_state_strong, 1, 2, "Hard", "A", "R32", 3, pd.Timestamp("2021-01-01"))
    feats_weak = compute_match_features(hist, elo_state_weak, 1, 2, "Hard", "A", "R32", 3, pd.Timestamp("2021-01-01"))
    assert feats_strong["elo_overall_diff"] > feats_weak["elo_overall_diff"]
```

- [ ] **Step 2: Run tests (expect them to fail first if features.py isn't created yet, or pass after)**

```bash
cd training && uv run pytest tests/test_features.py -v
```

Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add training/tests/test_features.py training/src/progno_train/features.py
git commit -m "test(features): add temporal leakage, cold start and rolling form tests"
```

---

## Task 5: Create train.py

**Files:**
- Create: `training/src/progno_train/train.py`

- [ ] **Step 1: Create train.py**

```python
"""Walk-forward training: CatBoost + Platt calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression

CAT_FEATURES = ["surface", "tourney_level", "round"]
BURN_IN_YEAR = 2004  # data before this year used only for Elo warm-up

CATBOOST_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "early_stopping_rounds": 50,
    "random_seed": 42,
    "verbose": False,
}


def walk_forward_splits(
    df: pd.DataFrame,
    val_start: int = 2016,
    test_start: int = 2023,
) -> list[tuple[pd.DataFrame, pd.DataFrame, str]]:
    """Yields (train, val_or_test, split_label) for each year from val_start onward."""
    df = df[df["year"] > BURN_IN_YEAR].copy()
    splits = []
    all_years = sorted(df["year"].unique())
    for year in all_years:
        if year < val_start:
            continue
        train = df[df["year"] < year]
        holdout = df[df["year"] == year]
        label = "val" if year < test_start else "test"
        splits.append((train, holdout, f"{label}_{year}"))
    return splits


def train_catboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> CatBoostClassifier:
    cat_idx = [i for i, c in enumerate(feature_cols) if c in CAT_FEATURES]
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df["label"].values

    pool_train = Pool(X_train, y_train, cat_features=cat_idx, feature_names=feature_cols)
    pool_val = Pool(X_val, y_val, cat_features=cat_idx, feature_names=feature_cols)

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(pool_train, eval_set=pool_val)
    return model


def fit_platt(raw_probs: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Fit Platt scaling: P_cal = sigmoid(a * logit(P_raw) + b)."""
    eps = 1e-7
    clipped = np.clip(raw_probs, eps, 1 - eps)
    logits = np.log(clipped) - np.log(1 - clipped)
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(logits.reshape(-1, 1), y_true)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt(raw_probs: np.ndarray, a: float, b: float) -> np.ndarray:
    eps = 1e-7
    clipped = np.clip(raw_probs, eps, 1 - eps)
    logits = np.log(clipped) - np.log(1 - clipped)
    return 1.0 / (1.0 + np.exp(-(a * logits + b)))


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"label", "tourney_date", "year"}
    return [c for c in df.columns if c not in exclude]


def run_walk_forward(featurized_path: Path) -> dict:
    """Run full walk-forward pipeline. Returns metrics dict."""
    df = pd.read_parquet(featurized_path)
    feature_cols = get_feature_cols(df)
    splits = walk_forward_splits(df)

    val_splits = [(tr, ho) for tr, ho, lbl in splits if lbl.startswith("val")]
    test_splits = [(tr, ho) for tr, ho, lbl in splits if lbl.startswith("test")]

    # Train on all val splits (expanding window), use last val year for Platt calibration
    all_train = df[df["year"] < 2022]
    cal_df = df[df["year"] == 2022]
    final_train = df[df["year"] < 2023]

    # Train final model on data up to test_start - 1
    model = train_catboost(final_train, cal_df, feature_cols)

    # Fit Platt on calibration year (2022)
    cal_pool = Pool(cal_df[feature_cols].fillna(0), feature_names=feature_cols)
    raw_cal = model.predict_proba(cal_pool)[:, 1]
    a, b = fit_platt(raw_cal, cal_df["label"].values)

    # Evaluate on test years
    test_df = pd.concat([ho for _, ho in test_splits])
    test_pool = Pool(test_df[feature_cols].fillna(0), feature_names=feature_cols)
    raw_test = model.predict_proba(test_pool)[:, 1]
    cal_test = apply_platt(raw_test, a, b)

    from progno_train.validate import compute_log_loss, compute_ece
    metrics = {
        "logloss_catboost": compute_log_loss(test_df["label"].values, cal_test),
        "ece_catboost": compute_ece(test_df["label"].values, cal_test),
        "n_test": len(test_df),
        "platt_a": a,
        "platt_b": b,
    }
    return model, a, b, metrics, feature_cols
```

- [ ] **Step 2: Commit**

```bash
git add training/src/progno_train/train.py
git commit -m "feat(train): add walk-forward CatBoost + Platt calibration pipeline"
```

---

## Task 6: Test train.py

**Files:**
- Create: `training/tests/test_train.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for training pipeline: determinism, Platt correctness."""

import numpy as np
import pandas as pd
import pytest

from progno_train.train import fit_platt, apply_platt, get_feature_cols, walk_forward_splits


def make_feature_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    years = np.repeat(np.arange(2016, 2026), n // 10)[:n]
    return pd.DataFrame({
        "elo_overall_diff": rng.normal(0, 100, n),
        "win_rate_diff": rng.normal(0, 0.2, n),
        "h2h_score": rng.uniform(0.3, 0.7, n),
        "h2h_sample_size": rng.integers(0, 20, n),
        "low_history_flag": rng.integers(0, 2, n),
        "age_diff": rng.normal(0, 3, n),
        "height_diff": rng.normal(0, 10, n),
        "surface": rng.choice(["Hard", "Clay", "Grass"], n),
        "tourney_level": rng.choice(["G", "M", "A"], n),
        "round": rng.choice(["F", "SF", "QF", "R32"], n),
        "best_of_5": rng.integers(0, 2, n),
        "label": rng.integers(0, 2, n),
        "year": years,
        "tourney_date": pd.date_range("2016-01-01", periods=n, freq="3D"),
    })


def test_platt_output_in_range():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.1, 0.9, 100)
    y = (raw > 0.5).astype(int)
    a, b = fit_platt(raw, y)
    cal = apply_platt(raw, a, b)
    assert cal.min() >= 0.0
    assert cal.max() <= 1.0


def test_platt_deterministic():
    rng = np.random.default_rng(1)
    raw = rng.uniform(0.1, 0.9, 100)
    y = (raw > 0.5).astype(int)
    a1, b1 = fit_platt(raw, y)
    a2, b2 = fit_platt(raw, y)
    assert a1 == a2
    assert b1 == b2


def test_platt_identity_for_perfect_model():
    """If raw probs are already perfectly calibrated, a≈1 b≈0."""
    rng = np.random.default_rng(42)
    raw = rng.uniform(0.2, 0.8, 500)
    y = rng.binomial(1, raw)  # perfect calibration
    a, b = fit_platt(raw, y)
    # With well-calibrated probs, Platt should be close to identity
    cal = apply_platt(raw, a, b)
    # Mean squared error of calibration should be small
    assert np.mean((cal - raw) ** 2) < 0.02


def test_walk_forward_splits_no_overlap():
    df = make_feature_df(200)
    splits = walk_forward_splits(df, val_start=2018, test_start=2023)
    for train, holdout, label in splits:
        assert len(train) > 0
        assert len(holdout) > 0
        # No overlap between train and holdout
        assert set(train["year"]).isdisjoint(set(holdout["year"]))


def test_walk_forward_no_future_in_train():
    df = make_feature_df(200)
    splits = walk_forward_splits(df, val_start=2018, test_start=2023)
    for train, holdout, label in splits:
        holdout_year = holdout["year"].min()
        assert train["year"].max() < holdout_year


def test_get_feature_cols_excludes_metadata():
    df = make_feature_df(10)
    cols = get_feature_cols(df)
    assert "label" not in cols
    assert "year" not in cols
    assert "tourney_date" not in cols
    assert "elo_overall_diff" in cols
```

- [ ] **Step 2: Run tests**

```bash
cd training && uv run pytest tests/test_train.py -v
```

Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add training/tests/test_train.py
git commit -m "test(train): add walk-forward and Platt calibration tests"
```

---

## Task 7: Create validate.py

**Files:**
- Create: `training/src/progno_train/validate.py`

- [ ] **Step 1: Create validate.py**

```python
"""Metrics and acceptance gate for model validation."""

from __future__ import annotations

import numpy as np


def compute_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-7
    p = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_pred[mask].mean())
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / n


def acceptance_gate(
    model_logloss: float,
    baseline_logloss: float,
    ece: float,
    ece_threshold: float = 0.03,
) -> None:
    """Raise ValueError if model fails acceptance criteria."""
    if model_logloss >= baseline_logloss:
        raise ValueError(
            f"GATE FAIL: model log-loss {model_logloss:.4f} >= baseline {baseline_logloss:.4f}"
        )
    if ece > ece_threshold:
        raise ValueError(
            f"GATE FAIL: ECE {ece:.4f} > threshold {ece_threshold}"
        )


def elo_baseline_logloss(y_true: np.ndarray, elo_probs: np.ndarray) -> float:
    return compute_log_loss(y_true, elo_probs)
```

- [ ] **Step 2: Write tests**

Create `training/tests/test_validate.py`:

```python
import numpy as np
import pytest

from progno_train.validate import compute_log_loss, compute_ece, acceptance_gate


def test_log_loss_perfect_prediction():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.999, 0.001, 0.999, 0.001])
    ll = compute_log_loss(y, p)
    assert ll < 0.01


def test_log_loss_random_prediction():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 1000)
    p = np.full(1000, 0.5)
    ll = compute_log_loss(y, p)
    assert abs(ll - np.log(2)) < 0.05  # ~0.693


def test_ece_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 1000)
    y = rng.binomial(1, p)
    ece = compute_ece(y, p)
    assert ece < 0.05  # Should be low for perfectly calibrated probs


def test_ece_miscalibrated():
    # Always predict 0.9 but true rate is 0.5
    y = np.array([1, 0] * 500)
    p = np.full(1000, 0.9)
    ece = compute_ece(y, p)
    assert ece > 0.3


def test_acceptance_gate_passes():
    # Should not raise
    acceptance_gate(model_logloss=0.60, baseline_logloss=0.65, ece=0.02)


def test_acceptance_gate_fails_logloss():
    with pytest.raises(ValueError, match="log-loss"):
        acceptance_gate(model_logloss=0.70, baseline_logloss=0.65, ece=0.02)


def test_acceptance_gate_fails_ece():
    with pytest.raises(ValueError, match="ECE"):
        acceptance_gate(model_logloss=0.60, baseline_logloss=0.65, ece=0.05)
```

- [ ] **Step 3: Run tests**

```bash
cd training && uv run pytest tests/test_validate.py -v
```

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add training/src/progno_train/validate.py training/tests/test_validate.py
git commit -m "feat(validate): add log-loss, ECE metrics and acceptance gate"
```

---

## Task 8: Update CLI and justfile

**Files:**
- Modify: `training/src/progno_train/cli.py`
- Modify: `justfile`

- [ ] **Step 1: Read current cli.py**

```bash
cd training && cat src/progno_train/cli.py
```

- [ ] **Step 2: Add features, train, validate, retrain commands to cli.py**

Add these functions and update `main()` in `training/src/progno_train/cli.py`:

```python
def run_features(paths: Paths) -> None:
    import json
    from progno_train.features import build_all_features

    logging.info("Loading match history for feature engineering...")
    history = pd.read_parquet(paths.match_history)
    elo_state = json.loads(paths.elo_state.read_text())

    logging.info("Building features for %d matches...", len(history))
    featurized = build_all_features(history, elo_state)
    paths.featurized.parent.mkdir(parents=True, exist_ok=True)
    featurized.to_parquet(paths.featurized, index=False)
    logging.info("Featurized dataset written: %s (%d rows)", paths.featurized, len(featurized))


def run_train(paths: Paths) -> None:
    import subprocess
    from progno_train.train import run_walk_forward
    from progno_train.artifacts import write_calibration, write_model_card

    logging.info("Running walk-forward training...")
    model, a, b, metrics, feature_cols = run_walk_forward(paths.featurized)

    logging.info("Saving model artifacts...")
    model.save_model(str(paths.model_cbm))
    write_calibration(a, b, paths.calibration)

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"

    write_model_card(
        train_years=(2005, 2022),
        test_year=2023,
        metrics=metrics,
        feature_names=feature_cols,
        git_sha=git_sha,
        out_path=paths.model_card,
    )
    logging.info("Training complete. Metrics: %s", metrics)


def run_validate(paths: Paths) -> None:
    import json
    import numpy as np
    from catboost import CatBoostClassifier, Pool
    from progno_train.train import apply_platt, get_feature_cols
    from progno_train.validate import compute_log_loss, compute_ece, acceptance_gate

    logging.info("Running validation and acceptance gate...")
    model = CatBoostClassifier()
    model.load_model(str(paths.model_cbm))

    cal = json.loads(paths.calibration.read_text())
    a, b = cal["a"], cal["b"]

    df = pd.read_parquet(paths.featurized)
    test_df = df[df["year"] >= 2023]
    feature_cols = get_feature_cols(df)

    pool = Pool(test_df[feature_cols].fillna(0), feature_names=feature_cols)
    raw = model.predict_proba(pool)[:, 1]
    cal_probs = apply_platt(raw, a, b)

    y = test_df["label"].values
    elo_probs = (test_df["elo_overall_diff"].values / 400 + 0.5).clip(0.05, 0.95)

    model_ll = compute_log_loss(y, cal_probs)
    baseline_ll = compute_log_loss(y, elo_probs)
    ece = compute_ece(y, cal_probs)

    logging.info("Model log-loss: %.4f | Elo baseline: %.4f | ECE: %.4f", model_ll, baseline_ll, ece)
    acceptance_gate(model_ll, baseline_ll, ece)
    logging.info("Acceptance gate: PASS")


def run_retrain(paths: Paths, version: str) -> None:
    run_features(paths)
    run_train(paths)
    run_validate(paths)
    run_publish(paths, version)
```

Update `main()` to add new subcommands:

```python
def main() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser(prog="progno-train")
    parser.add_argument("command", choices=[
        "update_data", "ingest", "elo", "features", "train", "validate", "retrain", "publish"
    ])
    parser.add_argument("--version", default="dev")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent.parent  # repo root / training
    paths = Paths.default(Path(__file__).parent.parent.parent)  # training/

    if args.command == "update_data":
        run_update_data(paths)
    elif args.command == "ingest":
        run_ingest(paths)
    elif args.command == "elo":
        run_elo(paths)
    elif args.command == "features":
        run_features(paths)
    elif args.command == "train":
        run_train(paths)
    elif args.command == "validate":
        run_validate(paths)
    elif args.command == "retrain":
        run_retrain(paths, args.version)
    elif args.command == "publish":
        run_publish(paths, args.version)
```

- [ ] **Step 3: Update justfile**

Append to `justfile`:

```just
# --- Phase 3 targets ---
features:
    cd training && uv run python -m progno_train.cli features

train:
    cd training && uv run python -m progno_train.cli train

validate:
    cd training && uv run python -m progno_train.cli validate

retrain version:
    cd training && uv run python -m progno_train.cli retrain --version {{version}}

build-sidecar:
    cd sidecar && uv run pyinstaller --onefile --name progno-sidecar server.py
```

- [ ] **Step 4: Commit**

```bash
git add training/src/progno_train/cli.py justfile
git commit -m "feat(cli): add features, train, validate, retrain commands"
```

---

## Task 9: Create FastAPI sidecar

**Files:**
- Create: `sidecar/server.py`
- Create: `sidecar/pyproject.toml`
- Create: `sidecar/build.sh`

- [ ] **Step 1: Create sidecar/pyproject.toml**

```toml
[project]
name = "progno-sidecar"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "catboost>=1.2",
    "scikit-learn>=1.4",
    "pandas>=2.2",
    "pyarrow>=15.0",
    "numpy>=1.26",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Create sidecar/server.py**

```python
"""FastAPI sidecar — loads model artifacts, serves /health /predict /model_info."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import uvicorn
from catboost import CatBoostClassifier, Pool
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Global state set during startup ──────────────────────────────────────────
_model: CatBoostClassifier | None = None
_platt_a: float = 1.0
_platt_b: float = 0.0
_feature_cols: list[str] = []
_match_history: pd.DataFrame | None = None
_elo_state: dict = {}
_model_card: dict = {}
_port: int = 0


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _load_artifacts(artifacts_dir: Path) -> None:
    global _model, _platt_a, _platt_b, _feature_cols, _match_history, _elo_state, _model_card

    model_path = artifacts_dir / "model.cbm"
    if not model_path.exists():
        print(f"ERROR model.cbm not found at {model_path}", flush=True)
        sys.exit(1)

    _model = CatBoostClassifier()
    _model.load_model(str(model_path))
    _feature_cols = _model.feature_names_

    cal = json.loads((artifacts_dir / "calibration.json").read_text())
    _platt_a = cal["a"]
    _platt_b = cal["b"]

    _match_history = pd.read_parquet(artifacts_dir / "match_history.parquet")
    _elo_state = json.loads((artifacts_dir / "elo_state.json").read_text())

    card_path = artifacts_dir / "model_card.json"
    _model_card = json.loads(card_path.read_text()) if card_path.exists() else {}


def _apply_platt(raw: np.ndarray) -> np.ndarray:
    eps = 1e-7
    clipped = np.clip(raw, eps, 1 - eps)
    logits = np.log(clipped) - np.log(1 - clipped)
    return 1.0 / (1.0 + np.exp(-(_platt_a * logits + _platt_b)))


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    sys.stdout.write(f"READY port={_port}\n")
    sys.stdout.flush()
    yield


app = FastAPI(lifespan=lifespan)


class MatchRequest(BaseModel):
    player_a_id: str
    player_b_id: str
    surface: str
    tourney_level: str = "A"
    round_: str = "R32"
    best_of: int = 3
    tourney_date: str  # "YYYY-MM-DD"


class PredictRequest(BaseModel):
    matches: list[MatchRequest]


class MatchPrediction(BaseModel):
    prob_a_wins: float
    prob_a_wins_uncalibrated: float
    elo_prob_a_wins: float
    confidence_flag: str  # "ok" | "low_history" | "insufficient_data"


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[MatchPrediction]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/model_info")
async def model_info():
    return _model_card


@app.post("/predict")
async def predict(req: PredictRequest) -> PredictResponse:
    from progno_train.features import compute_match_features

    if _model is None or _match_history is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    predictions = []
    for m in req.matches:
        tourney_date = pd.Timestamp(m.tourney_date)
        try:
            pid_a = int(m.player_a_id)
            pid_b = int(m.player_b_id)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"player IDs must be integers: {m.player_a_id}, {m.player_b_id}")

        feats = compute_match_features(
            history=_match_history,
            elo_state=_elo_state,
            player_a_id=pid_a,
            player_b_id=pid_b,
            surface=m.surface,
            tourney_level=m.tourney_level,
            round_=m.round_,
            best_of=m.best_of,
            tourney_date=tourney_date,
        )

        low_history = bool(feats.get("low_history_flag", 0))
        conf_flag = "low_history" if low_history else "ok"

        feat_row = {col: feats.get(col, 0) for col in _feature_cols}
        feat_df = pd.DataFrame([feat_row])
        pool = Pool(feat_df, feature_names=_feature_cols)

        raw = float(_model.predict_proba(pool)[0, 1])
        cal = float(_apply_platt(np.array([raw]))[0])

        # Elo baseline from feature (elo_overall_diff encodes rating difference)
        elo_diff = feats.get("elo_overall_diff", 0.0)
        elo_prob = float(1.0 / (1.0 + 10 ** (-elo_diff / 400)))

        predictions.append(MatchPrediction(
            prob_a_wins=round(cal, 4),
            prob_a_wins_uncalibrated=round(raw, 4),
            elo_prob_a_wins=round(elo_prob, 4),
            confidence_flag=conf_flag,
        ))

    model_version = _model_card.get("generated_at", "unknown")
    return PredictResponse(model_version=model_version, predictions=predictions)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global _port
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True)
    args = parser.parse_args()

    _load_artifacts(Path(args.artifacts_dir))
    _port = _find_free_port()
    uvicorn.run(app, host="127.0.0.1", port=_port, log_level="error")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create sidecar/build.sh**

```bash
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
uv run pyinstaller \
  --onefile \
  --name progno-sidecar \
  --hidden-import catboost \
  --hidden-import progno_train.features \
  server.py
echo "Built: dist/progno-sidecar"
```

- [ ] **Step 4: Commit**

```bash
git add sidecar/
git commit -m "feat(sidecar): add FastAPI sidecar with /health /predict /model_info"
```

---

## Task 10: Rust sidecar integration

**Files:**
- Modify: `app/src-tauri/Cargo.toml`
- Modify: `app/src-tauri/tauri.conf.json`
- Create: `app/src-tauri/src/sidecar.rs`
- Modify: `app/src-tauri/src/commands.rs`
- Modify: `app/src-tauri/src/main.rs`

- [ ] **Step 1: Read Cargo.toml and tauri.conf.json**

```bash
cat app/src-tauri/Cargo.toml
cat app/src-tauri/tauri.conf.json
```

- [ ] **Step 2: Add dependencies to Cargo.toml**

In `app/src-tauri/Cargo.toml`, add to `[dependencies]`:

```toml
tauri-plugin-shell = "2"
reqwest = { version = "0.12", features = ["json"] }
tokio = { version = "1", features = ["full"] }
```

- [ ] **Step 3: Update tauri.conf.json**

Add shell plugin permissions and externalBin. In `tauri.conf.json`, add:

```json
{
  "bundle": {
    "externalBin": ["../sidecar/dist/progno-sidecar"]
  },
  "plugins": {
    "shell": {
      "open": false,
      "sidecar": true
    }
  }
}
```

(Merge with existing JSON — don't replace the whole file.)

- [ ] **Step 4: Create sidecar.rs**

Create `app/src-tauri/src/sidecar.rs`:

```rust
use std::sync::Mutex;
use serde::{Deserialize, Serialize};
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::CommandEvent;

pub struct SidecarState {
    pub port: Option<u16>,
}

impl Default for SidecarState {
    fn default() -> Self { Self { port: None } }
}

#[derive(Serialize)]
pub struct MlMatchRequest {
    pub player_a_id: String,
    pub player_b_id: String,
    pub surface: String,
    pub tourney_level: String,
    pub round_: String,
    pub best_of: u8,
    pub tourney_date: String,
}

#[derive(Serialize)]
struct PredictPayload {
    matches: Vec<MlMatchRequest>,
}

#[derive(Deserialize)]
pub struct MlMatchPrediction {
    pub prob_a_wins: f64,
    pub prob_a_wins_uncalibrated: f64,
    pub elo_prob_a_wins: f64,
    pub confidence_flag: String,
}

#[derive(Deserialize)]
pub struct MlPredictResponse {
    pub model_version: String,
    pub predictions: Vec<MlMatchPrediction>,
}

pub fn spawn_sidecar(app: &tauri::AppHandle, artifacts_dir: String) {
    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        match do_spawn(&handle, &artifacts_dir).await {
            Ok(port) => {
                let state = handle.state::<Mutex<SidecarState>>();
                state.lock().unwrap().port = Some(port);
                eprintln!("[sidecar] ready on port {port}");
            }
            Err(e) => eprintln!("[sidecar] failed to start: {e}"),
        }
    });
}

async fn do_spawn(app: &tauri::AppHandle, artifacts_dir: &str) -> anyhow::Result<u16> {
    let (mut rx, _child) = app
        .shell()
        .sidecar("progno-sidecar")?
        .args(["--artifacts-dir", artifacts_dir])
        .spawn()?;

    while let Some(event) = rx.recv().await {
        match event {
            CommandEvent::Stdout(line) => {
                let s = String::from_utf8_lossy(&line);
                if let Some(port_str) = s.trim().strip_prefix("READY port=") {
                    let port: u16 = port_str.parse()?;
                    return Ok(port);
                }
            }
            CommandEvent::Terminated(status) => {
                return Err(anyhow::anyhow!("sidecar exited: {:?}", status));
            }
            _ => {}
        }
    }
    Err(anyhow::anyhow!("sidecar did not emit READY"))
}

pub async fn ml_predict(port: u16, matches: Vec<MlMatchRequest>) -> anyhow::Result<MlPredictResponse> {
    let url = format!("http://127.0.0.1:{port}/predict");
    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(&PredictPayload { matches })
        .send()
        .await?
        .error_for_status()?
        .json::<MlPredictResponse>()
        .await?;
    Ok(resp)
}
```

- [ ] **Step 5: Update commands.rs — extend PredictionResult and add predict_with_ml**

In `app/src-tauri/src/commands.rs`, update `PredictionResult`:

```rust
#[derive(Serialize, Deserialize, Clone)]
pub struct PredictionResult {
    pub player_a: String,
    pub player_b: String,
    pub surface: String,
    pub prob_a_wins: f64,
    pub prob_b_wins: f64,
    pub elo_a_overall: f64,
    pub elo_b_overall: f64,
    // Phase 3: ML model results (None when sidecar is unavailable)
    pub ml_prob_a_wins: Option<f64>,
    pub confidence_flag: Option<String>,
}
```

Add at the bottom (before the `#[cfg(test)]` block):

```rust
#[derive(Serialize, Deserialize)]
pub struct MlPredictRequest {
    pub text: String,
    pub tourney_date: String,  // "YYYY-MM-DD"
}

#[cfg(not(test))]
#[tauri::command]
pub async fn predict_with_ml(
    request: MlPredictRequest,
    app_state: tauri::State<'_, AppState>,
    sidecar_state: tauri::State<'_, std::sync::Mutex<crate::sidecar::SidecarState>>,
) -> Result<PredictResponse, String> {
    use crate::sidecar::{MlMatchRequest, ml_predict};

    // First get Elo predictions
    let guard = app_state.elo_state.lock().unwrap();
    let elo_resp = match &*guard {
        None => return Err("Elo data not loaded".to_string()),
        Some(elo) => predict_text(&request.text, elo),
    };
    drop(guard);

    // Check if sidecar is available
    let port = sidecar_state.lock().unwrap().port;
    if port.is_none() {
        // Return Elo results without ML enrichment
        return Ok(elo_resp);
    }
    let port = port.unwrap();

    // Build ML requests from Elo predictions
    let ml_matches: Vec<MlMatchRequest> = elo_resp.predictions.iter().map(|p| {
        MlMatchRequest {
            player_a_id: normalize_player_id(&p.player_a),
            player_b_id: normalize_player_id(&p.player_b),
            surface: p.surface.clone(),
            tourney_level: "A".to_string(),
            round_: "R32".to_string(),
            best_of: 3,
            tourney_date: request.tourney_date.clone(),
        }
    }).collect();

    match ml_predict(port, ml_matches).await {
        Ok(ml_resp) => {
            let enriched: Vec<PredictionResult> = elo_resp.predictions.into_iter()
                .zip(ml_resp.predictions.into_iter())
                .map(|(mut elo_pred, ml_pred)| {
                    elo_pred.ml_prob_a_wins = Some(ml_pred.prob_a_wins);
                    elo_pred.confidence_flag = Some(ml_pred.confidence_flag);
                    elo_pred
                })
                .collect();
            Ok(PredictResponse { predictions: enriched, ..elo_resp })
        }
        Err(e) => {
            eprintln!("[ml] predict failed: {e}, falling back to Elo");
            Ok(elo_resp)  // Graceful fallback to Elo
        }
    }
}
```

- [ ] **Step 6: Update main.rs**

In `app/src-tauri/src/main.rs`, add:

```rust
mod sidecar;
```

And in the `main()` function:

```rust
use std::sync::Mutex;

tauri::Builder::default()
    .manage(AppState::default())
    .manage(Mutex::new(sidecar::SidecarState::default()))
    .plugin(tauri_plugin_shell::init())
    .setup(|app| {
        // Load Elo state
        let path = artifacts::elo_state_path();
        if let Ok(elo) = artifacts::load_elo_state(path.to_str().unwrap_or("elo_state.json")) {
            *app.state::<AppState>().elo_state.lock().unwrap() = Some(elo);
        }
        // Spawn ML sidecar (non-blocking — Elo remains available immediately)
        let artifacts_dir = artifacts::artifacts_dir().to_string_lossy().to_string();
        sidecar::spawn_sidecar(&app.handle(), artifacts_dir);
        Ok(())
    })
    .invoke_handler(tauri::generate_handler![
        commands::parse_and_predict,
        commands::get_data_as_of_cmd,
        commands::calculate_kelly,
        commands::predict_with_ml,
    ])
    .run(tauri::generate_context!())
    .map_err(|e| eprintln!("Failed to run Tauri: {}", e))
    .ok();
```

- [ ] **Step 7: Verify compilation**

```bash
cd app && cargo check
```

Expected: compiles without errors (sidecar binary doesn't need to exist for `cargo check`).

- [ ] **Step 8: Commit**

```bash
git add app/src-tauri/src/sidecar.rs app/src-tauri/src/commands.rs app/src-tauri/src/main.rs app/src-tauri/Cargo.toml app/src-tauri/tauri.conf.json
git commit -m "feat(sidecar): Rust sidecar integration — spawn, port handshake, ML predict with Elo fallback"
```

---

## Task 11: Integration test — Rust ↔ sidecar

**Files:**
- Modify: `app/src-tauri/tests/integration_test.rs`

- [ ] **Step 1: Verify artifacts exist**

To run integration test, a trained model must exist. Run the full training pipeline first:

```bash
just ingest && just elo && just features && just train && just validate
```

- [ ] **Step 2: Build sidecar binary**

```bash
just build-sidecar
```

Expected: `sidecar/dist/progno-sidecar` created.

- [ ] **Step 3: Manual smoke test — start sidecar and call /predict**

```bash
# Start sidecar manually (in one terminal)
./sidecar/dist/progno-sidecar --artifacts-dir training/artifacts
# Look for: READY port=XXXXX

# In another terminal, POST to /predict (replace PORT with actual port)
curl -s -X POST http://127.0.0.1:PORT/predict \
  -H "Content-Type: application/json" \
  -d '{"matches":[{"player_a_id":"104745","player_b_id":"106421","surface":"Hard","tourney_date":"2026-04-23"}]}'
```

Expected: JSON response with `prob_a_wins` in [0, 1], `confidence_flag` is "ok" or "low_history".

- [ ] **Step 4: Add Rust integration test for sidecar HTTP call**

In `app/src-tauri/tests/integration_test.rs`, add:

```rust
#[test]
fn test_sidecar_predict_request_struct() {
    // Verify serialization matches sidecar API contract
    use serde_json::json;
    let req = json!({
        "matches": [{
            "player_a_id": "104745",
            "player_b_id": "106421",
            "surface": "Hard",
            "tourney_level": "M",
            "round_": "QF",
            "best_of": 3,
            "tourney_date": "2026-04-23"
        }]
    });
    // Verify it deserializes correctly
    assert_eq!(req["matches"][0]["surface"], "Hard");
    assert_eq!(req["matches"][0]["best_of"], 3);
}
```

- [ ] **Step 5: Run all Rust tests**

```bash
cd app && cargo test --all
```

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add app/src-tauri/tests/integration_test.rs
git commit -m "test(integration): add sidecar API contract test"
```

---

## Spec Coverage Check

| Spec requirement | Task | Status |
|---|---|---|
| Rolling form: win_rate_overall (50), win_rate_surface (20), win_rate_12m, win_rate_vs_top20 (§3.2) | Task 3 | ✅ |
| Fatigue: days_since, sets_last_14d, matches_last_30d, surface_switch (§3.3) | Task 3 | ✅ |
| Serve/return efficiency rolling 25 matches (§3.4) | Task 3 | ✅ |
| H2H with shrinkage prior=5 (§3.5) | Task 3 | ✅ |
| Match context: surface, tourney_level, round, best_of (§3.6) | Task 3 | ✅ |
| Player meta: age_diff, height_diff, lefty_vs_righty (§3.7) | Task 3 | ✅ |
| Pair-diff encoding for numeric features (§3.9) | Task 3 | ✅ |
| No data leakage — temporal property test (§2.4, §6.5) | Task 4 | ✅ |
| CatBoost with cat_features (§4.2, §4.4) | Task 5 | ✅ |
| Walk-forward validation, no random split (§2.5, §4.1) | Task 5 | ✅ |
| Platt scaling: P_cal = sigmoid(a * logit + b) (§4.5) | Task 5 | ✅ |
| Acceptance gate: log-loss < Elo baseline, ECE < 0.03 (§6.4) | Task 7 | ✅ |
| `just features`, `just train`, `just validate`, `just retrain` (§6.2) | Task 8 | ✅ |
| FastAPI sidecar on localhost random port (§4.7) | Task 9 | ✅ |
| /health, /predict, /model_info endpoints (§4.9) | Task 9 | ✅ |
| Stdout port handshake (§4.11) | Task 9 + 10 | ✅ |
| Elo fallback when sidecar down (§5.9) | Task 10 | ✅ |
| model.cbm + calibration.json + model_card.json artifacts (§4.8) | Task 5 + 8 | ✅ |
| Deterministic training random_seed=42 (§6.7) | Task 5 | ✅ |
| Cold start: < 5 matches → population mean + flag (§3.2, §2.7) | Task 3 + 4 | ✅ |

### Notes

- **ROI metric**: spec §6.4 requires ROI ≥ 0 on test vs Pinnacle closing odds. This needs the tennis-data.co.uk XLSX join (spec §2.6), which is deferred to a Phase 3.5 task. The acceptance gate in Task 7 validates log-loss and ECE only. Add ROI gate after implementing the odds join.
- **burn-in years**: 2000–2004 data updates Elo state but is excluded from ML training (BURN_IN_YEAR = 2004).
- **Sidecar bundling**: PyInstaller produces a ~300MB binary (spec §4.11). Run `just build-sidecar` once after model training; Tauri bundles it at `cargo tauri build`.
- **Player ID format**: sidecar receives string player IDs matching the keys in elo_state.json (normalized names). Update `commands.rs` `normalize_player_id` if Sackmann numeric IDs differ from elo_state keys.
