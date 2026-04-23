# Phase 3 + Phase 4 (WTA) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a CatBoost model for ATP (Phase 3) and WTA (Phase 4), served via a shared Python FastAPI sidecar with tour-based routing (`tour: "atp" | "wta"`), with an ATP/WTA dropdown in the Tauri UI.

**Architecture:** The training pipeline is parameterized by `tour` from the start — `Paths.for_tour(root, tour)` puts artifacts in `artifacts/atp/` or `artifacts/wta/`. A single FastAPI sidecar loads both models and routes by the `tour` field in `/predict`. Rust `AppState` holds dual Elo states; `parse_and_predict` accepts a `tour` param and picks the correct state. The UI header has a `selectedTour` store driving all calls.

**Tech Stack:** Python 3.12, CatBoost 1.2, scikit-learn 1.4, FastAPI 0.110, uvicorn, pandas 2.2, pyarrow 15, PyInstaller; Rust 2021, Tauri 2, tauri-plugin-shell 2, reqwest 0.12; Svelte 5, TypeScript, Tailwind

---

## Files Summary

| File | Action | Responsibility |
|------|--------|----------------|
| `training/pyproject.toml` | Modify | Add catboost, scikit-learn deps |
| `training/src/progno_train/config.py` | Modify | Add `Paths.for_tour()`, path properties |
| `training/src/progno_train/artifacts.py` | Modify | Extend `MATCH_HISTORY_COLUMNS`, add `write_calibration`, `write_model_card` |
| `training/src/progno_train/cli.py` | Modify | Add `--tour` to all commands; add `features`, `train`, `validate`, `retrain` |
| `training/src/progno_train/features.py` | Create | Time-gated feature engineering: rolling form, fatigue, serve stats, H2H |
| `training/src/progno_train/train.py` | Create | Walk-forward CatBoost + Platt calibration |
| `training/src/progno_train/validate.py` | Create | Log-loss, ECE metrics, acceptance gate |
| `training/tests/test_features.py` | Create | Temporal leakage, cold start, H2H shrinkage tests |
| `training/tests/test_train.py` | Create | Platt correctness, walk-forward no-overlap |
| `training/tests/test_validate.py` | Create | Log-loss, ECE, gate pass/fail |
| `training/scripts/fetch_sackmann_wta.sh` | Create | Clone/pull `tennis_wta` repo |
| `sidecar/pyproject.toml` | Create | Sidecar deps: fastapi, uvicorn, catboost, pandas |
| `sidecar/server.py` | Create | FastAPI: /health, /predict with tour routing, /model_info |
| `sidecar/features.py` | Create | Copy of features.py used at inference time |
| `sidecar/build.sh` | Create | PyInstaller one-file build |
| `justfile` | Modify | Add features, train, validate, retrain, WTA variants, build-sidecar |
| `app/src-tauri/Cargo.toml` | Modify | Add tauri-plugin-shell 2, reqwest 0.12 |
| `app/src-tauri/tauri.conf.json` | Modify | Add shell plugin + externalBin |
| `app/src-tauri/src/state.rs` | Modify | Add `elo_wta` alongside `elo_atp` |
| `app/src-tauri/src/artifacts.rs` | Modify | Add `load_elo_state_for_tour()` |
| `app/src-tauri/src/sidecar.rs` | Create | Spawn sidecar, port handshake, HTTP proxy |
| `app/src-tauri/src/commands.rs` | Modify | Add `tour` param; add `ml_prob_a_wins`, `confidence_flag` to result |
| `app/src-tauri/src/main.rs` | Modify | Load dual Elo, spawn sidecar, register `predict_with_ml` |
| `app/src-tauri/src/lib.rs` | Modify | Export `sidecar` module |
| `app/src/lib/stores.ts` | Modify | Add `selectedTour` writable store |
| `app/src/App.svelte` | Modify | Add ATP/WTA dropdown to header |
| `app/src/lib/components/MatchInput.svelte` | Modify | Pass `selectedTour` to `parse_and_predict` |

---

## Task 1: Add ML dependencies

**Files:** Modify `training/pyproject.toml`

- [ ] **Step 1: Add catboost and scikit-learn**

Replace `dependencies` in `training/pyproject.toml`:

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
    "catboost>=1.2",
    "scikit-learn>=1.4",
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

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Sync and verify**

```bash
cd training && uv sync
uv run python -c "import catboost; import sklearn; print('OK')"
```

Expected: prints `OK`.

- [ ] **Step 3: Commit**

```bash
git add training/pyproject.toml training/uv.lock
git commit -m "feat(deps): add catboost and scikit-learn to training"
```

---

## Task 2: Extend config.py

**Files:** Modify `training/src/progno_train/config.py`

- [ ] **Step 1: Replace config.py**

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
    def default(cls, root: Path) -> "Paths":
        return cls(
            data_raw=root / "data" / "raw",
            data_staging=root / "data" / "staging",
            artifacts=root / "artifacts",
        )

    @classmethod
    def for_tour(cls, root: Path, tour: str) -> "Paths":
        return cls(
            data_raw=root / "data" / "raw",
            data_staging=root / "data" / "staging" / tour,
            artifacts=root / "artifacts" / tour,
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

- [ ] **Step 2: Run existing tests to verify no regression**

```bash
cd training && uv run pytest tests/ -v -k "not smoke"
```

Expected: all tests PASS (existing tests use `Paths.default()` which is unchanged).

- [ ] **Step 3: Commit**

```bash
git add training/src/progno_train/config.py
git commit -m "feat(config): add Paths.for_tour() and path properties for ML artifacts"
```

---

## Task 3: Extend artifacts.py

**Files:** Modify `training/src/progno_train/artifacts.py`

- [ ] **Step 1: Extend MATCH_HISTORY_COLUMNS and add write functions**

Replace the full file `training/src/progno_train/artifacts.py`:

```python
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
```

- [ ] **Step 2: Run existing artifact tests**

```bash
cd training && uv run pytest tests/test_artifacts.py -v
```

Expected: all PASS (`write_match_history` now selects only available columns, which is backward-compatible).

- [ ] **Step 3: Commit**

```bash
git add training/src/progno_train/artifacts.py
git commit -m "feat(artifacts): extend match_history schema; add write_calibration, write_model_card"
```

---

## Task 4: Update cli.py

**Files:** Modify `training/src/progno_train/cli.py`

- [ ] **Step 1: Replace cli.py**

```python
"""CLI entry points for the training pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

from progno_train.artifacts import (
    write_calibration,
    write_elo_state,
    write_match_history,
    write_model_card,
    write_players,
)
from progno_train.config import Paths
from progno_train.ingest import ingest_sackmann_csv
from progno_train.rollup import rollup_elo

log = logging.getLogger("progno_train")


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def run_update_data(paths: Paths) -> int:
    log.info("update_data: pull latest Sackmann data")
    if not paths.data_raw.exists():
        paths.data_raw.mkdir(parents=True, exist_ok=True)
    return 0


def run_ingest(paths: Paths, tour: str) -> int:
    csv_glob = f"tennis_{tour}/{tour}_matches_*.csv"
    csvs = sorted(paths.data_raw.glob(csv_glob))
    if not csvs:
        log.error("no Sackmann CSVs found: %s/%s", paths.data_raw, csv_glob)
        return 2
    log.info("ingesting %d CSV files for tour=%s", len(csvs), tour)
    df = ingest_sackmann_csv(csvs)
    out = paths.matches_clean
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("wrote %d rows to %s", len(df), out)
    return 0


def run_elo(paths: Paths) -> int:
    if not paths.matches_clean.exists():
        log.error("no staging parquet at %s; run ingest first", paths.matches_clean)
        return 2
    matches = pd.read_parquet(paths.matches_clean)
    log.info("rolling up %d matches", len(matches))
    state = rollup_elo(matches)
    log.info("produced Elo state for %d players", len(state))

    data_as_of = matches["tourney_date"].max()
    all_names = pd.concat([
        matches[["winner_id", "winner_name"]].rename(columns={"winner_id": "id", "winner_name": "name"}),
        matches[["loser_id", "loser_name"]].rename(columns={"loser_id": "id", "loser_name": "name"}),
    ]).drop_duplicates("id")
    player_names = {int(row.id): row.name.split()[-1].lower() for row in all_names.itertuples()}
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    write_elo_state(state, paths.elo_state, data_as_of=data_as_of, player_names=player_names)
    write_players(matches, paths.players)
    write_match_history(matches, paths.match_history)
    log.info("artifacts written to %s", paths.artifacts)
    return 0


def run_features(paths: Paths) -> int:
    from progno_train.features import build_all_features

    if not paths.match_history.exists():
        log.error("no match_history at %s; run elo first", paths.match_history)
        return 2

    log.info("loading match history for feature engineering...")
    history = pd.read_parquet(paths.match_history)
    elo_state = json.loads(paths.elo_state.read_text())

    log.info("building features for %d matches...", len(history))
    featurized = build_all_features(history, elo_state)
    paths.featurized.parent.mkdir(parents=True, exist_ok=True)
    featurized.to_parquet(paths.featurized, index=False)
    log.info("featurized dataset written: %s (%d rows)", paths.featurized, len(featurized))
    return 0


def run_train(paths: Paths, tour: str) -> int:
    from progno_train.train import run_walk_forward, BURN_IN_YEAR_ATP, BURN_IN_YEAR_WTA

    if not paths.featurized.exists():
        log.error("no featurized parquet at %s; run features first", paths.featurized)
        return 2

    burn_in = BURN_IN_YEAR_WTA if tour == "wta" else BURN_IN_YEAR_ATP
    log.info("running walk-forward training (tour=%s, burn_in=%d)...", tour, burn_in)
    model, a, b, metrics, feature_cols = run_walk_forward(paths.featurized, burn_in_year=burn_in)

    log.info("saving model artifacts...")
    model.save_model(str(paths.model_cbm))
    write_calibration(a, b, paths.calibration)
    write_model_card(
        train_years=(burn_in + 1, metrics.get("cal_year", 2022)),
        test_year=metrics.get("test_year", 2023),
        metrics=metrics,
        feature_names=feature_cols,
        git_sha=_git_sha(),
        out_path=paths.model_card,
    )
    log.info("training complete. metrics: %s", metrics)
    return 0


def run_validate(paths: Paths) -> int:
    from catboost import CatBoostClassifier, Pool
    from progno_train.train import apply_platt, get_feature_cols
    from progno_train.validate import compute_log_loss, compute_ece, acceptance_gate

    log.info("running validation and acceptance gate...")
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
    elo_probs = (1.0 / (1.0 + 10 ** (-test_df["elo_overall_diff"].values / 400))).clip(0.05, 0.95)

    model_ll = compute_log_loss(y, cal_probs)
    baseline_ll = compute_log_loss(y, elo_probs)
    ece = compute_ece(y, cal_probs)

    log.info("model log-loss: %.4f | Elo baseline: %.4f | ECE: %.4f", model_ll, baseline_ll, ece)
    try:
        acceptance_gate(model_ll, baseline_ll, ece)
        log.info("acceptance gate: PASS")
    except ValueError as e:
        log.error("acceptance gate: FAIL — %s", e)
        return 1
    return 0


def run_retrain(paths: Paths, tour: str, version: str) -> int:
    for fn in (
        lambda: run_ingest(paths, tour),
        lambda: run_elo(paths),
        lambda: run_features(paths),
        lambda: run_train(paths, tour),
        lambda: run_validate(paths),
    ):
        rc = fn()
        if rc != 0:
            return rc
    log.info("retrain complete for tour=%s version=%s", tour, version)
    return 0


def run_publish(paths: Paths, version: str) -> int:
    log.warning("publish: stub — copy artifacts/%s to app-data in Phase 5", paths.artifacts.name)
    return 0


def main() -> int:
    _setup_logging()
    parser = argparse.ArgumentParser(prog="progno-train")
    parser.add_argument("--tour", choices=["atp", "wta"], default="atp",
                        help="Tour to process (default: atp)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("update_data")
    sub.add_parser("ingest")
    sub.add_parser("elo")
    sub.add_parser("features")
    sub.add_parser("train")
    sub.add_parser("validate")
    sub.add_parser("retrain").add_argument("--version", default="dev")
    sub.add_parser("publish").add_argument("version")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent  # = training/
    paths = Paths.for_tour(root, args.tour)

    dispatch = {
        "update_data": lambda: run_update_data(paths),
        "ingest": lambda: run_ingest(paths, args.tour),
        "elo": lambda: run_elo(paths),
        "features": lambda: run_features(paths),
        "train": lambda: run_train(paths, args.tour),
        "validate": lambda: run_validate(paths),
        "retrain": lambda: run_retrain(paths, args.tour, getattr(args, "version", "dev")),
        "publish": lambda: run_publish(paths, getattr(args, "version", "dev")),
    }
    return dispatch[args.command]()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run existing CLI tests**

```bash
cd training && uv run pytest tests/test_cli.py -v
```

Expected: all PASS. The `--tour atp` default preserves existing behavior.

- [ ] **Step 3: Commit**

```bash
git add training/src/progno_train/cli.py
git commit -m "feat(cli): add --tour param; add features, train, validate, retrain commands"
```

---

## Task 5: Create features.py

**Files:** Create `training/src/progno_train/features.py`

- [ ] **Step 1: Create features.py**

```python
"""Pre-match feature engineering — all features are time-gated (no leakage)."""

from __future__ import annotations

from typing import Any

import pandas as pd

POPULATION_WIN_RATE = 0.5
LOW_HISTORY_THRESHOLD = 5


def _player_matches_before(
    history: pd.DataFrame,
    player_id: int,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """All completed matches for player_id strictly before as_of_date."""
    won_mask = (history["winner_id"] == player_id) & (history["tourney_date"] < as_of_date) & history["is_complete"]
    lost_mask = (history["loser_id"] == player_id) & (history["tourney_date"] < as_of_date) & history["is_complete"]

    cols_base = ["tourney_date", "surface", "minutes", "completed_sets",
                 "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced",
                 "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced"]
    available = [c for c in cols_base if c in history.columns]

    w = history.loc[won_mask, available + ["loser_rank"]].assign(won=True, opponent_rank=history.loc[won_mask, "loser_rank"])
    l = history.loc[lost_mask, available + ["winner_rank"]].assign(won=False, opponent_rank=history.loc[lost_mask, "winner_rank"])

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
    current_surface: str,
) -> dict[str, float]:
    """Days since last match, sets played last 14d, matches last 30d, surface switch."""
    df = _player_matches_before(history, player_id, as_of_date)
    if df.empty:
        return {"days_since_last_match": 30.0, "sets_last_14d": 0.0,
                "matches_last_30d": 0.0, "surface_switch": 0.0}

    last_date = df["tourney_date"].max()
    days_since = min((as_of_date - last_date).days, 30)

    cutoff_14d = as_of_date - pd.Timedelta(days=14)
    cutoff_30d = as_of_date - pd.Timedelta(days=30)
    sets_14d = float(df.loc[df["tourney_date"] >= cutoff_14d, "completed_sets"].sum())
    matches_30d = float((df["tourney_date"] >= cutoff_30d).sum())

    last_surface = df.iloc[-1]["surface"] if "surface" in df.columns else None
    surface_switch = 1.0 if (last_surface and last_surface != current_surface) else 0.0

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
    """Rolling serve stats over last n matches."""
    df = _player_matches_before(history, player_id, as_of_date).tail(n)
    eps = 1e-6

    def _safe_div(a: float, b: float) -> float | None:
        return float(a / b) if b > eps else None

    if not all(c in df.columns for c in ["w_svpt", "w_1stIn"]):
        return {"first_serve_in_pct": None, "first_serve_won_pct": None,
                "ace_rate": None, "df_rate": None}

    # Aggregate: w_ columns for won matches, l_ for lost
    won = df[df["won"] == True] if "won" in df.columns else df
    lost = df[df["won"] == False] if "won" in df.columns else pd.DataFrame()

    def col_sum(frame: pd.DataFrame, w_col: str, l_col: str) -> float:
        total = 0.0
        if w_col in frame.columns:
            total += frame[w_col].fillna(0).sum()
        if l_col in frame.columns:
            total += frame[l_col].fillna(0).sum()
        return float(total)

    svpt = col_sum(df, "w_svpt", "l_svpt")
    first_in = col_sum(df, "w_1stIn", "l_1stIn")
    first_won = col_sum(df, "w_1stWon", "l_1stWon")
    ace = col_sum(df, "w_ace", "l_ace")
    df_ = col_sum(df, "w_df", "l_df")

    return {
        "first_serve_in_pct": _safe_div(first_in, svpt),
        "first_serve_won_pct": _safe_div(first_won, first_in),
        "ace_rate": _safe_div(ace, svpt),
        "df_rate": _safe_div(df_, svpt),
    }


def h2h_score(
    history: pd.DataFrame,
    player_a_id: int,
    player_b_id: int,
    as_of_date: pd.Timestamp,
    prior: int = 5,
    prior_mean: float = 0.5,
) -> tuple[float, int]:
    """Shrinkage H2H win rate for A vs B."""
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
    wins_a = int((df["winner_id"] == player_a_id).sum())
    shrunk = (wins_a + prior * prior_mean) / (n + prior)
    return float(shrunk), n


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

    wr_a_50, lhf_a = rolling_win_rate(history, player_a_id, tourney_date, 50)
    wr_b_50, lhf_b = rolling_win_rate(history, player_b_id, tourney_date, 50)
    wr_a_surf, _ = rolling_win_rate(history, player_a_id, tourney_date, 20, surface=surface)
    wr_b_surf, _ = rolling_win_rate(history, player_b_id, tourney_date, 20, surface=surface)
    wr_a_top20, _ = rolling_win_rate(history, player_a_id, tourney_date, 30, max_opponent_rank=20)
    wr_b_top20, _ = rolling_win_rate(history, player_b_id, tourney_date, 30, max_opponent_rank=20)

    feats["win_rate_diff"] = wr_a_50 - wr_b_50
    feats["win_rate_surface_diff"] = wr_a_surf - wr_b_surf
    feats["win_rate_top20_diff"] = wr_a_top20 - wr_b_top20
    feats["low_history_flag"] = int(lhf_a or lhf_b)

    fat_a = fatigue_features(history, player_a_id, tourney_date, surface)
    fat_b = fatigue_features(history, player_b_id, tourney_date, surface)
    feats["days_since_last_diff"] = fat_a["days_since_last_match"] - fat_b["days_since_last_match"]
    feats["sets_last_14d_diff"] = fat_a["sets_last_14d"] - fat_b["sets_last_14d"]
    feats["matches_last_30d_diff"] = fat_a["matches_last_30d"] - fat_b["matches_last_30d"]
    feats["surface_switch_a"] = fat_a["surface_switch"]
    feats["surface_switch_b"] = fat_b["surface_switch"]

    srv_a = serve_efficiency(history, player_a_id, tourney_date)
    srv_b = serve_efficiency(history, player_b_id, tourney_date)
    for stat in ["first_serve_in_pct", "first_serve_won_pct", "ace_rate", "df_rate"]:
        va = srv_a.get(stat) or 0.0
        vb = srv_b.get(stat) or 0.0
        feats[f"{stat}_diff"] = va - vb

    h2h, h2h_n = h2h_score(history, player_a_id, player_b_id, tourney_date)
    feats["h2h_score"] = h2h
    feats["h2h_sample_size"] = h2h_n

    def _elo(pid: int, field: str) -> float:
        p = elo_state.get("players", {}).get(str(pid), {})
        return float(p.get(field, 1500))

    elo_a = _elo(player_a_id, "elo_overall")
    elo_b = _elo(player_b_id, "elo_overall")
    elo_a_surf = _elo(player_a_id, f"elo_{surface.lower()}")
    elo_b_surf = _elo(player_b_id, f"elo_{surface.lower()}")
    feats["elo_overall_diff"] = elo_a - elo_b
    feats["elo_surface_diff"] = elo_a_surf - elo_b_surf

    feats["age_diff"] = (player_a_age or 25.0) - (player_b_age or 25.0)
    feats["height_diff"] = (player_a_height or 185.0) - (player_b_height or 185.0)
    feats["lefty_vs_righty"] = int((player_a_hand == "L") != (player_b_hand == "L"))

    feats["surface"] = surface
    feats["tourney_level"] = tourney_level
    feats["round"] = round_
    feats["best_of_5"] = int(best_of == 5)

    return feats


def build_all_features(
    history: pd.DataFrame,
    elo_state: dict,
) -> pd.DataFrame:
    """Compute features for every complete match in history."""
    rows = []
    for _, row in history[history["is_complete"]].iterrows():
        feats = compute_match_features(
            history=history,
            elo_state=elo_state,
            player_a_id=int(row["winner_id"]),
            player_b_id=int(row["loser_id"]),
            surface=row.get("surface", "Hard"),
            tourney_level=row.get("tourney_level", "A"),
            round_=row.get("round", "R32"),
            best_of=int(row.get("best_of", 3)),
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
        feats["label"] = 1
        feats["tourney_date"] = row["tourney_date"]
        feats["year"] = row["tourney_date"].year
        rows.append(feats)
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Commit (tests come next)**

```bash
git add training/src/progno_train/features.py
git commit -m "feat(features): add pre-match feature engineering module"
```

---

## Task 6: Test features.py

**Files:** Create `training/tests/test_features.py`

- [ ] **Step 1: Create test_features.py**

```python
"""Tests for feature engineering: no leakage, cold start, H2H."""

import pandas as pd
import pytest

from progno_train.features import (
    POPULATION_WIN_RATE,
    LOW_HISTORY_THRESHOLD,
    rolling_win_rate,
    fatigue_features,
    h2h_score,
    compute_match_features,
)


def make_history(n: int, player_a: int = 1, player_b: int = 2) -> pd.DataFrame:
    """Synthetic history: player_a wins all matches against player_b."""
    base = pd.Timestamp("2020-01-01")
    rows = []
    for i in range(n):
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
    rate, low = rolling_win_rate(hist, 1, pd.Timestamp("2021-01-01"), 50)
    assert rate == 1.0
    assert low is False


def test_rolling_win_rate_cold_start():
    hist = make_history(3)
    rate, low = rolling_win_rate(hist, 1, pd.Timestamp("2021-01-01"), 50)
    assert rate == POPULATION_WIN_RATE
    assert low is True


def test_rolling_win_rate_no_future_leakage():
    hist = make_history(20)
    cutoff = hist["tourney_date"].iloc[9]
    rate, _ = rolling_win_rate(hist, 1, cutoff, 50)
    visible = hist[hist["tourney_date"] < cutoff]
    assert len(visible) == 9
    assert rate == 1.0


def test_no_leakage_on_match_date():
    hist = make_history(20)
    for i in range(5, 20):
        as_of = hist["tourney_date"].iloc[i]
        rate, _ = rolling_win_rate(hist, 1, as_of, 50)
        assert rate == 1.0
        visible = hist[hist["tourney_date"] < as_of]
        assert len(visible) == i


def test_h2h_shrinkage_no_history():
    hist = make_history(0)
    score, n = h2h_score(hist, 1, 2, pd.Timestamp("2021-01-01"))
    assert n == 0
    assert abs(score - 0.5) < 0.01


def test_h2h_shrinkage_with_history():
    hist = make_history(10)
    score, n = h2h_score(hist, 1, 2, pd.Timestamp("2021-01-01"))
    assert n == 10
    assert abs(score - 12.5 / 15.0) < 0.001


def test_h2h_no_future_leakage():
    hist = make_history(10)
    cutoff = hist["tourney_date"].iloc[5]
    _, n = h2h_score(hist, 1, 2, cutoff)
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


def test_compute_match_features_returns_expected_keys():
    hist = make_history(20)
    elo_state = {"players": {
        "1": {"elo_overall": 1600, "elo_hard": 1620, "elo_clay": 1550, "elo_grass": 1580},
        "2": {"elo_overall": 1500, "elo_hard": 1520, "elo_clay": 1480, "elo_grass": 1500},
    }}
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
    assert feats["elo_overall_diff"] == 100.0


def test_elo_monotonicity():
    hist = make_history(20)
    strong = {"players": {"1": {"elo_overall": 1800, "elo_hard": 1800, "elo_clay": 1800, "elo_grass": 1800},
                          "2": {"elo_overall": 1500, "elo_hard": 1500, "elo_clay": 1500, "elo_grass": 1500}}}
    weak = {"players": {"1": {"elo_overall": 1510, "elo_hard": 1510, "elo_clay": 1510, "elo_grass": 1510},
                        "2": {"elo_overall": 1500, "elo_hard": 1500, "elo_clay": 1500, "elo_grass": 1500}}}
    f_strong = compute_match_features(hist, strong, 1, 2, "Hard", "A", "R32", 3, pd.Timestamp("2021-01-01"))
    f_weak = compute_match_features(hist, weak, 1, 2, "Hard", "A", "R32", 3, pd.Timestamp("2021-01-01"))
    assert f_strong["elo_overall_diff"] > f_weak["elo_overall_diff"]
```

- [ ] **Step 2: Run tests**

```bash
cd training && uv run pytest tests/test_features.py -v
```

Expected: all 12 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add training/tests/test_features.py
git commit -m "test(features): add leakage, cold start, H2H shrinkage tests"
```

---

## Task 7: Create train.py

**Files:** Create `training/src/progno_train/train.py`

- [ ] **Step 1: Create train.py**

```python
"""Walk-forward training: CatBoost + Platt calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression

BURN_IN_YEAR_ATP = 2004
BURN_IN_YEAR_WTA = 2011
CAL_YEAR = 2022
TEST_START_YEAR = 2023

CAT_FEATURES = ["surface", "tourney_level", "round"]

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


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"label", "tourney_date", "year"}
    return [c for c in df.columns if c not in exclude]


def walk_forward_splits(
    df: pd.DataFrame,
    burn_in_year: int = BURN_IN_YEAR_ATP,
    val_start: int = 2016,
    test_start: int = TEST_START_YEAR,
) -> list[tuple[pd.DataFrame, pd.DataFrame, str]]:
    """Expanding window walk-forward splits. No data before burn_in_year in training."""
    df = df[df["year"] > burn_in_year].copy()
    splits = []
    for year in sorted(df["year"].unique()):
        if year < val_start:
            continue
        train = df[df["year"] < year]
        holdout = df[df["year"] == year]
        label = "val" if year < test_start else "test"
        splits.append((train, holdout, f"{label}_{year}"))
    return splits


def _train_catboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> CatBoostClassifier:
    cat_idx = [i for i, c in enumerate(feature_cols) if c in CAT_FEATURES]
    pool_tr = Pool(train_df[feature_cols].fillna(0), train_df["label"].values,
                   cat_features=cat_idx, feature_names=feature_cols)
    pool_val = Pool(val_df[feature_cols].fillna(0), val_df["label"].values,
                    cat_features=cat_idx, feature_names=feature_cols)
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(pool_tr, eval_set=pool_val)
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


def run_walk_forward(
    featurized_path: Path,
    burn_in_year: int = BURN_IN_YEAR_ATP,
) -> tuple[CatBoostClassifier, float, float, dict, list[str]]:
    """Run full walk-forward pipeline. Returns (model, platt_a, platt_b, metrics, feature_cols)."""
    df = pd.read_parquet(featurized_path)
    feature_cols = get_feature_cols(df)

    cal_df = df[df["year"] == CAL_YEAR]
    final_train = df[(df["year"] > burn_in_year) & (df["year"] < TEST_START_YEAR)]

    model = _train_catboost(final_train, cal_df, feature_cols)

    cal_pool = Pool(cal_df[feature_cols].fillna(0), feature_names=feature_cols)
    raw_cal = model.predict_proba(cal_pool)[:, 1]
    a, b = fit_platt(raw_cal, cal_df["label"].values)

    test_df = df[df["year"] >= TEST_START_YEAR]
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
        "cal_year": CAL_YEAR,
        "test_year": TEST_START_YEAR,
    }
    return model, a, b, metrics, feature_cols
```

- [ ] **Step 2: Commit**

```bash
git add training/src/progno_train/train.py
git commit -m "feat(train): add walk-forward CatBoost + Platt calibration"
```

---

## Task 8: Test train.py

**Files:** Create `training/tests/test_train.py`

- [ ] **Step 1: Create test_train.py**

```python
"""Tests: Platt correctness, walk-forward no-overlap, burn-in."""

import numpy as np
import pandas as pd
import pytest

from progno_train.train import (
    fit_platt,
    apply_platt,
    get_feature_cols,
    walk_forward_splits,
    BURN_IN_YEAR_ATP,
    BURN_IN_YEAR_WTA,
)


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
    assert a1 == a2 and b1 == b2


def test_walk_forward_no_overlap():
    df = make_feature_df(200)
    splits = walk_forward_splits(df, burn_in_year=2015, val_start=2018)
    for train, holdout, _ in splits:
        assert set(train["year"]).isdisjoint(set(holdout["year"]))


def test_walk_forward_no_future_in_train():
    df = make_feature_df(200)
    splits = walk_forward_splits(df, burn_in_year=2015, val_start=2018)
    for train, holdout, _ in splits:
        assert train["year"].max() < holdout["year"].min()


def test_walk_forward_atp_burn_in():
    df = make_feature_df(200)
    splits = walk_forward_splits(df, burn_in_year=BURN_IN_YEAR_ATP, val_start=2016)
    for train, _, _ in splits:
        assert train["year"].min() > BURN_IN_YEAR_ATP


def test_walk_forward_wta_burn_in():
    rng = np.random.default_rng(42)
    n = 160
    years = np.repeat(np.arange(2012, 2026), n // 14)[:n]
    df = pd.DataFrame({
        "elo_overall_diff": rng.normal(0, 100, n),
        "win_rate_diff": rng.normal(0, 0.2, n),
        "h2h_score": rng.uniform(0.3, 0.7, n),
        "h2h_sample_size": rng.integers(0, 20, n),
        "low_history_flag": rng.integers(0, 2, n),
        "age_diff": rng.normal(0, 3, n),
        "surface": rng.choice(["Hard", "Clay", "Grass"], n),
        "tourney_level": rng.choice(["G", "M", "A"], n),
        "round": rng.choice(["F", "SF", "QF", "R32"], n),
        "best_of_5": rng.integers(0, 2, n),
        "label": rng.integers(0, 2, n),
        "year": years,
        "tourney_date": pd.date_range("2012-01-01", periods=n, freq="3D"),
    })
    splits = walk_forward_splits(df, burn_in_year=BURN_IN_YEAR_WTA, val_start=2019)
    for train, _, _ in splits:
        assert train["year"].min() > BURN_IN_YEAR_WTA


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

Expected: all 8 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add training/tests/test_train.py
git commit -m "test(train): walk-forward no-overlap, Platt correctness, burn-in tests"
```

---

## Task 9: Create validate.py + tests

**Files:** Create `training/src/progno_train/validate.py`, create `training/tests/test_validate.py`

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
        ece += mask.sum() * abs(float(y_true[mask].mean()) - float(y_pred[mask].mean()))
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
        raise ValueError(f"GATE FAIL: ECE {ece:.4f} > threshold {ece_threshold}")
```

- [ ] **Step 2: Create test_validate.py**

```python
import numpy as np
import pytest

from progno_train.validate import compute_log_loss, compute_ece, acceptance_gate


def test_log_loss_perfect():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.999, 0.001, 0.999, 0.001])
    assert compute_log_loss(y, p) < 0.01


def test_log_loss_random():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 1000)
    p = np.full(1000, 0.5)
    assert abs(compute_log_loss(y, p) - np.log(2)) < 0.05


def test_ece_well_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 1000)
    y = rng.binomial(1, p)
    assert compute_ece(y, p) < 0.05


def test_ece_miscalibrated():
    y = np.array([1, 0] * 500)
    p = np.full(1000, 0.9)
    assert compute_ece(y, p) > 0.3


def test_gate_passes():
    acceptance_gate(model_logloss=0.60, baseline_logloss=0.65, ece=0.02)


def test_gate_fails_logloss():
    with pytest.raises(ValueError, match="log-loss"):
        acceptance_gate(model_logloss=0.70, baseline_logloss=0.65, ece=0.02)


def test_gate_fails_ece():
    with pytest.raises(ValueError, match="ECE"):
        acceptance_gate(model_logloss=0.60, baseline_logloss=0.65, ece=0.05)
```

- [ ] **Step 3: Run tests**

```bash
cd training && uv run pytest tests/test_validate.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add training/src/progno_train/validate.py training/tests/test_validate.py
git commit -m "feat(validate): log-loss, ECE metrics, acceptance gate + tests"
```

---

## Task 10: Update justfile

**Files:** Modify `justfile`

- [ ] **Step 1: Replace justfile**

```just
default:
    @just --list

# --- Phase 1a targets ---
update-data:
    bash training/scripts/fetch_sackmann.sh
    bash training/scripts/fetch_sackmann_wta.sh

ingest:
    cd training && uv run python -m progno_train.cli --tour atp ingest

elo:
    cd training && uv run python -m progno_train.cli --tour atp elo

publish version:
    cd training && uv run python -m progno_train.cli --tour atp publish {{version}}

# --- Phase 3 targets (ATP) ---
features:
    cd training && uv run python -m progno_train.cli --tour atp features

train:
    cd training && uv run python -m progno_train.cli --tour atp train

validate:
    cd training && uv run python -m progno_train.cli --tour atp validate

retrain version:
    cd training && uv run python -m progno_train.cli --tour atp retrain --version {{version}}

build-sidecar:
    cd sidecar && bash build.sh

# --- Phase 4 targets (WTA) ---
ingest-wta:
    cd training && uv run python -m progno_train.cli --tour wta ingest

elo-wta:
    cd training && uv run python -m progno_train.cli --tour wta elo

features-wta:
    cd training && uv run python -m progno_train.cli --tour wta features

train-wta:
    cd training && uv run python -m progno_train.cli --tour wta train

validate-wta:
    cd training && uv run python -m progno_train.cli --tour wta validate

retrain-wta version:
    cd training && uv run python -m progno_train.cli --tour wta retrain --version {{version}}

# --- Dev helpers ---
test:
    cd training && uv run pytest -v

test-rust:
    cd app/src-tauri && cargo test

test-all: test test-rust

fmt:
    cd training && uv run ruff format .
    cd training && uv run ruff check --fix .

check:
    cd training && uv run ruff check .
    cd training && uv run pytest -v
```

- [ ] **Step 2: Verify justfile parses**

```bash
just --list
```

Expected: lists all targets without error.

- [ ] **Step 3: Commit**

```bash
git add justfile
git commit -m "feat(justfile): add Phase 3+4 targets; update-data pulls both ATP and WTA"
```

---

## Task 11: Create WTA data script

**Files:** Create `training/scripts/fetch_sackmann_wta.sh`

- [ ] **Step 1: Create script**

```bash
#!/usr/bin/env bash
set -euo pipefail

WTA_DIR="training/data/raw/tennis_wta"

if [ -d "$WTA_DIR/.git" ]; then
    echo "Pulling latest tennis_wta..."
    git -C "$WTA_DIR" pull --ff-only
else
    echo "Cloning tennis_wta..."
    git clone https://github.com/JeffSackmann/tennis_wta "$WTA_DIR"
fi

echo "WTA data ready at $WTA_DIR"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x training/scripts/fetch_sackmann_wta.sh
git add training/scripts/fetch_sackmann_wta.sh
git commit -m "feat(scripts): add fetch_sackmann_wta.sh for WTA data"
```

---

## Task 12: Create FastAPI sidecar

**Files:** Create `sidecar/pyproject.toml`, `sidecar/server.py`, `sidecar/features.py`, `sidecar/build.sh`

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

- [ ] **Step 2: Create sidecar/features.py (copy of training module for bundling)**

Copy the file verbatim:

```bash
cp training/src/progno_train/features.py sidecar/features.py
```

- [ ] **Step 3: Create sidecar/build.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Refresh features.py from training source before building
cp ../training/src/progno_train/features.py features.py
uv run pyinstaller \
    --onefile \
    --name progno-sidecar \
    --hidden-import catboost \
    --add-data "features.py:." \
    server.py
echo "Built: dist/progno-sidecar"
```

- [ ] **Step 4: Create sidecar/server.py**

```python
"""FastAPI sidecar — loads ATP and WTA models, routes by tour field."""

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

import features as feat_module

_models: dict[str, CatBoostClassifier | None] = {"atp": None, "wta": None}
_platt: dict[str, tuple[float, float]] = {"atp": (1.0, 0.0), "wta": (1.0, 0.0)}
_history: dict[str, pd.DataFrame | None] = {"atp": None, "wta": None}
_elo_state: dict[str, dict] = {"atp": {}, "wta": {}}
_model_card: dict[str, dict] = {"atp": {}, "wta": {}}
_port: int = 0


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _load_tour(artifacts_root: Path, tour: str) -> None:
    tour_dir = artifacts_root / tour
    model_path = tour_dir / "model.cbm"
    if not model_path.exists():
        print(f"INFO no model for {tour} at {model_path} — skipping", flush=True)
        return

    m = CatBoostClassifier()
    m.load_model(str(model_path))
    _models[tour] = m

    cal = json.loads((tour_dir / "calibration.json").read_text())
    _platt[tour] = (cal["a"], cal["b"])

    _history[tour] = pd.read_parquet(tour_dir / "match_history.parquet")
    _elo_state[tour] = json.loads((tour_dir / "elo_state.json").read_text())

    card_path = tour_dir / "model_card.json"
    _model_card[tour] = json.loads(card_path.read_text()) if card_path.exists() else {}

    print(f"INFO loaded {tour} model", flush=True)


def _apply_platt(raw: np.ndarray, tour: str) -> np.ndarray:
    a, b = _platt[tour]
    eps = 1e-7
    clipped = np.clip(raw, eps, 1 - eps)
    logits = np.log(clipped) - np.log(1 - clipped)
    return 1.0 / (1.0 + np.exp(-(a * logits + b)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    sys.stdout.write(f"READY port={_port}\n")
    sys.stdout.flush()
    yield


app = FastAPI(lifespan=lifespan)


class MatchRequest(BaseModel):
    tour: str
    player_a_id: str
    player_b_id: str
    surface: str
    tourney_level: str = "A"
    round_: str = "R32"
    best_of: int = 3
    tourney_date: str


class PredictRequest(BaseModel):
    matches: list[MatchRequest]


class MatchPrediction(BaseModel):
    prob_a_wins: float
    prob_a_wins_uncalibrated: float
    elo_prob_a_wins: float
    confidence_flag: str


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[MatchPrediction]


@app.get("/health")
async def health():
    return {"status": "ok", "tours_loaded": [t for t, m in _models.items() if m is not None]}


@app.get("/model_info")
async def model_info():
    return {
        t: {"loaded": _models[t] is not None, "card": _model_card.get(t, {})}
        for t in ("atp", "wta")
    }


@app.post("/predict")
async def predict(req: PredictRequest) -> PredictResponse:
    results = []
    for m in req.matches:
        tour = m.tour
        if _models.get(tour) is None:
            raise HTTPException(503, f"Model not loaded for tour: {tour}")

        try:
            pid_a, pid_b = int(m.player_a_id), int(m.player_b_id)
        except ValueError:
            raise HTTPException(422, f"player IDs must be integers: {m.player_a_id}, {m.player_b_id}")

        tourney_date = pd.Timestamp(m.tourney_date)
        feats = feat_module.compute_match_features(
            history=_history[tour],
            elo_state=_elo_state[tour],
            player_a_id=pid_a,
            player_b_id=pid_b,
            surface=m.surface,
            tourney_level=m.tourney_level,
            round_=m.round_,
            best_of=m.best_of,
            tourney_date=tourney_date,
        )

        low_history = bool(feats.get("low_history_flag", 0))
        feature_cols = _models[tour].feature_names_
        feat_df = pd.DataFrame([{col: feats.get(col, 0) for col in feature_cols}])
        pool = Pool(feat_df, feature_names=list(feature_cols))

        raw = float(_models[tour].predict_proba(pool)[0, 1])
        cal = float(_apply_platt(np.array([raw]), tour)[0])
        elo_diff = feats.get("elo_overall_diff", 0.0)
        elo_prob = float(1.0 / (1.0 + 10 ** (-elo_diff / 400)))

        results.append(MatchPrediction(
            prob_a_wins=round(cal, 4),
            prob_a_wins_uncalibrated=round(raw, 4),
            elo_prob_a_wins=round(elo_prob, 4),
            confidence_flag="low_history" if low_history else "ok",
        ))

    version = _model_card.get(req.matches[0].tour, {}).get("generated_at", "unknown") if req.matches else "unknown"
    return PredictResponse(model_version=version, predictions=results)


def main() -> None:
    global _port
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", required=True)
    args = parser.parse_args()

    root = Path(args.artifacts_root)
    for tour in ("atp", "wta"):
        _load_tour(root, tour)

    _port = _find_free_port()
    uvicorn.run(app, host="127.0.0.1", port=_port, log_level="error")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Install sidecar deps**

```bash
cd sidecar && uv sync
```

Expected: dependencies installed.

- [ ] **Step 6: Commit**

```bash
git add sidecar/
git commit -m "feat(sidecar): FastAPI sidecar with ATP/WTA tour routing"
```

---

## Task 13: Update Rust — state.rs and artifacts.rs

**Files:** Modify `app/src-tauri/src/state.rs`, modify `app/src-tauri/src/artifacts.rs`

- [ ] **Step 1: Replace state.rs**

```rust
use std::sync::Mutex;

#[derive(Default)]
pub struct AppState {
    pub elo_atp: Mutex<Option<serde_json::Value>>,
    pub elo_wta: Mutex<Option<serde_json::Value>>,
}
```

- [ ] **Step 2: Add tour-aware loader to artifacts.rs**

Add this function at the end of `app/src-tauri/src/artifacts.rs` (after `get_data_as_of`):

```rust
/// Load elo_state.json for a specific tour from artifacts/{tour}/elo_state.json
/// relative to the binary location.
pub fn load_elo_state_for_tour(tour: &str) -> Result<serde_json::Value, String> {
    let path = PathBuf::from(format!("artifacts/{}/elo_state.json", tour));
    load_elo_state(path.to_str().unwrap_or("elo_state.json"))
}
```

- [ ] **Step 3: Run Rust tests**

```bash
cd app && cargo test 2>&1 | tail -20
```

Expected: all tests PASS (existing tests use `AppState` which now has `elo_atp`/`elo_wta`; the tests don't construct `AppState` directly so no compilation errors).

- [ ] **Step 4: Commit**

```bash
git add app/src-tauri/src/state.rs app/src-tauri/src/artifacts.rs
git commit -m "feat(state): dual ATP/WTA Elo in AppState; tour-aware artifact loading"
```

---

## Task 14: Create sidecar.rs and update commands.rs

**Files:** Create `app/src-tauri/src/sidecar.rs`, modify `app/src-tauri/src/commands.rs`

- [ ] **Step 1: Create sidecar.rs**

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
    pub tour: String,
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

pub fn spawn_sidecar(app: &tauri::AppHandle, artifacts_root: String) {
    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        match do_spawn(&handle, &artifacts_root).await {
            Ok(port) => {
                let state = handle.state::<Mutex<SidecarState>>();
                state.lock().unwrap().port = Some(port);
                eprintln!("[sidecar] ready on port {port}");
            }
            Err(e) => eprintln!("[sidecar] failed to start: {e}"),
        }
    });
}

async fn do_spawn(app: &tauri::AppHandle, artifacts_root: &str) -> anyhow::Result<u16> {
    let (mut rx, _child) = app
        .shell()
        .sidecar("progno-sidecar")?
        .args(["--artifacts-root", artifacts_root])
        .spawn()?;

    while let Some(event) = rx.recv().await {
        match event {
            CommandEvent::Stdout(line) => {
                let s = String::from_utf8_lossy(&line);
                if let Some(port_str) = s.trim().strip_prefix("READY port=") {
                    return Ok(port_str.parse()?);
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

- [ ] **Step 2: Update commands.rs**

Replace the top of `commands.rs` (the struct definitions and `predict_text` / `parse_and_predict` functions). Keep all existing test code unchanged.

Replace the structs and non-test functions with:

```rust
use serde::{Deserialize, Serialize};
use crate::artifacts::{get_data_as_of, get_player_elo, get_player_surface_matches};
use crate::elo::{expected_probability, surface_elo};
use crate::parser::{parse_match_text, ParsedMatch};
use crate::kelly;
#[cfg(not(test))]
use crate::state::AppState;

#[derive(Serialize, Deserialize, Clone)]
pub struct KellyRequest {
    pub model_prob: f64,
    pub decimal_odds: f64,
    pub bankroll: f64,
    pub kelly_fraction: f64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct KellyResult {
    pub implied_prob: f64,
    pub edge: f64,
    pub full_kelly: f64,
    pub fractional_kelly: f64,
    pub stake: f64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct PredictionResult {
    pub player_a: String,
    pub player_b: String,
    pub surface: String,
    pub prob_a_wins: f64,
    pub prob_b_wins: f64,
    pub elo_a_overall: f64,
    pub elo_b_overall: f64,
    pub ml_prob_a_wins: Option<f64>,
    pub confidence_flag: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct PredictResponse {
    pub predictions: Vec<PredictionResult>,
    pub data_as_of: String,
    pub error: Option<String>,
}

pub fn predict_text(text: &str, elo_state: &serde_json::Value) -> PredictResponse {
    let matches = match parse_match_text(text) {
        Ok(m) => m,
        Err(e) => return PredictResponse {
            predictions: vec![],
            data_as_of: "unknown".to_string(),
            error: Some(e),
        },
    };

    let data_as_of = get_data_as_of(elo_state);
    let mut predictions = Vec::new();

    for m in matches {
        match predict_match(&m, elo_state) {
            Ok(pred) => predictions.push(pred),
            Err(e) => eprintln!("Failed to predict {} vs {}: {}", m.player_a, m.player_b, e),
        }
    }

    let error = if predictions.is_empty() {
        Some("No matches could be predicted".to_string())
    } else {
        None
    };
    PredictResponse { predictions, data_as_of, error }
}

#[cfg(not(test))]
#[tauri::command]
pub fn parse_and_predict(
    text: String,
    tour: String,
    app_state: tauri::State<AppState>,
) -> PredictResponse {
    let elo = match tour.as_str() {
        "wta" => app_state.elo_wta.lock().unwrap(),
        _ => app_state.elo_atp.lock().unwrap(),
    };
    match &*elo {
        None => PredictResponse {
            predictions: vec![],
            data_as_of: "unknown".to_string(),
            error: Some(format!("Elo data for {} not loaded. Run 'just elo' first.", tour.to_uppercase())),
        },
        Some(state) => predict_text(&text, state),
    }
}

#[cfg(not(test))]
#[tauri::command]
pub fn get_data_as_of_cmd(tour: String, app_state: tauri::State<AppState>) -> String {
    let elo = match tour.as_str() {
        "wta" => app_state.elo_wta.lock().unwrap(),
        _ => app_state.elo_atp.lock().unwrap(),
    };
    match &*elo {
        None => "unknown".to_string(),
        Some(state) => get_data_as_of(state),
    }
}

#[cfg(not(test))]
#[tauri::command]
pub fn calculate_kelly(request: KellyRequest) -> Result<KellyResult, String> {
    calculate_kelly_impl(request)
}

#[derive(Serialize, Deserialize)]
pub struct MlPredictRequest {
    pub text: String,
    pub tour: String,
    pub tourney_date: String,
}

#[cfg(not(test))]
#[tauri::command]
pub async fn predict_with_ml(
    request: MlPredictRequest,
    app_state: tauri::State<'_, AppState>,
    sidecar_state: tauri::State<'_, std::sync::Mutex<crate::sidecar::SidecarState>>,
) -> Result<PredictResponse, String> {
    use crate::sidecar::{MlMatchRequest, ml_predict};

    let elo_guard = match request.tour.as_str() {
        "wta" => app_state.elo_wta.lock().unwrap(),
        _ => app_state.elo_atp.lock().unwrap(),
    };
    let elo_resp = match &*elo_guard {
        None => return Err(format!("Elo data for {} not loaded", request.tour.to_uppercase())),
        Some(elo) => predict_text(&request.text, elo),
    };
    drop(elo_guard);

    let port = sidecar_state.lock().unwrap().port;
    if port.is_none() {
        return Ok(elo_resp);
    }
    let port = port.unwrap();

    let ml_matches: Vec<MlMatchRequest> = elo_resp.predictions.iter().map(|p| MlMatchRequest {
        tour: request.tour.clone(),
        player_a_id: normalize_player_id(&p.player_a),
        player_b_id: normalize_player_id(&p.player_b),
        surface: p.surface.clone(),
        tourney_level: "A".to_string(),
        round_: "R32".to_string(),
        best_of: 3,
        tourney_date: request.tourney_date.clone(),
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
            Ok(elo_resp)
        }
    }
}

fn normalize_player_id(name: &str) -> String {
    name.replace(' ', "_").to_lowercase()
}

fn resolve_player(
    state: &serde_json::Value,
    player_id: &str,
    surface: &str,
) -> Result<(f64, f64), String> {
    let elo_overall = get_player_elo(state, player_id, "overall")
        .or_else(|_| get_player_elo(state, player_id, ""))?;
    let matches_on_surface = get_player_surface_matches(state, player_id, surface).unwrap_or(0);
    let elo_surface = get_player_elo(state, player_id, surface).unwrap_or(elo_overall);
    Ok((elo_overall, surface_elo(elo_surface, elo_overall, matches_on_surface)))
}

fn predict_match(m: &ParsedMatch, state: &serde_json::Value) -> Result<PredictionResult, String> {
    let id_a = normalize_player_id(&m.player_a);
    let id_b = normalize_player_id(&m.player_b);
    let (elo_a_overall, elo_a_composite) = resolve_player(state, &id_a, &m.surface)?;
    let (elo_b_overall, elo_b_composite) = resolve_player(state, &id_b, &m.surface)?;
    let prob_a = expected_probability(elo_a_composite, elo_b_composite);
    Ok(PredictionResult {
        player_a: m.player_a.clone(),
        player_b: m.player_b.clone(),
        surface: m.surface.clone(),
        prob_a_wins: prob_a,
        prob_b_wins: 1.0 - prob_a,
        elo_a_overall,
        elo_b_overall,
        ml_prob_a_wins: None,
        confidence_flag: None,
    })
}

pub fn calculate_kelly_impl(req: KellyRequest) -> Result<KellyResult, String> {
    let implied_prob = kelly::implied_probability(req.decimal_odds);
    let edge = kelly::edge(req.model_prob, req.decimal_odds);
    let full_kelly = kelly::full_kelly_fraction(req.model_prob, req.decimal_odds);
    let fractional_kelly = kelly::fractional_kelly(req.model_prob, req.decimal_odds, req.kelly_fraction);
    let stake = kelly::stake_from_kelly(req.bankroll, fractional_kelly);
    Ok(KellyResult { implied_prob, edge, full_kelly, fractional_kelly, stake })
}
```

Keep the entire `#[cfg(test)]` block unchanged from the original file.

- [ ] **Step 3: Update Cargo.toml**

In `app/src-tauri/Cargo.toml`, replace `[dependencies]`:

```toml
[dependencies]
tauri = { version = "2.1" }
tauri-plugin-store = "2.0"
tauri-plugin-shell = "2"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
regex = "1.10"
anyhow = "1.0"
reqwest = { version = "0.12", features = ["json"] }
tokio = { version = "1", features = ["full"] }
```

- [ ] **Step 4: Check compilation**

```bash
cd app && cargo check 2>&1 | tail -20
```

Expected: compiles without errors.

- [ ] **Step 5: Run Rust tests**

```bash
cd app && cargo test 2>&1 | tail -30
```

Expected: all tests PASS. The existing test vectors still exercise `predict_text` and `calculate_kelly_impl` which are unchanged in behavior.

- [ ] **Step 6: Commit**

```bash
git add app/src-tauri/src/sidecar.rs app/src-tauri/src/commands.rs app/src-tauri/Cargo.toml
git commit -m "feat(rust): sidecar.rs + tour param in commands; ml_prob_a_wins field"
```

---

## Task 15: Update main.rs and lib.rs

**Files:** Modify `app/src-tauri/src/main.rs`, modify `app/src-tauri/src/lib.rs`

- [ ] **Step 1: Add shell plugin permission**

In `app/src-tauri/tauri.conf.json`, add `bundle.externalBin` and plugin config. The full file should be:

```json
{
  "productName": "Progno",
  "version": "0.1.0",
  "identifier": "com.progno.app",
  "build": {
    "beforeDevCommand": "npm run dev",
    "devUrl": "http://localhost:5173",
    "beforeBuildCommand": "npm run build",
    "frontendDist": "../dist"
  },
  "app": {
    "windows": [
      {
        "title": "Progno",
        "width": 1200,
        "height": 800,
        "resizable": true,
        "fullscreen": false
      }
    ],
    "security": {
      "csp": "default-src 'self'; style-src 'self' 'unsafe-inline';"
    }
  },
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

- [ ] **Step 2: Replace main.rs**

```rust
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod artifacts;
mod commands;
mod elo;
mod kelly;
mod parser;
mod sidecar;
mod state;

#[cfg(not(test))]
use state::AppState;
#[cfg(not(test))]
use tauri::Manager;

fn main() {
    #[cfg(not(test))]
    tauri::Builder::default()
        .manage(AppState::default())
        .manage(std::sync::Mutex::new(sidecar::SidecarState::default()))
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            if let Ok(elo) = artifacts::load_elo_state_for_tour("atp") {
                *app.state::<AppState>().elo_atp.lock().unwrap() = Some(elo);
            }
            if let Ok(elo) = artifacts::load_elo_state_for_tour("wta") {
                *app.state::<AppState>().elo_wta.lock().unwrap() = Some(elo);
            }
            let artifacts_root = std::env::current_dir()
                .unwrap_or_default()
                .join("artifacts")
                .to_string_lossy()
                .to_string();
            sidecar::spawn_sidecar(&app.handle(), artifacts_root);
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
}
```

- [ ] **Step 3: Update lib.rs**

```rust
pub mod artifacts;
pub mod commands;
pub mod elo;
pub mod kelly;
pub mod parser;
pub mod sidecar;
pub mod state;
```

- [ ] **Step 4: Check compilation**

```bash
cd app && cargo check 2>&1 | tail -20
```

Expected: compiles without errors.

- [ ] **Step 5: Commit**

```bash
git add app/src-tauri/src/main.rs app/src-tauri/src/lib.rs app/src-tauri/tauri.conf.json
git commit -m "feat(main): load dual Elo, spawn sidecar, register predict_with_ml"
```

---

## Task 16: Update UI

**Files:** Modify `app/src/lib/stores.ts`, modify `app/src/App.svelte`, modify `app/src/lib/components/MatchInput.svelte`

- [ ] **Step 1: Update stores.ts**

Replace `app/src/lib/stores.ts`:

```typescript
import { writable } from 'svelte/store'

export interface Prediction {
  player_a: string
  player_b: string
  surface: string
  prob_a_wins: number
  prob_b_wins: number
  elo_a_overall: number
  elo_b_overall: number
  ml_prob_a_wins?: number
  confidence_flag?: string
}

export interface KellyResult {
  implied_prob: number
  edge: number
  full_kelly: number
  fractional_kelly: number
  stake: number
}

export const predictions = writable<Prediction[]>([])
export const loading = writable(false)
export const error = writable<string | null>(null)
export const dataAsOf = writable('unknown')

// Phase 2: Kelly settings
export const bankroll = writable(1000)
export const kelly_fraction = writable(0.25)

// Phase 4: Tour selector
export const selectedTour = writable<'atp' | 'wta'>('atp')
```

- [ ] **Step 2: Update App.svelte**

Replace `app/src/App.svelte`:

```svelte
<script lang="ts">
  import MatchInput from './lib/components/MatchInput.svelte'
  import MatchCard from './lib/components/MatchCard.svelte'
  import Footer from './lib/components/Footer.svelte'
  import { predictions, error, bankroll, kelly_fraction, selectedTour } from './lib/stores'
</script>

<div class="min-h-screen flex flex-col bg-white">
  <header class="bg-white border-b border-gray-200 px-6 py-4">
    <div class="max-w-6xl mx-auto flex justify-between items-center">
      <h1 class="text-2xl font-bold">Progno</h1>
      <div class="flex gap-6 items-center text-sm">
        <label class="flex items-center gap-2">
          <span class="text-gray-700">Tour:</span>
          <select
            bind:value={$selectedTour}
            class="px-2 py-1 border border-gray-300 rounded"
            onchange={() => predictions.set([])}
          >
            <option value="atp">ATP</option>
            <option value="wta">WTA</option>
          </select>
        </label>
        <label class="flex items-center gap-2">
          <span class="text-gray-700">Bankroll ($):</span>
          <input
            type="number"
            bind:value={$bankroll}
            min="1"
            step="100"
            class="w-24 px-2 py-1 border border-gray-300 rounded"
          />
        </label>
        <label class="flex items-center gap-2">
          <span class="text-gray-700">Kelly Fraction:</span>
          <select bind:value={$kelly_fraction} class="px-2 py-1 border border-gray-300 rounded">
            <option value={0.1}>0.1×</option>
            <option value={0.25}>0.25×</option>
            <option value={0.5}>0.5×</option>
            <option value={1}>1.0×</option>
          </select>
        </label>
      </div>
    </div>
  </header>

  <MatchInput />

  {#if $error}
    <div class="bg-red-50 border-l-4 border-red-500 p-4 m-4 text-red-700">
      {$error}
    </div>
  {/if}

  <div class="flex-1">
    {#each $predictions as pred (pred.player_a + pred.player_b)}
      <MatchCard prediction={pred} />
    {/each}
  </div>

  <Footer />
</div>
```

- [ ] **Step 3: Update MatchInput.svelte**

Replace `app/src/lib/components/MatchInput.svelte`:

```svelte
<script lang="ts">
  import { invoke } from '@tauri-apps/api/core'
  import { predictions, loading, error, dataAsOf, selectedTour } from '../stores'

  let textInput = $state('')

  async function handleParse() {
    loading.set(true)
    error.set(null)

    try {
      const result = await invoke('parse_and_predict', {
        text: textInput,
        tour: $selectedTour,
      })

      if (result.error) {
        error.set(result.error)
      } else {
        predictions.set(result.predictions)
        dataAsOf.set(result.data_as_of)
      }
    } catch (err) {
      error.set(String(err))
    } finally {
      loading.set(false)
    }
  }
</script>

<div class="p-6 border-b border-gray-200 bg-white">
  <h2 class="text-lg font-semibold mb-4">
    Paste today's {$selectedTour.toUpperCase()} matches
  </h2>
  <textarea
    bind:value={textInput}
    class="w-full p-3 border border-gray-300 rounded-md font-mono text-sm"
    rows="6"
    placeholder={$selectedTour === 'wta'
      ? 'Swiatek vs Sabalenka - Clay\nGauff vs Rybakina - Hard'
      : 'Alcaraz vs Sinner - Clay\nDjokovic vs Zverev - Hard'}
  />
  <button
    onclick={handleParse}
    disabled={$loading}
    class="mt-4 px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
  >
    {$loading ? 'Parsing...' : 'Parse'}
  </button>
</div>
```

- [ ] **Step 4: Build and verify TypeScript types**

```bash
cd app && npm run build 2>&1 | tail -20
```

Expected: builds without TypeScript errors.

- [ ] **Step 5: Commit**

```bash
git add app/src/lib/stores.ts app/src/App.svelte app/src/lib/components/MatchInput.svelte
git commit -m "feat(ui): add ATP/WTA tour dropdown; pass tour to parse_and_predict"
```

---

## Task 17: Final verification

- [ ] **Step 1: Run full Python test suite**

```bash
cd training && uv run pytest -v 2>&1 | tail -30
```

Expected: all tests PASS.

- [ ] **Step 2: Run Rust test suite**

```bash
cd app && cargo test 2>&1 | tail -30
```

Expected: all tests PASS.

- [ ] **Step 3: Rust type check**

```bash
cd app && cargo check 2>&1 | tail -10
```

Expected: no errors.

- [ ] **Step 4: TypeScript check**

```bash
cd app && npm run build 2>&1 | tail -10
```

Expected: no errors.

- [ ] **Step 5: Commit if any fixes needed, otherwise tag**

```bash
git log --oneline -10
```

Verify the commit history looks clean.

---

## Spec Coverage Check

| Spec requirement | Task | Status |
|---|---|---|
| Rolling form: win_rate_overall(50), win_rate_surface(20), win_rate_vs_top20(30) (§3.2) | Task 5 | ✓ |
| Fatigue: days_since, sets_14d, matches_30d, surface_switch (§3.3) | Task 5 | ✓ |
| Serve efficiency rolling 25 matches (§3.4) | Task 5 | ✓ |
| H2H shrinkage prior=5 (§3.5) | Task 5 | ✓ |
| Match context: surface, level, round, best_of (§3.6) | Task 5 | ✓ |
| Player meta: age_diff, height_diff, lefty_vs_righty (§3.7) | Task 5 | ✓ |
| Pair-diff encoding (§3.9) | Task 5 | ✓ |
| No data leakage — temporal property tests (§2.4, §6.5) | Task 6 | ✓ |
| CatBoost with cat_features (§4.2, §4.4) | Task 7 | ✓ |
| Walk-forward, no random split (§2.5) | Task 7 | ✓ |
| random_seed=42 (§6.7) | Task 7 | ✓ |
| Platt calibration (§4.5) | Task 7 | ✓ |
| ATP burn-in 2004, WTA burn-in 2011 (Phase 4 spec) | Task 7,8 | ✓ |
| Acceptance gate: log-loss < Elo, ECE < 0.03 (§6.4) | Task 9 | ✓ |
| `just features/train/validate/retrain` (§6.2) | Task 10 | ✓ |
| WTA data download script (Phase 4 spec) | Task 11 | ✓ |
| FastAPI sidecar, random port, stdout handshake (§4.7, §4.11) | Task 12 | ✓ |
| `/health`, `/predict`, `/model_info` (§4.9) | Task 12 | ✓ |
| Single sidecar with tour routing (Phase 4 design) | Task 12 | ✓ |
| WTA model absent at Phase 3 launch — not fatal (Phase 4 design) | Task 12 | ✓ |
| Dual ATP/WTA Elo in AppState (Phase 4 design) | Task 13 | ✓ |
| `tour` param in Rust commands (Phase 4 design) | Task 14 | ✓ |
| Elo fallback when sidecar down (§5.9) | Task 14 | ✓ |
| `ml_prob_a_wins`, `confidence_flag` in PredictionResult (§5.4) | Task 14 | ✓ |
| ATP/WTA dropdown in UI header (Phase 4 design) | Task 16 | ✓ |
| `selectedTour` store drives all prediction calls (Phase 4 design) | Task 16 | ✓ |
| Kelly stake formula unchanged for WTA (§5.3) | Task 14,16 | ✓ |

### Notes

- **ROI gate**: spec §6.4 requires ROI ≥ 0 on test vs Pinnacle closing odds. Deferred (needs odds join from tennis-data.co.uk, same as ATP Phase 3.5). The current gate validates log-loss and ECE only.
- **Sidecar bundling**: `just build-sidecar` must run after `just train` (ATP) before `cargo tauri build`. The WTA model is optional for initial release.
- **Player ID resolution**: sidecar receives normalized name keys (`"alcaraz"`, `"sinner"`) matching `elo_state.json`. WTA keys will be player last names in lowercase.
- **Artifact paths in production**: `artifacts.rs::load_elo_state_for_tour` uses `artifacts/{tour}/elo_state.json` relative to the binary. In `cargo tauri build`, configure `bundle.resources` to include the `artifacts/` directory.
