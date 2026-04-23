# Phase 4 — WTA Model Design

**Status**: Approved
**Date**: 2026-04-24
**Prerequisite**: Phase 3 (CatBoost + Python sidecar) must be implemented first per existing plan `docs/superpowers/plans/2026-04-23-phase-3-catboost-sidecar.md`.

## Goal

Add WTA as a separate model with its own ETL, Elo state, and CatBoost model, sharing the same training infrastructure as ATP via a `tour` parameter. Users select ATP or WTA via a dropdown in the app header.

## Scope

Phase 4 adds to Phase 3:
1. WTA data download (`tennis_wta` Sackmann repo)
2. Parameterized training pipeline (`tour: "atp" | "wta"`)
3. Artifact directory split: `artifacts/atp/` and `artifacts/wta/`
4. Shared sidecar loading both models, `/predict` with `tour` field
5. Rust tour parameter threading
6. UI dropdown "ATP / WTA" in header

## Architecture

### Principle

A single `tour: "atp" | "wta"` parameter flows from CLI → Python pipeline → artifacts → sidecar → Rust → UI. No code duplication: WTA uses the same `ingest.py`, `elo.py`, `features.py`, `train.py` with a different path root.

### Directory layout

```
training/
├── data/raw/
│   ├── tennis_atp/                    (existing)
│   └── tennis_wta/                    (new — git clone JeffSackmann/tennis_wta)
├── scripts/
│   ├── fetch_sackmann.sh              (existing — ATP)
│   └── fetch_sackmann_wta.sh          (new)
├── src/progno_train/
│   ├── config.py                      (Paths.for_tour factory method)
│   ├── ingest.py                      (--tour param, selects data/raw/tennis_{tour}/)
│   ├── elo.py                         (--tour param)
│   ├── features.py                    (no changes — tour-agnostic)
│   ├── train.py                       (BURN_IN_YEAR per tour)
│   └── cli.py                         (all commands get --tour)
└── artifacts/
    ├── atp/                           (migrated from artifacts/ flat layout)
    │   ├── elo_state.json
    │   ├── match_history.parquet
    │   ├── players.parquet
    │   ├── model.cbm
    │   ├── calibration.json
    │   └── model_card.json
    └── wta/
        └── (same files)

sidecar/server.py                      (loads both tours, /predict with tour field)

app/src-tauri/src/
├── sidecar.rs                         (tour field in MlMatchRequest)
├── commands.rs                        (tour param in PredictRequest, dual EloState)
└── main.rs                            (loads both ATP and WTA elo_state.json)

app/src/
├── stores.ts                          (selectedTour store)
└── App.svelte                         (ATP/WTA dropdown in header)
```

## Data Pipeline

### WTA data source

- **Sackmann `tennis_wta`**: `https://github.com/JeffSackmann/tennis_wta` — same CSV format as `tennis_atp`, same column names. Files: `wta_matches_YYYY.csv`.
- **tennis-data.co.uk WTA**: available from 2007 (vs ATP from 2000). Used for odds join and ROI backtest (Phase 3.5 for WTA, same as ATP).
- License: CC-BY-NC-SA 4.0 (same as ATP, personal use OK).

### `Paths.for_tour(root, tour)`

```python
@classmethod
def for_tour(cls, root: Path, tour: str) -> "Paths":
    return cls(
        data_raw=root / "data" / "raw",
        data_staging=root / "data" / "staging" / tour,
        artifacts=root / "artifacts" / tour,
    )
```

All existing `Paths` properties (`matches_raw`, `elo_state`, `model_cbm`, etc.) remain unchanged — they resolve under the tour-specific root.

### ingest.py

`tour` param selects the source glob:
```python
pattern = paths.data_raw / f"tennis_{tour}" / f"{tour}_matches_*.csv"
```

All cleaning, retirement handling, and column selection logic is identical for ATP and WTA.

### WTA walk-forward parameters

| Parameter | ATP | WTA |
|-----------|-----|-----|
| Data start | 2000 | 2007 |
| Burn-in cutoff | 2004 | 2011 |
| Val start | 2016 | 2019 |
| Test start | 2023 | 2023 |

WTA has shorter history; burn-in 2011 allows at least 4 years of warm-up Elo before training.

### CLI

All subcommands gain `--tour` (default: `"atp"` for backwards compatibility):

```
python -m progno_train.cli ingest --tour wta
python -m progno_train.cli elo --tour wta
python -m progno_train.cli features --tour wta
python -m progno_train.cli train --tour wta
python -m progno_train.cli validate --tour wta
python -m progno_train.cli retrain --tour wta --version v1
```

### justfile additions

```just
ingest-wta:
    cd training && uv run python -m progno_train.cli ingest --tour wta

elo-wta:
    cd training && uv run python -m progno_train.cli elo --tour wta

features-wta:
    cd training && uv run python -m progno_train.cli features --tour wta

train-wta:
    cd training && uv run python -m progno_train.cli train --tour wta

validate-wta:
    cd training && uv run python -m progno_train.cli validate --tour wta

retrain-wta version:
    cd training && uv run python -m progno_train.cli retrain --tour wta --version {{version}}

update-data:
    # extended to pull both repos
    bash training/scripts/fetch_sackmann.sh
    bash training/scripts/fetch_sackmann_wta.sh
```

## Sidecar

### Startup

Sidecar receives `--artifacts-root` (parent of `atp/` and `wta/`) instead of `--artifacts-dir`:

```python
def _load_artifacts(artifacts_root: Path) -> None:
    for tour in ("atp", "wta"):
        tour_dir = artifacts_root / tour
        if not (tour_dir / "model.cbm").exists():
            continue  # WTA absent at Phase 3 launch — not fatal
        _models[tour] = CatBoostClassifier()
        _models[tour].load_model(str(tour_dir / "model.cbm"))
        # load platt, match_history, elo_state per tour
```

If only ATP model exists, sidecar starts normally. WTA requests return 503 until WTA model is trained.

### API

Single `/predict` endpoint, `tour` field added to each match:

```python
class MatchRequest(BaseModel):
    tour: str           # "atp" | "wta"
    player_a_id: str
    player_b_id: str
    surface: str
    tourney_level: str = "A"
    round_: str = "R32"
    best_of: int = 3
    tourney_date: str

@app.post("/predict")
async def predict(req: PredictRequest) -> PredictResponse:
    tour = req.matches[0].tour
    if _models.get(tour) is None:
        raise HTTPException(503, f"Model not loaded for tour: {tour}")
    # feature engineering + inference using tour-specific history and elo_state
```

`/health` and `/model_info` remain unchanged. `/model_info` extended to return status per tour:
```json
{ "atp": { "version": "...", "loaded": true },
  "wta": { "version": null, "loaded": false } }
```

## Rust

### AppState

Dual Elo state:

```rust
pub struct AppState {
    pub elo_atp: Mutex<Option<EloState>>,
    pub elo_wta: Mutex<Option<EloState>>,
}
```

`main.rs` setup loads both from `artifacts/atp/elo_state.json` and `artifacts/wta/elo_state.json`. Missing WTA file is logged and skipped.

### Commands

`PredictRequest` and `MlMatchRequest` gain `tour: String`:

```rust
#[derive(Deserialize)]
pub struct PredictRequest {
    pub text: String,
    pub tour: String,       // "atp" | "wta"
    pub tourney_date: String,
}
```

`parse_and_predict` selects the correct Elo state based on `tour`. `predict_with_ml` passes `tour` to sidecar. Fallback: if sidecar returns 503 for the requested tour, Rust uses the tour-specific Elo.

### SidecarState

No structural changes. The sidecar serves both tours on the same port. `port: Option<u16>` stays as-is.

## UI

### `stores.ts`

```ts
export const selectedTour = writable<'atp' | 'wta'>('atp');
```

Persisted to `settings.json` via `tauri-plugin-store` so the user's last choice is remembered.

### Header dropdown

```svelte
<select bind:value={$selectedTour} class="...">
  <option value="atp">ATP</option>
  <option value="wta">WTA</option>
</select>
```

Changing the dropdown clears current predictions (the paste buffer is tour-specific context).

### Prediction call

```ts
await invoke('parse_and_predict', {
    text: pasteBuffer,
    tour: $selectedTour,
    tourney_date: today,
});
```

### Footer

```svelte
<span>· {$selectedTour.toUpperCase()} tour</span>
```

### Match cards

No structural changes. Cards show whatever the command returns. If WTA Elo data is absent, the card shows "Insufficient data" (same mechanism as ATP cold start).

## Testing

### Python additions

```python
# test_ingest.py
@pytest.mark.parametrize("tour", ["atp", "wta"])
def test_paths_for_tour(tmp_path, tour):
    paths = Paths.for_tour(tmp_path, tour)
    assert tour in str(paths.artifacts)
    assert tour in str(paths.data_staging)

# test_train.py
@pytest.mark.parametrize("tour,burn_in", [("atp", 2004), ("wta", 2011)])
def test_walk_forward_burn_in(tour, burn_in):
    df = make_feature_df_spanning_years(2007, 2025)
    splits = walk_forward_splits(df, burn_in_year=burn_in, val_start=burn_in + 8)
    for train, _, _ in splits:
        assert train["year"].min() > burn_in
```

### Smoke test after `just retrain-wta`

```python
# Top-10 WTA Elo overall must contain known leaders
WTA_EXPECTED = {"Swiatek", "Sabalenka", "Gauff"}
top10 = sorted(elo_state["players"].items(), key=lambda x: -x[1]["elo_overall"])[:10]
top10_names = {name for name, _ in top10}
assert WTA_EXPECTED & top10_names, f"WTA Elo sanity fail: top10={top10_names}"
```

### Rust additions

```rust
#[test]
fn test_predict_request_tour_field() {
    let json = serde_json::json!({"text": "", "tour": "wta", "tourney_date": "2026-04-24"});
    let req: PredictRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.tour, "wta");
}
```

## Non-negotiable invariants (from spec, applied to WTA)

- No data leakage: WTA features use same time-gated windows as ATP.
- Walk-forward only: no random split on WTA data.
- Deterministic training: `random_seed=42` for WTA CatBoost.
- Kelly stake formula unchanged: WTA predictions feed the same EV/Kelly calculation.
- Acceptance gate: WTA model must pass log-loss < WTA Elo baseline, ECE < 0.03 before publishing.

## Migration: artifacts/ → artifacts/atp/

Existing ATP artifacts at `artifacts/*.json/parquet/cbm` move to `artifacts/atp/`. This is a one-time migration step (rename directory, update sidecar `--artifacts-root` arg). Existing Phase 3 justfile targets (`just elo`, `just train`, etc.) remain as `--tour atp` shortcuts.

## Out of scope for Phase 4

- WTA-specific odds join (tennis-data.co.uk WTA XLSX) — deferred to a future task, same as ATP Phase 3.5.
- WTA-specific hyperparameter tuning — use same ATP defaults initially.
- Separate WTA injury toggle — same mechanism as ATP Phase 5.
