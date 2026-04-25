# Phase 3 Enhancement — WElo + Feature Engineering Expansion

**Status**: Approved  
**Date**: 2026-04-25  
**Prerequisite**: Phase 3 (CatBoost + Python sidecar) base implementation per `docs/superpowers/plans/2026-04-23-phase-3-catboost-sidecar.md`

## Goal

Improve CatBoost model signal quality by:
1. Adding Weighted Elo (WElo) — margin-of-victory variant of Elo — as two new parallel features alongside existing Elo
2. Adding three missing rolling serve/return statistics: `bp_save_pct`, `second_won_pct`, `return_pts_pct`

Both changes target Phase 3 training only. Rust runtime and UI are unaffected.

## Scope

- `training/src/progno_train/elo.py` — WElo update logic
- `training/src/progno_train/rollup.py` — WElo rollup alongside standard Elo
- `training/src/progno_train/features.py` — new rolling stats + WElo diff features
- `training/artifacts/*/elo_state.json` — additive schema (new `welo_*` keys)
- `training/tests/test_elo.py`, `test_features.py` — new tests

Not in scope: time-decay rolling (§7.6), game-based MoV, momentum/EWMA features (Phase 5+), ROI acceptance gate (separate backlog item).

## Design

### 1. Weighted Elo (WElo)

**Motivation:** Standard Elo treats a 6:0 6:0 win identically to a 7:6 7:6 win. WElo weights the K-factor by margin of victory (Angelini, Candila, De Angelis 2022 — EJOR). Per research, WElo Brier Score 0.212 vs 0.215 for GNN MagNet and matches Pinnacle odds accuracy.

**MoV formula (set-based):**

```python
def mov_multiplier(sets_winner: int, sets_loser: int) -> float:
    """Linear interpolation: [0.6, 1.0] → [0.70, 1.50]."""
    mov = sets_winner / (sets_winner + sets_loser)
    return 2.0 * mov - 0.5
```

Calibration:
- 3-2 or 2-1 → MoV=0.60–0.67 → multiplier=0.70–0.83 (close match, less K)
- 3-1       → MoV=0.75       → multiplier=1.00 (neutral, same as standard Elo)
- 3-0 or 2-0 → MoV=1.00      → multiplier=1.50 (dominant win, more K)

**Implementation in `elo.py`:**

```python
def apply_welo_update(
    state: dict, winner: str, loser: str,
    sets_w: int, sets_l: int,
    surface: str, ctx: float
) -> None:
    """WElo update: same as apply_elo_update but K scaled by MoV."""
    multiplier = mov_multiplier(sets_w, sets_l)
    # reuse existing k_factor, context, expected_probability logic
    # K_welo = K_standard * multiplier
    # update welo_overall, welo_<surface> for winner and loser
```

WElo uses the same `k_factor(n)` formula (`250 / (n + 5)^0.4`) and context multipliers as standard Elo. Only K is scaled by MoV. The `matches` counter is shared (no new counter needed).

**`elo_state.json` schema — additive only:**

```json
{
  "Novak Djokovic": {
    "elo_overall": 2156, "elo_hard": 2089, "elo_clay": 2201, "elo_grass": 2134,
    "welo_overall": 2148, "welo_hard": 2081, "welo_clay": 2194, "welo_grass": 2127,
    "matches": 1247
  }
}
```

Initial WElo value: 1500 (same as standard Elo). Rust reads only `elo_overall`/`elo_surface` — new `welo_*` keys are ignored by existing Rust code.

**`rollup.py` changes:**

`rollup_elo()` already iterates matches chronologically. After calling `apply_elo_update()`, also call `apply_welo_update()` with `sets_w`/`sets_l` extracted from `match_history.parquet`. If `w_sets`/`l_sets` columns are not already stored there, extend `ingest.py` to extract them from the parsed score string (count completed sets per side) before writing parquet — this is a small addition since the score is already parsed for `is_complete`.

**New features in `features.py`:**

```python
"welo_overall_diff" = p1_welo_overall - p2_welo_overall
"welo_surface_diff" = p1_welo_surface_composite - p2_welo_surface_composite
```

Surface composite rule — identical to existing `elo_surface` logic:
```python
if player_matches_on_surface >= 20:
    welo_surface_composite = 0.5 * welo_surface + 0.5 * welo_overall
else:
    welo_surface_composite = welo_overall  # fallback until enough surface data
```

Lookup identical to existing `elo_overall_diff`: read from `elo_state.json` before match date, default to 1500 if missing.

### 2. New Rolling Serve/Return Features

All three added to `_rolling_serve_stats()` (window=25 matches, `min_periods=5`).

**Formulas:**

```python
# When player won (w_* columns); when player lost (l_* columns):

second_won_pct = w_2ndWon / (w_svpt - w_1stIn)      # or l_* equivalent

bp_save_pct = w_bpSaved / w_bpFaced                  # or l_* equivalent
# NaN when bpFaced == 0 → fill with population median

return_pts_pct = (l_svpt - l_1stWon - l_2ndWon) / l_svpt   # when won (opp's serve)
               = (w_svpt - w_1stWon - w_2ndWon) / w_svpt   # when lost (opp's serve)
```

`return_pts_pct` captures return effectiveness from the opponent's serve columns — the only feature that explicitly models receiving performance.

**Edge cases:**
- `bp_save_pct` at `bpFaced == 0`: → `NaN` → fill with population median over training fold (same strategy as win_rate cold start uses 0.5)
- `second_won_pct` at `1stIn == svpt` (degenerate, all serves land as first): → `NaN` → fill with population median
- CatBoost `Pool` handles `NaN` natively; no additional encoding needed

**Output diff features (+3 total):**

```python
"second_won_pct_diff"  = p1_second_won_pct  - p2_second_won_pct
"bp_save_pct_diff"     = p1_bp_save_pct     - p2_bp_save_pct
"return_pts_pct_diff"  = p1_return_pts_pct  - p2_return_pts_pct
```

Total features: 30 existing + 3 new rolling + 2 WElo = **35 features**.

### 3. Tests

**`test_elo.py` additions:**
- `test_welo_mov_multiplier_values` — spot-check 3-0, 3-1, 3-2, 2-0, 2-1 against expected multipliers
- `test_welo_dominant_win_higher_k` — after 3-0, WElo diff increases more than standard Elo diff
- `test_welo_close_win_lower_k` — after 3-2, WElo diff increases less than standard Elo diff
- `test_welo_total_mass_not_conserved` — WElo is intentionally non-zero-sum; verify winner always gains

**`test_features.py` additions:**
- `test_second_won_pct_correct_formula` — verify numerator/denominator against known match data
- `test_bp_save_pct_zero_faced_returns_median` — bpFaced=0 → no crash, returns median
- `test_return_pts_pct_correct_columns` — uses opponent's serve columns (not player's own)
- `test_new_features_no_future_leakage` — same temporal gate test pattern as existing serve stats

### 4. Acceptance Criteria

The model passes Phase 3 gate with these additional checks:
- Val log-loss ≤ current val log-loss (no regression from new features)
- ECE < 0.03 (unchanged)
- `welo_overall_diff` or `welo_surface_diff` appears in top-10 CatBoost feature importance (confirms WElo carries signal beyond standard Elo)

### 5. Implementation Order

```
1. elo.py        → apply_welo_update() + mov_multiplier()
2. rollup.py     → call apply_welo_update() in rollup_elo() loop
3. test_elo.py   → WElo tests pass
4. features.py   → _rolling_serve_stats() + welo_*_diff
5. test_features.py → new feature tests pass
6. Re-run rollup + train → verify acceptance criteria
```

## Non-Goals

- Replacing standard Elo with WElo — standard Elo stays for Rust Phase 1/2 fallback
- Game-based MoV (requires score parser extension; marginal gain over set-based)
- Hyperparameter search for window sizes — deferred to Phase 3.5
- ROI acceptance gate — separate backlog item
