# Phase 2: EV / Fractional Kelly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bookmaker odds input and compute expected value (EV), implied probability, and fractional Kelly stake recommendation (0.25×) for each match.

**Architecture:** 
- **Rust kelly.rs module** computes Kelly stake given model probability, decimal odds, and bankroll. Formulas per spec §5.3.
- **Tauri command** `calculate_kelly` exposes the computation.
- **Svelte state** tracks bankroll (USD) and kelly_fraction (default 0.25). Each MatchCard accepts decimal odds input.
- **UI display** shows implied probability, edge%, Kelly stake in the card. Negative edge → stake grayed out.

**Tech Stack:** 
- Rust 2021 + anyhow
- Tauri 2 commands
- Svelte 5 runes + TypeScript
- TDD: write tests first, implement minimal code

---

## Files Summary

| File | Action | Responsibility |
|------|--------|-----------------|
| `src-tauri/src/kelly.rs` | Create | Kelly calculation logic: `implied_probability`, `edge`, `full_kelly_fraction`, `fractional_kelly` |
| `src-tauri/src/main.rs` | Modify | Add `mod kelly;` |
| `src-tauri/src/commands.rs` | Modify | Add `KellyRequest`, `KellyResult`, `calculate_kelly` command |
| `src-tauri/tests/kelly.rs` | Create | Unit tests for Kelly formulas with known vectors |
| `app/src/lib/stores.ts` | Modify | Add `bankroll` and `kelly_fraction` stores |
| `app/src/lib/components/MatchCard.svelte` | Modify | Add odds input field, display Kelly calculations |
| `app/src/App.svelte` | Modify | Add bankroll input in top bar (before match list) |

---

## Task 1: Create kelly.rs module with test vectors

**Files:**
- Create: `src-tauri/src/kelly.rs`
- Create: `src-tauri/tests/kelly.rs`

### Step 1: Write tests for Kelly calculations

In `src-tauri/tests/kelly.rs`:

```rust
#[cfg(test)]
mod tests {
    use progno::kelly::*;

    #[test]
    fn test_implied_probability_from_decimal_odds() {
        // 2.0 decimal odds → 50% implied
        assert!((implied_probability(2.0) - 0.5).abs() < 0.001);
        // 1.5 decimal odds → 66.67% implied
        assert!((implied_probability(1.5) - (2.0 / 3.0)).abs() < 0.001);
        // 3.0 decimal odds → 33.33% implied
        assert!((implied_probability(3.0) - (1.0 / 3.0)).abs() < 0.001);
    }

    #[test]
    fn test_edge_calculation() {
        // If model says 60% but implied is 50%, edge is +10%
        let edge = edge(0.6, 2.0);
        assert!((edge - 0.1).abs() < 0.001);
        
        // If model says 30% but implied is 50%, edge is -20%
        let edge = edge(0.3, 2.0);
        assert!((edge - (-0.2)).abs() < 0.001);
    }

    #[test]
    fn test_full_kelly_fraction_positive_edge() {
        // Model 60%, decimal_odds 2.0
        // full_kelly = (0.6 * 2.0 - 1) / (2.0 - 1) = 0.2 / 1.0 = 0.2
        let kelly = full_kelly_fraction(0.6, 2.0);
        assert!((kelly - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_full_kelly_fraction_negative_edge() {
        // Model 40%, decimal_odds 2.0
        // full_kelly = (0.4 * 2.0 - 1) / (2.0 - 1) = -0.2 / 1.0 = -0.2
        let kelly = full_kelly_fraction(0.4, 2.0);
        assert!((kelly - (-0.2)).abs() < 0.001);
    }

    #[test]
    fn test_fractional_kelly_applies_fraction() {
        // Model 60%, decimal_odds 2.0, fraction 0.25
        // full = 0.2, frac = 0.25 * 0.2 = 0.05
        let kelly = fractional_kelly(0.6, 2.0, 0.25);
        assert!((kelly - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_fractional_kelly_clamps_negative_to_zero() {
        // Model 30%, decimal_odds 2.0, fraction 0.25
        // full = -0.5, frac = max(0, 0.25 * -0.5) = 0.0
        let kelly = fractional_kelly(0.3, 2.0, 0.25);
        assert_eq!(kelly, 0.0);
    }

    #[test]
    fn test_stake_from_kelly() {
        // bankroll $1000, kelly 0.05 → $50
        let stake = stake_from_kelly(1000.0, 0.05);
        assert!((stake - 50.0).abs() < 0.01);
    }
}
```

### Step 2: Run tests to verify they fail

```bash
cd app
cargo test --test kelly
```

Expected output: `error[E0432]: unresolved import 'progno'` and similar compilation errors.

### Step 3: Create minimal kelly.rs

In `src-tauri/src/kelly.rs`:

```rust
use anyhow::Result;

/// Implied probability from decimal odds.
/// odds = 2.0 → 0.5 (50%)
pub fn implied_probability(decimal_odds: f64) -> f64 {
    1.0 / decimal_odds
}

/// Expected value edge: model probability minus implied probability.
pub fn edge(model_prob: f64, decimal_odds: f64) -> f64 {
    model_prob - implied_probability(decimal_odds)
}

/// Full Kelly fraction: (p * odds - 1) / (odds - 1)
pub fn full_kelly_fraction(model_prob: f64, decimal_odds: f64) -> f64 {
    (model_prob * decimal_odds - 1.0) / (decimal_odds - 1.0)
}

/// Fractional Kelly: max(0, fraction * full_kelly)
pub fn fractional_kelly(model_prob: f64, decimal_odds: f64, fraction: f64) -> f64 {
    (fraction * full_kelly_fraction(model_prob, decimal_odds)).max(0.0)
}

/// Stake in currency units from bankroll and Kelly fraction.
pub fn stake_from_kelly(bankroll: f64, kelly_fraction: f64) -> f64 {
    bankroll * kelly_fraction
}
```

### Step 4: Run tests to verify they pass

Update `src-tauri/src/lib.rs` to export the kelly module (create if it doesn't exist):

```rust
pub mod kelly;
```

```bash
cd app
cargo test --test kelly
```

Expected: All tests PASS.

### Step 5: Commit

```bash
cd app
git add src/kelly.rs tests/kelly.rs src/lib.rs
git commit -m "feat(kelly): add Kelly calculation functions with tests"
```

---

## Task 2: Add kelly module to main.rs and export public API

**Files:**
- Modify: `src-tauri/src/main.rs`

### Step 1: Add kelly mod to main.rs

In `src-tauri/src/main.rs`, after the other module declarations:

```rust
mod artifacts;
mod commands;
mod elo;
mod kelly;
mod parser;
mod state;
```

### Step 2: Verify it compiles

```bash
cd app
cargo check
```

Expected: Success.

### Step 3: Commit

```bash
cd app
git add src/main.rs
git commit -m "feat: add kelly module to main"
```

---

## Task 3: Extend commands.rs with Kelly calculation command

**Files:**
- Modify: `src-tauri/src/commands.rs`

### Step 1: Write tests for Kelly command

In `src-tauri/src/commands.rs`, add to the `tests` module:

```rust
#[cfg(test)]
mod kelly_tests {
    use super::*;

    #[test]
    fn test_kelly_request_struct() {
        let req = KellyRequest {
            model_prob: 0.6,
            decimal_odds: 2.0,
            bankroll: 1000.0,
            kelly_fraction: 0.25,
        };
        assert_eq!(req.model_prob, 0.6);
    }

    #[test]
    fn test_calculate_kelly_response() {
        let req = KellyRequest {
            model_prob: 0.6,
            decimal_odds: 2.0,
            bankroll: 1000.0,
            kelly_fraction: 0.25,
        };
        let result = calculate_kelly_impl(req).unwrap();
        assert!((result.edge - 0.1).abs() < 0.001);
        assert!((result.implied_prob - 0.5).abs() < 0.001);
        assert!((result.stake - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_calculate_kelly_negative_edge() {
        let req = KellyRequest {
            model_prob: 0.3,
            decimal_odds: 2.0,
            bankroll: 1000.0,
            kelly_fraction: 0.25,
        };
        let result = calculate_kelly_impl(req).unwrap();
        assert_eq!(result.stake, 0.0);
    }
}
```

### Step 2: Run tests to verify they fail

```bash
cd app
cargo test --lib commands::
```

Expected: Tests fail with "KellyRequest not found", etc.

### Step 3: Add structures and implementation to commands.rs

At the top of `src-tauri/src/commands.rs`, add after imports:

```rust
use crate::kelly;

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
```

Add the implementation function (before the `#[cfg(test)]` section):

```rust
pub fn calculate_kelly_impl(req: KellyRequest) -> Result<KellyResult, String> {
    let implied_prob = kelly::implied_probability(req.decimal_odds);
    let edge = kelly::edge(req.model_prob, req.decimal_odds);
    let full_kelly = kelly::full_kelly_fraction(req.model_prob, req.decimal_odds);
    let fractional_kelly = kelly::fractional_kelly(req.model_prob, req.decimal_odds, req.kelly_fraction);
    let stake = kelly::stake_from_kelly(req.bankroll, fractional_kelly);

    Ok(KellyResult {
        implied_prob,
        edge,
        full_kelly,
        fractional_kelly,
        stake,
    })
}
```

Add the Tauri command (in the non-test section, after `get_data_as_of_cmd`):

```rust
#[cfg(not(test))]
#[tauri::command]
pub fn calculate_kelly(request: KellyRequest) -> Result<KellyResult, String> {
    calculate_kelly_impl(request)
}
```

### Step 4: Run tests to verify they pass

```bash
cd app
cargo test --lib commands::kelly_tests
```

Expected: All tests PASS.

### Step 5: Register command in main.rs

In `src-tauri/src/main.rs`, update the invoke_handler:

```rust
.invoke_handler(tauri::generate_handler![
    commands::parse_and_predict,
    commands::get_data_as_of_cmd,
    commands::calculate_kelly,
])
```

### Step 6: Verify compilation

```bash
cd app
cargo check
```

Expected: Success.

### Step 7: Commit

```bash
cd app
git add src/commands.rs src/main.rs
git commit -m "feat(commands): add calculate_kelly Tauri command"
```

---

## Task 4: Update Svelte stores with bankroll and kelly_fraction

**Files:**
- Modify: `app/src/lib/stores.ts`

### Step 1: Extend stores.ts

Replace the entire `app/src/lib/stores.ts`:

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
export const bankroll = writable(1000) // USD, default $1000
export const kelly_fraction = writable(0.25) // default 0.25×
```

### Step 2: Verify type checking

```bash
cd app
npm run check
```

Expected: Success (or only unrelated type errors).

### Step 3: Commit

```bash
cd app
git add src/lib/stores.ts
git commit -m "feat(stores): add bankroll and kelly_fraction stores"
```

---

## Task 5: Update MatchCard component with odds input and Kelly display

**Files:**
- Modify: `app/src/lib/components/MatchCard.svelte`

### Step 1: Update MatchCard.svelte

Replace the entire `app/src/lib/components/MatchCard.svelte`:

```svelte
<script lang="ts">
  import type { Prediction, KellyResult } from '../stores'
  import { bankroll, kelly_fraction } from '../stores'
  import { invoke } from '@tauri-apps/api/core'

  export let prediction: Prediction

  let odds: number | null = null
  let kellyResult: KellyResult | null = null
  let loading = false
  let error: string | null = null

  const probA = Math.round(prediction.prob_a_wins * 1000) / 10
  const probB = Math.round(prediction.prob_b_wins * 1000) / 10

  async function onOddsChange() {
    if (!odds || odds <= 1) {
      kellyResult = null
      error = null
      return
    }

    loading = true
    error = null

    try {
      const result = await invoke<KellyResult>('calculate_kelly', {
        request: {
          model_prob: prediction.prob_a_wins,
          decimal_odds: odds,
          bankroll: $bankroll,
          kelly_fraction: $kelly_fraction,
        },
      })
      kellyResult = result
    } catch (e) {
      error = `Kelly calculation failed: ${e}`
      kellyResult = null
    } finally {
      loading = false
    }
  }

  $: if (odds) {
    onOddsChange()
  }
</script>

<div class="p-6 border-b border-gray-100 hover:bg-gray-50">
  <div class="mb-2 text-sm font-semibold text-gray-700">
    {prediction.player_a} vs {prediction.player_b}
  </div>
  <div class="text-xs text-gray-500 mb-4">{prediction.surface}</div>

  <div class="space-y-3">
    <div>
      <div class="flex justify-between items-center mb-1">
        <span class="text-sm font-medium">{prediction.player_a}</span>
        <span class="text-sm font-bold text-blue-600">{probA}%</span>
      </div>
      <div class="h-6 bg-gray-200 rounded-sm overflow-hidden">
        <div
          class="h-full bg-blue-500"
          style="width: {probA}%"
        />
      </div>
    </div>

    <div>
      <div class="flex justify-between items-center mb-1">
        <span class="text-sm font-medium">{prediction.player_b}</span>
        <span class="text-sm font-bold text-red-600">{probB}%</span>
      </div>
      <div class="h-6 bg-gray-200 rounded-sm overflow-hidden">
        <div
          class="h-full bg-red-500"
          style="width: {probB}%"
        />
      </div>
    </div>
  </div>

  <div class="mt-4 p-4 bg-gray-50 rounded">
    <label class="text-xs font-semibold text-gray-600 block mb-2">
      Bookmaker Odds (decimal)
    </label>
    <input
      type="number"
      bind:value={odds}
      placeholder="e.g. 2.50"
      min="1"
      step="0.01"
      class="w-full px-2 py-1 border border-gray-300 rounded text-sm"
    />
  </div>

  {#if error}
    <div class="mt-2 text-xs text-red-600">{error}</div>
  {/if}

  {#if kellyResult}
    <div class="mt-4 p-4 bg-blue-50 rounded space-y-2">
      <div class="flex justify-between text-xs">
        <span class="text-gray-700">Implied Prob:</span>
        <span class="font-semibold">{Math.round(kellyResult.implied_prob * 1000) / 10}%</span>
      </div>
      <div
        class="flex justify-between text-xs"
        class:text-green-700={kellyResult.edge > 0}
        class:text-red-700={kellyResult.edge < 0}
      >
        <span>Edge:</span>
        <span class="font-semibold">
          {kellyResult.edge > 0 ? '+' : ''}{Math.round(kellyResult.edge * 1000) / 10}%
        </span>
      </div>
      <div class="flex justify-between text-xs pt-2 border-t border-blue-200">
        <span class="font-medium">Stake ({Math.round($kelly_fraction * 100)}× Kelly):</span>
        <span
          class="font-bold"
          class:text-blue-600={kellyResult.stake > 0}
          class:text-gray-400={kellyResult.stake === 0}
        >
          ${Math.round(kellyResult.stake * 100) / 100}
        </span>
      </div>
    </div>
  {/if}

  <div class="text-xs text-gray-500 mt-4">
    Elo: {prediction.player_a} {Math.round(prediction.elo_a_overall)} vs {Math.round(prediction.elo_b_overall)}
  </div>
</div>

<style>
  input {
    font-size: 14px;
  }
</style>
```

### Step 2: Verify type checking and visual

```bash
cd app
npm run check
npm run dev
```

Expected: App loads, no TypeScript errors. MatchCard shows odds input field.

### Step 3: Manually test in the dev app

1. Run the dev server (`npm run dev`)
2. Parse a match
3. Enter a decimal odds value (e.g., 2.50)
4. Verify Kelly calculation appears below with implied prob, edge, stake

### Step 4: Commit

```bash
cd app
git add src/lib/components/MatchCard.svelte
git commit -m "feat(ui): add odds input and Kelly calculation display to MatchCard"
```

---

## Task 6: Add bankroll input to App.svelte top bar

**Files:**
- Modify: `app/src/App.svelte`

### Step 1: Read current App.svelte

```bash
cd app
head -50 src/App.svelte
```

### Step 2: Update App.svelte top bar

Add bankroll input to the header. Find the section that displays the header and add:

```svelte
<script lang="ts">
  import { bankroll, kelly_fraction } from './lib/stores'
  // ... other imports
</script>

<div class="bg-white shadow-sm border-b border-gray-200">
  <div class="max-w-6xl mx-auto px-6 py-3 flex justify-between items-center">
    <h1 class="text-lg font-bold text-gray-900">Progno</h1>
    <div class="flex gap-4 items-center text-sm">
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
</div>
```

(Exact placement depends on your current App.svelte structure — adjust accordingly.)

### Step 3: Verify and test

```bash
cd app
npm run dev
```

Expected: Top bar shows bankroll input and Kelly fraction selector. Values persist as you update them.

### Step 4: Commit

```bash
cd app
git add src/App.svelte
git commit -m "feat(ui): add bankroll and kelly_fraction controls to header"
```

---

## Task 7: Integration test — end-to-end Kelly flow

**Files:**
- Test: manual browser testing

### Step 1: Start dev server

```bash
cd app
npm run dev
```

### Step 2: Run end-to-end test

1. Copy a match into the input (e.g., "Alcaraz vs Sinner - Hard")
2. Verify prediction displays with Elo probability
3. Set bankroll to $500 in header
4. Set Kelly fraction to 0.25× in header
5. Enter decimal odds in the MatchCard (e.g., 2.50 for Alcaraz)
6. Verify Kelly calculation appears:
   - Implied Prob: 40%
   - Edge: (Alcaraz's prob) - 40%
   - Stake: should compute correctly
7. Change odds to 1.50 (lower probability implied by odds)
8. Verify stake increases (more edge)
9. Change odds to a bad one (e.g., 1.20 for a 83% implied probability)
10. Verify stake becomes $0 (negative edge)
11. Change bankroll to $2000
12. Verify all stakes scale by 4×

### Step 3: Document results

If all pass, record: "Phase 2 integration test: PASS"

### Step 4: Commit

No code changes; this is manual verification. If you made any manual fixes during testing, commit those.

---

## Task 8: Run full test suite and final checks

**Files:**
- Test: all tests

### Step 1: Run Rust tests

```bash
cd app
cargo test --all
```

Expected: All tests pass, including kelly, commands, and existing tests.

### Step 2: Run TypeScript type check

```bash
cd app
npm run check
```

Expected: No errors.

### Step 3: Build production

```bash
cd app
npm run build
cargo tauri build
```

Expected: Build succeeds.

### Step 4: Verify spec §5.3 compliance

Checklist:
- ✅ Implied probability formula: `1 / decimal_odds`
- ✅ Edge formula: `model_p - implied_p`
- ✅ Full Kelly formula: `(model_p * odds - 1) / (odds - 1)`
- ✅ Fractional Kelly: `max(0, 0.25 × full_kelly)`
- ✅ Stake: `bankroll × frac_kelly`
- ✅ UI shows all values
- ✅ Negative edge → stake = $0, grayed out

### Step 5: Commit final status

```bash
git add -A
git commit -m "feat: Phase 2 complete — Kelly EV and stake calculation"
```

---

## Spec Coverage Check

| Spec §5.3 requirement | Task | Status |
|---|---|---|
| `implied_p = 1 / decimal_odds` | Task 1 | ✅ |
| `edge = model_p - implied_p` | Task 1 | ✅ |
| `full_kelly = (p × odds - 1) / (odds - 1)` | Task 1 | ✅ |
| `frac_kelly = max(0, 0.25 × full_kelly)` | Task 1 | ✅ |
| `stake = bankroll × frac_kelly` | Task 1 | ✅ |
| Bankroll input (user-configurable, default $1000) | Task 6 | ✅ |
| Kelly fraction input (default 0.25×) | Task 6 | ✅ |
| Display implied prob | Task 5 | ✅ |
| Display edge% | Task 5 | ✅ |
| Display stake | Task 5 | ✅ |
| Negative edge → stake disabled/grayed | Task 5 | ✅ |

---

## Notes

- **Error handling**: Kelly calculation is deterministic; only Tauri communication can fail. Handle gracefully in UI (Task 5).
- **Rounding**: Display to 1 decimal place for probabilities/edges, 2 decimals for stake (USD).
- **Validation**: Odds must be > 1. Frontend validates in input.
- **Cold start**: If elo_state missing, matches won't predict, so Kelly won't calculate. This is expected.
- **Data persistence**: Bankroll/kelly_fraction are Svelte stores (in-memory). Persist to localStorage if desired (Post-Phase 2 enhancement).
