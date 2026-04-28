# Autocomplete Match Input

**Date:** 2026-04-28  
**Status:** Approved  
**Scope:** Replace the free-text textarea in `MatchInput.svelte` with a structured row-based form where each player field has live autocomplete driven by the loaded Elo player list.

---

## Problem

The current textarea requires the user to know the exact spelling of player names as stored in the Elo state (e.g. `Alcaraz`, `Del Potro`). A typo causes a silent prediction failure. Manual entry is error-prone and slow.

---

## Solution

Replace the textarea with a structured form:

- Each row: **[Player A input] vs [Player B input] [Surface dropdown] [× remove]**
- `[+ Add match]` button appends a new empty row
- `[Predict]` submits all complete rows

Each player input has a live-filtered autocomplete dropdown sourced from the Elo state.

---

## Backend

### New Tauri command: `get_player_names`

```rust
#[tauri::command]
pub fn get_player_names(tour: String, app_state: State<AppState>) -> Vec<String>
```

**Logic:**
1. Lock the correct Elo state (`atp` or `wta`) from `AppState`.
2. Iterate `state["players"]` entries.
3. Filter: `matches_played >= 10`.
4. Sort: `elo_overall` descending (most active/strong players first).
5. Convert each key to a display name: replace `_` with space, title-case each word (`del_potro` → `Del Potro`, `alcaraz` → `Alcaraz`).
6. Return `Vec<String>`.

**Why this works end-to-end:** The user selects e.g. `"Del Potro"` → submitted as the player name → `normalize_player_id("Del Potro")` → `"del_potro"` → found in Elo state. No changes to prediction logic.

**Registration:** add `get_player_names` to `invoke_handler` in `main.rs`.

---

## Frontend

### Loading

`MatchInput.svelte` loads player names once on mount via `invoke('get_player_names', { tour })`. Reloads whenever `$selectedTour` changes (reactive `$effect`).

### State

```ts
let rows = $state([{ playerA: '', playerB: '', surface: 'Hard' }])
let playerNames: string[] = $state([])
let activeField: { rowIdx: number; field: 'a' | 'b' } | null = $state(null)
let query = $state('')
```

### UI layout

```
┌──────────────────────────────────────────────────┐
│ [Player A        ▾]  vs  [Player B        ▾]  [Hard ▾]  [×] │
│ [Player A        ▾]  vs  [Player B        ▾]  [Hard ▾]  [×] │
│ [+ Add match]                        [Predict]              │
└──────────────────────────────────────────────────┘
```

### Autocomplete dropdown

- Appears below the focused input when the field has content.
- Filters `playerNames` by case-insensitive substring match on the current value.
- Shows at most **10 suggestions**.
- Click a suggestion → sets the field value, closes the dropdown.
- `Escape` or click outside → closes without selecting.
- Dropdown is scoped per field (only one open at a time via `activeField`).

### Submit

On `[Predict]`:
1. Skip rows where either player is empty.
2. Build text: `"Alcaraz vs Sinner - Clay\nDjokovic vs Medvedev - Hard"`.
3. Call `invoke('predict_with_ml', { request: { text, tour, tourney_date } })` — identical to current behaviour.

### Tour change

When `$selectedTour` changes, reload `playerNames` and clear `rows` (same as current `predictions.set([])`).

---

## What does NOT change

- `predict_with_ml` command — unchanged.
- `normalize_player_id` — unchanged.
- Prediction pipeline, ML sidecar, Kelly logic — unchanged.
- `MatchCard.svelte`, `HistoryPanel.svelte`, `SchedulePanel.svelte` — unchanged.

---

## Out of scope

- Country suffix in dropdown (e.g. `Alcaraz (ESP)`) — can be added later if disambiguation is needed.
- Keeping the free-text textarea as a fallback mode.
- Fuzzy matching (substring is sufficient for manual entry).
