# Autocomplete Match Input — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the free-text textarea in `MatchInput.svelte` with a structured row-based form where each player field has live autocomplete driven by the Elo player list.

**Architecture:** Add a `get_player_names(tour)` Tauri command that extracts player display names from the in-memory Elo state (filtered to ≥10 matches, sorted by Elo desc). The frontend loads this list once on mount, then filters it locally on each keystroke to populate an autocomplete dropdown. On submit, rows are serialised to the existing text format and sent to `predict_with_ml` unchanged.

**Tech Stack:** Rust (Tauri 2, serde_json), Svelte 5 runes, TypeScript, Tailwind CSS.

---

## File Map

| File | Change |
|---|---|
| `app/src-tauri/src/commands.rs` | Add `key_to_display_name`, `players_from_elo_state`, `get_player_names` |
| `app/src-tauri/src/main.rs` | Register `get_player_names` in `invoke_handler` |
| `app/src/lib/components/MatchInput.svelte` | Full rewrite — structured rows + autocomplete |

---

## Task 1: Backend — helper functions + unit tests

**Files:**
- Modify: `app/src-tauri/src/commands.rs`

- [ ] **Step 1: Add failing tests to `commands.rs`**

Open `app/src-tauri/src/commands.rs`. At the bottom, inside the existing `#[cfg(test)] mod tests { ... }` block, append:

```rust
    #[test]
    fn test_key_to_display_name_single() {
        assert_eq!(key_to_display_name("alcaraz"), "Alcaraz");
    }

    #[test]
    fn test_key_to_display_name_underscore() {
        assert_eq!(key_to_display_name("del_potro"), "Del Potro");
    }

    #[test]
    fn test_key_to_display_name_single_char_suffix() {
        assert_eq!(key_to_display_name("murray_a"), "Murray A");
    }

    #[test]
    fn test_players_from_elo_state_filters_low_matches() {
        let state = json!({
            "players": {
                "alcaraz": { "elo_overall": 2300.0, "matches_played": 50 },
                "unknown": { "elo_overall": 1500.0, "matches_played": 3 }
            }
        });
        let names = players_from_elo_state(&state);
        assert_eq!(names, vec!["Alcaraz"]);
    }

    #[test]
    fn test_players_from_elo_state_sorted_by_elo() {
        let state = json!({
            "players": {
                "alcaraz": { "elo_overall": 2300.0, "matches_played": 50 },
                "sinner":  { "elo_overall": 2400.0, "matches_played": 100 },
                "medvedev":{ "elo_overall": 2200.0, "matches_played": 80 }
            }
        });
        let names = players_from_elo_state(&state);
        assert_eq!(names, vec!["Sinner", "Alcaraz", "Medvedev"]);
    }

    #[test]
    fn test_players_from_elo_state_empty() {
        let state = json!({ "players": {} });
        let names = players_from_elo_state(&state);
        assert!(names.is_empty());
    }
```

- [ ] **Step 2: Run tests — expect compile error (functions not defined yet)**

```bash
cd app && cargo test -p app 2>&1 | tail -20
```

Expected: compile error mentioning `key_to_display_name` and `players_from_elo_state` not found.

- [ ] **Step 3: Add the two helper functions to `commands.rs`**

Add these two functions directly above the `pub fn predict_text` function (around line 53):

```rust
pub(crate) fn key_to_display_name(key: &str) -> String {
    key.split('_')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn players_from_elo_state(state: &serde_json::Value) -> Vec<String> {
    let players = match state.get("players").and_then(|p| p.as_object()) {
        Some(p) => p,
        None => return vec![],
    };
    let mut entries: Vec<(&str, f64)> = players
        .iter()
        .filter_map(|(key, val)| {
            let matches_played = val.get("matches_played")?.as_u64()?;
            if matches_played < 10 { return None; }
            let elo_overall = val.get("elo_overall")?.as_f64()?;
            Some((key.as_str(), elo_overall))
        })
        .collect();
    entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    entries.iter().map(|(key, _)| key_to_display_name(key)).collect()
}
```

- [ ] **Step 4: Run tests — expect all new tests to pass**

```bash
cd app && cargo test -p app 2>&1 | tail -20
```

Expected: all tests pass, including the 6 new ones.

- [ ] **Step 5: Commit**

```bash
git add app/src-tauri/src/commands.rs
git commit -m "feat(commands): add key_to_display_name and players_from_elo_state helpers"
```

---

## Task 2: Backend — `get_player_names` Tauri command

**Files:**
- Modify: `app/src-tauri/src/commands.rs`
- Modify: `app/src-tauri/src/main.rs`

- [ ] **Step 1: Add `get_player_names` command to `commands.rs`**

Add this block after `players_from_elo_state` and before `pub fn predict_text`:

```rust
#[cfg(not(test))]
#[tauri::command]
pub fn get_player_names(tour: String, app_state: tauri::State<AppState>) -> Vec<String> {
    let elo_guard = match tour.as_str() {
        "wta" => app_state.elo_wta.lock().unwrap(),
        _ => app_state.elo_atp.lock().unwrap(),
    };
    match &*elo_guard {
        None => vec![],
        Some(state) => players_from_elo_state(state),
    }
}
```

- [ ] **Step 2: Register the command in `main.rs`**

In `app/src-tauri/src/main.rs`, find the `invoke_handler` block and add `commands::get_player_names` to the list:

```rust
        .invoke_handler(tauri::generate_handler![
            commands::parse_and_predict,
            commands::get_data_as_of_cmd,
            commands::calculate_kelly,
            commands::predict_with_ml,
            commands::parse_with_llm,
            commands::get_player_names,   // ← add this line
            schedule::fetch_and_predict,
            bets::add_bet,
            bets::get_bets,
            bets::update_bet_result,
            bets::delete_bet,
            llm::check_player_news,
            rapidapi::fetch_rapidapi_schedule,
            rapidapi::fetch_rankings,
            rapidapi::fetch_schedule_auto,
            config::load_api_keys,
            commands::trigger_retrain,
        ])
```

- [ ] **Step 3: Build to verify no compile errors**

```bash
cd app && cargo build 2>&1 | tail -20
```

Expected: `Finished` with no errors.

- [ ] **Step 4: Commit**

```bash
git add app/src-tauri/src/commands.rs app/src-tauri/src/main.rs
git commit -m "feat(commands): add get_player_names Tauri command"
```

---

## Task 3: Frontend — rewrite `MatchInput.svelte`

**Files:**
- Modify: `app/src/lib/components/MatchInput.svelte`

- [ ] **Step 1: Replace the entire file with the structured form**

Replace the full content of `app/src/lib/components/MatchInput.svelte` with:

```svelte
<script lang="ts">
  import { invoke } from '@tauri-apps/api/core'
  import { predictions, loading, error, dataAsOf, mlAvailable, selectedTour } from '../stores'

  type Row = { playerA: string; playerB: string; surface: string }
  type ActiveField = { rowIdx: number; field: 'a' | 'b' } | null

  let rows = $state<Row[]>([{ playerA: '', playerB: '', surface: 'Hard' }])
  let playerNames = $state<string[]>([])
  let activeField = $state<ActiveField>(null)

  $effect(() => {
    loadPlayers($selectedTour)
  })

  async function loadPlayers(tour: string) {
    try {
      playerNames = await invoke<string[]>('get_player_names', { tour })
    } catch {
      playerNames = []
    }
  }

  function suggestions(query: string): string[] {
    if (!query.trim()) return []
    const q = query.toLowerCase()
    return playerNames.filter(n => n.toLowerCase().includes(q)).slice(0, 10)
  }

  function selectSuggestion(name: string) {
    if (!activeField) return
    const { rowIdx, field } = activeField
    if (field === 'a') rows[rowIdx].playerA = name
    else rows[rowIdx].playerB = name
    activeField = null
  }

  function addRow() {
    rows = [...rows, { playerA: '', playerB: '', surface: 'Hard' }]
  }

  function removeRow(idx: number) {
    rows = rows.filter((_, i) => i !== idx)
    if (rows.length === 0) rows = [{ playerA: '', playerB: '', surface: 'Hard' }]
  }

  async function handlePredict() {
    const lines = rows
      .filter(r => r.playerA.trim() && r.playerB.trim())
      .map(r => `${r.playerA} vs ${r.playerB} - ${r.surface}`)
    if (lines.length === 0) return

    loading.set(true)
    error.set(null)
    try {
      const result = await invoke<any>('predict_with_ml', {
        request: {
          text: lines.join('\n'),
          tour: $selectedTour,
          tourney_date: new Date().toISOString().slice(0, 10),
        },
      })
      if (result.error) {
        error.set(result.error)
      } else {
        predictions.set(result.predictions)
        dataAsOf.set(result.data_as_of)
        mlAvailable.set(result.ml_available ?? false)
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
    {$selectedTour.toUpperCase()} matches
  </h2>

  <div class="flex flex-col gap-2">
    {#each rows as row, idx}
      <div class="flex items-center gap-2">

        <!-- Player A -->
        <div class="relative flex-1">
          <input
            type="text"
            bind:value={row.playerA}
            onfocus={() => (activeField = { rowIdx: idx, field: 'a' })}
            onblur={() => (activeField = null)}
            onkeydown={e => e.key === 'Escape' && (activeField = null)}
            placeholder="Player A"
            class="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
          />
          {#if activeField?.rowIdx === idx && activeField?.field === 'a' && suggestions(row.playerA).length > 0}
            <ul class="absolute z-10 w-full bg-white border border-gray-200 rounded-md shadow-lg mt-1 max-h-48 overflow-y-auto">
              {#each suggestions(row.playerA) as name}
                <li>
                  <button
                    class="w-full text-left px-3 py-1.5 text-sm hover:bg-blue-50"
                    onmousedown={() => selectSuggestion(name)}
                  >{name}</button>
                </li>
              {/each}
            </ul>
          {/if}
        </div>

        <span class="text-gray-400 text-sm font-medium">vs</span>

        <!-- Player B -->
        <div class="relative flex-1">
          <input
            type="text"
            bind:value={row.playerB}
            onfocus={() => (activeField = { rowIdx: idx, field: 'b' })}
            onblur={() => (activeField = null)}
            onkeydown={e => e.key === 'Escape' && (activeField = null)}
            placeholder="Player B"
            class="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
          />
          {#if activeField?.rowIdx === idx && activeField?.field === 'b' && suggestions(row.playerB).length > 0}
            <ul class="absolute z-10 w-full bg-white border border-gray-200 rounded-md shadow-lg mt-1 max-h-48 overflow-y-auto">
              {#each suggestions(row.playerB) as name}
                <li>
                  <button
                    class="w-full text-left px-3 py-1.5 text-sm hover:bg-blue-50"
                    onmousedown={() => selectSuggestion(name)}
                  >{name}</button>
                </li>
              {/each}
            </ul>
          {/if}
        </div>

        <!-- Surface -->
        <select
          bind:value={row.surface}
          class="px-2 py-2 border border-gray-300 rounded-md text-sm"
        >
          <option>Hard</option>
          <option>Clay</option>
          <option>Grass</option>
        </select>

        <!-- Remove -->
        <button
          onclick={() => removeRow(idx)}
          class="text-gray-400 hover:text-red-500 text-lg leading-none px-1"
          aria-label="Remove row"
        >×</button>

      </div>
    {/each}
  </div>

  <div class="flex justify-between items-center mt-4">
    <button
      onclick={addRow}
      class="px-4 py-2 text-sm text-blue-600 border border-blue-300 rounded-md hover:bg-blue-50"
    >+ Add match</button>
    <button
      onclick={handlePredict}
      disabled={$loading}
      class="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 text-sm"
    >{$loading ? 'Predicting…' : 'Predict'}</button>
  </div>
</div>
```

- [ ] **Step 2: Build the frontend to verify no type errors**

```bash
cd app && npm run build 2>&1 | tail -30
```

Expected: build completes with no errors (warnings about `any` are acceptable).

- [ ] **Step 3: Commit**

```bash
git add app/src/lib/components/MatchInput.svelte
git commit -m "feat(ui): replace textarea with structured autocomplete match input"
```

---

## Self-review

**Spec coverage:**
- ✅ Replace textarea with structured rows
- ✅ Separate Player A / Player B / Surface fields per row
- ✅ Autocomplete dropdown, ≤10 suggestions, substring match
- ✅ `get_player_names` filters matches_played ≥ 10, sorts by elo_overall desc
- ✅ Display name = title-cased key (underscore → space)
- ✅ `onmousedown` prevents blur-before-click race condition
- ✅ Reload player list on tour change via `$effect`
- ✅ Submit path unchanged (`predict_with_ml`)
- ✅ `[+ Add match]` / `[×]` row management

**Type consistency:**
- `key_to_display_name` defined in Task 1, used by `players_from_elo_state` in same task ✅
- `players_from_elo_state` defined in Task 1, used by `get_player_names` in Task 2 ✅
- `Row`, `ActiveField` types defined and used within Task 3 only ✅
