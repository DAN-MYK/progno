# CLV Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Stats" tab with three Chart.js charts — ROI/win-rate by edge bucket, model edge vs CLV vs ROI by month, and cumulative P&L — backed by a new `closing_odds` field in `BetRecord`.

**Architecture:** `closing_odds` added to the SQLite `bets` table via an idempotent `ALTER TABLE` migration in `open()`. Frontend reads from the existing `bets` store. `StatsPanel.svelte` is a new component; no existing components are modified except `MatchCard.svelte` (new input) and `App.svelte` (new tab). All chart data is computed via `$derived` / `$effect` — no new Tauri commands.

**Tech Stack:** Rust/rusqlite (migration), Svelte 5 runes, TypeScript, Chart.js (direct canvas API, `chart.js/auto`), Tailwind CSS.

---

## File Map

| File | Change |
|------|--------|
| `app/src-tauri/src/bets.rs` | Add `closing_odds: Option<f64>` to `BetRecord`, `open()` migration, update INSERT/SELECT in all commands |
| `app/src/lib/stores.ts` | Add `closing_odds?: number` to `BetRecord` interface |
| `app/src/lib/components/MatchCard.svelte` | Add optional "Closing odds" input to Log bet form |
| `app/src/App.svelte` | Add `'stats'` tab, import and render `StatsPanel` |
| `app/src/lib/components/StatsPanel.svelte` | **New** — three Chart.js charts |

---

## Task 1: Rust — `closing_odds` in `BetRecord` + SQLite migration

**Files:**
- Modify: `app/src-tauri/src/bets.rs`

- [ ] **Step 1: Write failing tests**

In `app/src-tauri/src/bets.rs`, inside `#[cfg(test)] mod tests`, add at the bottom:

```rust
    #[test]
    fn test_closing_odds_round_trip() {
        let dir = TempDir::new().unwrap();
        let conn = tmp_conn(dir.path());
        let mut bet = make_bet("co1", None);
        bet.closing_odds = Some(1.75);
        conn.execute(
            "INSERT INTO bets (id,date,player_a,player_b,surface,tournament,bet_on,our_prob,odds,stake,result,pnl,closing_odds)
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
            rusqlite::params![
                bet.id, bet.date, bet.player_a, bet.player_b, bet.surface,
                bet.tournament, bet.bet_on, bet.our_prob, bet.odds, bet.stake,
                bet.result, bet.pnl, bet.closing_odds,
            ],
        ).unwrap();
        let mut stmt = conn.prepare(
            "SELECT id,date,player_a,player_b,surface,tournament,bet_on,our_prob,odds,stake,result,pnl,closing_odds FROM bets"
        ).unwrap();
        let loaded: Vec<BetRecord> = stmt.query_map([], row_to_bet).unwrap()
            .collect::<Result<_, _>>().unwrap();
        assert_eq!(loaded[0].closing_odds, Some(1.75));
    }

    #[test]
    fn test_closing_odds_null_round_trip() {
        let dir = TempDir::new().unwrap();
        let conn = tmp_conn(dir.path());
        let bet = make_bet("co2", None);
        conn.execute(
            "INSERT INTO bets (id,date,player_a,player_b,surface,tournament,bet_on,our_prob,odds,stake,result,pnl,closing_odds)
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
            rusqlite::params![
                bet.id, bet.date, bet.player_a, bet.player_b, bet.surface,
                bet.tournament, bet.bet_on, bet.our_prob, bet.odds, bet.stake,
                bet.result, bet.pnl, bet.closing_odds,
            ],
        ).unwrap();
        let mut stmt = conn.prepare(
            "SELECT id,date,player_a,player_b,surface,tournament,bet_on,our_prob,odds,stake,result,pnl,closing_odds FROM bets"
        ).unwrap();
        let loaded: Vec<BetRecord> = stmt.query_map([], row_to_bet).unwrap()
            .collect::<Result<_, _>>().unwrap();
        assert_eq!(loaded[0].closing_odds, None);
    }
```

- [ ] **Step 2: Run tests — expect compile error**

```bash
cd /home/mykhailo_dan/apps/progno/app && cargo test -p app 2>&1 | tail -20
```

Expected: compile error — `BetRecord` has no field `closing_odds`.

- [ ] **Step 3: Add `closing_odds` to `BetRecord` struct**

In `bets.rs`, replace the `BetRecord` struct:

```rust
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BetRecord {
    pub id: String,
    pub date: String,
    pub player_a: String,
    pub player_b: String,
    pub surface: String,
    pub tournament: Option<String>,
    /// "a" | "b"
    pub bet_on: String,
    pub our_prob: f64,
    pub odds: f64,
    pub stake: f64,
    /// "win" | "loss" | "void" — None means pending
    pub result: Option<String>,
    pub pnl: Option<f64>,
    #[serde(default)]
    pub closing_odds: Option<f64>,
}
```

(`#[serde(default)]` allows deserialising old JSON records without this field — they get `None`.)

- [ ] **Step 4: Add migration to `open()`**

In `open()`, add one line after the `execute_batch` call that creates the table:

```rust
fn open(path: &std::path::Path) -> Result<rusqlite::Connection, String> {
    let conn = rusqlite::Connection::open(path)
        .map_err(|e| format!("Cannot open bets DB: {e}"))?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS bets (
            id          TEXT    PRIMARY KEY,
            date        TEXT    NOT NULL,
            player_a    TEXT    NOT NULL,
            player_b    TEXT    NOT NULL,
            surface     TEXT    NOT NULL,
            tournament  TEXT,
            bet_on      TEXT    NOT NULL,
            our_prob    REAL    NOT NULL,
            odds        REAL    NOT NULL,
            stake       REAL    NOT NULL,
            result      TEXT,
            pnl         REAL
        );",
    )
    .map_err(|e| format!("Cannot init bets table: {e}"))?;
    // Idempotent migration: ignored if column already exists
    let _ = conn.execute("ALTER TABLE bets ADD COLUMN closing_odds REAL", []);
    Ok(conn)
}
```

- [ ] **Step 5: Update `row_to_bet` to read column 12**

Replace `row_to_bet`:

```rust
fn row_to_bet(row: &rusqlite::Row<'_>) -> rusqlite::Result<BetRecord> {
    Ok(BetRecord {
        id:           row.get(0)?,
        date:         row.get(1)?,
        player_a:     row.get(2)?,
        player_b:     row.get(3)?,
        surface:      row.get(4)?,
        tournament:   row.get(5)?,
        bet_on:       row.get(6)?,
        our_prob:     row.get(7)?,
        odds:         row.get(8)?,
        stake:        row.get(9)?,
        result:       row.get(10)?,
        pnl:          row.get(11)?,
        closing_odds: row.get(12)?,
    })
}
```

- [ ] **Step 6: Update `get_bets` SELECT**

Replace the `prepare` SQL in `get_bets`:

```rust
    let mut stmt = conn
        .prepare(
            "SELECT id,date,player_a,player_b,surface,tournament,bet_on,
                    our_prob,odds,stake,result,pnl,closing_odds
             FROM bets ORDER BY date DESC, rowid DESC",
        )
        .map_err(|e| format!("prepare: {e}"))?;
```

- [ ] **Step 7: Update `add_bet` INSERT**

Replace the `conn.execute` call in `add_bet`:

```rust
    conn.execute(
        "INSERT OR REPLACE INTO bets
         (id,date,player_a,player_b,surface,tournament,bet_on,our_prob,odds,stake,result,pnl,closing_odds)
         VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
        rusqlite::params![
            record.id, record.date, record.player_a, record.player_b,
            record.surface, record.tournament, record.bet_on, record.our_prob,
            record.odds, record.stake, record.result, record.pnl, record.closing_odds,
        ],
    )
    .map_err(|e| format!("Failed to insert bet: {e}"))?;
```

- [ ] **Step 8: Update `maybe_migrate_json` INSERT**

Replace the `conn.execute` call inside `maybe_migrate_json`:

```rust
        let _ = conn.execute(
            "INSERT OR IGNORE INTO bets
             (id,date,player_a,player_b,surface,tournament,bet_on,our_prob,odds,stake,result,pnl,closing_odds)
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
            rusqlite::params![
                r.id, r.date, r.player_a, r.player_b, r.surface, r.tournament,
                r.bet_on, r.our_prob, r.odds, r.stake, r.result, r.pnl, r.closing_odds,
            ],
        );
```

- [ ] **Step 9: Update test helpers**

In the `#[cfg(test)] mod tests` block, update `make_bet` to include the new field:

```rust
    fn make_bet(id: &str, result: Option<&str>) -> BetRecord {
        BetRecord {
            id: id.to_string(),
            date: "2026-04-25".to_string(),
            player_a: "Alcaraz".to_string(),
            player_b: "Sinner".to_string(),
            surface: "Clay".to_string(),
            tournament: None,
            bet_on: "a".to_string(),
            our_prob: 0.65,
            odds: 1.80,
            stake: 50.0,
            result: result.map(str::to_string),
            pnl: None,
            closing_odds: None,
        }
    }
```

Update `test_sqlite_insert_and_query` — the SELECT must include `closing_odds` so `row_to_bet` finds column 12:

```rust
    #[test]
    fn test_sqlite_insert_and_query() {
        let dir = TempDir::new().unwrap();
        let conn = tmp_conn(dir.path());
        let bet = make_bet("abc", None);
        conn.execute(
            "INSERT INTO bets (id,date,player_a,player_b,surface,tournament,bet_on,our_prob,odds,stake,result,pnl)
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
            rusqlite::params![
                bet.id, bet.date, bet.player_a, bet.player_b, bet.surface,
                bet.tournament, bet.bet_on, bet.our_prob, bet.odds, bet.stake,
                bet.result, bet.pnl,
            ],
        ).unwrap();
        let mut stmt = conn.prepare(
            "SELECT id,date,player_a,player_b,surface,tournament,bet_on,our_prob,odds,stake,result,pnl,closing_odds FROM bets"
        ).unwrap();
        let loaded: Vec<BetRecord> = stmt.query_map([], row_to_bet).unwrap()
            .collect::<Result<_, _>>().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "abc");
        assert_eq!(loaded[0].closing_odds, None);
    }
```

- [ ] **Step 10: Run all tests — expect pass**

```bash
cd /home/mykhailo_dan/apps/progno/app && cargo test -p app 2>&1 | tail -25
```

Expected: all tests PASS including the 2 new ones.

- [ ] **Step 11: Commit**

```bash
cd /home/mykhailo_dan/apps/progno
git add app/src-tauri/src/bets.rs
git commit -m "feat(bets): add closing_odds field + SQLite migration"
```

---

## Task 2: TypeScript + MatchCard — `closing_odds` frontend

**Files:**
- Modify: `app/src/lib/stores.ts`
- Modify: `app/src/lib/components/MatchCard.svelte`

- [ ] **Step 1: Add `closing_odds` to `BetRecord` in stores.ts**

In `app/src/lib/stores.ts`, replace the `BetRecord` interface:

```typescript
export interface BetRecord {
  id: string
  date: string
  player_a: string
  player_b: string
  surface: string
  tournament?: string
  bet_on: 'a' | 'b'
  our_prob: number
  odds: number
  stake: number
  result?: 'win' | 'loss' | 'void'
  pnl?: number
  closing_odds?: number
}
```

- [ ] **Step 2: Add `closingOdds` state to `MatchCard.svelte`**

In `app/src/lib/components/MatchCard.svelte`, in the `<script>` block, after the `let logBetLoading` line (around line 19), add:

```typescript
  let closingOdds = $state<number | null>(null)
```

- [ ] **Step 3: Add closing odds input to the log bet form**

In `MatchCard.svelte`, inside the log bet `div` (the one with `class="mt-3 p-3 border border-green-300 rounded bg-green-50 text-xs space-y-2"`), add the closing odds input after the stake/odds display row (after the `<div class="flex justify-between text-gray-600">` block):

```svelte
        <div>
          <label class="text-gray-600 block mb-1">Closing odds <span class="text-gray-400">(optional)</span></label>
          <input
            type="number"
            bind:value={closingOdds}
            placeholder="e.g. 1.85"
            min="1"
            step="0.01"
            class="w-full px-2 py-1 border border-gray-300 rounded text-xs"
          />
        </div>
```

- [ ] **Step 4: Include `closing_odds` in the `logBet` record**

In `MatchCard.svelte`, inside `logBet()`, replace the `record` object construction:

```typescript
    const record: BetRecord = {
      id: crypto.randomUUID(),
      date: new Date().toISOString().slice(0, 10),
      player_a: prediction.player_a,
      player_b: prediction.player_b,
      surface: prediction.surface,
      bet_on: betOnA ? 'a' : 'b',
      our_prob: betProb,
      odds,
      stake: kellyResult.stake,
      closing_odds: closingOdds ?? undefined,
    }
```

- [ ] **Step 5: Reset `closingOdds` when log bet panel closes**

In `MatchCard.svelte`, find the Cancel button's `onclick`:

```svelte
              onclick={() => (showLogBet = false)}
```

Replace with:

```svelte
              onclick={() => { showLogBet = false; closingOdds = null }}
```

Also reset when Confirm succeeds — in `logBet()`, after `showLogBet = false`:

```typescript
      showLogBet = false
      closingOdds = null
```

- [ ] **Step 6: Build frontend to verify no type errors**

```bash
cd /home/mykhailo_dan/apps/progno/app && npm run build 2>&1 | tail -20
```

Expected: build completes with no errors.

- [ ] **Step 7: Commit**

```bash
cd /home/mykhailo_dan/apps/progno
git add app/src/lib/stores.ts app/src/lib/components/MatchCard.svelte
git commit -m "feat(ui): add closing_odds field to BetRecord and Log bet form"
```

---

## Task 3: App.svelte — Stats tab

**Files:**
- Modify: `app/src/App.svelte`

- [ ] **Step 1: Add Stats tab and StatsPanel**

Replace the entire content of `app/src/App.svelte` with:

```svelte
<script lang="ts">
  import MatchInput from './lib/components/MatchInput.svelte'
  import MatchCard from './lib/components/MatchCard.svelte'
  import HistoryPanel from './lib/components/HistoryPanel.svelte'
  import SchedulePanel from './lib/components/SchedulePanel.svelte'
  import StatsPanel from './lib/components/StatsPanel.svelte'
  import Footer from './lib/components/Footer.svelte'
  import { invoke } from '@tauri-apps/api/core'
  import { predictions, error, bankroll, kelly_fraction, selectedTour, dataAsOf, mlAvailable } from './lib/stores'

  let activeTab = $state<'predict' | 'history' | 'schedule' | 'stats'>('predict')

  let retrainLoading = $state(false)
  let retrainMsg = $state<{ ok: boolean; text: string } | null>(null)

  async function triggerRetrain() {
    retrainLoading = true
    retrainMsg = null
    try {
      const out = await invoke<string>('trigger_retrain')
      retrainMsg = { ok: true, text: out || 'Retrain completed.' }
    } catch (e) {
      retrainMsg = { ok: false, text: String(e) }
    } finally {
      retrainLoading = false
    }
  }

  function isDataStale(dateStr: string): boolean {
    if (!dateStr || dateStr === 'unknown') return false
    const asOf = new Date(dateStr)
    return (Date.now() - asOf.getTime()) / 86_400_000 > 14
  }
</script>

<div class="min-h-screen flex flex-col bg-white">
  <header class="bg-white border-b border-gray-200 px-6 py-3">
    <div class="max-w-6xl mx-auto flex justify-between items-center">
      <div class="flex items-center gap-6">
        <h1 class="text-xl font-bold">Progno</h1>
        <nav class="flex gap-1">
          {#each (['predict', 'history', 'stats', 'schedule'] as const) as tab}
            <button
              onclick={() => (activeTab = tab)}
              class="px-3 py-1.5 text-sm rounded {activeTab === tab
                ? 'bg-blue-600 text-white'
                : 'text-gray-600 hover:bg-gray-100'}"
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          {/each}
        </nav>
      </div>
      <div class="flex gap-4 items-center text-sm">
        <label class="flex items-center gap-2">
          <span class="text-gray-600">Tour:</span>
          <select
            bind:value={$selectedTour}
            class="px-2 py-1 border border-gray-300 rounded text-sm"
            onchange={() => predictions.set([])}
          >
            <option value="atp">ATP</option>
            <option value="wta">WTA</option>
          </select>
        </label>
        <label class="flex items-center gap-2">
          <span class="text-gray-600">Bankroll:</span>
          <input
            type="number"
            bind:value={$bankroll}
            min="1"
            step="100"
            class="w-20 px-2 py-1 border border-gray-300 rounded text-sm"
          />
        </label>
        <label class="flex items-center gap-2">
          <span class="text-gray-600">Kelly:</span>
          <select bind:value={$kelly_fraction} class="px-2 py-1 border border-gray-300 rounded text-sm">
            <option value={0.1}>0.1×</option>
            <option value={0.25}>0.25×</option>
            <option value={0.5}>0.5×</option>
            <option value={1}>1.0×</option>
          </select>
        </label>
        <button
          onclick={triggerRetrain}
          disabled={retrainLoading}
          class="px-3 py-1 text-xs rounded border border-gray-300 text-gray-600 hover:bg-gray-100 disabled:opacity-50"
          title="Re-run full training pipeline (just retrain)"
        >
          {retrainLoading ? 'Retraining…' : 'Retrain'}
        </button>
      </div>
    </div>
  </header>

  {#if retrainMsg}
    <div
      class="px-6 py-2 text-xs border-l-4"
      class:bg-green-50={retrainMsg.ok}
      class:border-green-400={retrainMsg.ok}
      class:text-green-800={retrainMsg.ok}
      class:bg-red-50={!retrainMsg.ok}
      class:border-red-400={!retrainMsg.ok}
      class:text-red-800={!retrainMsg.ok}
    >
      {retrainMsg.ok ? 'Retrain succeeded.' : 'Retrain failed:'} {retrainMsg.text}
      <button
        onclick={() => (retrainMsg = null)}
        class="ml-2 underline opacity-70 hover:opacity-100"
      >dismiss</button>
    </div>
  {/if}

  {#if $mlAvailable === false && $predictions.length > 0}
    <div class="bg-yellow-50 border-l-4 border-yellow-400 px-6 py-2 text-xs text-yellow-800">
      ML service unavailable — showing Elo predictions only. Run <code class="font-mono">just build-sidecar</code> to enable the ML model.
    </div>
  {/if}

  {#if isDataStale($dataAsOf)}
    <div class="bg-orange-50 border-l-4 border-orange-400 px-6 py-2 text-xs text-orange-800">
      Model data is stale (as of {$dataAsOf}). Consider retraining: <code class="font-mono">just retrain &lt;version&gt;</code>
    </div>
  {/if}

  {#if activeTab === 'predict'}
    <MatchInput />
    {#if $error}
      <div class="bg-red-50 border-l-4 border-red-500 p-4 m-4 text-red-700 text-sm">
        {$error}
      </div>
    {/if}
    <div class="flex-1">
      {#each $predictions as pred (pred.player_a + pred.player_b)}
        <MatchCard prediction={pred} />
      {/each}
    </div>
  {:else if activeTab === 'history'}
    <div class="flex-1 overflow-auto">
      <HistoryPanel />
    </div>
  {:else if activeTab === 'stats'}
    <div class="flex-1 overflow-auto">
      <StatsPanel />
    </div>
  {:else}
    <div class="flex-1 overflow-auto">
      <SchedulePanel />
    </div>
  {/if}

  <Footer />
</div>
```

- [ ] **Step 2: Verify build (StatsPanel doesn't exist yet — expect import error)**

```bash
cd /home/mykhailo_dan/apps/progno/app && npm run build 2>&1 | grep -i "error\|cannot find" | head -5
```

Expected: error about `StatsPanel.svelte` not found. This confirms the import is wired correctly and the file just needs to be created.

- [ ] **Step 3: Commit**

```bash
cd /home/mykhailo_dan/apps/progno
git add app/src/App.svelte
git commit -m "feat(ui): add Stats tab to App.svelte"
```

---

## Task 4: StatsPanel.svelte — three charts

**Files:**
- Create: `app/src/lib/components/StatsPanel.svelte`

- [ ] **Step 1: Install chart.js**

```bash
cd /home/mykhailo_dan/apps/progno/app && npm install chart.js
```

Expected: `chart.js` added to `node_modules` and `package.json`.

- [ ] **Step 2: Create `StatsPanel.svelte`**

Create `app/src/lib/components/StatsPanel.svelte` with the following content:

```svelte
<script lang="ts">
  import { Chart } from 'chart.js/auto'
  import { bets, type BetRecord } from '../stores'

  // Canvas refs must be $state so $effect tracks the bind:this assignment
  let canvas1 = $state<HTMLCanvasElement | undefined>(undefined)
  let canvas2 = $state<HTMLCanvasElement | undefined>(undefined)
  let canvas3 = $state<HTMLCanvasElement | undefined>(undefined)

  // ── Helpers ──────────────────────────────────────────────────────────────

  function modelEdge(b: BetRecord): number {
    return b.our_prob - 1 / b.odds
  }

  function clv(b: BetRecord): number | null {
    if (b.closing_odds == null) return null
    return 1 / b.closing_odds - 1 / b.odds
  }

  function expectedClv(b: BetRecord): number | null {
    const c = clv(b)
    return c !== null ? c * b.stake : null
  }

  type EdgeBucket = '<0%' | '0–3%' | '3–6%' | '6–10%' | '>10%'
  const BUCKETS: EdgeBucket[] = ['<0%', '0–3%', '3–6%', '6–10%', '>10%']

  function edgeBucket(edge: number): EdgeBucket {
    if (edge < 0) return '<0%'
    if (edge < 0.03) return '0–3%'
    if (edge < 0.06) return '3–6%'
    if (edge < 0.10) return '6–10%'
    return '>10%'
  }

  // ── Chart 1 data — ROI & Win rate by edge bucket ─────────────────────────

  function c1Data(allBets: BetRecord[]) {
    const settled = allBets.filter(b => b.result === 'win' || b.result === 'loss')
    const map = new Map<EdgeBucket, { pnl: number; stake: number; wins: number; n: number }>()
    for (const b of settled) {
      const bucket = edgeBucket(modelEdge(b))
      const cur = map.get(bucket) ?? { pnl: 0, stake: 0, wins: 0, n: 0 }
      cur.pnl += b.pnl ?? 0
      cur.stake += b.stake
      cur.wins += b.result === 'win' ? 1 : 0
      cur.n++
      map.set(bucket, cur)
    }
    const labels = BUCKETS.filter(b => map.has(b))
    const ns = labels.map(b => map.get(b)!.n)
    const roi = labels.map(b => {
      const d = map.get(b)!
      return d.stake > 0 ? Math.round(d.pnl / d.stake * 1000) / 10 : 0
    })
    const winRate = labels.map(b => {
      const d = map.get(b)!
      return d.n > 0 ? Math.round(d.wins / d.n * 1000) / 10 : 0
    })
    return { labels, ns, roi, winRate }
  }

  // ── Chart 2 data — Model edge vs CLV vs ROI by month ────────────────────

  function c2Data(allBets: BetRecord[]) {
    const monthMap = new Map<string, {
      edgeWSum: number; edgeSSum: number
      clvWSum: number; clvSSum: number
      pnl: number; stake: number
    }>()
    for (const b of allBets) {
      const month = b.date.slice(0, 7)
      const cur = monthMap.get(month) ?? { edgeWSum: 0, edgeSSum: 0, clvWSum: 0, clvSSum: 0, pnl: 0, stake: 0 }
      const edge = modelEdge(b)
      cur.edgeWSum += edge * b.stake
      cur.edgeSSum += b.stake
      const c = clv(b)
      if (c !== null) { cur.clvWSum += c * b.stake; cur.clvSSum += b.stake }
      if (b.result === 'win' || b.result === 'loss') { cur.pnl += b.pnl ?? 0; cur.stake += b.stake }
      monthMap.set(month, cur)
    }
    const months = [...monthMap.keys()].sort()
    const edgeLine = months.map(m => {
      const d = monthMap.get(m)!
      return d.edgeSSum > 0 ? Math.round(d.edgeWSum / d.edgeSSum * 1000) / 10 : null
    })
    const clvLine = months.map(m => {
      const d = monthMap.get(m)!
      return d.clvSSum > 0 ? Math.round(d.clvWSum / d.clvSSum * 1000) / 10 : null
    })
    const roiLine = months.map(m => {
      const d = monthMap.get(m)!
      return d.stake > 0 ? Math.round(d.pnl / d.stake * 1000) / 10 : null
    })
    return { months, edgeLine, clvLine, roiLine }
  }

  // ── Chart 3 data — Cumulative P&L + Expected CLV ─────────────────────────

  function c3Data(allBets: BetRecord[]) {
    const settled = allBets
      .filter(b => b.result === 'win' || b.result === 'loss')
      .sort((a, b) => a.date.localeCompare(b.date))
    let cumPnl = 0
    let cumClv = 0
    let hasClv = false
    const labels: string[] = []
    const pnlLine: number[] = []
    const clvLine: (number | null)[] = []
    for (const b of settled) {
      cumPnl += b.pnl ?? 0
      const ec = expectedClv(b)
      if (ec !== null) { cumClv += ec; hasClv = true }
      labels.push(b.date)
      pnlLine.push(Math.round(cumPnl * 100) / 100)
      clvLine.push(hasClv ? Math.round(cumClv * 100) / 100 : null)
    }
    return { labels, pnlLine, clvLine: hasClv ? clvLine : [] as (number | null)[] }
  }

  // ── Reactive flags ────────────────────────────────────────────────────────

  let hasSettled = $derived($bets.some(b => b.result === 'win' || b.result === 'loss'))
  let hasClosingOdds = $derived($bets.some(b => b.closing_odds != null))

  // ── Chart 1 effect ────────────────────────────────────────────────────────

  $effect(() => {
    if (!canvas1 || !hasSettled) return
    const { labels, ns, roi, winRate } = c1Data($bets)
    if (labels.length === 0) return
    const bgRoi = labels.map((l, i) =>
      l === '<0%' ? 'rgba(239,68,68,0.7)' :
      ns[i] < 10 ? 'rgba(156,163,175,0.7)' : 'rgba(59,130,246,0.7)'
    )
    const bgWin = labels.map((l, i) =>
      l === '<0%' ? 'rgba(239,68,68,0.4)' :
      ns[i] < 10 ? 'rgba(156,163,175,0.4)' : 'rgba(16,185,129,0.7)'
    )
    const chart = new Chart(canvas1, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'ROI%', data: roi, backgroundColor: bgRoi, borderWidth: 1 },
          { label: 'Win rate%', data: winRate, backgroundColor: bgWin, borderWidth: 1 },
        ]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top' },
          tooltip: {
            callbacks: {
              afterLabel(ctx) {
                const n = ns[ctx.dataIndex]
                return n < 10 ? `n=${n} ⚠ low sample` : `n=${n}`
              }
            }
          }
        },
        scales: { x: { title: { display: true, text: '%' } } }
      }
    })
    return () => chart.destroy()
  })

  // ── Chart 2 effect ────────────────────────────────────────────────────────

  $effect(() => {
    if (!canvas2 || $bets.length === 0) return
    const { months, edgeLine, clvLine, roiLine } = c2Data($bets)
    if (months.length === 0) return
    const datasets: any[] = [
      {
        label: 'Model edge%',
        data: edgeLine,
        borderColor: 'rgb(59,130,246)',
        backgroundColor: 'rgba(59,130,246,0.1)',
        tension: 0.3,
        spanGaps: false,
      }
    ]
    if (clvLine.some(v => v !== null)) {
      datasets.push({
        label: 'True CLV%',
        data: clvLine,
        borderColor: 'rgb(16,185,129)',
        backgroundColor: 'rgba(16,185,129,0.1)',
        tension: 0.3,
        spanGaps: false,
      })
    }
    if (roiLine.some(v => v !== null)) {
      datasets.push({
        label: 'Realized ROI%',
        data: roiLine,
        borderColor: 'rgb(245,158,11)',
        backgroundColor: 'rgba(245,158,11,0.1)',
        tension: 0.3,
        spanGaps: false,
        borderDash: [4, 4],
      })
    }
    const chart = new Chart(canvas2, {
      type: 'line',
      data: { labels: months, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'top' } },
        scales: { y: { title: { display: true, text: '%' } } }
      }
    })
    return () => chart.destroy()
  })

  // ── Chart 3 effect ────────────────────────────────────────────────────────

  $effect(() => {
    if (!canvas3 || !hasSettled) return
    const { labels, pnlLine, clvLine } = c3Data($bets)
    if (labels.length === 0) return
    const datasets: any[] = [
      {
        label: 'Cumulative P&L',
        data: pnlLine,
        borderColor: 'rgb(59,130,246)',
        backgroundColor: 'rgba(59,130,246,0.05)',
        fill: true,
        tension: 0.2,
        pointRadius: 2,
      }
    ]
    if (clvLine.length > 0) {
      datasets.push({
        label: 'Expected CLV',
        data: clvLine,
        borderColor: 'rgb(16,185,129)',
        borderDash: [5, 5],
        tension: 0.2,
        pointRadius: 2,
      })
    }
    const chart = new Chart(canvas3, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'top' } },
        scales: { y: { title: { display: true, text: '$' } } }
      }
    })
    return () => chart.destroy()
  })
</script>

<div class="p-6 space-y-10">
  {#if $bets.length === 0}
    <div class="text-center text-gray-400 py-16 text-sm">
      No bets logged yet. Use "Log bet" on a MatchCard.
    </div>
  {:else}

    <!-- Chart 1: ROI & Win rate by edge bucket -->
    <section>
      <h2 class="text-sm font-semibold text-gray-700 mb-1">ROI & Win Rate by Edge Bucket</h2>
      <p class="text-xs text-gray-400 mb-3">Settled bets only. Grey bars = n &lt; 10 (low sample).</p>
      {#if !hasSettled}
        <div class="text-gray-400 text-sm">No settled bets yet.</div>
      {:else}
        <div class="h-56"><canvas bind:this={canvas1}></canvas></div>
      {/if}
    </section>

    <!-- Chart 2: Model edge vs CLV vs ROI by month -->
    <section>
      <h2 class="text-sm font-semibold text-gray-700 mb-1">Model Edge vs CLV vs ROI by Month</h2>
      <p class="text-xs text-gray-400 mb-3">Stake-weighted averages. Dashed = realized ROI.</p>
      <div class="h-56"><canvas bind:this={canvas2}></canvas></div>
      {#if !hasClosingOdds}
        <p class="text-xs text-gray-400 mt-2">
          Log closing odds on bets to enable the True CLV line.
        </p>
      {/if}
    </section>

    <!-- Chart 3: Cumulative P&L -->
    <section>
      <h2 class="text-sm font-semibold text-gray-700 mb-1">Cumulative P&L</h2>
      <p class="text-xs text-gray-400 mb-3">Settled bets in chronological order.</p>
      {#if !hasSettled}
        <div class="text-gray-400 text-sm">No settled bets yet.</div>
      {:else}
        <div class="h-56"><canvas bind:this={canvas3}></canvas></div>
        {#if !hasClosingOdds}
          <p class="text-xs text-gray-400 mt-2">
            Log closing odds to enable the Expected CLV line.
          </p>
        {/if}
      {/if}
    </section>

  {/if}
</div>
```

- [ ] **Step 3: Build to verify no errors**

```bash
cd /home/mykhailo_dan/apps/progno/app && npm run build 2>&1 | tail -20
```

Expected: build completes with no errors. Warnings about `any` type in datasets are acceptable.

- [ ] **Step 4: Commit**

```bash
cd /home/mykhailo_dan/apps/progno
git add app/src/lib/components/StatsPanel.svelte app/package.json app/package-lock.json
git commit -m "feat(ui): add StatsPanel with 3 Chart.js charts (edge bucket, monthly CLV, cumulative P&L)"
```

---

## Self-Review Checklist

**Spec coverage:**

- [x] §1.1 `closing_odds?: number` added to `BetRecord` (TypeScript + Rust) — Task 1 + Task 2 Step 1
- [x] §1.2 SQLite migration `ALTER TABLE bets ADD COLUMN closing_odds REAL` — Task 1 Step 4
- [x] §1.3 `add_bet` / `get_bets` updated — Task 1 Steps 6–8
- [x] §1.4 "Closing odds (optional)" input in `MatchCard.svelte` — Task 2 Steps 2–5
- [x] §2.1 `StatsPanel.svelte` created — Task 4
- [x] §2.2 "Stats" tab in `App.svelte` — Task 3
- [x] §2.3 Chart.js direct canvas, `$effect` lifecycle — Task 4
- [x] §3 `model_edge`, `clv`, `expected_clv` formulas — Task 4 helpers
- [x] §3 Chart 1: ROI & win rate by edge bucket, low-n grey, `<0%` red — Task 4
- [x] §3 Chart 2: triple line stake-weighted, `spanGaps: false` — Task 4
- [x] §3 Chart 3: dual line cumulative P&L + expected CLV — Task 4
- [x] §4 Empty states (0 bets, 0 settled, no closing_odds) — Task 4 template
- [x] §4 `chart.destroy()` on unmount via `$effect` return — Task 4
- [x] §5 No new Tauri commands — confirmed
