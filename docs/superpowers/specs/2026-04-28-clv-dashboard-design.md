# CLV Dashboard — Design

**Date**: 2026-04-28
**Status**: Approved

## Goal

Add a "Stats" tab to the Progno app with three Chart.js charts that visualise bet performance. The key addition is a proper CLV (Closing Line Value) tracker — the gold-standard metric for separating skill from luck in sports betting.

---

## Section 1 — Data model change

### 1.1 BetRecord

Add one nullable field to `BetRecord` (TypeScript) and the SQLite `bets` table:

```
closing_odds?: number   // decimal odds at market close; absent for bets logged before this feature
```

**Why nullable**: existing records have no closing odds. The charts degrade gracefully — CLV-dependent lines are omitted when the field is absent.

### 1.2 SQLite migration

`bets.rs` runs on startup:

```sql
ALTER TABLE bets ADD COLUMN closing_odds REAL;
```

SQLite ignores the statement if the column already exists (wrapped in a `PRAGMA table_info` check or caught error).

### 1.3 Rust changes

- `add_bet` command: accept optional `closing_odds` in payload; store NULL when absent.
- `get_bets` command: include `closing_odds` in SELECT.
- `update_bet_result` command: unchanged.

### 1.4 "Log bet" UI

`MatchCard.svelte` — add an optional "Closing odds" input below the existing odds field. Label: "Closing odds (optional)". Placeholder: "e.g. 1.85". Sent as `null` if empty.

---

## Section 2 — New StatsPanel

### 2.1 File

`app/src/lib/components/StatsPanel.svelte` — new file. No changes to `HistoryPanel.svelte`; the existing monthly table stays there.

### 2.2 Navigation

`App.svelte`: add "Stats" as a third tab alongside "Predict" and "History". `StatsPanel` mounts only when the tab is active.

### 2.3 Chart library

**Chart.js** (`chart.js` npm package, direct canvas API — no `svelte-chartjs` wrapper).

Lifecycle pattern (Svelte 5 runes):

```ts
$effect(() => {
  const chart = new Chart(canvas, config)
  return () => chart.destroy()
})
```

This handles both initial mount and reactive updates in one block. One `$effect` per chart canvas.

---

## Section 3 — Charts

### Shared definitions

| Symbol | Formula | Notes |
|--------|---------|-------|
| `model_edge(b)` | `b.our_prob − 1/b.odds` | Always computable |
| `clv(b)` | `1/b.closing_odds − 1/b.odds` | Only when `closing_odds` present |
| `expected_clv(b)` | `clv(b) × b.stake` | Stake-weighted expected gain |
| settled | `b.result === 'win' \| 'loss'` | void excluded everywhere |

Averages are **stake-weighted** where stakes vary (CLV%, ROI% by month).

### Chart 1 — ROI & Win rate by edge bucket

**Type**: grouped horizontal bar chart  
**Data source**: settled bets only  
**Buckets**: `<0%` / `0–3%` / `3–6%` / `6–10%` / `>10%`  
**Two bars per bucket**:
- ROI% = `sum(pnl) / sum(stake) × 100`
- Win rate% = `wins / n × 100`

**Below each bucket**: show `n` (number of bets).  
**Low-n treatment**: when `n < 10` — bar colour is grey, tooltip shows "⚠ low sample (n=X)".  
**Empty bucket**: not rendered.  
**Negative edge bucket** (`<0%`): rendered in red regardless of n.

Diagnostic purpose: confirms whether higher model edge correlates with better ROI and win rate, validating model calibration.

### Chart 2 — Model edge vs CLV vs ROI by month

**Type**: multi-line chart (up to 3 lines)  
**X-axis**: months (YYYY-MM), chronological  
**Y-axis**: % (stake-weighted)

| Line | Data | Shown when |
|------|------|-----------|
| Model edge | stake-weighted mean `model_edge` across all bets in month | always |
| True CLV | stake-weighted mean `clv` across bets with `closing_odds` | ≥1 bet with closing_odds in month |
| Realized ROI | `sum(pnl)/sum(stake)×100` for settled bets | ≥1 settled bet in month |

**Null gap**: months where a line has no data show a gap (Chart.js `spanGaps: false`).  
**Interpretation**: if CLV > ROI consistently → variance or line-shopping problem; if model edge ≈ CLV ≈ ROI → model is working.

### Chart 3 — Cumulative P&L + Expected CLV

**Type**: dual-line chart  
**X-axis**: individual bet dates (chronological)  
**Two lines**:
- Actual cumulative P&L: running sum of `pnl` for settled bets
- Expected CLV: running sum of `expected_clv` for bets with `closing_odds` present (only shown if ≥1 such bet)

**Interpretation**: the gap between lines is visible variance. Long-run convergence of the two lines is the skill signal.

---

## Section 4 — Error and empty states

| State | Behaviour |
|-------|-----------|
| 0 bets total | "No bets logged yet." — full-panel message |
| 0 settled bets | Charts 1 and 3 show empty state text inline; Chart 2 shows model edge line only |
| No `closing_odds` on any bet | CLV line and Expected CLV line absent; tooltip on chart 2 explains "Log closing odds to enable CLV tracking" |
| Tab switch (unmount) | `chart.destroy()` called via `$effect` cleanup — no memory leak |

---

## Section 5 — What is NOT in scope

- No new Tauri commands beyond `bets.rs` changes.
- No changes to `HistoryPanel.svelte` — existing tables remain as-is.
- No server-side/sidecar involvement.
- No closing odds auto-fetch — user enters manually.
- No stop-loss guards or hedge calculator (§7.5 P2/P3 — future).
