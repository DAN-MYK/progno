<script lang="ts">
  import type { Prediction, KellyResult } from '../stores'
  import { bankroll, kelly_fraction } from '../stores'
  import { invoke } from '@tauri-apps/api/core'

  let { prediction }: { prediction: Prediction } = $props()

  let expanded = $state(false)
  let bettingSide = $state<'a' | 'b'>('a')
  let oddsRaw = $state('')
  let kellyResult = $state<KellyResult | null>(null)
  let kellyLoading = $state(false)

  let probA = $derived(Math.round(prediction.prob_a_wins * 1000) / 10)
  let probB = $derived(Math.round(prediction.prob_b_wins * 1000) / 10)
  let mlProbA = $derived(
    prediction.ml_prob_a_wins != null
      ? Math.round(prediction.ml_prob_a_wins * 1000) / 10
      : null
  )
  let modelProb = $derived(bettingSide === 'a' ? prediction.prob_a_wins : prediction.prob_b_wins)
  let odds = $derived(parseFloat(oddsRaw) || null)
  let hasValue = $derived(kellyResult !== null && kellyResult.edge > 0)
  let accentColor = $derived(
    kellyResult === null
      ? 'var(--no-value-border)'
      : hasValue
        ? 'var(--value-border)'
        : 'var(--no-value-border)'
  )
  let rowOpacity = $derived(kellyResult !== null && !hasValue ? '0.6' : '1')

  $effect(() => {
    const o = odds
    const mp = modelProb
    const br = $bankroll
    const kf = $kelly_fraction
    if (!o || o <= 1) {
      kellyResult = null
      return
    }
    computeKelly(mp, o, br, kf)
  })

  async function computeKelly(mp: number, o: number, br: number, kf: number) {
    kellyLoading = true
    try {
      kellyResult = await invoke<KellyResult>('calculate_kelly', {
        request: { model_prob: mp, decimal_odds: o, bankroll: br, kelly_fraction: kf },
      })
    } catch {
      kellyResult = null
    } finally {
      kellyLoading = false
    }
  }

  function formatStake(k: KellyResult): string {
    return `$${k.stake.toFixed(2)}`
  }

  function formatEdge(k: KellyResult): string {
    return `${k.edge >= 0 ? '+' : ''}${(k.edge * 100).toFixed(1)}%`
  }
</script>

<div class="card" style:border-left-color={accentColor} style:opacity={rowOpacity}>
  <!-- Collapsed row header -->
  <button class="row-header" onclick={() => (expanded = !expanded)} aria-expanded={expanded}>
    <div class="row-main">
      <span class="player-prob">
        <span class="pname">{prediction.player_a}</span>
        <span class="prob blue">{probA}%</span>
        <span class="dot">·</span>
        <span class="pname">{prediction.player_b}</span>
        <span class="prob red">{probB}%</span>
      </span>
      <span class="toggle-icon">{expanded ? '▲' : '▼'}</span>
    </div>
    <div class="row-meta">
      <span>{prediction.surface}</span>
      {#if kellyResult}
        <span>·</span>
        <span class:value-text={hasValue} class:muted-text={!hasValue}>
          {hasValue ? `Stake ${formatStake(kellyResult)}` : 'No value'}
        </span>
        {#if hasValue}
          <span>·</span>
          <span class="value-text">Edge {formatEdge(kellyResult)}</span>
        {/if}
      {/if}
    </div>
  </button>

  <!-- Expanded body -->
  {#if expanded}
    <div class="row-body">
      <div class="prob-bar-wrap">
        <span class="bar-label pname">{prediction.player_a}</span>
        <div class="bar-track">
          <div class="bar-fill fav" style:width="{probA}%"></div>
        </div>
        <span class="bar-pct">{probA}%</span>
      </div>
      <div class="prob-bar-wrap">
        <span class="bar-label pname">{prediction.player_b}</span>
        <div class="bar-track">
          <div class="bar-fill dog" style:width="{probB}%"></div>
        </div>
        <span class="bar-pct">{probB}%</span>
      </div>

      <div class="bet-controls">
        <select class="side-select" bind:value={bettingSide} aria-label="Bet on">
          <option value="a">Bet on: {prediction.player_a}</option>
          <option value="b">Bet on: {prediction.player_b}</option>
        </select>

        <input
          type="number"
          class="odds-input"
          bind:value={oddsRaw}
          placeholder="Odds e.g. 2.20"
          min="1.01"
          step="0.01"
        />

        <div class="kelly-pill" class:value={hasValue} class:no-value={!hasValue || kellyResult === null}>
          {#if kellyResult && hasValue}
            <span>Stake {formatStake(kellyResult)}</span>
            <span class="edge-badge">{formatEdge(kellyResult)}</span>
          {:else if kellyResult}
            <span>No value</span>
          {:else}
            <span>{kellyLoading ? '…' : '—'}</span>
          {/if}
        </div>
      </div>
    </div>
  {/if}

  {#if mlProbA != null}
    <div class="ml-section">
      <span class="ml-label">ML {prediction.player_a}:</span>
      <span class="ml-prob">{mlProbA}%</span>
      {#if prediction.confidence_flag === 'low_history'}
        <span class="ml-warn">low history</span>
      {/if}
    </div>
  {/if}
</div>

<style>
  .card {
    border-left: 3px solid var(--no-value-border);
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    transition: opacity 0.15s, border-left-color 0.2s;
  }

  .row-header {
    width: 100%;
    background: none;
    border: none;
    padding: 10px 14px;
    cursor: pointer;
    text-align: left;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }

  .row-header:hover {
    background: color-mix(in srgb, var(--text-primary) 3%, transparent);
  }

  .row-main {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .player-prob {
    display: flex;
    align-items: center;
    gap: 5px;
    flex-wrap: wrap;
  }

  .pname {
    font-size: 12px;
    font-weight: 700;
    color: var(--text-primary);
  }

  .prob {
    font-size: 12px;
    font-weight: 600;
  }

  .prob.blue { color: var(--accent-blue); }
  .prob.red  { color: var(--accent-red); }

  .dot { color: var(--text-muted); font-size: 10px; }

  .toggle-icon {
    font-size: 9px;
    color: var(--text-muted);
    flex-shrink: 0;
    margin-left: 8px;
  }

  .row-meta {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 10px;
    color: var(--text-secondary);
  }

  .value-text { color: var(--accent-green); }
  .muted-text { color: var(--text-muted); }

  /* ── Expanded body ───────────────────────────────────────── */
  .row-body {
    padding: 10px 14px 14px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    border-top: 1px solid var(--border);
    background: color-mix(in srgb, var(--bg) 50%, var(--surface));
  }

  .prob-bar-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .bar-label {
    width: 80px;
    flex-shrink: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .bar-track {
    flex: 1;
    height: 14px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }

  .bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
  }

  .bar-fill.fav { background: var(--accent-blue); }
  .bar-fill.dog { background: var(--accent-red); }

  .bar-pct {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-secondary);
    width: 32px;
    text-align: right;
    flex-shrink: 0;
  }

  .bet-controls {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    margin-top: 4px;
  }

  .side-select,
  .odds-input {
    font-size: 11px;
    padding: 5px 8px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--surface);
    color: var(--text-primary);
    font-family: inherit;
    outline: none;
  }

  .side-select:focus,
  .odds-input:focus {
    border-color: var(--accent-blue);
  }

  .odds-input {
    width: 120px;
    flex-shrink: 0;
  }

  .kelly-pill {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    font-weight: 600;
    padding: 5px 10px;
    border-radius: 20px;
    flex-shrink: 0;
  }

  .kelly-pill.value {
    background: color-mix(in srgb, var(--accent-green) 12%, transparent);
    color: var(--accent-green);
  }

  .kelly-pill.no-value {
    background: color-mix(in srgb, var(--text-muted) 12%, transparent);
    color: var(--text-muted);
  }

  .edge-badge {
    font-size: 10px;
    font-weight: 700;
  }

  .ml-section {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 6px 14px;
    font-size: 10px;
    border-top: 1px solid var(--border);
    background: color-mix(in srgb, var(--accent-blue) 4%, var(--surface));
  }

  .ml-label { color: var(--text-secondary); }
  .ml-prob { font-weight: 700; color: var(--accent-blue); }
  .ml-warn { color: #b45309; margin-left: auto; }
</style>
