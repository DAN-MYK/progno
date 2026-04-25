<script lang="ts">
  import type { Prediction, KellyResult, BetRecord } from '../stores'
  import { bankroll, kelly_fraction, bets, llmProvider, llmApiKey } from '../stores'
  import { invoke } from '@tauri-apps/api/core'

  interface NewsResult {
    injury_flag: boolean
    summary: string
    items: string[]
  }

  let { prediction, tournament }: { prediction: Prediction; tournament?: string } = $props()

  let odds = $state<number | null>(null)
  let kellyResult = $state<KellyResult | null>(null)
  let loading = $state(false)
  let error = $state<string | null>(null)
  let betLogged = $state(false)
  let injA = $state(false)
  let injB = $state(false)
  let newsResult = $state<NewsResult | null>(null)
  let newsLoading = $state(false)
  let newsError = $state<string | null>(null)

  // §5.5: P_adj = 0.7 × P_model + 0.3 × 0.5
  function adjustedProbs(pa: number, pb: number, ia: boolean, ib: boolean) {
    let a = ia ? 0.7 * pa + 0.15 : pa
    let b = ib ? 0.7 * pb + 0.15 : pb
    const total = a + b
    return { a: a / total, b: b / total }
  }

  let adjProbs = $derived(adjustedProbs(
    prediction.prob_a_wins, prediction.prob_b_wins, injA, injB
  ))

  let probA = $derived(Math.round(adjProbs.a * 1000) / 10)
  let probB = $derived(Math.round(adjProbs.b * 1000) / 10)
  let mlProbA = $derived(
    prediction.ml_prob_a_wins != null
      ? Math.round(prediction.ml_prob_a_wins * 1000) / 10
      : null
  )
  let isInjured = $derived(injA || injB)

  async function recalcKelly() {
    if (!odds || odds <= 1) {
      kellyResult = null
      error = null
      return
    }
    loading = true
    error = null
    try {
      kellyResult = await invoke<KellyResult>('calculate_kelly', {
        request: {
          model_prob: adjProbs.a,
          decimal_odds: odds,
          bankroll: $bankroll,
          kelly_fraction: $kelly_fraction,
        },
      })
    } catch (e) {
      error = `Kelly calculation failed: ${e}`
      kellyResult = null
    } finally {
      loading = false
    }
  }

  $effect(() => {
    odds; adjProbs.a; // track both — recalc on odds change OR injury toggle
    recalcKelly()
  })

  async function checkNews() {
    if (!$llmApiKey) {
      newsError = 'Set an AI API key in "AI settings" (Manual Input tab).'
      return
    }
    newsLoading = true
    newsError = null
    newsResult = null
    try {
      newsResult = await invoke<NewsResult>('check_player_news', {
        playerA: prediction.player_a,
        playerB: prediction.player_b,
        provider: $llmProvider,
        apiKey: $llmApiKey,
      })
    } catch (e) {
      newsError = String(e)
    } finally {
      newsLoading = false
    }
  }

  async function logBet(betOn: 'a' | 'b') {
    if (!kellyResult || !odds) return
    const prob = betOn === 'a' ? adjProbs.a : adjProbs.b
    const record: BetRecord = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      date: new Date().toISOString().slice(0, 10),
      player_a: prediction.player_a,
      player_b: prediction.player_b,
      surface: prediction.surface,
      tournament: tournament ?? undefined,
      bet_on: betOn,
      our_prob: prob,
      odds: odds,
      stake: kellyResult.stake,
      result: undefined,
      pnl: undefined,
    }
    try {
      await invoke('add_bet', { record })
      bets.update(prev => [...prev, record])
      betLogged = true
      setTimeout(() => (betLogged = false), 2000)
    } catch (e) {
      error = `Failed to log bet: ${e}`
    }
  }
</script>

<div
  class="p-6 border-b border-gray-100 hover:bg-gray-50 transition-colors"
  class:border-l-4={isInjured}
  class:border-l-yellow-400={isInjured}
  class:bg-yellow-50={isInjured}
>
  <div class="mb-4 space-y-2">
    <!-- Player A -->
    <div>
      <div class="flex justify-between items-center mb-1">
        <div class="flex items-center gap-2">
          <span class="text-sm font-medium">{prediction.player_a}</span>
          <button
            onclick={() => { injA = !injA }}
            class="text-xs px-1.5 py-0.5 rounded border transition-colors {injA
              ? 'bg-yellow-400 border-yellow-500 text-yellow-900 font-semibold'
              : 'border-gray-300 text-gray-400 hover:border-yellow-400 hover:text-yellow-500'}"
            title="Injury flag — shrinks probability toward 50%"
          >INJ</button>
        </div>
        <span class="text-sm font-bold text-blue-600">{probA}%</span>
      </div>
      <div class="h-6 bg-gray-200 rounded-sm overflow-hidden">
        <div class="h-full bg-blue-500 transition-all" style="width: {probA}%" />
      </div>
    </div>

    <!-- Player B -->
    <div>
      <div class="flex justify-between items-center mb-1">
        <div class="flex items-center gap-2">
          <span class="text-sm font-medium">{prediction.player_b}</span>
          <button
            onclick={() => { injB = !injB }}
            class="text-xs px-1.5 py-0.5 rounded border transition-colors {injB
              ? 'bg-yellow-400 border-yellow-500 text-yellow-900 font-semibold'
              : 'border-gray-300 text-gray-400 hover:border-yellow-400 hover:text-yellow-500'}"
            title="Injury flag — shrinks probability toward 50%"
          >INJ</button>
        </div>
        <span class="text-sm font-bold text-red-600">{probB}%</span>
      </div>
      <div class="h-6 bg-gray-200 rounded-sm overflow-hidden">
        <div class="h-full bg-red-500 transition-all" style="width: {probB}%" />
      </div>
    </div>
  </div>

  {#if isInjured}
    <div class="mb-3 text-xs text-yellow-700 bg-yellow-100 px-3 py-1.5 rounded">
      Injury adjustment active — P_adj = 0.7 × P_model + 0.3 × 0.5
    </div>
  {/if}

  <div class="p-4 bg-gray-50 rounded">
    <label class="text-xs font-semibold text-gray-600 block mb-2">
      Bookmaker Odds on {prediction.player_a} (decimal)
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
        <span class="font-medium">Stake ({$kelly_fraction}× Kelly):</span>
        <span
          class="font-bold"
          class:text-blue-600={kellyResult.stake > 0}
          class:text-gray-400={kellyResult.stake === 0}
        >
          ${Math.round(kellyResult.stake * 100) / 100}
        </span>
      </div>

      {#if kellyResult.stake > 0}
        <div class="flex gap-2 pt-2 border-t border-blue-200">
          <button
            onclick={() => logBet('a')}
            class="flex-1 py-1 text-xs rounded bg-blue-600 text-white hover:bg-blue-700"
          >
            {betLogged ? '✓ Logged' : `Log bet on ${prediction.player_a}`}
          </button>
          <button
            onclick={() => logBet('b')}
            class="flex-1 py-1 text-xs rounded bg-slate-600 text-white hover:bg-slate-700"
          >
            {betLogged ? '✓ Logged' : `Log bet on ${prediction.player_b}`}
          </button>
        </div>
      {/if}
    </div>
  {/if}

  {#if mlProbA != null}
    <div class="mt-3 p-3 bg-purple-50 rounded text-xs space-y-1">
      <div class="flex justify-between">
        <span class="text-purple-700 font-medium">ML model ({prediction.player_a}):</span>
        <span class="font-bold text-purple-800">{mlProbA}%</span>
      </div>
      {#if prediction.confidence_flag === 'low_history'}
        <div class="text-orange-600">⚠ Low match history — prediction less reliable</div>
      {/if}
    </div>
  {/if}

  <!-- News / injury check -->
  <div class="mt-3">
    {#if !newsResult && !newsLoading}
      <button
        onclick={checkNews}
        class="text-xs text-gray-400 hover:text-gray-600 underline"
      >
        Check news & injuries ({$llmProvider === 'grok' ? 'Grok' : 'Gemini'})
      </button>
    {:else if newsLoading}
      <span class="text-xs text-gray-400">Checking news…</span>
    {/if}

    {#if newsError}
      <div class="mt-1 text-xs text-red-600">{newsError}</div>
    {/if}

    {#if newsResult}
      <div
        class="mt-2 p-3 rounded text-xs space-y-1"
        class:bg-red-50={newsResult.injury_flag}
        class:border={newsResult.injury_flag}
        class:border-red-200={newsResult.injury_flag}
        class:bg-gray-50={!newsResult.injury_flag}
      >
        <div class="flex items-start gap-2">
          {#if newsResult.injury_flag}
            <span class="text-red-600 font-semibold shrink-0">⚠ Injury concern</span>
          {:else}
            <span class="text-green-700 font-semibold shrink-0">✓ No injury flags</span>
          {/if}
        </div>
        <p class="text-gray-700">{newsResult.summary}</p>
        {#if newsResult.items.length > 0}
          <ul class="mt-1 space-y-0.5 text-gray-600 list-none">
            {#each newsResult.items as item}
              <li>· {item}</li>
            {/each}
          </ul>
        {/if}
        <button
          onclick={() => { newsResult = null }}
          class="text-gray-400 hover:text-gray-600 text-xs mt-1"
        >dismiss</button>
      </div>
    {/if}
  </div>

  <div class="text-xs text-gray-500 mt-3">
    Elo: {prediction.player_a} {Math.round(prediction.elo_a_overall)} vs {Math.round(prediction.elo_b_overall)}
  </div>
</div>

<style>
  input { font-size: 14px; }
</style>
