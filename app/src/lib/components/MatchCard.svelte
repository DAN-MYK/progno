<script lang="ts">
  import type { Prediction, KellyResult, BetRecord } from '../stores'
  import { bankroll, kelly_fraction, bets } from '../stores'
  import { invoke } from '@tauri-apps/api/core'

  let { prediction }: { prediction: Prediction } = $props()

  let odds = $state<number | null>(null)
  let kellyResult = $state<KellyResult | null>(null)
  let loading = $state(false)
  let error = $state<string | null>(null)
  let showBreakdown = $state(false)
  let showLogBet = $state(false)
  let betOnA = $state(true)
  let logBetError = $state<string | null>(null)
  let logBetLoading = $state(false)

  let probA = $derived(Math.round(prediction.prob_a_wins * 1000) / 10)
  let probB = $derived(Math.round(prediction.prob_b_wins * 1000) / 10)
  let mlProbA = $derived(
    prediction.ml_prob_a_wins != null
      ? Math.round(prediction.ml_prob_a_wins * 1000) / 10
      : null
  )
  let eloProbA = $derived(Math.round(prediction.prob_a_wins * 1000) / 10)
  let mlAdjustment = $derived(
    prediction.ml_prob_a_wins != null
      ? Math.round((prediction.ml_prob_a_wins - prediction.prob_a_wins) * 1000) / 10
      : null
  )

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
          model_prob: prediction.ml_prob_a_wins ?? prediction.prob_a_wins,
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

  $effect(() => {
    if (odds) {
      onOddsChange()
    }
  })

  async function logBet() {
    if (!odds || !kellyResult) return
    logBetLoading = true
    logBetError = null
    const betProb = betOnA
      ? (prediction.ml_prob_a_wins ?? prediction.prob_a_wins)
      : 1 - (prediction.ml_prob_a_wins ?? prediction.prob_a_wins)
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
    }
    try {
      await invoke('add_bet', { record })
      bets.update(prev => [...prev, record])
      showLogBet = false
    } catch (e) {
      logBetError = String(e)
    } finally {
      logBetLoading = false
    }
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
        <span class="text-sm font-bold text-blue-600">
          {mlProbA ?? probA}%
        </span>
      </div>
      <div class="h-6 bg-gray-200 rounded-sm overflow-hidden">
        <div
          class="h-full bg-blue-500"
          style="width: {mlProbA ?? probA}%"
        />
      </div>
    </div>

    <div>
      <div class="flex justify-between items-center mb-1">
        <span class="text-sm font-medium">{prediction.player_b}</span>
        <span class="text-sm font-bold text-red-600">
          {mlProbA != null ? Math.round((100 - mlProbA) * 10) / 10 : probB}%
        </span>
      </div>
      <div class="h-6 bg-gray-200 rounded-sm overflow-hidden">
        <div
          class="h-full bg-red-500"
          style="width: {mlProbA != null ? 100 - mlProbA : probB}%"
        />
      </div>
    </div>
  </div>

  {#if mlProbA != null}
    <div class="mt-3">
      <button
        onclick={() => showBreakdown = !showBreakdown}
        class="text-xs text-purple-600 hover:text-purple-800 underline"
      >
        {showBreakdown ? 'Hide' : 'Show'} model breakdown
      </button>

      {#if showBreakdown}
        <div class="mt-2 p-3 bg-purple-50 rounded text-xs space-y-1 font-mono">
          <div class="flex justify-between">
            <span class="text-gray-600">Elo baseline ({prediction.player_a}):</span>
            <span class="font-semibold">{eloProbA}%</span>
          </div>
          <div
            class="flex justify-between"
            class:text-green-700={mlAdjustment != null && mlAdjustment > 0}
            class:text-red-700={mlAdjustment != null && mlAdjustment < 0}
          >
            <span>ML adjustment:</span>
            <span class="font-semibold">
              {mlAdjustment != null ? (mlAdjustment > 0 ? '+' : '') + mlAdjustment + '%' : '—'}
            </span>
          </div>
          <div class="flex justify-between border-t border-purple-200 pt-1">
            <span class="font-medium text-purple-800">CatBoost calibrated:</span>
            <span class="font-bold text-purple-800">{mlProbA}%</span>
          </div>
          {#if prediction.confidence_flag === 'low_history'}
            <div class="text-orange-600 pt-1">Warning: low match history</div>
          {:else if prediction.confidence_flag === 'insufficient_data'}
            <div class="text-red-600 pt-1">Warning: insufficient data</div>
          {:else if prediction.confidence_flag === 'low_context'}
            <div class="text-gray-500 pt-1">Note: default tournament context used</div>
          {/if}
        </div>
      {/if}
    </div>
  {/if}

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
        <span class="font-medium">Stake ({$kelly_fraction}× Kelly):</span>
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

  {#if kellyResult && kellyResult.stake > 0}
    {#if !showLogBet}
      <button
        onclick={() => { showLogBet = true; betOnA = true }}
        class="mt-3 w-full py-1.5 text-xs bg-green-600 text-white rounded hover:bg-green-700"
      >
        Log bet
      </button>
    {:else}
      <div class="mt-3 p-3 border border-green-300 rounded bg-green-50 text-xs space-y-2">
        <div class="font-semibold text-green-800">Log this bet</div>
        <div class="flex gap-2">
          <button
            onclick={() => (betOnA = true)}
            class="flex-1 py-1 rounded border {betOnA ? 'bg-green-600 text-white border-green-600' : 'border-gray-300 text-gray-600'}"
          >
            {prediction.player_a}
          </button>
          <button
            onclick={() => (betOnA = false)}
            class="flex-1 py-1 rounded border {!betOnA ? 'bg-green-600 text-white border-green-600' : 'border-gray-300 text-gray-600'}"
          >
            {prediction.player_b}
          </button>
        </div>
        <div class="flex justify-between text-gray-600">
          <span>Stake: <strong>${Math.round(kellyResult.stake * 100) / 100}</strong></span>
          <span>Odds: <strong>{odds}</strong></span>
        </div>
        {#if logBetError}
          <div class="text-red-600">{logBetError}</div>
        {/if}
        <div class="flex gap-2">
          <button
            onclick={logBet}
            disabled={logBetLoading}
            class="flex-1 py-1 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
          >
            {logBetLoading ? 'Saving…' : 'Confirm'}
          </button>
          <button
            onclick={() => (showLogBet = false)}
            class="px-3 py-1 border border-gray-300 text-gray-600 rounded hover:bg-gray-100"
          >
            Cancel
          </button>
        </div>
      </div>
    {/if}
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
