<script lang="ts">
  import type { Prediction, KellyResult } from '../stores'
  import { bankroll, kelly_fraction } from '../stores'
  import { invoke } from '@tauri-apps/api/core'

  let { prediction }: { prediction: Prediction } = $props()

  let odds = $state<number | null>(null)
  let kellyResult = $state<KellyResult | null>(null)
  let loading = $state(false)
  let error = $state<string | null>(null)

  let probA = $derived(Math.round(prediction.prob_a_wins * 1000) / 10)
  let probB = $derived(Math.round(prediction.prob_b_wins * 1000) / 10)
  let mlProbA = $derived(
    prediction.ml_prob_a_wins != null
      ? Math.round(prediction.ml_prob_a_wins * 1000) / 10
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

  $effect(() => {
    if (odds) {
      onOddsChange()
    }
  })
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

  <div class="text-xs text-gray-500 mt-4">
    Elo: {prediction.player_a} {Math.round(prediction.elo_a_overall)} vs {Math.round(prediction.elo_b_overall)}
  </div>
</div>

<style>
  input {
    font-size: 14px;
  }
</style>
