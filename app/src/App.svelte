<script lang="ts">
  import MatchInput from './lib/components/MatchInput.svelte'
  import MatchCard from './lib/components/MatchCard.svelte'
  import HistoryPanel from './lib/components/HistoryPanel.svelte'
  import SchedulePanel from './lib/components/SchedulePanel.svelte'
  import Footer from './lib/components/Footer.svelte'
  import { predictions, error, bankroll, kelly_fraction, selectedTour, dataAsOf, mlAvailable } from './lib/stores'

  // localPersist in stores.ts handles bankroll/kelly_fraction/selectedTour persistence via localStorage

  let activeTab = $state<'predict' | 'history' | 'schedule'>('predict')

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
          {#each (['predict', 'history', 'schedule'] as const) as tab}
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
      </div>
    </div>
  </header>

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
  {:else}
    <div class="flex-1 overflow-auto">
      <SchedulePanel />
    </div>
  {/if}

  <Footer />
</div>
