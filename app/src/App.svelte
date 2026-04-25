<script lang="ts">
  import { onMount } from 'svelte'
  import { invoke } from '@tauri-apps/api/core'
  import MatchInput from './lib/components/MatchInput.svelte'
  import MatchCard from './lib/components/MatchCard.svelte'
  import SchedulePanel from './lib/components/SchedulePanel.svelte'
  import HistoryPanel from './lib/components/HistoryPanel.svelte'
  import Footer from './lib/components/Footer.svelte'
  import {
    predictions, error, bankroll, kelly_fraction, selectedTour,
    geminiApiKey, grokApiKey, rapidApiKey, apiKey,
  } from './lib/stores'

  let activeTab = $state<'manual' | 'schedule' | 'history'>('manual')

  onMount(async () => {
    try {
      const keys = await invoke<{
        gemini: string; grok: string; rapidapi: string; api_tennis: string
      }>('load_api_keys')
      if (keys.gemini)    geminiApiKey.set(keys.gemini)
      if (keys.grok)      grokApiKey.set(keys.grok)
      if (keys.rapidapi)  rapidApiKey.set(keys.rapidapi)
      if (keys.api_tennis) apiKey.set(keys.api_tennis)
    } catch {
      // file not found — user can enter keys manually
    }
  })
</script>

<div class="min-h-screen flex flex-col bg-white">
  <header class="bg-white border-b border-gray-200 px-6 py-4">
    <div class="max-w-6xl mx-auto flex justify-between items-center">
      <h1 class="text-2xl font-bold">Progno</h1>
      <div class="flex gap-6 items-center text-sm">
        <label class="flex items-center gap-2">
          <span class="text-gray-700">Tour:</span>
          <select
            bind:value={$selectedTour}
            class="px-2 py-1 border border-gray-300 rounded"
            onchange={() => predictions.set([])}
          >
            <option value="atp">ATP</option>
            <option value="wta">WTA</option>
          </select>
        </label>
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
  </header>

  <!-- tab nav -->
  <nav class="border-b border-gray-200 bg-white px-6">
    <div class="max-w-6xl mx-auto flex gap-0">
      <button
        onclick={() => (activeTab = 'manual')}
        class="px-5 py-2.5 text-sm font-medium border-b-2 transition-colors {activeTab === 'manual'
          ? 'border-blue-600 text-blue-600'
          : 'border-transparent text-gray-500 hover:text-gray-700'}"
      >
        Manual Input
      </button>
      <button
        onclick={() => (activeTab = 'schedule')}
        class="px-5 py-2.5 text-sm font-medium border-b-2 transition-colors {activeTab === 'schedule'
          ? 'border-blue-600 text-blue-600'
          : 'border-transparent text-gray-500 hover:text-gray-700'}"
      >
        Schedule
      </button>
      <button
        onclick={() => (activeTab = 'history')}
        class="px-5 py-2.5 text-sm font-medium border-b-2 transition-colors {activeTab === 'history'
          ? 'border-blue-600 text-blue-600'
          : 'border-transparent text-gray-500 hover:text-gray-700'}"
      >
        History
      </button>
    </div>
  </nav>

  {#if activeTab === 'manual'}
    <MatchInput />

    {#if $error}
      <div class="bg-red-50 border-l-4 border-red-500 p-4 m-4 text-red-700">
        {$error}
      </div>
    {/if}

    <div class="flex-1">
      {#each $predictions as pred (pred.player_a + pred.player_b)}
        <MatchCard prediction={pred} />
      {/each}
    </div>
  {:else if activeTab === 'schedule'}
    <div class="flex-1">
      <SchedulePanel />
    </div>
  {:else}
    <div class="flex-1">
      <HistoryPanel />
    </div>
  {/if}

  <Footer />
</div>
