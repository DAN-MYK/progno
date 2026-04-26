<script lang="ts">
  import { invoke } from '@tauri-apps/api/core'
  import type { ScheduledPrediction } from '../stores'
  import {
    apiKey,
    rapidApiKey,
    scheduleDate,
    scheduledPredictions,
    scheduleLoading,
    scheduleError,
    selectedTour,
  } from '../stores'
  import MatchCard from './MatchCard.svelte'
  import { predictionsToCSV, downloadCSV } from '../csv'

  let showKey = $state(false)
  let keyDraft = $state('')
  let editingSource: 'api-tennis' | 'rapidapi' | null = $state(null)
  let activeSource = $state<string | null>(null)

  function startEdit(src: 'api-tennis' | 'rapidapi') {
    editingSource = src
    keyDraft = src === 'rapidapi' ? $rapidApiKey : $apiKey
    showKey = true
  }

  function saveKey() {
    if (editingSource === 'rapidapi') rapidApiKey.set(keyDraft.trim())
    else apiKey.set(keyDraft.trim())
    showKey = false
    editingSource = null
  }

  function cancelKey() {
    showKey = false
    editingSource = null
  }

  async function fetchSchedule() {
    const hasTennisKey = !!$apiKey
    const hasRapidKey = !!$rapidApiKey

    if (!hasTennisKey && !hasRapidKey) {
      scheduleError.set('Enter at least one API key.')
      startEdit('api-tennis')
      return
    }

    scheduleLoading.set(true)
    scheduleError.set(null)
    activeSource = null
    scheduledPredictions.set([])

    try {
      const [result, src] = await invoke<[ScheduledPrediction[], string]>(
        'fetch_schedule_auto',
        {
          apiTennisKey: $apiKey,
          rapidKey: $rapidApiKey,
          tour: $selectedTour,
          date: $scheduleDate,
        }
      )
      scheduledPredictions.set(result)
      activeSource = src
      if (result.length === 0) {
        scheduleError.set('No upcoming matches found for this date.')
      }
    } catch (err) {
      scheduleError.set(String(err))
    } finally {
      scheduleLoading.set(false)
    }
  }
</script>

<!-- toolbar -->
<div class="p-4 border-b border-gray-200 bg-white flex flex-wrap gap-3 items-center">
  <h2 class="text-base font-semibold text-gray-800">
    {$selectedTour.toUpperCase()} Schedule
  </h2>

  <input
    type="date"
    bind:value={$scheduleDate}
    class="px-2 py-1 border border-gray-300 rounded text-sm"
  />

  <button
    onclick={fetchSchedule}
    disabled={$scheduleLoading}
    class="px-4 py-1.5 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50"
  >
    {$scheduleLoading ? 'Fetching…' : 'Fetch Matches'}
  </button>

  {#if $scheduledPredictions.length > 0}
    <button
      onclick={() => downloadCSV(predictionsToCSV($scheduledPredictions), `progno-schedule-${$selectedTour}-${$scheduleDate}.csv`)}
      class="px-3 py-1.5 border border-gray-300 text-gray-600 rounded text-sm hover:bg-gray-50"
    >
      Export CSV
    </button>
  {/if}

  <!-- key controls (auto-fallback: api-tennis → RapidAPI) -->
  <div class="ml-auto flex items-center gap-3 text-xs">
    {#if activeSource}
      <span class="text-green-600 font-medium">via {activeSource}</span>
      <span class="text-gray-300">|</span>
    {/if}

    {#if !showKey}
      <!-- api-tennis.com key -->
      <span class="text-gray-500">api-tennis:</span>
      {#if $apiKey}
        <span class="text-gray-400">●●●●</span>
        <button onclick={() => startEdit('api-tennis')} class="text-blue-500 hover:underline">Change</button>
      {:else}
        <button onclick={() => startEdit('api-tennis')} class="text-orange-500 hover:underline">Set key</button>
      {/if}

      <span class="text-gray-300">|</span>

      <!-- RapidAPI key -->
      <span class="text-gray-500">RapidAPI:</span>
      {#if $rapidApiKey}
        <span class="text-gray-400">●●●●</span>
        <button onclick={() => startEdit('rapidapi')} class="text-blue-500 hover:underline">Change</button>
      {:else}
        <button onclick={() => startEdit('rapidapi')} class="text-orange-500 hover:underline">Set key</button>
      {/if}
    {:else}
      <input
        type="text"
        bind:value={keyDraft}
        placeholder="{editingSource === 'rapidapi' ? 'RapidAPI' : 'api-tennis.com'} key…"
        class="w-64 px-2 py-1 border border-gray-300 rounded text-xs font-mono"
      />
      <button
        onclick={saveKey}
        class="px-2 py-1 bg-green-600 text-white rounded text-xs hover:bg-green-700"
      >Save</button>
      <button onclick={cancelKey} class="text-gray-400 hover:text-gray-600">✕</button>
    {/if}
  </div>
</div>

<!-- error banner -->
{#if $scheduleError}
  <div class="bg-red-50 border-l-4 border-red-500 p-4 mx-4 mt-4 text-red-700 text-sm rounded">
    {$scheduleError}
  </div>
{/if}

<!-- match list -->
{#each $scheduledPredictions as pred (pred.player_a + pred.player_b)}
  <div class="border-b border-gray-100">
    <div class="px-6 py-1.5 bg-amber-50 border-b border-amber-100 flex gap-2 text-xs text-amber-800">
      <span class="font-semibold">{pred.tournament}</span>
      <span>·</span>
      <span>{pred.round}</span>
      {#if pred.event_time}
        <span>·</span>
        <span>{pred.event_time}</span>
      {/if}
      <span class="ml-auto">{pred.surface}</span>
    </div>
    <MatchCard prediction={pred} tournament={pred.tournament} />
  </div>
{/each}
