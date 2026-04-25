<script lang="ts">
  import { invoke } from '@tauri-apps/api/core'
  import {
    predictions, loading, error, dataAsOf, selectedTour,
    llmProvider, llmApiKey, geminiApiKey, grokApiKey,
  } from '../stores'
  import { predictionsToCSV, downloadCSV } from '../csv'

  let textInput = $state('')
  let llmLoading = $state(false)
  let showLlmSettings = $state(false)
  // Draft tracks the key for whichever provider is currently selected
  let llmKeyDraft = $state($llmProvider === 'grok' ? $grokApiKey : $geminiApiKey)

  async function handleParse() {
    loading.set(true)
    error.set(null)
    try {
      const result = await invoke('predict_with_ml', {
        request: {
          text: textInput,
          tour: $selectedTour,
          tourney_date: new Date().toISOString().slice(0, 10),
        },
      })
      if (result.error) error.set(result.error)
      else { predictions.set(result.predictions); dataAsOf.set(result.data_as_of) }
    } catch (err) {
      error.set(String(err))
    } finally {
      loading.set(false)
    }
  }

  async function handleParseLlm() {
    if (!$llmApiKey) { showLlmSettings = true; return }
    llmLoading = true
    error.set(null)
    try {
      const result = await invoke('parse_with_llm', {
        text: textInput,
        tour: $selectedTour,
        provider: $llmProvider,
        apiKey: $llmApiKey,
      })
      if (result.error) error.set(result.error)
      else { predictions.set(result.predictions); dataAsOf.set(result.data_as_of) }
    } catch (err) {
      error.set(String(err))
    } finally {
      llmLoading = false
    }
  }

  function onProviderChange() {
    // Sync draft to the newly selected provider's stored key
    llmKeyDraft = $llmProvider === 'grok' ? $grokApiKey : $geminiApiKey
  }

  function saveLlmKey() {
    if ($llmProvider === 'grok') grokApiKey.set(llmKeyDraft.trim())
    else geminiApiKey.set(llmKeyDraft.trim())
    showLlmSettings = false
  }
</script>

<div class="p-6 border-b border-gray-200 bg-white">
  <h2 class="text-lg font-semibold mb-4">
    Paste today's {$selectedTour.toUpperCase()} matches
  </h2>
  <textarea
    bind:value={textInput}
    class="w-full p-3 border border-gray-300 rounded-md font-mono text-sm"
    rows="6"
    placeholder={$selectedTour === 'wta'
      ? 'Swiatek vs Sabalenka - Clay\nGauff vs Rybakina - Hard'
      : 'Alcaraz vs Sinner - Clay\nDjokovic vs Zverev - Hard'}
  ></textarea>

  <div class="mt-4 flex flex-wrap items-center gap-3">
    <button
      onclick={handleParse}
      disabled={$loading}
      class="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
    >
      {$loading ? 'Parsing...' : 'Parse'}
    </button>

    <button
      onclick={handleParseLlm}
      disabled={llmLoading || !textInput.trim()}
      class="px-4 py-2 bg-violet-600 text-white rounded-md hover:bg-violet-700 disabled:opacity-50 text-sm"
    >
      {llmLoading ? 'AI parsing…' : `Try with AI (${$llmProvider === 'grok' ? 'Grok' : 'Gemini'})`}
    </button>

    {#if $predictions.length > 0}
      <button
        onclick={() => downloadCSV(predictionsToCSV($predictions), `progno-${$selectedTour}-${new Date().toISOString().slice(0,10)}.csv`)}
        class="px-3 py-2 border border-gray-300 text-gray-600 rounded-md text-sm hover:bg-gray-50"
      >
        Export CSV
      </button>
    {/if}

    <button
      onclick={() => { llmKeyDraft = $llmProvider === 'grok' ? $grokApiKey : $geminiApiKey; showLlmSettings = !showLlmSettings }}
      class="ml-auto text-xs text-gray-400 hover:text-gray-600"
    >
      AI settings
    </button>
  </div>

  {#if showLlmSettings}
    <div class="mt-3 p-3 bg-gray-50 rounded border border-gray-200 flex flex-wrap gap-2 items-center text-sm">
      <select
        bind:value={$llmProvider}
        onchange={onProviderChange}
        class="px-2 py-1 border border-gray-300 rounded text-xs"
      >
        <option value="grok">Grok (xAI)</option>
        <option value="gemini">Gemini Flash</option>
      </select>
      <div class="flex items-center gap-1 text-xs text-gray-500">
        {#if ($llmProvider === 'grok' ? $grokApiKey : $geminiApiKey)}
          <span class="text-green-600">●</span> key set
        {:else}
          <span class="text-red-500">●</span> no key
        {/if}
      </div>
      <input
        type="text"
        bind:value={llmKeyDraft}
        placeholder="{$llmProvider === 'grok' ? 'xai-…' : 'AIzaSy…'}"
        class="flex-1 min-w-48 px-2 py-1 border border-gray-300 rounded text-xs font-mono"
      />
      <button
        onclick={saveLlmKey}
        class="px-3 py-1 bg-green-600 text-white rounded text-xs hover:bg-green-700"
      >
        Save
      </button>
      <button
        onclick={() => (showLlmSettings = false)}
        class="text-gray-400 hover:text-gray-600 text-xs"
      >
        ✕
      </button>
    </div>
  {/if}
</div>
