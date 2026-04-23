<script lang="ts">
  import { invoke } from '@tauri-apps/api/core'
  import { predictions, loading, error, dataAsOf } from '../stores'

  let textInput = ''

  async function handleParse() {
    loading.set(true)
    error.set(null)

    try {
      const eloJson = localStorage.getItem('elo_state') || '{}'
      const result = await invoke('parse_and_predict', {
        text: textInput,
        eloJson: eloJson,
      })

      if (result.error) {
        error.set(result.error)
      } else {
        predictions.set(result.predictions)
        dataAsOf.set(result.data_as_of)
      }
    } catch (err) {
      error.set(String(err))
    } finally {
      loading.set(false)
    }
  }
</script>

<div class="p-6 border-b border-gray-200 bg-white">
  <h2 class="text-lg font-semibold mb-4">Paste today's matches</h2>
  <textarea
    bind:value={textInput}
    class="w-full p-3 border border-gray-300 rounded-md font-mono text-sm"
    rows="6"
    placeholder="Alcaraz vs Sinner - Clay&#10;Djokovic vs Zverev - Hard"
  />
  <button
    on:click={handleParse}
    disabled={$loading}
    class="mt-4 px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
  >
    {$loading ? 'Parsing...' : 'Parse'}
  </button>
</div>
