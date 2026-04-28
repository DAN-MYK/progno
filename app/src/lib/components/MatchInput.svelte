<script lang="ts">
  import { invoke } from '@tauri-apps/api/core'
  import { predictions, loading, error, dataAsOf, mlAvailable, selectedTour } from '../stores'

  type Row = { playerA: string; playerB: string; surface: string }
  type ActiveField = { rowIdx: number; field: 'a' | 'b' } | null

  let rows = $state<Row[]>([{ playerA: '', playerB: '', surface: 'Hard' }])
  let playerNames = $state<string[]>([])
  let activeField = $state<ActiveField>(null)

  $effect(() => {
    loadPlayers($selectedTour)
  })

  async function loadPlayers(tour: string) {
    try {
      playerNames = await invoke<string[]>('get_player_names', { tour })
    } catch {
      playerNames = []
    }
  }

  function suggestions(query: string): string[] {
    if (!query.trim()) return []
    const q = query.toLowerCase()
    return playerNames.filter(n => n.toLowerCase().includes(q)).slice(0, 10)
  }

  function selectSuggestion(name: string) {
    if (!activeField) return
    const { rowIdx, field } = activeField
    if (field === 'a') rows[rowIdx].playerA = name
    else rows[rowIdx].playerB = name
    activeField = null
  }

  function addRow() {
    rows = [...rows, { playerA: '', playerB: '', surface: 'Hard' }]
  }

  function removeRow(idx: number) {
    rows = rows.filter((_, i) => i !== idx)
    if (rows.length === 0) rows = [{ playerA: '', playerB: '', surface: 'Hard' }]
  }

  async function handlePredict() {
    const lines = rows
      .filter(r => r.playerA.trim() && r.playerB.trim())
      .map(r => `${r.playerA} vs ${r.playerB} - ${r.surface}`)
    if (lines.length === 0) return

    loading.set(true)
    error.set(null)
    try {
      const result = await invoke<any>('predict_with_ml', {
        request: {
          text: lines.join('\n'),
          tour: $selectedTour,
          tourney_date: new Date().toISOString().slice(0, 10),
        },
      })
      if (result.error) {
        error.set(result.error)
        predictions.set([])
      } else {
        predictions.set(result.predictions)
        dataAsOf.set(result.data_as_of)
        mlAvailable.set(result.ml_available ?? false)
      }
    } catch (err) {
      error.set(String(err))
    } finally {
      loading.set(false)
    }
  }
</script>

<div class="p-6 border-b border-gray-200 bg-white">
  <h2 class="text-lg font-semibold mb-4">
    {$selectedTour.toUpperCase()} matches
  </h2>

  <div class="flex flex-col gap-2">
    {#each rows as row, idx}
      {@const suggsA = suggestions(row.playerA)}
      {@const suggsB = suggestions(row.playerB)}
      <div class="flex items-center gap-2">

        <!-- Player A -->
        <div class="relative flex-1">
          <input
            type="text"
            bind:value={row.playerA}
            onfocus={() => (activeField = { rowIdx: idx, field: 'a' })}
            onblur={() => (activeField = null)}
            onkeydown={e => e.key === 'Escape' && (activeField = null)}
            placeholder="Player A"
            class="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
          />
          {#if activeField?.rowIdx === idx && activeField?.field === 'a' && suggsA.length > 0}
            <ul class="absolute z-10 w-full bg-white border border-gray-200 rounded-md shadow-lg mt-1 max-h-48 overflow-y-auto">
              {#each suggsA as name}
                <li>
                  <button
                    class="w-full text-left px-3 py-1.5 text-sm hover:bg-blue-50"
                    onmousedown={() => selectSuggestion(name)}
                  >{name}</button>
                </li>
              {/each}
            </ul>
          {/if}
        </div>

        <span class="text-gray-400 text-sm font-medium">vs</span>

        <!-- Player B -->
        <div class="relative flex-1">
          <input
            type="text"
            bind:value={row.playerB}
            onfocus={() => (activeField = { rowIdx: idx, field: 'b' })}
            onblur={() => (activeField = null)}
            onkeydown={e => e.key === 'Escape' && (activeField = null)}
            placeholder="Player B"
            class="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
          />
          {#if activeField?.rowIdx === idx && activeField?.field === 'b' && suggsB.length > 0}
            <ul class="absolute z-10 w-full bg-white border border-gray-200 rounded-md shadow-lg mt-1 max-h-48 overflow-y-auto">
              {#each suggsB as name}
                <li>
                  <button
                    class="w-full text-left px-3 py-1.5 text-sm hover:bg-blue-50"
                    onmousedown={() => selectSuggestion(name)}
                  >{name}</button>
                </li>
              {/each}
            </ul>
          {/if}
        </div>

        <!-- Surface -->
        <select
          bind:value={row.surface}
          class="px-2 py-2 border border-gray-300 rounded-md text-sm"
        >
          <option value="Hard">Hard</option>
          <option value="Clay">Clay</option>
          <option value="Grass">Grass</option>
        </select>

        <!-- Remove -->
        <button
          onclick={() => removeRow(idx)}
          class="text-gray-400 hover:text-red-500 text-lg leading-none px-1"
          aria-label="Remove row"
        >×</button>

      </div>
    {/each}
  </div>

  <div class="flex justify-between items-center mt-4">
    <button
      onclick={addRow}
      class="px-4 py-2 text-sm text-blue-600 border border-blue-300 rounded-md hover:bg-blue-50"
    >+ Add match</button>
    <button
      onclick={handlePredict}
      disabled={$loading}
      class="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 text-sm"
    >{$loading ? 'Predicting…' : 'Predict'}</button>
  </div>
</div>
