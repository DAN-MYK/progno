<script lang="ts">
  import { invoke } from '@tauri-apps/api/core'
  import { loading, error, appendPredictions, dataAsOf, selectedTour } from '../stores'

  interface PlayerSuggestion {
    name: string
    player_id: string
    elo: number
    matches: number
  }

  let inputEl = $state<HTMLInputElement | null>(null)
  let textInput = $state('')
  let suggestions = $state<PlayerSuggestion[]>([])
  let activeIndex = $state(-1)
  let showDropdown = $state(false)
  let cursorSide = $state<'a' | 'b'>('a')
  let confirmedA = $state<string | null>(null)
  let confirmedB = $state<string | null>(null)

  // ── Live preview ────────────────────────────────────────────
  const MATCH_RE = /^(.+?)\s+vs\s+(.+?)(?:\s*[-–]\s*(.+))?$/i

  function computePreview(): { state: 'ok' | 'warn' | 'hidden'; text: string } {
    const m = textInput.trim().match(MATCH_RE)
    if (!m) return { state: 'hidden', text: '' }
    const pa = m[1].trim()
    const pb = m[2].trim()
    const surf = m[3]?.trim() ?? null

    const aOk = confirmedA !== null && confirmedA.toLowerCase() === pa.toLowerCase()
    const bOk = confirmedB !== null && confirmedB.toLowerCase() === pb.toLowerCase()

    const label = surf ? `${pa} vs ${pb} · ${surf}` : `${pa} vs ${pb}`
    if (aOk && bOk) return { state: 'ok', text: `OK ${label}` }
    const unknown = !aOk ? pa : pb
    return { state: 'warn', text: `Warning: "${unknown}" not recognised` }
  }

  let previewResult = $derived(computePreview())
  let canPredict = $derived(previewResult.state !== 'hidden' && !$loading)

  // ── Autocomplete helpers ────────────────────────────────────
  function currentToken(): { token: string; side: 'a' | 'b' } {
    const val = textInput
    const vsIdx = val.toLowerCase().indexOf(' vs ')
    const cursor = inputEl?.selectionStart ?? val.length

    if (vsIdx === -1 || cursor <= vsIdx + 1) {
      return { token: val.substring(0, cursor).trim(), side: 'a' }
    }
    const afterVs = val.substring(vsIdx + 4)
    const dashIdx = afterVs.indexOf(' - ')
    const bPart = dashIdx === -1 ? afterVs : afterVs.substring(0, dashIdx)
    return { token: bPart.trim(), side: 'b' }
  }

  async function onInput() {
    const { token, side } = currentToken()
    cursorSide = side

    // Invalidate confirmed player when text changes
    if (side === 'a' && confirmedA) {
      const vsIdx = textInput.toLowerCase().indexOf(' vs ')
      const leftText = vsIdx === -1 ? textInput.trim() : textInput.substring(0, vsIdx).trim()
      if (leftText.toLowerCase() !== confirmedA.toLowerCase()) confirmedA = null
    }
    if (side === 'b' && confirmedB) {
      const vsIdx = textInput.toLowerCase().indexOf(' vs ')
      const rightRaw = vsIdx === -1 ? '' : textInput.substring(vsIdx + 4)
      const dashIdx = rightRaw.indexOf(' - ')
      const rightText = (dashIdx === -1 ? rightRaw : rightRaw.substring(0, dashIdx)).trim()
      if (rightText.toLowerCase() !== confirmedB.toLowerCase()) confirmedB = null
    }

    if (token.length >= 2) {
      try {
        suggestions = await invoke<PlayerSuggestion[]>('search_players', { query: token })
        showDropdown = suggestions.length > 0
        activeIndex = -1
      } catch {
        showDropdown = false
      }
    } else {
      suggestions = []
      showDropdown = false
    }
  }

  function selectSuggestion(s: PlayerSuggestion) {
    const val = textInput
    const vsIdx = val.toLowerCase().indexOf(' vs ')

    if (cursorSide === 'a') {
      const rightPart = vsIdx !== -1 ? val.substring(vsIdx + 4) : ''
      textInput = rightPart ? `${s.name} vs ${rightPart}` : `${s.name} vs `
      confirmedA = s.name
      requestAnimationFrame(() => {
        if (inputEl) {
          const pos = `${s.name} vs `.length
          inputEl.setSelectionRange(pos, pos)
        }
      })
    } else {
      const leftPart = vsIdx !== -1 ? val.substring(0, vsIdx) : val.trim()
      const afterVs = vsIdx !== -1 ? val.substring(vsIdx + 4) : ''
      const dashIdx = afterVs.indexOf(' - ')
      const surfPart = dashIdx !== -1 ? ` - ${afterVs.substring(dashIdx + 3)}` : ''
      textInput = `${leftPart} vs ${s.name}${surfPart}`
      confirmedB = s.name
    }

    showDropdown = false
    suggestions = []
    inputEl?.focus()
  }

  function onKeydown(e: KeyboardEvent) {
    if (!showDropdown) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      activeIndex = Math.min(activeIndex + 1, suggestions.length - 1)
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      activeIndex = Math.max(activeIndex - 1, -1)
    } else if ((e.key === 'Enter' || e.key === 'Tab') && activeIndex >= 0) {
      e.preventDefault()
      selectSuggestion(suggestions[activeIndex])
    } else if (e.key === 'Escape') {
      showDropdown = false
    }
  }

  function highlightMatch(text: string, query: string): string {
    if (!query) return text
    const idx = text.toLowerCase().indexOf(query.toLowerCase())
    if (idx === -1) return text
    return (
      text.substring(0, idx) +
      `<mark>${text.substring(idx, idx + query.length)}</mark>` +
      text.substring(idx + query.length)
    )
  }

  // ── Predict ─────────────────────────────────────────────────
  async function handlePredict() {
    if (!canPredict) return
    loading.set(true)
    error.set(null)
    try {
      const result = await invoke<{ predictions: any[]; data_as_of: string; error: string | null }>(
        'predict_with_ml',
        {
          request: {
            text: textInput,
            tour: $selectedTour,
            tourney_date: new Date().toISOString().slice(0, 10),
          },
        }
      )
      if (result.error) {
        error.set(result.error)
      } else {
        appendPredictions(result.predictions)
        dataAsOf.set(result.data_as_of)
        textInput = ''
        confirmedA = null
        confirmedB = null
      }
    } catch (err) {
      error.set(String(err))
    } finally {
      loading.set(false)
    }
  }
</script>

<div class="input-section">
  <div class="field-wrap" role="combobox" aria-expanded={showDropdown} aria-haspopup="listbox" aria-controls="autocomplete-list">
    <input
      bind:this={inputEl}
      bind:value={textInput}
      oninput={onInput}
      onkeydown={onKeydown}
      onblur={() => setTimeout(() => { showDropdown = false }, 150)}
      type="text"
      placeholder="Alcaraz vs Sinner - Clay"
      class="text-field"
      aria-autocomplete="list"
      aria-controls="autocomplete-list"
    />

    {#if showDropdown}
      <ul class="dropdown" id="autocomplete-list" role="listbox">
        {#each suggestions as s, i}
          <li
            role="option"
            aria-selected={i === activeIndex}
            class="dropdown-item"
            class:active={i === activeIndex}
            onmousedown={() => selectSuggestion(s)}
          >
            <!-- eslint-disable-next-line svelte/no-at-html-tags -->
            {@html highlightMatch(s.name, currentToken().token)}
          </li>
        {/each}
      </ul>
    {/if}
  </div>

  {#if previewResult.state !== 'hidden'}
    <div class="preview-chip" class:ok={previewResult.state === 'ok'} class:warn={previewResult.state === 'warn'}>
      {previewResult.text}
    </div>
  {/if}

  <button onclick={handlePredict} disabled={!canPredict} class="predict-btn">
    {$loading ? 'Predicting…' : 'Predict'}
  </button>
</div>

<style>
  .input-section {
    padding: 16px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .field-wrap {
    position: relative;
  }

  .text-field {
    width: 100%;
    padding: 9px 12px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg);
    color: var(--text-primary);
    font-size: 13px;
    font-family: inherit;
    transition: border-color 0.15s, background 0.15s;
    outline: none;
  }

  .text-field:focus {
    border-color: var(--accent-blue);
    background: var(--surface);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent-blue) 15%, transparent);
  }

  .text-field::placeholder {
    color: var(--text-muted);
  }

  .dropdown {
    position: absolute;
    top: calc(100% + 4px);
    left: 0;
    right: 0;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 4px 0;
    list-style: none;
    margin: 0;
    z-index: 50;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  }

  .dropdown-item {
    padding: 7px 12px;
    font-size: 12px;
    color: var(--text-primary);
    cursor: pointer;
  }

  .dropdown-item:hover,
  .dropdown-item.active {
    background: color-mix(in srgb, var(--accent-blue) 8%, transparent);
    color: var(--accent-blue);
  }

  .dropdown-item :global(mark) {
    background: none;
    color: var(--accent-blue);
    font-weight: 700;
  }

  .preview-chip {
    font-size: 11px;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 20px;
    align-self: flex-start;
  }

  .preview-chip.ok {
    background: color-mix(in srgb, var(--accent-green) 12%, transparent);
    color: var(--accent-green);
  }

  .preview-chip.warn {
    background: color-mix(in srgb, #f59e0b 12%, transparent);
    color: #b45309;
  }

  .predict-btn {
    width: 100%;
    padding: 9px;
    background: var(--accent-blue);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.15s;
  }

  .predict-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .predict-btn:not(:disabled):hover {
    opacity: 0.88;
  }
</style>
