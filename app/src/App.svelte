<script lang="ts">
  import { onMount } from 'svelte'
  import MatchInput from './lib/components/MatchInput.svelte'
  import MatchList from './lib/components/MatchList.svelte'
  import { selectedTour, predictions } from './lib/stores'

  let dark = $state(false)

  onMount(() => {
    dark = localStorage.getItem('theme') === 'dark'
    applyTheme()
  })

  function applyTheme() {
    document.body.classList.toggle('dark', dark)
  }

  function toggleTheme() {
    dark = !dark
    localStorage.setItem('theme', dark ? 'dark' : 'light')
    applyTheme()
  }
</script>

<div class="shell">
  <header class="header">
    <div class="header-inner">
      <div class="header-brand">
        <span class="brand-title">Progno</span>
        <span class="brand-sub">{$selectedTour.toUpperCase()} · Elo model</span>
      </div>
      <div class="header-controls">
        <select
          class="tour-select"
          bind:value={$selectedTour}
          onchange={() => predictions.set([])}
          aria-label="Tour"
        >
          <option value="atp">ATP</option>
          <option value="wta">WTA</option>
        </select>
        <button class="theme-toggle" onclick={toggleTheme} aria-label="Toggle theme">
          {dark ? 'Light' : 'Dark'}
        </button>
      </div>
    </div>
  </header>

  <main class="main">
    <MatchInput />
    <MatchList />
  </main>
</div>

<style>
  .shell {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .header {
    width: 100%;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 10px 16px;
  }

  .header-inner {
    max-width: 520px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .header-brand {
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .brand-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
  }

  .brand-sub {
    font-size: 11px;
    color: var(--text-muted);
  }

  .header-controls {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .tour-select {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-secondary);
    background: none;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 8px;
    cursor: pointer;
    font-family: inherit;
    transition: border-color 0.1s, color 0.1s;
  }

  .tour-select:focus {
    outline: none;
    border-color: var(--accent-blue);
    color: var(--text-primary);
  }

  .theme-toggle {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-secondary);
    background: none;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 10px;
    cursor: pointer;
    transition: border-color 0.1s, color 0.1s;
  }

  .theme-toggle:hover {
    color: var(--text-primary);
    border-color: var(--text-secondary);
  }

  .main {
    width: 100%;
    max-width: 520px;
    padding: 0;
    flex: 1;
  }
</style>
