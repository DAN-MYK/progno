<script lang="ts">
  import MatchInput from './lib/components/MatchInput.svelte'
  import MatchCard from './lib/components/MatchCard.svelte'
  import Footer from './lib/components/Footer.svelte'
  import { predictions, error, bankroll, kelly_fraction } from './lib/stores'
</script>

<div class="min-h-screen flex flex-col bg-white">
  <header class="bg-white border-b border-gray-200 px-6 py-4">
    <div class="max-w-6xl mx-auto flex justify-between items-center">
      <h1 class="text-2xl font-bold">Progno</h1>
      <div class="flex gap-6 items-center text-sm">
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

  <Footer />
</div>
