<script lang="ts">
  import MatchInput from './lib/components/MatchInput.svelte'
  import MatchCard from './lib/components/MatchCard.svelte'
  import Footer from './lib/components/Footer.svelte'
  import { predictions, error } from './lib/stores'
</script>

<div class="min-h-screen flex flex-col bg-white">
  <header class="bg-white border-b border-gray-200 px-6 py-4">
    <h1 class="text-2xl font-bold">Progno</h1>
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
