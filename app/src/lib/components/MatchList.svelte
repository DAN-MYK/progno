<script lang="ts">
  import { predictions, error } from '../stores'
  import MatchCard from './MatchCard.svelte'
</script>

{#if $error}
  <div class="error-banner">{$error}</div>
{/if}

{#if $predictions.length === 0}
  <div class="empty-state">
    Введи матч вище щоб побачити прогноз
  </div>
{:else}
  <div class="match-list">
    {#each $predictions as pred, i (pred.player_a + pred.player_b + i)}
      <MatchCard prediction={pred} />
    {/each}
  </div>
{/if}

<style>
  .error-banner {
    margin: 12px 14px 0;
    padding: 8px 12px;
    background: color-mix(in srgb, var(--accent-red) 10%, transparent);
    color: var(--accent-red);
    border-radius: 6px;
    font-size: 12px;
  }

  .empty-state {
    padding: 48px 16px;
    text-align: center;
    font-size: 12px;
    color: var(--text-muted);
  }

  .match-list {
    display: flex;
    flex-direction: column;
  }
</style>
