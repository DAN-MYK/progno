<script lang="ts">
  import { dataAsOf, selectedTour } from '../stores'

  let isStale = $derived(() => {
    if ($dataAsOf === 'unknown') return false
    const d = new Date($dataAsOf)
    if (isNaN(d.getTime())) return false
    return (Date.now() - d.getTime()) > 14 * 24 * 60 * 60 * 1000
  })
</script>

{#if isStale()}
  <div class="bg-amber-50 border-t border-amber-200 px-6 py-2 text-xs text-amber-800 text-center">
    Model data is stale (as of {$dataAsOf}). Consider retraining.
  </div>
{/if}

<footer class="bg-gray-50 border-t border-gray-200 p-4 text-xs text-gray-600">
  <div class="text-center">
    <p>Model: Elo baseline · Data as of {$dataAsOf} · {$selectedTour.toUpperCase()}</p>
    <p class="mt-1 text-gray-500">Not financial advice.</p>
  </div>
</footer>
