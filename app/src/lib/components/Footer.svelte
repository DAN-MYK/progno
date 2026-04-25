<script lang="ts">
  import { dataAsOf, selectedTour } from '../stores'
</script>

<footer class="bg-gray-50 border-t border-gray-200 p-4 text-xs text-gray-600">
  <div class="text-center">
    <p>Model: Elo + ML · Data as of {$dataAsOf} · {$selectedTour.toUpperCase()}</p>
    <p class="mt-1 text-gray-500">Not financial advice.</p>
  </div>
</footer>
