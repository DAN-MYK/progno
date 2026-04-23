<script lang="ts">
  import type { Prediction } from '../stores'

  export let prediction: Prediction

  const probA = Math.round(prediction.prob_a_wins * 1000) / 10
  const probB = Math.round(prediction.prob_b_wins * 1000) / 10
</script>

<div class="p-6 border-b border-gray-100 hover:bg-gray-50">
  <div class="mb-2 text-sm font-semibold text-gray-700">
    {prediction.player_a} vs {prediction.player_b}
  </div>
  <div class="text-xs text-gray-500 mb-4">{prediction.surface}</div>

  <div class="space-y-3">
    <div>
      <div class="flex justify-between items-center mb-1">
        <span class="text-sm font-medium">{prediction.player_a}</span>
        <span class="text-sm font-bold text-blue-600">{probA}%</span>
      </div>
      <div class="h-6 bg-gray-200 rounded-sm overflow-hidden">
        <div
          class="h-full bg-blue-500"
          style="width: {probA}%"
        />
      </div>
    </div>

    <div>
      <div class="flex justify-between items-center mb-1">
        <span class="text-sm font-medium">{prediction.player_b}</span>
        <span class="text-sm font-bold text-red-600">{probB}%</span>
      </div>
      <div class="h-6 bg-gray-200 rounded-sm overflow-hidden">
        <div
          class="h-full bg-red-500"
          style="width: {probB}%"
        />
      </div>
    </div>
  </div>

  <div class="text-xs text-gray-500 mt-4">
    Elo: {prediction.player_a} {Math.round(prediction.elo_a_overall)} vs {Math.round(prediction.elo_b_overall)}
  </div>
</div>
