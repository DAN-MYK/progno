<script lang="ts">
  import { invoke } from '@tauri-apps/api/core'
  import { bets, type BetRecord } from '../stores'
  import { onMount } from 'svelte'

  onMount(async () => {
    try {
      const loaded = await invoke<BetRecord[]>('get_bets')
      bets.set(loaded)
    } catch (e) {
      console.error('Failed to load bets:', e)
    }
  })

  let filterResult = $state<'all' | 'pending' | 'win' | 'loss'>('all')

  let filtered = $derived(
    filterResult === 'all'
      ? $bets
      : filterResult === 'pending'
        ? $bets.filter(b => !b.result)
        : $bets.filter(b => b.result === filterResult)
  )

  function betClv(b: BetRecord): number {
    return b.our_prob - 1 / b.odds
  }

  let stats = $derived.by(() => {
    const settled = $bets.filter(b => b.result && b.result !== 'void')
    const wins = settled.filter(b => b.result === 'win').length
    const totalPnl = $bets.reduce((s, b) => s + (b.pnl ?? 0), 0)
    const totalStake = settled.reduce((s, b) => s + b.stake, 0)
    const roi = totalStake > 0 ? (totalPnl / totalStake) * 100 : 0
    const clvSum = $bets.reduce((s, b) => s + betClv(b), 0)
    const meanClv = $bets.length > 0 ? (clvSum / $bets.length) * 100 : 0
    return { total: $bets.length, settled: settled.length, wins, totalPnl, roi, meanClv }
  })

  let surfaceStats = $derived.by(() => {
    const map = new Map<string, { pnl: number; stake: number; n: number }>()
    for (const b of $bets) {
      if (!b.result || b.result === 'void') continue
      const s = b.surface || 'Unknown'
      const cur = map.get(s) ?? { pnl: 0, stake: 0, n: 0 }
      cur.pnl += b.pnl ?? 0
      cur.stake += b.stake
      cur.n++
      map.set(s, cur)
    }
    return [...map.entries()]
      .map(([surface, d]) => ({ surface, roi: d.stake > 0 ? (d.pnl / d.stake) * 100 : 0, pnl: d.pnl, n: d.n }))
      .sort((a, b) => b.n - a.n)
  })

  async function setResult(id: string, result: 'win' | 'loss' | 'void') {
    try {
      await invoke('update_bet_result', { id, result })
      bets.update(prev =>
        prev.map(b => {
          if (b.id !== id) return b
          const pnl = result === 'win' ? b.stake * (b.odds - 1) : result === 'loss' ? -b.stake : 0
          return { ...b, result, pnl }
        })
      )
    } catch (e) {
      console.error('Failed to update bet:', e)
    }
  }

  async function removeBet(id: string) {
    try {
      await invoke('delete_bet', { id })
      bets.update(prev => prev.filter(b => b.id !== id))
    } catch (e) {
      console.error('Failed to delete bet:', e)
    }
  }

  function fmt(n: number) {
    return (n >= 0 ? '+' : '') + n.toFixed(2)
  }
</script>

<div class="p-4 border-b border-gray-200 bg-white flex flex-wrap gap-4 items-center">
  <h2 class="text-base font-semibold text-gray-800">Bet History</h2>

  <!-- Summary stats -->
  {#if $bets.length > 0}
    {@const s = stats}
    <div class="flex gap-4 text-xs text-gray-600">
      <span><strong>{s.total}</strong> bets</span>
      <span><strong>{s.wins}/{s.settled}</strong> won</span>
      <span
        class="font-semibold"
        class:text-green-700={s.totalPnl >= 0}
        class:text-red-700={s.totalPnl < 0}
      >
        P&L: {fmt(s.totalPnl)}
      </span>
      <span
        class="font-semibold"
        class:text-green-700={s.roi >= 0}
        class:text-red-700={s.roi < 0}
      >
        ROI: {s.roi.toFixed(1)}%
      </span>
      <span
        class="font-semibold"
        title="Mean edge (our prob − implied prob) at bet time"
        class:text-green-700={s.meanClv >= 0}
        class:text-red-700={s.meanClv < 0}
      >
        Edge: {s.meanClv >= 0 ? '+' : ''}{s.meanClv.toFixed(1)}%
      </span>
    </div>
  {/if}

  <!-- Filter -->
  <div class="ml-auto flex gap-1">
    {#each ['all', 'pending', 'win', 'loss'] as f}
      <button
        onclick={() => (filterResult = f)}
        class="px-2 py-1 text-xs rounded border {filterResult === f
          ? 'bg-blue-600 text-white border-blue-600'
          : 'border-gray-300 text-gray-500 hover:border-gray-400'}"
      >
        {f.charAt(0).toUpperCase() + f.slice(1)}
      </button>
    {/each}
  </div>
</div>

{#if filtered.length === 0}
  <div class="p-8 text-center text-gray-400 text-sm">
    {$bets.length === 0
      ? 'No bets logged yet. Use "Log bet" on a MatchCard.'
      : 'No bets match this filter.'}
  </div>
{:else}
  <div class="overflow-x-auto">
    <table class="w-full text-xs">
      <thead class="bg-gray-50 border-b border-gray-200">
        <tr>
          <th class="px-4 py-2 text-left font-medium text-gray-600">Date</th>
          <th class="px-4 py-2 text-left font-medium text-gray-600">Match</th>
          <th class="px-4 py-2 text-left font-medium text-gray-600">Bet on</th>
          <th class="px-4 py-2 text-right font-medium text-gray-600">Our %</th>
          <th class="px-4 py-2 text-right font-medium text-gray-600">Edge</th>
          <th class="px-4 py-2 text-right font-medium text-gray-600">Odds</th>
          <th class="px-4 py-2 text-right font-medium text-gray-600">Stake</th>
          <th class="px-4 py-2 text-center font-medium text-gray-600">Result</th>
          <th class="px-4 py-2 text-right font-medium text-gray-600">P&L</th>
          <th class="px-4 py-2"></th>
        </tr>
      </thead>
      <tbody>
        {#each filtered as bet (bet.id)}
          <tr class="border-b border-gray-100 hover:bg-gray-50">
            <td class="px-4 py-2 text-gray-500">{bet.date}</td>
            <td class="px-4 py-2">
              <div class="font-medium">{bet.player_a} vs {bet.player_b}</div>
              <div class="text-gray-400">{bet.surface}{bet.tournament ? ` · ${bet.tournament}` : ''}</div>
            </td>
            <td class="px-4 py-2 font-medium">
              {bet.bet_on === 'a' ? bet.player_a : bet.player_b}
            </td>
            <td class="px-4 py-2 text-right">{Math.round(bet.our_prob * 1000) / 10}%</td>
            <td
              class="px-4 py-2 text-right font-medium"
              class:text-green-700={betClv(bet) >= 0}
              class:text-red-700={betClv(bet) < 0}
            >
              {betClv(bet) >= 0 ? '+' : ''}{Math.round(betClv(bet) * 1000) / 10}%
            </td>
            <td class="px-4 py-2 text-right">{bet.odds.toFixed(2)}</td>
            <td class="px-4 py-2 text-right">${bet.stake.toFixed(2)}</td>
            <td class="px-4 py-2 text-center">
              {#if bet.result}
                <span
                  class="px-2 py-0.5 rounded text-xs font-medium"
                  class:bg-green-100={bet.result === 'win'}
                  class:text-green-800={bet.result === 'win'}
                  class:bg-red-100={bet.result === 'loss'}
                  class:text-red-800={bet.result === 'loss'}
                  class:bg-gray-100={bet.result === 'void'}
                  class:text-gray-600={bet.result === 'void'}
                >
                  {bet.result}
                </span>
              {:else}
                <div class="flex justify-center gap-1">
                  <button
                    onclick={() => setResult(bet.id, 'win')}
                    class="px-1.5 py-0.5 bg-green-600 text-white rounded hover:bg-green-700"
                  >W</button>
                  <button
                    onclick={() => setResult(bet.id, 'loss')}
                    class="px-1.5 py-0.5 bg-red-600 text-white rounded hover:bg-red-700"
                  >L</button>
                  <button
                    onclick={() => setResult(bet.id, 'void')}
                    class="px-1.5 py-0.5 bg-gray-400 text-white rounded hover:bg-gray-500"
                  >V</button>
                </div>
              {/if}
            </td>
            <td
              class="px-4 py-2 text-right font-semibold"
              class:text-green-700={bet.pnl != null && bet.pnl > 0}
              class:text-red-700={bet.pnl != null && bet.pnl < 0}
              class:text-gray-400={bet.pnl == null}
            >
              {bet.pnl != null ? fmt(bet.pnl) : '—'}
            </td>
            <td class="px-4 py-2">
              <button
                onclick={() => removeBet(bet.id)}
                class="text-gray-300 hover:text-red-500"
                title="Delete"
              >✕</button>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
{/if}

{#if surfaceStats.length > 0}
  <div class="p-4 border-t border-gray-100">
    <h3 class="text-xs font-semibold text-gray-600 mb-2">ROI by Surface</h3>
    <div class="flex gap-4 flex-wrap">
      {#each surfaceStats as s}
        <div class="text-xs bg-gray-50 rounded px-3 py-2 border border-gray-200">
          <div class="font-medium text-gray-700">{s.surface}</div>
          <div class="text-gray-500">{s.n} bets</div>
          <div
            class="font-semibold"
            class:text-green-700={s.roi >= 0}
            class:text-red-700={s.roi < 0}
          >
            {s.roi >= 0 ? '+' : ''}{s.roi.toFixed(1)}%
          </div>
        </div>
      {/each}
    </div>
  </div>
{/if}
