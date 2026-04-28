<script lang="ts">
  import { Chart } from 'chart.js/auto'
  import { bets, type BetRecord } from '../stores'

  // Canvas refs must be $state so $effect tracks the bind:this assignment
  let canvas1 = $state<HTMLCanvasElement | undefined>(undefined)
  let canvas2 = $state<HTMLCanvasElement | undefined>(undefined)
  let canvas3 = $state<HTMLCanvasElement | undefined>(undefined)

  // ── Helpers ──────────────────────────────────────────────────────────────

  function modelEdge(b: BetRecord): number {
    return b.our_prob - 1 / b.odds
  }

  function clv(b: BetRecord): number | null {
    if (b.closing_odds == null || b.closing_odds <= 0) return null
    return 1 / b.closing_odds - 1 / b.odds
  }

  function expectedClv(b: BetRecord): number | null {
    const c = clv(b)
    return c !== null ? c * b.stake : null
  }

  type EdgeBucket = '<0%' | '0–3%' | '3–6%' | '6–10%' | '>10%'
  const BUCKETS: EdgeBucket[] = ['<0%', '0–3%', '3–6%', '6–10%', '>10%']

  function edgeBucket(edge: number): EdgeBucket {
    if (edge < 0) return '<0%'
    if (edge < 0.03) return '0–3%'
    if (edge < 0.06) return '3–6%'
    if (edge < 0.10) return '6–10%'
    return '>10%'
  }

  // ── Chart 1 data — ROI & Win rate by edge bucket ─────────────────────────

  function c1Data(allBets: BetRecord[]) {
    const settled = allBets.filter(b => b.result === 'win' || b.result === 'loss')
    const map = new Map<EdgeBucket, { pnl: number; stake: number; wins: number; n: number }>()
    for (const b of settled) {
      const bucket = edgeBucket(modelEdge(b))
      const cur = map.get(bucket) ?? { pnl: 0, stake: 0, wins: 0, n: 0 }
      cur.pnl += b.pnl ?? 0
      cur.stake += b.stake
      cur.wins += b.result === 'win' ? 1 : 0
      cur.n++
      map.set(bucket, cur)
    }
    const labels = BUCKETS.filter(b => map.has(b))
    const ns = labels.map(b => map.get(b)!.n)
    const roi = labels.map(b => {
      const d = map.get(b)!
      return d.stake > 0 ? Math.round(d.pnl / d.stake * 1000) / 10 : 0
    })
    const winRate = labels.map(b => {
      const d = map.get(b)!
      return d.n > 0 ? Math.round(d.wins / d.n * 1000) / 10 : 0
    })
    return { labels, ns, roi, winRate }
  }

  // ── Chart 2 data — Model edge vs CLV vs ROI by month ────────────────────

  function c2Data(allBets: BetRecord[]) {
    const monthMap = new Map<string, {
      edgeWSum: number; edgeSSum: number
      clvWSum: number; clvSSum: number
      pnl: number; stake: number
    }>()
    for (const b of allBets) {
      const month = b.date.slice(0, 7)
      const cur = monthMap.get(month) ?? { edgeWSum: 0, edgeSSum: 0, clvWSum: 0, clvSSum: 0, pnl: 0, stake: 0 }
      const edge = modelEdge(b)
      cur.edgeWSum += edge * b.stake
      cur.edgeSSum += b.stake
      const c = clv(b)
      if (c !== null) { cur.clvWSum += c * b.stake; cur.clvSSum += b.stake }
      if (b.result === 'win' || b.result === 'loss') { cur.pnl += b.pnl ?? 0; cur.stake += b.stake }
      monthMap.set(month, cur)
    }
    const months = [...monthMap.keys()].sort()
    const edgeLine = months.map(m => {
      const d = monthMap.get(m)!
      return d.edgeSSum > 0 ? Math.round(d.edgeWSum / d.edgeSSum * 1000) / 10 : null
    })
    const clvLine = months.map(m => {
      const d = monthMap.get(m)!
      return d.clvSSum > 0 ? Math.round(d.clvWSum / d.clvSSum * 1000) / 10 : null
    })
    const roiLine = months.map(m => {
      const d = monthMap.get(m)!
      return d.stake > 0 ? Math.round(d.pnl / d.stake * 1000) / 10 : null
    })
    return { months, edgeLine, clvLine, roiLine }
  }

  // ── Chart 3 data — Cumulative P&L + Expected CLV ─────────────────────────

  function c3Data(allBets: BetRecord[]) {
    const settled = allBets
      .filter(b => b.result === 'win' || b.result === 'loss')
      .map((b, i) => ({ ...b, idx: i }))
      .sort((a, b) => {
        const d = a.date.localeCompare(b.date)
        return d !== 0 ? d : a.idx - b.idx
      })
    let cumPnl = 0
    let cumClv = 0
    let hasClv = false
    const labels: string[] = []
    const pnlLine: number[] = []
    const clvLine: (number | null)[] = []
    for (const b of settled) {
      cumPnl += b.pnl ?? 0
      const ec = expectedClv(b)
      if (ec !== null) { cumClv += ec; hasClv = true }
      labels.push(b.date)
      pnlLine.push(Math.round(cumPnl * 100) / 100)
      clvLine.push(hasClv ? Math.round(cumClv * 100) / 100 : null)
    }
    return { labels, pnlLine, clvLine: hasClv ? clvLine : [] as (number | null)[] }
  }

  // ── Reactive flags ────────────────────────────────────────────────────────

  let hasSettled = $derived($bets.some(b => b.result === 'win' || b.result === 'loss'))
  let hasClosingOdds = $derived($bets.some(b => b.closing_odds != null))

  // ── Chart 1 effect ────────────────────────────────────────────────────────

  $effect(() => {
    if (!canvas1 || !hasSettled) return
    const { labels, ns, roi, winRate } = c1Data($bets)
    if (labels.length === 0) return
    const bgRoi = labels.map((l, i) =>
      l === '<0%' ? 'rgba(239,68,68,0.7)' :
      ns[i] < 10 ? 'rgba(156,163,175,0.7)' : 'rgba(59,130,246,0.7)'
    )
    const bgWin = labels.map((l, i) =>
      l === '<0%' ? 'rgba(239,68,68,0.4)' :
      ns[i] < 10 ? 'rgba(156,163,175,0.4)' : 'rgba(16,185,129,0.7)'
    )
    const chart = new Chart(canvas1, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'ROI%', data: roi, backgroundColor: bgRoi, borderWidth: 1 },
          { label: 'Win rate%', data: winRate, backgroundColor: bgWin, borderWidth: 1 },
        ]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top' },
          tooltip: {
            callbacks: {
              afterLabel(ctx) {
                const n = ns[ctx.dataIndex]
                return n < 10 ? `n=${n} ⚠ low sample` : `n=${n}`
              }
            }
          }
        },
        scales: { x: { title: { display: true, text: '%' } } }
      }
    })
    return () => chart.destroy()
  })

  // ── Chart 2 effect ────────────────────────────────────────────────────────

  $effect(() => {
    if (!canvas2 || $bets.length === 0) return
    const { months, edgeLine, clvLine, roiLine } = c2Data($bets)
    if (months.length === 0) return
    const datasets: any[] = [
      {
        label: 'Model edge%',
        data: edgeLine,
        borderColor: 'rgb(59,130,246)',
        backgroundColor: 'rgba(59,130,246,0.1)',
        tension: 0.3,
        spanGaps: false,
      }
    ]
    if (clvLine.some(v => v !== null)) {
      datasets.push({
        label: 'True CLV%',
        data: clvLine,
        borderColor: 'rgb(16,185,129)',
        backgroundColor: 'rgba(16,185,129,0.1)',
        tension: 0.3,
        spanGaps: false,
      })
    }
    if (roiLine.some(v => v !== null)) {
      datasets.push({
        label: 'Realized ROI%',
        data: roiLine,
        borderColor: 'rgb(245,158,11)',
        backgroundColor: 'rgba(245,158,11,0.1)',
        tension: 0.3,
        spanGaps: false,
        borderDash: [4, 4],
      })
    }
    const chart = new Chart(canvas2, {
      type: 'line',
      data: { labels: months, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'top' } },
        scales: { y: { title: { display: true, text: '%' } } }
      }
    })
    return () => chart.destroy()
  })

  // ── Chart 3 effect ────────────────────────────────────────────────────────

  $effect(() => {
    if (!canvas3 || !hasSettled) return
    const { labels, pnlLine, clvLine } = c3Data($bets)
    if (labels.length === 0) return
    const datasets: any[] = [
      {
        label: 'Cumulative P&L',
        data: pnlLine,
        borderColor: 'rgb(59,130,246)',
        backgroundColor: 'rgba(59,130,246,0.05)',
        fill: true,
        tension: 0.2,
        pointRadius: 2,
      }
    ]
    if (clvLine.length > 0) {
      datasets.push({
        label: 'Expected CLV',
        data: clvLine,
        borderColor: 'rgb(16,185,129)',
        borderDash: [5, 5],
        tension: 0.2,
        pointRadius: 2,
      })
    }
    const chart = new Chart(canvas3, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'top' } },
        scales: { y: { title: { display: true, text: '$' } } }
      }
    })
    return () => chart.destroy()
  })
</script>

<div class="p-6 space-y-10">
  {#if $bets.length === 0}
    <div class="text-center text-gray-400 py-16 text-sm">
      No bets logged yet. Use "Log bet" on a MatchCard.
    </div>
  {:else}

    <!-- Chart 1: ROI & Win rate by edge bucket -->
    <section>
      <h2 class="text-sm font-semibold text-gray-700 mb-1">ROI & Win Rate by Edge Bucket</h2>
      <p class="text-xs text-gray-400 mb-3">Settled bets only. Grey bars = n &lt; 10 (low sample).</p>
      {#if !hasSettled}
        <div class="text-gray-400 text-sm">No settled bets yet.</div>
      {:else}
        <div class="h-56"><canvas bind:this={canvas1}></canvas></div>
      {/if}
    </section>

    <!-- Chart 2: Model edge vs CLV vs ROI by month -->
    <section>
      <h2 class="text-sm font-semibold text-gray-700 mb-1">Model Edge vs CLV vs ROI by Month</h2>
      <p class="text-xs text-gray-400 mb-3">Stake-weighted averages. Dashed = realized ROI.</p>
      <div class="h-56"><canvas bind:this={canvas2}></canvas></div>
      {#if !hasClosingOdds}
        <p class="text-xs text-gray-400 mt-2">
          Log closing odds on bets to enable the True CLV line.
        </p>
      {/if}
    </section>

    <!-- Chart 3: Cumulative P&L -->
    <section>
      <h2 class="text-sm font-semibold text-gray-700 mb-1">Cumulative P&L</h2>
      <p class="text-xs text-gray-400 mb-3">Settled bets in chronological order.</p>
      {#if !hasSettled}
        <div class="text-gray-400 text-sm">No settled bets yet.</div>
      {:else}
        <div class="h-56"><canvas bind:this={canvas3}></canvas></div>
        {#if !hasClosingOdds}
          <p class="text-xs text-gray-400 mt-2">
            Log closing odds to enable the Expected CLV line.
          </p>
        {/if}
      {/if}
    </section>

  {/if}
</div>
