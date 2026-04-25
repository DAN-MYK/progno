import type { Prediction } from './stores'

export function predictionsToCSV(preds: Prediction[]): string {
  const header = 'player_a,player_b,surface,prob_a,prob_b,ml_prob_a,elo_a,elo_b'
  const rows = preds.map(p => [
    `"${p.player_a}"`,
    `"${p.player_b}"`,
    `"${p.surface}"`,
    p.prob_a_wins.toFixed(4),
    p.prob_b_wins.toFixed(4),
    p.ml_prob_a_wins != null ? p.ml_prob_a_wins.toFixed(4) : '',
    Math.round(p.elo_a_overall),
    Math.round(p.elo_b_overall),
  ].join(','))
  return [header, ...rows].join('\n')
}

export function downloadCSV(content: string, filename: string) {
  const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
