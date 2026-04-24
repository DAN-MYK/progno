import { writable } from 'svelte/store'

export interface Prediction {
  player_a: string
  player_b: string
  surface: string
  prob_a_wins: number
  prob_b_wins: number
  elo_a_overall: number
  elo_b_overall: number
  ml_prob_a_wins?: number | null
  confidence_flag?: string | null
}

export interface KellyResult {
  implied_prob: number
  edge: number
  full_kelly: number
  fractional_kelly: number
  stake: number
}

export const predictions = writable<Prediction[]>([])
export const loading = writable(false)
export const error = writable<string | null>(null)
export const dataAsOf = writable('unknown')

// Defaults — not exposed in UI per UX redesign spec
export const bankroll = writable(1000)
export const kelly_fraction = writable(0.25)

// Tour selector
export const selectedTour = writable<'atp' | 'wta'>('atp')

export function appendPredictions(newPreds: Prediction[]) {
  predictions.update(current => [...current, ...newPreds])
}
