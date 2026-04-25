import { writable, derived } from 'svelte/store'

export interface Prediction {
  player_a: string
  player_b: string
  surface: string
  prob_a_wins: number
  prob_b_wins: number
  elo_a_overall: number
  elo_b_overall: number
  ml_prob_a_wins?: number
  confidence_flag?: string
}

export interface ScheduledPrediction extends Prediction {
  player_a_full: string
  player_b_full: string
  tournament: string
  round: string
  event_time: string
  event_date: string
}

export interface KellyResult {
  implied_prob: number
  edge: number
  full_kelly: number
  fractional_kelly: number
  stake: number
}

function localPersist<T>(key: string, initial: T) {
  let stored = initial
  try {
    const raw = localStorage.getItem(key)
    if (raw !== null) stored = JSON.parse(raw) as T
  } catch {}
  const store = writable<T>(stored)
  store.subscribe(v => {
    try { localStorage.setItem(key, JSON.stringify(v)) } catch {}
  })
  return store
}

export const predictions = writable<Prediction[]>([])
export const loading = writable(false)
export const error = writable<string | null>(null)
export const dataAsOf = writable('unknown')

// Phase 2: Kelly settings
export const bankroll = localPersist('progno_bankroll', 1000)
export const kelly_fraction = localPersist('progno_kelly_fraction', 0.25)

// Phase 4: Tour selector
export const selectedTour = localPersist<'atp' | 'wta'>('progno_selected_tour', 'atp')

// Phase 5: Schedule
export const apiKey = localPersist('progno_api_key', '')
export const scheduleDate = localPersist('progno_schedule_date', new Date().toISOString().slice(0, 10))
export const scheduledPredictions = writable<ScheduledPrediction[]>([])
export const scheduleLoading = writable(false)
export const scheduleError = writable<string | null>(null)

// LLM parser fallback — separate key per provider
export const llmProvider = localPersist<'gemini' | 'grok'>('progno_llm_provider', 'grok')
export const geminiApiKey = localPersist('progno_gemini_key', '')
export const grokApiKey   = localPersist('progno_grok_key', '')
// Active key — automatically switches with provider
export const llmApiKey = derived(
  [llmProvider, geminiApiKey, grokApiKey],
  ([$p, $g, $gr]) => ($p === 'grok' ? $gr : $g)
)

// RapidAPI Tennis
export const rapidApiKey = localPersist('progno_rapidapi_key', '')
export const scheduleSource = localPersist<'api-tennis' | 'rapidapi'>('progno_schedule_source', 'api-tennis')

// Bet history
export interface BetRecord {
  id: string
  date: string
  player_a: string
  player_b: string
  surface: string
  tournament?: string
  bet_on: 'a' | 'b'
  our_prob: number
  odds: number
  stake: number
  result?: 'win' | 'loss' | 'void'
  pnl?: number
}

export const bets = writable<BetRecord[]>([])
