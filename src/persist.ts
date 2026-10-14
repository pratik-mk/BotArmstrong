import type { BrainWeights } from './rl/dqn'

const KEY = 'rocket-lander-brain-v1'

export interface SavedRun {
  version: 1
  brain: BrainWeights
  returns: number[]
  lands: number[]
  epsilons?: number[]
  losses?: number[]
  episode: number
  /** True once landings were steady enough to freeze the weights. */
  locked?: boolean
  /** True when the flight is using the fixed landing weights. */
  forceGreedy?: boolean
}

export function saveRun(run: SavedRun): void {
  localStorage.setItem(KEY, JSON.stringify(run))
}

export function loadRun(): SavedRun | null {
  const raw = localStorage.getItem(KEY)
  if (!raw) return null
  const data = JSON.parse(raw) as SavedRun
  if (data.version !== 1 || !data.brain) return null
  return data
}

export function hasSavedRun(): boolean {
  return localStorage.getItem(KEY) !== null
}
