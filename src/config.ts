export const DT = 1 / 30
export const START_Y = 24
export const MAX_STEPS = 600
/** Physics steps one chosen action is held. The learner discounts once per hold. */
export const ACTION_REPEAT = 1
export const ARENA_HALF = 80
export const MAX_ALT = 150
/** Gear proximity sensor, metres above the ground. */
export const SENSOR_RANGE = 1.2
export const TRAIL = 100

export const LIMITS = {
  vx: 1.5,
  vy: 2,
  angle: 0.35,
  omega: 0.5,
} as const

export interface WorldParams {
  gravity: number
  mainAcc: number
  rcsAcc: number
  /** Seconds of continuous main-engine burn in a full tank. */
  burnSeconds: number
  wind: number
  padHalf: number
  /** 0 = identical start, 1 = wide scatter of position, tilt, and velocity. */
  scatter: number
}

export function defaultParams(): WorldParams {
  return {
    gravity: 1.62,
    mainAcc: 4.2,
    rcsAcc: 1.1,
    burnSeconds: 10,
    wind: 0,
    padHalf: 14,
    scatter: 0.45,
  }
}

export function mainBurnPerSec(params: WorldParams): number {
  return 1 / Math.max(1, params.burnSeconds)
}

export const OBS_DIM = 10
export const ACTIONS = 4
export const HIDDEN = 32

export const OBS_LABELS = ['x', 'alt', 'vx', 'vy', 'tilt', 'spin', 'fuel', 'wind', 'L', 'R'] as const
export const ACTION_LABELS = ['Idle', 'Main', 'Left', 'Right'] as const

export const LEARN = {
  gamma: 0.99,
  lr: 5e-4,
  /** After this many decisions the step size drops so a landing policy is not trained off the pad. */
  lrCutSteps: 70_000,
  lrFinal: 2.5e-5,
  tau: 0.005,
  batch: 64,
  buffer: 20_000,
  warmup: 800,
  epsStart: 1,
  epsEnd: 0.05,
  epsDecaySteps: 60_000,
  /** Rewards are backed up this many decisions at once so a landing or a timeout reaches the start of the flight. */
  nStep: 1,
  huber: 1,
  gradClip: 5,
}

/**
 * Potential weights. Φ is higher when the rocket is closer, slower near the
 * ground, upright, and not spinning. Shaping is γΦ(s') − Φ(s).
 */
export const REWARD = {
  dist: 14,
  speed: 10,
  tilt: 5,
  spin: 2.5,
  distX: 20,
  distY: 32,
  speedScale: 6,
  land: 60,
  crash: -40,
  timeout: -70,
  fuelMain: 0.01,
  fuelRcs: 0.004,
}
