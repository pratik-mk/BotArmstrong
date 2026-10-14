import { clamp } from '../rng'

/** Half-distance between footpads, metres. Origin of the body is the midpoint of the feet. */
export const LEG_HALF = 2.35
/** Nose length from the foot line, metres. */
export const NOSE_LEN = 6.6

export interface RocketState {
  x: number
  y: number
  vx: number
  vy: number
  /** Radians. 0 is upright. Positive tips the nose toward +x. */
  angle: number
  omega: number
  fuel: number
  leftContact: number
  rightContact: number
}

export interface Point {
  x: number
  y: number
}

export function createState(): RocketState {
  return {
    x: 0,
    y: 0,
    vx: 0,
    vy: 0,
    angle: 0,
    omega: 0,
    fuel: 1,
    leftContact: 0,
    rightContact: 0,
  }
}

export function normalizeAngle(angle: number): number {
  const tau = Math.PI * 2
  let x = angle % tau
  if (x > Math.PI) x -= tau
  if (x < -Math.PI) x += tau
  return x
}

/** side −1 is the left foot, +1 the right foot. */
export function foot(state: RocketState, side: -1 | 1): Point {
  const c = Math.cos(state.angle)
  const s = Math.sin(state.angle)
  return {
    x: state.x + side * LEG_HALF * c,
    y: state.y - side * LEG_HALF * s,
  }
}

export function noseTip(state: RocketState): Point {
  return {
    x: state.x + NOSE_LEN * Math.sin(state.angle),
    y: state.y + NOSE_LEN * Math.cos(state.angle),
  }
}

/**
 * Observation, all dimensionless:
 * pad-relative x, altitude, vx, vy, angle, spin, fuel, wind, left foot, right foot.
 */
export function writeObservation(state: RocketState, wind: number, out: Float64Array): void {
  out[0] = clamp(state.x / 20, -3, 3)
  out[1] = clamp(state.y / 32, -3, 3)
  out[2] = clamp(state.vx / 8, -3, 3)
  out[3] = clamp(state.vy / 8, -3, 3)
  out[4] = clamp(normalizeAngle(state.angle) / Math.PI, -1, 1)
  out[5] = clamp(state.omega / 2, -3, 3)
  out[6] = clamp(state.fuel, 0, 1)
  out[7] = clamp(wind / 2, -1.5, 1.5)
  out[8] = state.leftContact
  out[9] = state.rightContact
}
