import { LEARN, REWARD } from '../config'
import { normalizeAngle, type RocketState } from '../sim/physics'

/**
 * State potential. Higher is better.
 * The step bonus is γΦ(s') − Φ(s). It densifies the signal and does not change
 * the best policy for the discounted task.
 */
export function potential(state: RocketState): number {
  const dist = Math.hypot(state.x / REWARD.distX, Math.max(state.y, 0) / REWARD.distY)
  const speed = Math.hypot(state.vx / REWARD.speedScale, state.vy / REWARD.speedScale)
  const closeness = Math.exp(-Math.max(state.y, 0) / 16)
  const tilt = Math.abs(normalizeAngle(state.angle)) / Math.PI
  const spin = Math.abs(state.omega) / 2
  return -(
    REWARD.dist * dist +
    REWARD.speed * speed * (0.25 + closeness) +
    REWARD.tilt * tilt +
    REWARD.spin * spin
  )
}

export type Outcome = 'fly' | 'land' | 'crash' | 'timeout'

export function fuelCost(action: number, fired: boolean): number {
  if (!fired) return 0
  if (action === 1) return REWARD.fuelMain
  if (action === 2 || action === 3) return REWARD.fuelRcs
  return 0
}

export function terminalBonus(outcome: Outcome): number {
  if (outcome === 'land') return REWARD.land
  if (outcome === 'crash') return REWARD.crash
  if (outcome === 'timeout') return REWARD.timeout
  return 0
}

/**
 * Potential difference, fuel cost, and the terminal outcome.
 * Shaping uses the same discount as the action-value backup.
 */
export function stepReward(
  prevPhi: number,
  nextPhi: number,
  action: number,
  outcome: Outcome,
  fired: boolean,
  crashScore = REWARD.crash,
): number {
  const shaping = LEARN.gamma * nextPhi - prevPhi
  const terminal = outcome === 'crash' ? crashScore : terminalBonus(outcome)
  return shaping - fuelCost(action, fired) + terminal
}
