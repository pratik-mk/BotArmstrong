import {
  ACTION_REPEAT,
  ARENA_HALF,
  DT,
  LIMITS,
  MAX_ALT,
  MAX_STEPS,
  SENSOR_RANGE,
  START_Y,
  TRAIL,
  mainBurnPerSec,
  type WorldParams,
} from '../config'
import { potential, stepReward, type Outcome } from '../rl/reward'
import {
  createState,
  foot,
  noseTip,
  normalizeAngle,
  writeObservation,
  type RocketState,
} from './physics'

export interface StepResult {
  reward: number
  done: boolean
  outcome: Outcome
}

/**
 * Fixed-step lander. Thrust is along the nose. Positive angle tips the nose to +x.
 * Action 0 idle, 1 main engine, 2 rotate left, 3 rotate right.
 */
export class LanderEnv {
  readonly state: RocketState = createState()
  readonly obs = new Float64Array(10)
  readonly prevObs = new Float64Array(10)
  readonly trailX = new Float32Array(TRAIL)
  readonly trailY = new Float32Array(TRAIL)
  trailCount = 0
  trailHead = 0

  done = false
  outcome: Outcome = 'fly'
  steps = 0
  episodeReturn = 0
  lastReward = 0
  lastAction = 0
  lastFired = false

  private prevPhi = 0

  constructor(
    readonly params: WorldParams,
    private readonly rng: () => number,
  ) {
    this.reset()
  }

  reset(): void {
    const s = this.params.scatter
    const r = this.rng
    const st = this.state
    st.x = (r() * 2 - 1) * s * 14
    st.y = START_Y + (r() * 2 - 1) * s * 2
    st.vx = (r() * 2 - 1) * s * 1.2
    st.vy = (r() * 2 - 1) * s * 0.8
    st.angle = (r() * 2 - 1) * s * 0.45
    st.omega = (r() * 2 - 1) * s * 0.25
    st.fuel = 1
    this.sense()
    this.prevPhi = potential(st)
    this.done = false
    this.outcome = 'fly'
    this.steps = 0
    this.episodeReturn = 0
    this.lastReward = 0
    this.lastAction = 0
    this.lastFired = false
    this.trailCount = 0
    this.trailHead = 0
    writeObservation(st, this.params.wind, this.obs)
    this.prevObs.set(this.obs)
    this.pushTrail()
  }

  /**
   * Hold one action across several physics steps and sum the reward.
   * The caller stores one transition for the whole hold.
   */
  hold(action: number): StepResult {
    let reward = 0
    let done = false
    let outcome: Outcome = this.outcome
    for (let i = 0; i < ACTION_REPEAT && !this.done; i++) {
      const result = this.step(action)
      reward += result.reward
      done = result.done
      outcome = result.outcome
    }
    return { reward, done, outcome }
  }

  step(action: number): StepResult {
    if (this.done) {
      return { reward: 0, done: true, outcome: this.outcome }
    }
    this.prevObs.set(this.obs)
    const st = this.state
    const fired = this.integrate(action)
    this.steps += 1
    this.sense()

    const left = foot(st, -1)
    const right = foot(st, 1)
    const nose = noseTip(st)
    const feetHit = left.y <= 0 || right.y <= 0
    const noseFirst = nose.y <= 0 && nose.y <= left.y && nose.y <= right.y && !feetHit

    let outcome: Outcome = 'fly'
    let onPad = false
    if (noseFirst) {
      outcome = 'crash'
    } else if (feetHit) {
      onPad = Math.abs(left.x) <= this.params.padHalf && Math.abs(right.x) <= this.params.padHalf
      const soft =
        Math.abs(st.vx) <= LIMITS.vx &&
        Math.abs(st.vy) <= LIMITS.vy &&
        Math.abs(normalizeAngle(st.angle)) <= LIMITS.angle &&
        Math.abs(st.omega) <= LIMITS.omega
      outcome = onPad && soft ? 'land' : 'crash'
    } else if (Math.abs(st.x) > ARENA_HALF || st.y > MAX_ALT) {
      outcome = 'crash'
    } else if (this.steps >= MAX_STEPS) {
      outcome = 'timeout'
    }

    const phi = potential(st)
    const reward = stepReward(
      this.prevPhi,
      phi,
      action,
      outcome,
      fired,
    )
    this.prevPhi = phi
    this.episodeReturn += reward
    this.lastReward = reward
    this.lastAction = action
    this.lastFired = fired
    this.outcome = outcome
    this.done = outcome !== 'fly'
    writeObservation(st, this.params.wind, this.obs)
    this.pushTrail()
    return { reward, done: this.done, outcome }
  }

  private sense(): void {
    const left = foot(this.state, -1)
    const right = foot(this.state, 1)
    this.state.leftContact = left.y <= SENSOR_RANGE ? 1 : 0
    this.state.rightContact = right.y <= SENSOR_RANGE ? 1 : 0
  }

  /** Returns whether an engine actually fired. */
  private integrate(action: number): boolean {
    const st = this.state
    const dt = DT
    let thrust = 0
    let torque = 0
    let burned = 0
    if (st.fuel > 0) {
      const burn = mainBurnPerSec(this.params)
      if (action === 1) {
        thrust = this.params.mainAcc
        burned = burn * dt
      } else if (action === 2) {
        torque = -this.params.rcsAcc
        burned = burn * 0.22 * dt
      } else if (action === 3) {
        torque = this.params.rcsAcc
        burned = burn * 0.22 * dt
      }
      if (burned > 0) {
        if (burned >= st.fuel) {
          const scale = st.fuel / burned
          thrust *= scale
          torque *= scale
          st.fuel = 0
        } else {
          st.fuel -= burned
        }
      }
    }
    const s = Math.sin(st.angle)
    const c = Math.cos(st.angle)
    st.vx += (thrust * s + this.params.wind) * dt
    st.vy += (thrust * c - this.params.gravity) * dt
    st.omega += torque * dt
    st.x += st.vx * dt
    st.y += st.vy * dt
    st.angle = normalizeAngle(st.angle + st.omega * dt)
    return thrust !== 0 || torque !== 0
  }

  private pushTrail(): void {
    this.trailX[this.trailHead] = this.state.x
    this.trailY[this.trailHead] = this.state.y
    this.trailHead = (this.trailHead + 1) % TRAIL
    if (this.trailCount < TRAIL) this.trailCount += 1
  }
}
