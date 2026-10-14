import { ACTIONS, LEARN, OBS_DIM } from '../config'

interface PendingStep {
  state: Float64Array
  action: number
  reward: number
  next: Float64Array
  done: boolean
}
import { QNetwork } from './network'
import { Replay } from './replay'

export interface BrainWeights {
  version: 1
  steps: number
  adamT: number
  weights: number[]
  target: number[]
  m: number[]
  v: number[]
}

function pack(net: QNetwork): number[] {
  const flat = new Float64Array(net.paramCount())
  net.writeFlat(flat)
  return Array.from(flat)
}

/** Double DQN with a soft target network and Adam. */
export class Agent {
  readonly net: QNetwork
  readonly target: QNetwork
  steps = 0
  exploring = false
  lastAction = 0
  lastLoss: number | null = null
  /** Set once recent flights are landing, so later updates cannot erase the policy. */
  locked = false
  /** Greedy flight. Learning still runs until the landing lock. */
  forceGreedy = false

  private readonly recentLands: number[] = []

  private readonly replay: Replay
  private readonly m: Float64Array
  private readonly v: Float64Array
  private adamT = 0
  private readonly sBuf: Float64Array
  private readonly spBuf: Float64Array
  private readonly aBuf: Uint8Array
  private readonly rBuf: Float64Array
  private readonly dBuf: Uint8Array
  private readonly gBuf: Float64Array
  private readonly dQ: Float64Array
  private readonly best: Int8Array
  private readonly pending: PendingStep[] = []

  constructor(private readonly rng: () => number) {
    const batch = LEARN.batch
    this.net = new QNetwork(batch, rng)
    this.target = new QNetwork(batch, rng)
    this.target.copyFrom(this.net)
    const count = this.net.paramCount()
    this.m = new Float64Array(count)
    this.v = new Float64Array(count)
    this.replay = new Replay(LEARN.buffer, rng)
    this.sBuf = new Float64Array(batch * OBS_DIM)
    this.spBuf = new Float64Array(batch * OBS_DIM)
    this.aBuf = new Uint8Array(batch)
    this.rBuf = new Float64Array(batch)
    this.dBuf = new Uint8Array(batch)
    this.gBuf = new Float64Array(batch)
    this.dQ = new Float64Array(batch * ACTIONS)
    this.best = new Int8Array(batch)
  }

  epsilon(): number {
    if (this.locked || this.forceGreedy) return 0
    const t = Math.min(1, this.steps / LEARN.epsDecaySteps)
    return LEARN.epsStart + (LEARN.epsEnd - LEARN.epsStart) * t
  }

  /** Call once per finished flight. Locks the weights after landings become regular. */
  noteEpisode(landed: boolean, episodeNumber: number): void {
    this.recentLands.push(landed ? 1 : 0)
    if (this.recentLands.length > 20) this.recentLands.shift()
    if (this.locked || this.recentLands.length < 20) return
    if (!this.forceGreedy && episodeNumber < 200) return
    let sum = 0
    for (const flag of this.recentLands) sum += flag
    if (sum >= 10) this.locked = true
  }

  act(obs: ArrayLike<number>): number {
    this.net.forward(obs)
    let greedy = 0
    let bestQ = this.net.q[0]
    for (let i = 1; i < ACTIONS; i++) {
      if (this.net.q[i] > bestQ) {
        bestQ = this.net.q[i]
        greedy = i
      }
    }
    if (this.rng() < (this.locked ? 0 : this.epsilon())) {
      this.exploring = true
      this.lastAction = (this.rng() * ACTIONS) | 0
    } else {
      this.exploring = false
      this.lastAction = greedy
    }
    return this.lastAction
  }

  remember(state: ArrayLike<number>, action: number, reward: number, next: ArrayLike<number>, done: boolean): void {
    const s = new Float64Array(OBS_DIM)
    const sp = new Float64Array(OBS_DIM)
    s.set(state)
    sp.set(next)
    this.pending.push({ state: s, action, reward, next: sp, done })
    this.steps += 1
    const n = LEARN.nStep
    while (this.pending.length >= n) this.commitFront()
    if (done) while (this.pending.length > 0) this.commitFront()
  }

  /** Write the oldest decision, with the return over the following decisions. */
  private commitFront(): void {
    const first = this.pending[0]
    let gain = 0
    let discount = 1
    let last = first
    for (const step of this.pending) {
      gain += discount * step.reward
      discount *= LEARN.gamma
      last = step
      if (step.done) break
    }
    this.replay.add(first.state, first.action, gain, last.next, last.done, last.done ? 0 : discount)
    this.pending.shift()
  }

  clearReplay(): void {
    this.replay.clear()
    this.pending.length = 0
    this.locked = false
    this.forceGreedy = false
    this.recentLands.length = 0
  }

  resetBrain(): void {
    this.net.init(this.rng)
    this.target.copyFrom(this.net)
    this.m.fill(0)
    this.v.fill(0)
    this.adamT = 0
    this.steps = 0
    this.replay.clear()
    this.pending.length = 0
    this.lastLoss = null
    this.exploring = false
    this.lastAction = 0
    this.locked = false
    this.forceGreedy = false
    this.recentLands.length = 0
  }

  learn(): number | null {
    if (this.locked || this.forceGreedy) return this.lastLoss
    if (this.replay.size < LEARN.warmup) return null
    const batch = LEARN.batch
    this.replay.sample(batch, this.sBuf, this.aBuf, this.rBuf, this.spBuf, this.dBuf, this.gBuf)

    this.net.forwardBatch(this.spBuf, batch)
    for (let b = 0; b < batch; b++) {
      const base = b * ACTIONS
      let greedy = 0
      let bestQ = this.net.qBatch[base]
      for (let k = 1; k < ACTIONS; k++) {
        const q = this.net.qBatch[base + k]
        if (q > bestQ) {
          bestQ = q
          greedy = k
        }
      }
      this.best[b] = greedy
    }

    this.target.forwardBatch(this.spBuf, batch)
    this.net.forwardBatch(this.sBuf, batch)

    this.dQ.fill(0)
    let loss = 0
    const delta = LEARN.huber
    for (let b = 0; b < batch; b++) {
      let target = this.rBuf[b]
      if (this.dBuf[b] === 0) {
        target += this.gBuf[b] * this.target.qBatch[b * ACTIONS + this.best[b]]
      }
      const q = this.net.qBatch[b * ACTIONS + this.aBuf[b]]
      const td = q - target
      const abs = Math.abs(td)
      loss += abs <= delta ? 0.5 * td * td : delta * (abs - 0.5 * delta)
      const grad = abs <= delta ? td : delta * Math.sign(td)
      this.dQ[b * ACTIONS + this.aBuf[b]] = grad / batch
    }

    this.net.backward(this.sBuf, this.dQ, batch)
    this.clipGrads()
    this.adam()
    this.target.softUpdateFrom(this.net, LEARN.tau)
    this.lastLoss = loss / batch
    return this.lastLoss
  }

  exportWeights(): BrainWeights {
    return {
      version: 1,
      steps: this.steps,
      adamT: this.adamT,
      weights: pack(this.net),
      target: pack(this.target),
      m: Array.from(this.m),
      v: Array.from(this.v),
    }
  }

  restoreWeights(data: BrainWeights): boolean {
    const count = this.net.paramCount()
    if (data.version !== 1 || data.weights.length !== count || data.target.length !== count) return false
    if (data.m.length !== count || data.v.length !== count) return false
    this.net.readFlat(data.weights)
    this.target.readFlat(data.target)
    this.m.set(data.m)
    this.v.set(data.v)
    this.adamT = data.adamT
    this.steps = data.steps
    this.replay.clear()
    this.pending.length = 0
    this.recentLands.length = 0
    this.lastLoss = null
    return true
  }

  private clipGrads(): void {
    let sum = 0
    for (const grad of this.net.gradTensors()) {
      for (let i = 0; i < grad.length; i++) sum += grad[i] * grad[i]
    }
    const norm = Math.sqrt(sum)
    if (norm > LEARN.gradClip) {
      const scale = LEARN.gradClip / norm
      for (const grad of this.net.gradTensors()) {
        for (let i = 0; i < grad.length; i++) grad[i] *= scale
      }
    }
  }

  private adam(): void {
    this.adamT += 1
    const b1 = 0.9
    const b2 = 0.999
    const c1 = 1 - b1 ** this.adamT
    const c2 = 1 - b2 ** this.adamT
    let offset = 0
    const tensors = this.net.tensors()
    const grads = this.net.gradTensors()
    for (let t = 0; t < tensors.length; t++) {
      const values = tensors[t]
      const grad = grads[t]
      for (let i = 0; i < values.length; i++) {
        const g = grad[i]
        const m = (this.m[offset] = b1 * this.m[offset] + (1 - b1) * g)
        const v = (this.v[offset] = b2 * this.v[offset] + (1 - b2) * g * g)
        const lr = this.steps < LEARN.lrCutSteps ? LEARN.lr : LEARN.lrFinal
        values[i] -= (lr * (m / c1)) / (Math.sqrt(v / c2) + 1e-8)
        offset += 1
      }
    }
  }
}
