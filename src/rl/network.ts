import { ACTIONS, HIDDEN, OBS_DIM } from '../config'

const IN = OBS_DIM
const H1 = HIDDEN
const H2 = HIDDEN
const OUT = ACTIONS

function fillNormal(dst: Float64Array, std: number, rng: () => number): void {
  for (let i = 0; i < dst.length; i++) {
    let u = rng()
    let v = rng()
    if (u < 1e-9) u = 1e-9
    if (v < 1e-9) v = 1e-9
    dst[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(Math.PI * 2 * v) * std
  }
}

/** Two ReLU hidden layers and a linear action-value head. */
export class QNetwork {
  readonly w1: Float64Array
  readonly b1: Float64Array
  readonly w2: Float64Array
  readonly b2: Float64Array
  readonly w3: Float64Array
  readonly b3: Float64Array

  readonly gw1: Float64Array
  readonly gb1: Float64Array
  readonly gw2: Float64Array
  readonly gb2: Float64Array
  readonly gw3: Float64Array
  readonly gb3: Float64Array

  readonly h1 = new Float64Array(H1)
  readonly h2 = new Float64Array(H2)
  readonly q = new Float64Array(OUT)

  readonly maxBatch: number
  private readonly z1 = new Float64Array(H1)
  private readonly z2 = new Float64Array(H2)
  private readonly bz1: Float64Array
  private readonly bh1: Float64Array
  private readonly bz2: Float64Array
  private readonly bh2: Float64Array
  readonly qBatch: Float64Array
  private readonly dh1: Float64Array
  private readonly dh2: Float64Array

  constructor(maxBatch: number, rng: () => number) {
    this.maxBatch = maxBatch
    this.w1 = new Float64Array(H1 * IN)
    this.b1 = new Float64Array(H1)
    this.w2 = new Float64Array(H2 * H1)
    this.b2 = new Float64Array(H2)
    this.w3 = new Float64Array(OUT * H2)
    this.b3 = new Float64Array(OUT)
    this.gw1 = new Float64Array(this.w1.length)
    this.gb1 = new Float64Array(H1)
    this.gw2 = new Float64Array(this.w2.length)
    this.gb2 = new Float64Array(H2)
    this.gw3 = new Float64Array(this.w3.length)
    this.gb3 = new Float64Array(OUT)
    this.bz1 = new Float64Array(maxBatch * H1)
    this.bh1 = new Float64Array(maxBatch * H1)
    this.bz2 = new Float64Array(maxBatch * H2)
    this.bh2 = new Float64Array(maxBatch * H2)
    this.qBatch = new Float64Array(maxBatch * OUT)
    this.dh1 = new Float64Array(maxBatch * H1)
    this.dh2 = new Float64Array(maxBatch * H2)
    this.init(rng)
  }

  init(rng: () => number): void {
    fillNormal(this.w1, Math.sqrt(2 / IN), rng)
    fillNormal(this.w2, Math.sqrt(2 / H1), rng)
    fillNormal(this.w3, 0.01, rng)
    this.b1.fill(0.01)
    this.b2.fill(0.01)
    this.b3.fill(0)
  }

  paramCount(): number {
    return this.w1.length + this.b1.length + this.w2.length + this.b2.length + this.w3.length + this.b3.length
  }

  tensors(): Float64Array[] {
    return [this.w1, this.b1, this.w2, this.b2, this.w3, this.b3]
  }

  gradTensors(): Float64Array[] {
    return [this.gw1, this.gb1, this.gw2, this.gb2, this.gw3, this.gb3]
  }

  copyFrom(other: QNetwork): void {
    const dst = this.tensors()
    const src = other.tensors()
    for (let i = 0; i < dst.length; i++) dst[i].set(src[i])
  }

  softUpdateFrom(online: QNetwork, tau: number): void {
    const dst = this.tensors()
    const src = online.tensors()
    const keep = 1 - tau
    for (let t = 0; t < dst.length; t++) {
      const d = dst[t]
      const s = src[t]
      for (let i = 0; i < d.length; i++) d[i] = keep * d[i] + tau * s[i]
    }
  }

  writeFlat(dst: Float64Array): void {
    let offset = 0
    for (const tensor of this.tensors()) {
      dst.set(tensor, offset)
      offset += tensor.length
    }
  }

  readFlat(src: ArrayLike<number>): void {
    let offset = 0
    for (const tensor of this.tensors()) {
      for (let i = 0; i < tensor.length; i++) tensor[i] = src[offset + i]
      offset += tensor.length
    }
  }

  forward(x: ArrayLike<number>): void {
    for (let i = 0; i < H1; i++) {
      let sum = this.b1[i]
      const row = i * IN
      for (let j = 0; j < IN; j++) sum += this.w1[row + j] * x[j]
      this.z1[i] = sum
      this.h1[i] = sum > 0 ? sum : 0
    }
    for (let i = 0; i < H2; i++) {
      let sum = this.b2[i]
      const row = i * H1
      for (let j = 0; j < H1; j++) sum += this.w2[row + j] * this.h1[j]
      this.z2[i] = sum
      this.h2[i] = sum > 0 ? sum : 0
    }
    for (let k = 0; k < OUT; k++) {
      let sum = this.b3[k]
      const row = k * H2
      for (let j = 0; j < H2; j++) sum += this.w3[row + j] * this.h2[j]
      this.q[k] = sum
    }
  }

  forwardBatch(x: ArrayLike<number>, batch: number): void {
    for (let b = 0; b < batch; b++) {
      const x0 = b * IN
      const h1o = b * H1
      for (let i = 0; i < H1; i++) {
        let sum = this.b1[i]
        const row = i * IN
        for (let j = 0; j < IN; j++) sum += this.w1[row + j] * x[x0 + j]
        this.bz1[h1o + i] = sum
        this.bh1[h1o + i] = sum > 0 ? sum : 0
      }
      const h2o = b * H2
      for (let i = 0; i < H2; i++) {
        let sum = this.b2[i]
        const row = i * H1
        for (let j = 0; j < H1; j++) sum += this.w2[row + j] * this.bh1[h1o + j]
        this.bz2[h2o + i] = sum
        this.bh2[h2o + i] = sum > 0 ? sum : 0
      }
      const q0 = b * OUT
      for (let k = 0; k < OUT; k++) {
        let sum = this.b3[k]
        const row = k * H2
        for (let j = 0; j < H2; j++) sum += this.w3[row + j] * this.bh2[h2o + j]
        this.qBatch[q0 + k] = sum
      }
    }
  }

  /** dQ is ∂L/∂Q for each sample, including the 1/batch factor if the loss is a mean. */
  backward(x: ArrayLike<number>, dQ: Float64Array, batch: number): void {
    this.gw1.fill(0)
    this.gb1.fill(0)
    this.gw2.fill(0)
    this.gb2.fill(0)
    this.gw3.fill(0)
    this.gb3.fill(0)
    this.dh1.fill(0)
    this.dh2.fill(0)

    for (let b = 0; b < batch; b++) {
      const h2o = b * H2
      const q0 = b * OUT
      for (let k = 0; k < OUT; k++) {
        const g = dQ[q0 + k]
        if (g === 0) continue
        this.gb3[k] += g
        const row = k * H2
        for (let j = 0; j < H2; j++) {
          this.gw3[row + j] += g * this.bh2[h2o + j]
          this.dh2[h2o + j] += this.w3[row + j] * g
        }
      }
      const h1o = b * H1
      for (let i = 0; i < H2; i++) {
        const hidden = this.bz2[h2o + i] > 0 ? this.dh2[h2o + i] : 0
        if (hidden === 0) continue
        this.gb2[i] += hidden
        const row = i * H1
        for (let j = 0; j < H1; j++) {
          this.gw2[row + j] += hidden * this.bh1[h1o + j]
          this.dh1[h1o + j] += this.w2[row + j] * hidden
        }
      }
      const x0 = b * IN
      for (let i = 0; i < H1; i++) {
        const hidden = this.bz1[h1o + i] > 0 ? this.dh1[h1o + i] : 0
        if (hidden === 0) continue
        this.gb1[i] += hidden
        const row = i * IN
        for (let j = 0; j < IN; j++) this.gw1[row + j] += hidden * x[x0 + j]
      }
    }
  }
}

function huber(err: number, delta: number): number {
  const abs = Math.abs(err)
  if (abs <= delta) return 0.5 * err * err
  return delta * (abs - 0.5 * delta)
}

function huberGrad(err: number, delta: number): number {
  if (err > delta) return delta
  if (err < -delta) return -delta
  return err
}

/** Finite-difference check of the Huber backward pass. Throws if the gradient is wrong. */
export function verifyQGradients(rng: () => number): void {
  let last = 'gradient check did not run'
  for (let attempt = 0; attempt < 8; attempt++) {
    const mismatch = gradientMismatch(rng)
    if (mismatch === null) return
    last = mismatch
  }
  throw new Error(last)
}

function gradientMismatch(rng: () => number): string | null {
  const net = new QNetwork(1, rng)
  const x = new Float64Array(IN)
  for (let i = 0; i < IN; i++) x[i] = rng() * 2 - 1
  const action = (rng() * OUT) | 0
  const target = rng() * 2 - 1
  const delta = 1
  const eps = 1e-5

  const loss = (): number => {
    net.forwardBatch(x, 1)
    return huber(net.qBatch[action] - target, delta)
  }

  net.forwardBatch(x, 1)
  const dQ = new Float64Array(OUT)
  dQ[action] = huberGrad(net.qBatch[action] - target, delta)
  net.backward(x, dQ, 1)

  const tensors = net.tensors()
  const grads = net.gradTensors()
  for (let t = 0; t < tensors.length; t++) {
    const values = tensors[t]
    const grad = grads[t]
    const probes = Math.min(3, values.length)
    for (let p = 0; p < probes; p++) {
      const i = (rng() * values.length) | 0
      const orig = values[i]
      values[i] = orig + eps
      const plus = loss()
      values[i] = orig - eps
      const minus = loss()
      values[i] = orig
      const numeric = (plus - minus) / (2 * eps)
      const analytic = grad[i]
      const scale = Math.max(1, Math.abs(numeric))
      if (Math.abs(numeric - analytic) > 2e-4 * scale && Math.abs(numeric - analytic) > 2e-4) {
        return `gradient mismatch on tensor ${t} index ${i}: numeric ${numeric} analytic ${analytic}`
      }
    }
  }
  return null
}
