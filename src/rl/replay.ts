import { OBS_DIM } from '../config'

export class Replay {
  private readonly states: Float32Array
  private readonly next: Float32Array
  private readonly actions: Uint8Array
  private readonly rewards: Float32Array
  private readonly discounts: Float32Array
  private readonly dones: Uint8Array
  private head = 0
  size = 0

  constructor(
    readonly capacity: number,
    private readonly rng: () => number,
  ) {
    this.states = new Float32Array(capacity * OBS_DIM)
    this.next = new Float32Array(capacity * OBS_DIM)
    this.actions = new Uint8Array(capacity)
    this.rewards = new Float32Array(capacity)
    this.discounts = new Float32Array(capacity)
    this.dones = new Uint8Array(capacity)
  }

  clear(): void {
    this.head = 0
    this.size = 0
  }

  add(
    state: ArrayLike<number>,
    action: number,
    reward: number,
    next: ArrayLike<number>,
    done: boolean,
    discount: number,
  ): void {
    const o = this.head * OBS_DIM
    for (let k = 0; k < OBS_DIM; k++) {
      this.states[o + k] = state[k]
      this.next[o + k] = next[k]
    }
    this.actions[this.head] = action
    this.rewards[this.head] = reward
    this.discounts[this.head] = discount
    this.dones[this.head] = done ? 1 : 0
    this.head = (this.head + 1) % this.capacity
    if (this.size < this.capacity) this.size += 1
  }

  sample(
    batch: number,
    states: Float64Array,
    actions: Uint8Array,
    rewards: Float64Array,
    next: Float64Array,
    dones: Uint8Array,
    discounts: Float64Array,
  ): void {
    for (let b = 0; b < batch; b++) {
      const idx = (this.rng() * this.size) | 0
      const o = idx * OBS_DIM
      const bo = b * OBS_DIM
      for (let k = 0; k < OBS_DIM; k++) {
        states[bo + k] = this.states[o + k]
        next[bo + k] = this.next[o + k]
      }
      actions[b] = this.actions[idx]
      rewards[b] = this.rewards[idx]
      dones[b] = this.dones[idx]
      discounts[b] = this.discounts[idx]
    }
  }
}
