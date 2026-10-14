/**
 * Fit the Q network to the known landing policy and write public/landing-brain.json.
 * The page can load these weights and fly them greedily while learning stays available.
 */
import { mkdirSync, writeFileSync } from 'fs'
import { ACTIONS, LEARN, OBS_DIM, defaultParams } from '../src/config'
import { Agent } from '../src/rl/dqn'
import { QNetwork } from '../src/rl/network'
import { mulberry32 } from '../src/rng'
import { LanderEnv } from '../src/sim/env'

declare const process: { exit: (code: number) => never }

function heuristic(env: LanderEnv): number {
  const s = env.state
  const aUp = env.params.mainAcc - env.params.gravity
  const impactSq = s.vy * s.vy - 2 * aUp * Math.max(s.y, 0)
  const level = s.angle > 0 ? 2 : 3
  if (Math.abs(s.angle) > 0.55) return level
  if (s.y < 1.4 && s.vy > -1.8 && Math.abs(s.vx) < 1.4 && Math.abs(s.angle) < 0.25) {
    if (s.vy < -1.05) return 1
    if (Math.abs(s.angle) > 0.1) return level
    return 0
  }
  if (impactSq > 0.25 && Math.abs(s.angle) < 0.35) return 1
  if (Math.abs(s.angle) > 0.07) return level
  if (s.omega > 0.08) return 2
  if (s.omega < -0.08) return 3
  return 0
}

interface Sample {
  s: Float64Array
  a: number
}

function greedyAction(net: QNetwork, obs: ArrayLike<number>): number {
  net.forward(obs)
  let best = 0
  let bestQ = net.q[0]
  for (let i = 1; i < ACTIONS; i++) {
    if (net.q[i] > bestQ) {
      bestQ = net.q[i]
      best = i
    }
  }
  return best
}

function clipGrads(net: QNetwork): void {
  let sum = 0
  for (const grad of net.gradTensors()) {
    for (let i = 0; i < grad.length; i++) sum += grad[i] * grad[i]
  }
  const norm = Math.sqrt(sum)
  if (norm > 1) {
    const scale = 1 / norm
    for (const grad of net.gradTensors()) {
      for (let i = 0; i < grad.length; i++) grad[i] *= scale
    }
  }
}

function train(net: QNetwork, data: Sample[], epochs: number, seed: number): void {
  const rng = mulberry32(seed)
  const batch = 128
  const sBuf = new Float64Array(batch * OBS_DIM)
  const dQ = new Float64Array(batch * ACTIONS)
  const m = new Float64Array(net.paramCount())
  const v = new Float64Array(net.paramCount())
  let adamT = 0
  const lr = 2e-3
  const order = data.map((_, i) => i)
  for (let epoch = 0; epoch < epochs; epoch++) {
    for (let i = order.length - 1; i > 0; i--) {
      const j = (rng() * (i + 1)) | 0
      const swap = order[i]
      order[i] = order[j]
      order[j] = swap
    }
    let correct = 0
    let seen = 0
    for (let start = 0; start + batch <= order.length; start += batch) {
      for (let b = 0; b < batch; b++) sBuf.set(data[order[start + b]].s, b * OBS_DIM)
      net.forwardBatch(sBuf, batch)
      dQ.fill(0)
      for (let b = 0; b < batch; b++) {
        const sample = data[order[start + b]]
        const base = b * ACTIONS
        let picked = 0
        let pickedQ = net.qBatch[base]
        for (let k = 1; k < ACTIONS; k++) {
          if (net.qBatch[base + k] > pickedQ) {
            pickedQ = net.qBatch[base + k]
            picked = k
          }
        }
        if (picked === sample.a) correct += 1
        for (let k = 0; k < ACTIONS; k++) {
          const target = k === sample.a ? 1 : 0
          dQ[base + k] = (net.qBatch[base + k] - target) / batch
        }
        seen += 1
      }
      net.backward(sBuf, dQ, batch)
      clipGrads(net)
      adamT += 1
      const b1 = 0.9
      const b2 = 0.999
      const c1 = 1 - b1 ** adamT
      const c2 = 1 - b2 ** adamT
      let offset = 0
      const tensors = net.tensors()
      const grads = net.gradTensors()
      for (let t = 0; t < tensors.length; t++) {
        const values = tensors[t]
        const grad = grads[t]
        for (let i = 0; i < values.length; i++) {
          const g = grad[i]
          const mm = (m[offset] = b1 * m[offset] + (1 - b1) * g)
          const vv = (v[offset] = b2 * v[offset] + (1 - b2) * g * g)
          values[i] -= (lr * (mm / c1)) / (Math.sqrt(vv / c2) + 1e-8)
          offset += 1
        }
      }
    }
    if (epoch === epochs - 1) console.log(`fit accuracy ${((100 * correct) / Math.max(1, seen)).toFixed(1)}%`)
  }
}

function landRate(agent: Agent, env: LanderEnv, n: number, learn: boolean): { land: number; agree: number } {
  let lands = 0
  let agree = 0
  let decisions = 0
  for (let ep = 1; ep <= n; ep++) {
    env.reset()
    let guard = 0
    while (!env.done && guard < 2000) {
      guard += 1
      const label = heuristic(env)
      const obs = new Float64Array(env.obs)
      const action = agent.act(env.obs)
      if (action === label) agree += 1
      decisions += 1
      const result = env.step(action)
      if (learn) {
        agent.remember(obs, action, result.reward, env.obs, result.done)
        agent.learn()
      }
    }
    if (env.outcome === 'land') lands += 1
    if (learn) agent.noteEpisode(env.outcome === 'land', ep)
  }
  return { land: lands / n, agree: agree / Math.max(1, decisions) }
}

const rng = mulberry32(4)
const env = new LanderEnv(defaultParams(), mulberry32(21))
const net = new QNetwork(128, mulberry32(3))
const data: Sample[] = []

for (let round = 1; round <= 8; round++) {
  for (let ep = 0; ep < 40; ep++) {
    env.reset()
    let guard = 0
    while (!env.done && guard < 2000) {
      guard += 1
      const s = new Float64Array(env.obs)
      const label = heuristic(env)
      data.push({ s, a: label })
      const useNet = round > 1 && rng() < 0.65
      const exec = useNet ? greedyAction(net, s) : label
      env.step(exec)
    }
  }
  train(net, data, 10, 20 + round)
  const flat = new Float64Array(net.paramCount())
  net.writeFlat(flat)
  const agent = new Agent(mulberry32(11))
  agent.net.readFlat(flat)
  agent.target.copyFrom(agent.net)
  agent.forceGreedy = true
  const score = landRate(agent, env, 30, false)
  console.log(
    `round ${round} samples ${data.length} land ${(100 * score.land).toFixed(1)}% agree ${(100 * score.agree).toFixed(1)}%`,
  )
  if (score.land >= 0.95 && round >= 3) break
}

const flat = new Float64Array(net.paramCount())
net.writeFlat(flat)
const agent = new Agent(mulberry32(11))
agent.net.readFlat(flat)
agent.target.copyFrom(agent.net)
agent.forceGreedy = true
agent.steps = LEARN.lrCutSteps
const greedy = landRate(agent, env, 40, false)
console.log(`greedy land ${(100 * greedy.land).toFixed(1)}% agree ${(100 * greedy.agree).toFixed(1)}%`)

const learner = new Agent(mulberry32(15))
const brain = agent.exportWeights()
if (!learner.restoreWeights(brain)) throw new Error('restore failed')
learner.forceGreedy = true
const held = landRate(learner, env, 40, true)
console.log(
  `after 40 learning flights ${(100 * held.land).toFixed(1)}% agree ${(100 * held.agree).toFixed(1)}% locked ${learner.locked}`,
)

if (greedy.land < 0.9) {
  console.error('landing weights are not reliable')
  process.exit(1)
}

mkdirSync('public', { recursive: true })
writeFileSync('public/landing-brain.json', JSON.stringify(brain))
console.log('wrote public/landing-brain.json')
