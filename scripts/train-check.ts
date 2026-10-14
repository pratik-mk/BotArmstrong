import { defaultParams } from '../src/config'
import { verifyQGradients } from '../src/rl/network'
import { Agent } from '../src/rl/dqn'
import { LanderEnv } from '../src/sim/env'
import { mulberry32 } from '../src/rng'

declare const process: { argv: string[]; exit: (code: number) => never }

const seed = Number(process.argv[3] ?? 7)

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

function runEpisode(env: LanderEnv, choose: (env: LanderEnv) => number, agent: Agent | null): { ret: number; outcome: string; steps: number } {
  env.reset()
  const obs = new Float64Array(env.obs.length)
  let guard = 0
  while (!env.done && guard < 2000) {
    guard += 1
    obs.set(env.obs)
    const action = choose(env)
    const result = env.hold(action)
    if (agent) {
      agent.remember(obs, action, result.reward, env.obs, result.done)
      agent.learn()
    }
  }
  return { ret: env.episodeReturn, outcome: env.outcome, steps: env.steps }
}

function summarize(label: string, rows: { ret: number; outcome: string }[]): void {
  let lands = 0
  let crashes = 0
  let timeouts = 0
  let ret = 0
  for (const row of rows) {
    ret += row.ret
    if (row.outcome === 'land') lands += 1
    else if (row.outcome === 'timeout') timeouts += 1
    else crashes += 1
  }
  const n = rows.length
  console.log(
    `${label} n=${n} mean=${(ret / n).toFixed(2)} land=${((100 * lands) / n).toFixed(1)}% crash=${((100 * crashes) / n).toFixed(1)}% timeout=${((100 * timeouts) / n).toFixed(1)}%`,
  )
}

function mean(values: number[], start: number, count: number): number {
  let sum = 0
  for (let i = start; i < start + count; i++) sum += values[i]
  return sum / count
}

const mode = process.argv[2] ?? '400'
const rng = mulberry32(seed)
verifyQGradients(rng)
console.log('gradient check passed')

const params = defaultParams()
const env = new LanderEnv(params, mulberry32(seed + 3))

if (mode === 'debug') {
  env.reset()
  for (let i = 0; i < 500 && !env.done; i++) {
    const action = heuristic(env)
    if (i % 8 === 0) {
      const s = env.state
      console.log(
        `${i} a=${action} y=${s.y.toFixed(2)} vy=${s.vy.toFixed(2)} ang=${s.angle.toFixed(3)} w=${s.omega.toFixed(3)} fuel=${s.fuel.toFixed(2)}`,
      )
    }
    env.step(action)
  }
  const s = env.state
  console.log('end', env.outcome, `x=${s.x.toFixed(2)} vy=${s.vy.toFixed(2)} ang=${s.angle.toFixed(3)}`)
  process.exit(0)
}

if (mode === 'heuristic') {
  const rows = []
  for (let i = 0; i < 30; i++) {
    const row = runEpisode(env, (e) => heuristic(e), null)
    rows.push(row)
    if (row.outcome !== 'land') {
      const s = env.state
      console.log(
        row.outcome,
        `x=${s.x.toFixed(2)} y=${s.y.toFixed(2)} vx=${s.vx.toFixed(2)} vy=${s.vy.toFixed(2)} ang=${((s.angle * 180) / Math.PI).toFixed(1)} w=${s.omega.toFixed(2)} fuel=${s.fuel.toFixed(2)} steps=${row.steps}`,
      )
    }
  }
  summarize('heuristic', rows)
  process.exit(0)
}

const heuristicRows = []
for (let i = 0; i < 20; i++) heuristicRows.push(runEpisode(env, (e) => heuristic(e), null))
summarize('heuristic', heuristicRows)

const randomRows = []
const randomRng = mulberry32(99)
for (let i = 0; i < 20; i++) randomRows.push(runEpisode(env, () => (randomRng() * 4) | 0, null))
summarize('random', randomRows)

const episodes = Number(mode)
const agent = new Agent(mulberry32(seed + 11))
const returns: number[] = []
const lands: number[] = []
const outcomes: string[] = []
const started = Date.now()
let lossSum = 0
let lossCount = 0

for (let ep = 1; ep <= episodes; ep++) {
  const row = runEpisode(env, (e) => agent.act(e.obs), agent)
  returns.push(row.ret)
  lands.push(row.outcome === 'land' ? 1 : 0)
  outcomes.push(row.outcome)
  agent.noteEpisode(row.outcome === 'land', ep)
  if (agent.lastLoss !== null) {
    lossSum += agent.lastLoss
    lossCount += 1
  }
  if (ep % 25 === 0 || ep === episodes) {
    const from = Math.max(0, ep - 25)
    let recentRet = 0
    let recentLand = 0
    let recentTime = 0
    const span = ep - from
    for (let i = from; i < ep; i++) {
      recentRet += returns[i]
      recentLand += lands[i]
      if (outcomes[i] === 'timeout') recentTime += 1
    }
    const secs = (Date.now() - started) / 1000
    console.log(
      `ep ${ep} ret ${(recentRet / span).toFixed(1)} land ${((100 * recentLand) / span).toFixed(0)}% timeout ${((100 * recentTime) / span).toFixed(0)}% eps ${agent.epsilon().toFixed(2)}${agent.locked ? ' held' : ''} loss ${lossCount ? (lossSum / lossCount).toFixed(3) : '—'} ${((ep / secs) || 0).toFixed(1)} ep/s`,
    )
    lossSum = 0
    lossCount = 0
  }
}

const window = Math.min(100, Math.floor(episodes / 2))
const first = mean(returns, 0, window)
const last = mean(returns, returns.length - window, window)
let firstLands = 0
let lastLands = 0
for (let i = 0; i < window; i++) firstLands += lands[i]
for (let i = lands.length - window; i < lands.length; i++) lastLands += lands[i]
let tailLands = 0
const tail = Math.min(40, lands.length)
for (let i = lands.length - tail; i < lands.length; i++) tailLands += lands[i]
console.log(`RESULT first=${first.toFixed(2)} last=${last.toFixed(2)} lastLands=${lastLands}/${window} tail=${tailLands}/${tail} steps=${agent.steps}`)
const landingsImproved = lastLands > firstLands && lastLands >= Math.max(1, Math.round(window * 0.1))
const tailHeld = tailLands >= Math.round(tail * 0.35)
if (!(last > first && landingsImproved && tailHeld)) {
  console.error('training did not improve')
  process.exit(1)
}
console.log('training check passed')
