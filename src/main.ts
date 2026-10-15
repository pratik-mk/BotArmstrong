import { ACTION_REPEAT, DT, defaultParams } from './config'
import { loadRun, saveRun } from './persist'
import { mulberry32 } from './rng'
import { Agent } from './rl/dqn'
import type { Outcome } from './rl/reward'
import { LanderEnv } from './sim/env'
import { sizeCanvas } from './ui/canvas'
import { drawCharts } from './ui/charts'
import { drawNetwork } from './ui/networkView'
import { mountPanel } from './ui/panel'
import { createScene } from './ui/world'

const params = defaultParams()
// Same initialization as the headless check, so a fresh tab follows a run that learns to land.
const seed = 7
const env = new LanderEnv(params, mulberry32(seed + 3))
const agent = new Agent(mulberry32(seed + 11))
env.reset()
const scene = createScene()

const returns: number[] = []
const lands: number[] = []
const epsilons: number[] = []
const losses: number[] = []
let episode = 1
let speed: 1 | 4 | 16 = 4
let paused = false
let holdUntil = 0
let bannerUntil = 0
let shake = 0
let lastOutcome: Outcome = 'fly'
let acc = 0
let held = 0
let holdReward = 0
let manual = false
const heldKeys = new Set<string>()
const stickHeld = { burn: false, left: false, right: false }
const decisionObs = new Float64Array(10)
const stickNode = document.querySelector<HTMLElement>('#stick')
if (!stickNode) throw new Error('Missing controls')
const stick: HTMLElement = stickNode
const stickButtons = [...stick.querySelectorAll<HTMLButtonElement>('[data-pilot]')]

function pressStick(kind: string, down: boolean): void {
  if (kind === 'burn') stickHeld.burn = down
  else if (kind === 'left') stickHeld.left = down
  else stickHeld.right = down
}

for (const button of stickButtons) {
  const kind = button.dataset.pilot ?? 'right'
  button.addEventListener('pointerdown', (event) => {
    event.preventDefault()
    pressStick(kind, true)
    button.blur()
    try {
      button.setPointerCapture(event.pointerId)
    } catch {
      // The button still counts as held until pointerup.
    }
  })
  const release = () => pressStick(kind, false)
  button.addEventListener('pointerup', release)
  button.addEventListener('pointercancel', release)
}

function pilotInput(): { burn: boolean; turn: -1 | 0 | 1 } {
  const left = heldKeys.has('ArrowLeft') || heldKeys.has('KeyA') || stickHeld.left
  const right = heldKeys.has('ArrowRight') || heldKeys.has('KeyD') || stickHeld.right
  const burn = heldKeys.has('ArrowUp') || heldKeys.has('KeyW') || stickHeld.burn
  return { burn, turn: left === right ? 0 : left ? -1 : 1 }
}

function commandLabel(burn: boolean, turn: number): string {
  const spin = turn < 0 ? 'Left' : turn > 0 ? 'Right' : ''
  if (burn && spin) return `Main + ${spin}`
  if (burn) return 'Main'
  return spin || 'Idle'
}

const worldNode = document.querySelector<HTMLCanvasElement>('#world')
const chartNode = document.querySelector<HTMLCanvasElement>('#chart')
const netNode = document.querySelector<HTMLCanvasElement>('#net')
const bannerNode = document.querySelector<HTMLElement>('#banner')
if (!worldNode || !chartNode || !netNode || !bannerNode) throw new Error('Missing canvas')
const worldCanvas: HTMLCanvasElement = worldNode
const chartCanvas: HTMLCanvasElement = chartNode
const netCanvas: HTMLCanvasElement = netNode
const banner: HTMLElement = bannerNode

const panel = mountPanel(params, {
  onSpeed: (value) => {
    if (value === 1 || value === 4 || value === 16) setSpeed(value)
  },
  onPause: () => setPaused(!paused),
  onReset: () => resetBrain(),
  onSave: () => saveBrain(),
  onLoad: () => loadBrain(),
  onLanding: () => void loadLandingWeights(),
  onPlay: () => setManual(!manual),
  onParam: (key, value) => {
    if (params[key] === value) return
    params[key] = value
    agent.clearReplay()
    panel.setNote('Replay cleared. The weights stay, and the next flights adapt to the new world.')
  },
})

panel.setSpeed(speed)
panel.setPaused(false)
panel.setNote('Weights start random and keep learning. Landing weights fly a saved network that already touches down.')

window.addEventListener('keydown', (event) => {
  const tag = (event.target as HTMLElement | null)?.tagName
  if (tag === 'INPUT' || tag === 'TEXTAREA') return
  if (manual && (event.code === 'ArrowUp' || event.code === 'ArrowLeft' || event.code === 'ArrowRight' || event.code === 'KeyW' || event.code === 'KeyA' || event.code === 'KeyD')) {
    event.preventDefault()
    heldKeys.add(event.code)
    return
  }
  if (tag === 'BUTTON') return
  if (event.code === 'Space') {
    event.preventDefault()
    setPaused(!paused)
  } else if (event.key === '1') setSpeed(1)
  else if (event.key === '2') setSpeed(4)
  else if (event.key === '3') setSpeed(16)
})
window.addEventListener('keyup', (event) => {
  heldKeys.delete(event.code)
})
window.addEventListener('blur', () => heldKeys.clear())

function setSpeed(next: 1 | 4 | 16): void {
  speed = next
  panel.setSpeed(next)
}

function setPaused(next: boolean): void {
  paused = next
  panel.setPaused(next)
}

function leaveManual(): void {
  if (!manual) return
  manual = false
  panel.setManual(false)
  heldKeys.clear()
  stickHeld.burn = false
  stickHeld.left = false
  stickHeld.right = false
  stick.hidden = true
}

function setManual(next: boolean): void {
  if (next === manual) return
  leaveManual()
  if (!next) {
    env.reset()
    holdUntil = 0
    bannerUntil = 0
    acc = 0
    held = 0
    holdReward = 0
    banner.hidden = true
    lastOutcome = 'fly'
    panel.setNote('Back to training. The brain is unchanged.')
    return
  }
  manual = true
  panel.setManual(true)
  stick.hidden = false
  env.reset()
  holdUntil = 0
  bannerUntil = 0
  acc = 0
  held = 0
  holdReward = 0
  banner.hidden = true
  lastOutcome = 'fly'
  setSpeed(1)
  setPaused(false)
  panel.setNote('You have the stick. ↑ or W burns, ← → or A D rotate. You can do both at once. Space pauses.')
  if (document.activeElement instanceof HTMLElement) document.activeElement.blur()
}

function beginFlight(note: string): void {
  env.reset()
  returns.length = 0
  lands.length = 0
  epsilons.length = 0
  losses.length = 0
  episode = 1
  holdUntil = 0
  bannerUntil = 0
  shake = 0
  lastOutcome = 'fly'
  acc = 0
  held = 0
  holdReward = 0
  banner.hidden = true
  panel.setNote(note)
}

function resetBrain(): void {
  leaveManual()
  agent.resetBrain()
  beginFlight('New random brain. It has to learn the landing again.')
}

async function loadLandingWeights(): Promise<void> {
  leaveManual()
  try {
    const response = await fetch('./landing-brain.json')
    if (!response.ok) {
      panel.setNote('Landing weights are not in this build.')
      return
    }
    const data = (await response.json()) as Parameters<typeof agent.restoreWeights>[0]
    if (!agent.restoreWeights(data)) {
      panel.setNote('Landing weights do not match this network.')
      return
    }
    agent.forceGreedy = true
    agent.locked = false
    beginFlight('Flying the landing weights. They stay fixed so practice cannot erase the touchdown. Reset brain to learn from scratch.')
  } catch {
    panel.setNote('Could not load the landing weights.')
  }
}

function saveBrain(): void {
  try {
    saveRun({
      version: 1,
      brain: agent.exportWeights(),
      returns: returns.slice(),
      lands: lands.slice(),
      epsilons: epsilons.slice(),
      losses: losses.slice(),
      episode,
      locked: agent.locked,
      forceGreedy: agent.forceGreedy,
    })
    panel.setNote('Brain saved in this browser.')
  } catch {
    panel.setNote('Could not save the brain in this browser.')
  }
}

function loadBrain(): void {
  leaveManual()
  try {
    const saved = loadRun()
    if (!saved || !agent.restoreWeights(saved.brain)) {
      panel.setNote('No saved brain in this browser.')
      return
    }
    returns.splice(0, returns.length, ...saved.returns)
    lands.splice(0, lands.length, ...saved.lands)
    epsilons.splice(0, epsilons.length, ...(saved.epsilons ?? []))
    losses.splice(0, losses.length, ...(saved.losses ?? []))
    episode = saved.episode
    env.reset()
    holdUntil = 0
    acc = 0
    held = 0
    holdReward = 0
    banner.hidden = true
    agent.locked = saved.locked === true
    agent.forceGreedy = saved.forceGreedy === true
    panel.setNote(
      agent.forceGreedy
        ? 'Loaded the landing weights. Flights stay fixed. Reset brain to learn from scratch.'
        : agent.locked
          ? 'Loaded the saved brain. Weight updates stay paused.'
          : 'Loaded the saved brain. It keeps learning from here.',
    )
  } catch {
    panel.setNote('Could not read the saved brain.')
  }
}

function finishEpisode(outcome: Outcome): void {
  if (!manual) {
    returns.push(env.episodeReturn)
    lands.push(outcome === 'land' ? 1 : 0)
    epsilons.push(agent.epsilon())
    losses.push(agent.lastLoss ?? (losses.length > 0 ? losses[losses.length - 1] : 0))
    if (returns.length > 240) {
      returns.shift()
      lands.shift()
      epsilons.shift()
      losses.shift()
    }
  }
  lastOutcome = outcome
  banner.textContent = outcome === 'land' ? 'Touchdown' : outcome === 'crash' ? 'Vehicle lost' : 'Out of time'
  banner.dataset.kind = outcome
  banner.hidden = false
  const now = performance.now()
  bannerUntil = now + 1100
  holdUntil = now + (speed <= 1 ? 900 : speed <= 4 ? 160 : 0)
  if (outcome === 'crash') shake = 1
  if (outcome === 'land' || outcome === 'crash') scene.burst(env.state, outcome)
  if (!manual) {
    const wasLocked = agent.locked
    agent.noteEpisode(outcome === 'land', episode)
    if (!wasLocked && agent.locked && !agent.forceGreedy) {
      panel.setNote('Landing learned. Weight updates are paused so more flights do not erase it.')
    }
  }
}

function recent(values: number[], count: number): number {
  if (values.length === 0) return 0
  const start = Math.max(0, values.length - count)
  let sum = 0
  for (let i = start; i < values.length; i++) sum += values[i]
  return sum / (values.length - start)
}

let last = performance.now()
function frame(now: number): void {
  const realDt = Math.min(0.05, (now - last) / 1000)
  last = now
  if (holdUntil > 0 && now >= holdUntil) {
    holdUntil = 0
    episode += 1
    env.reset()
  }
  const pilot = pilotInput()
  let spawnFlame = false
  let spawnRcs = 0
  let steps = 0
  if (!paused && holdUntil === 0) {
    acc += realDt * speed
    while (acc >= DT && steps < 24) {
      acc -= DT
      steps += 1
      if (manual) {
        const result = env.stepPilot(pilot.burn, pilot.turn)
        if (env.lastFired && pilot.burn) spawnFlame = true
        if (env.lastFired && pilot.turn !== 0) spawnRcs = pilot.turn < 0 ? 2 : 3
        if (result.done) {
          finishEpisode(result.outcome)
          acc = 0
          break
        }
        continue
      }
      if (held === 0) {
        decisionObs.set(env.obs)
        agent.act(env.obs)
        holdReward = 0
      }
      const result = env.step(agent.lastAction)
      holdReward += result.reward
      held += 1
      if (env.lastFired && env.lastAction === 1) spawnFlame = true
      if (env.lastFired && (env.lastAction === 2 || env.lastAction === 3)) spawnRcs = env.lastAction
      if (result.done || held >= ACTION_REPEAT) {
        agent.remember(decisionObs, agent.lastAction, holdReward, env.obs, result.done)
        agent.learn()
        held = 0
      }
      if (result.done) {
        finishEpisode(result.outcome)
        acc = 0
        break
      }
    }
    if (steps >= 24) acc = 0
  }
  for (const button of stickButtons) {
    const kind = button.dataset.pilot
    const on = kind === 'burn' ? pilot.burn : kind === 'left' ? pilot.turn < 0 : pilot.turn > 0
    button.classList.toggle('on', on)
  }
  if (shake > 0) shake *= Math.exp(-2.8 * realDt)
  if (bannerUntil > 0 && now > bannerUntil) banner.hidden = true

  const world = sizeCanvas(worldCanvas)
  const mood: Outcome = now < bannerUntil ? lastOutcome : 'fly'
  scene.draw(world.ctx, world.w, world.h, {
    state: env.state,
    action: env.lastAction,
    fired: holdUntil === 0 && env.lastFired,
    padHalf: params.padHalf,
    trailX: env.trailX,
    trailY: env.trailY,
    trailCount: env.trailCount,
    trailHead: env.trailHead,
    episode,
    mood,
    paused,
    fixed: agent.forceGreedy,
    manual,
    time: now / 1000,
    shake,
    dt: realDt,
    spawnFlame: spawnFlame && holdUntil === 0,
    spawnRcs: holdUntil === 0 ? spawnRcs : 0,
  })

  const chart = sizeCanvas(chartCanvas)
  drawCharts(chart.ctx, chart.w, chart.h, returns, lands, epsilons, losses)
  const shownAction = manual ? (pilot.burn ? 1 : pilot.turn < 0 ? 2 : pilot.turn > 0 ? 3 : 0) : agent.lastAction
  if (manual || steps === 0) agent.net.forward(env.obs)
  const net = sizeCanvas(netCanvas)
  drawNetwork(net.ctx, net.w, net.h, {
    input: manual || steps === 0 ? env.obs : decisionObs,
    h1: agent.net.h1,
    h2: agent.net.h2,
    q: agent.net.q,
    action: shownAction,
    exploring: manual ? false : agent.exploring,
  })
  panel.update({
    state: env.state,
    action: shownAction,
    command: manual ? commandLabel(pilot.burn, pilot.turn) : undefined,
    wind: params.wind,
    padHalf: params.padHalf,
    episode,
    epsilon: agent.epsilon(),
    loss: agent.lastLoss,
    flightReturn: env.episodeReturn,
    successRate: recent(lands, 20),
    meanReturn: recent(returns, 20),
  })
  page.__rocketFrame = requestAnimationFrame(frame)
}

const page = window as Window & { __rocketFrame?: number }
if (page.__rocketFrame !== undefined) cancelAnimationFrame(page.__rocketFrame)
page.__rocketFrame = requestAnimationFrame(frame)
