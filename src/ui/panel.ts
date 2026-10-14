import { ACTION_LABELS, LIMITS, type WorldParams } from '../config'
import { LEG_HALF, type RocketState } from '../sim/physics'

export interface PanelView {
  state: RocketState
  action: number
  wind: number
  padHalf: number
  episode: number
  epsilon: number
  loss: number | null
  flightReturn: number
  successRate: number
  meanReturn: number
}

interface Handlers {
  onSpeed: (speed: number) => void
  onPause: () => void
  onReset: () => void
  onSave: () => void
  onLoad: () => void
  onLanding: () => void
  onParam: (key: keyof WorldParams, value: number) => void
}

export interface Panel {
  update(view: PanelView): void
  setSpeed(speed: number): void
  setPaused(paused: boolean): void
  setNote(text: string): void
}

export function mountPanel(params: WorldParams, handlers: Handlers): Panel {
  const pause = must<HTMLButtonElement>('pause')
  const note = must<HTMLParagraphElement>('note')
  const speeds = document.querySelectorAll<HTMLButtonElement>('[data-speed]')
  pause.addEventListener('click', handlers.onPause)
  must<HTMLButtonElement>('reset').addEventListener('click', handlers.onReset)
  must<HTMLButtonElement>('save').addEventListener('click', handlers.onSave)
  must<HTMLButtonElement>('load').addEventListener('click', handlers.onLoad)
  must<HTMLButtonElement>('landing').addEventListener('click', handlers.onLanding)
  speeds.forEach((button) => {
    button.addEventListener('click', () => handlers.onSpeed(Number(button.dataset.speed)))
  })

  const telemetry = must<HTMLElement>('telemetry')
  const alt = meter(telemetry, 'Altitude')
  const horizontal = meter(telemetry, 'Horizontal error')
  const vy = meter(telemetry, 'Vertical speed')
  const vx = meter(telemetry, 'Horizontal speed')
  const tilt = meter(telemetry, 'Tilt')
  const spin = meter(telemetry, 'Spin')
  const fuel = meter(telemetry, 'Fuel')
  const wind = meter(telemetry, 'Wind')
  const action = document.createElement('p')
  action.className = 'action'
  telemetry.append(action)

  const attempt = stat('Attempt')
  const eps = stat('Exploration')
  const loss = stat('Loss')
  const success = stat('Success, last 20')
  const mean = stat('Return, last 20')
  const flight = stat('This flight')

  const form = must<HTMLElement>('params')
  const sliders: { key: keyof WorldParams; input: HTMLInputElement; read: HTMLElement; format: (value: number) => string }[] = []
  for (const spec of SLIDERS) {
    const row = document.createElement('label')
    row.className = 'slider'
    const top = document.createElement('span')
    top.className = 'slider-top'
    const name = document.createElement('span')
    name.textContent = spec.label
    const read = document.createElement('span')
    read.textContent = spec.format(params[spec.key])
    top.append(name, read)
    const input = document.createElement('input')
    input.type = 'range'
    input.min = String(spec.min)
    input.max = String(spec.max)
    input.step = String(spec.step)
    input.value = String(params[spec.key])
    input.addEventListener('input', () => {
      const value = Number(input.value)
      read.textContent = spec.format(value)
      handlers.onParam(spec.key, value)
    })
    row.append(top, input)
    form.append(row)
    sliders.push({ key: spec.key, input, read, format: spec.format })
  }

  return {
    update(view) {
      const s = view.state
      alt.set(`${s.y.toFixed(1)} m`, unit(s.y, 0, 40), null)
      const usable = Math.max(0.5, view.padHalf - LEG_HALF)
      horizontal.set(
        `${s.x.toFixed(1)} m`,
        unit(s.x, -20, 20),
        Math.abs(s.x) <= usable,
        [unit(-usable, -20, 20), unit(usable, -20, 20)],
      )
      vy.set(`${s.vy.toFixed(2)} m/s`, unit(s.vy, -8, 4), Math.abs(s.vy) <= LIMITS.vy, [unit(-LIMITS.vy, -8, 4), unit(LIMITS.vy, -8, 4)])
      vx.set(`${s.vx.toFixed(2)} m/s`, unit(s.vx, -6, 6), Math.abs(s.vx) <= LIMITS.vx, [unit(-LIMITS.vx, -6, 6), unit(LIMITS.vx, -6, 6)])
      const deg = (s.angle * 180) / Math.PI
      const tiltLimit = (LIMITS.angle * 180) / Math.PI
      tilt.set(`${deg.toFixed(1)}°`, unit(deg, -45, 45), Math.abs(s.angle) <= LIMITS.angle, [unit(-tiltLimit, -45, 45), unit(tiltLimit, -45, 45)])
      const spinDeg = (s.omega * 180) / Math.PI
      const spinLimit = (LIMITS.omega * 180) / Math.PI
      spin.set(`${spinDeg.toFixed(1)}°/s`, unit(spinDeg, -90, 90), Math.abs(s.omega) <= LIMITS.omega, [unit(-spinLimit, -90, 90), unit(spinLimit, -90, 90)])
      fuel.set(`${Math.round(s.fuel * 100)}%`, unit(s.fuel, 0, 1), s.fuel > 0.15, [0.15, 1])
      wind.set(`${view.wind.toFixed(2)} m/s²`, unit(view.wind, -2, 2), null)
      action.textContent = `Engine command: ${ACTION_LABELS[view.action] ?? 'Idle'}`

      attempt.textContent = String(view.episode)
      eps.textContent = `${Math.round(view.epsilon * 100)}%`
      loss.textContent = view.loss === null ? '—' : view.loss.toFixed(3)
      success.textContent = `${Math.round(view.successRate * 100)}%`
      mean.textContent = view.meanReturn.toFixed(1)
      flight.textContent = view.flightReturn.toFixed(1)

      for (const slider of sliders) {
        const value = params[slider.key]
        if (document.activeElement !== slider.input) {
          slider.input.value = String(value)
          slider.read.textContent = slider.format(value)
        }
      }
    },
    setSpeed(speed) {
      speeds.forEach((button) => {
        const on = Number(button.dataset.speed) === speed
        button.classList.toggle('on', on)
        button.setAttribute('aria-pressed', on ? 'true' : 'false')
      })
    },
    setPaused(paused) {
      pause.textContent = paused ? 'Resume' : 'Pause'
      pause.setAttribute('aria-pressed', paused ? 'true' : 'false')
    },
    setNote(text) {
      note.textContent = text
    },
  }
}

const SLIDERS: { key: keyof WorldParams; label: string; min: number; max: number; step: number; format: (value: number) => string }[] = [
  { key: 'gravity', label: 'Gravity', min: 0.5, max: 8, step: 0.01, format: (v) => `${v.toFixed(2)} m/s²` },
  { key: 'mainAcc', label: 'Main engine', min: 1, max: 12, step: 0.1, format: (v) => `${v.toFixed(1)} m/s²` },
  { key: 'rcsAcc', label: 'RCS', min: 0.2, max: 4, step: 0.05, format: (v) => `${v.toFixed(2)} rad/s²` },
  { key: 'burnSeconds', label: 'Fuel load', min: 6, max: 30, step: 0.5, format: (v) => `${v.toFixed(1)} s` },
  { key: 'wind', label: 'Wind', min: -2, max: 2, step: 0.05, format: (v) => `${v.toFixed(2)} m/s²` },
  { key: 'padHalf', label: 'Pad half-width', min: 6, max: 22, step: 0.5, format: (v) => `${v.toFixed(1)} m` },
  { key: 'scatter', label: 'Start scatter', min: 0, max: 1, step: 0.01, format: (v) => v.toFixed(2) },
]

function meter(parent: HTMLElement, label: string) {
  const root = document.createElement('div')
  root.className = 'meter'
  const top = document.createElement('div')
  top.className = 'meter-top'
  const name = document.createElement('span')
  name.textContent = label
  const value = document.createElement('span')
  value.className = 'v'
  top.append(name, value)
  const track = document.createElement('div')
  track.className = 'track'
  const zone = document.createElement('div')
  zone.className = 'zone'
  zone.hidden = true
  const thumb = document.createElement('div')
  thumb.className = 'thumb'
  track.append(zone, thumb)
  root.append(top, track)
  parent.append(root)
  return {
    set(text: string, amount: number, good: boolean | null, zoneRange?: [number, number]) {
      value.textContent = text
      value.className = good === null ? 'v' : good ? 'v good' : 'v bad'
      thumb.style.left = `${amount * 100}%`
      if (zoneRange) {
        zone.hidden = false
        zone.style.left = `${zoneRange[0] * 100}%`
        zone.style.width = `${Math.max(0, zoneRange[1] - zoneRange[0]) * 100}%`
      } else {
        zone.hidden = true
      }
    },
  }
}

function stat(label: string): HTMLSpanElement {
  const row = document.createElement('div')
  row.className = 'stat'
  const name = document.createElement('span')
  name.textContent = label
  const value = document.createElement('span')
  value.className = 'v'
  value.textContent = '—'
  row.append(name, value)
  must<HTMLElement>('stats').append(row)
  return value
}

function unit(value: number, min: number, max: number): number {
  return (Math.max(min, Math.min(max, value)) - min) / (max - min)
}

function must<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id)
  if (!node) throw new Error(`Missing #${id}`)
  return node as T
}
