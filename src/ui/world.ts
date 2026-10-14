import { TRAIL } from '../config'
import { mulberry32 } from '../rng'
import { LEG_HALF, type RocketState } from '../sim/physics'
import type { Outcome } from '../rl/reward'

export interface SceneFrame {
  state: RocketState
  action: number
  fired: boolean
  padHalf: number
  trailX: Float32Array
  trailY: Float32Array
  trailCount: number
  trailHead: number
  episode: number
  mood: Outcome
  paused: boolean
  fixed: boolean
  time: number
  shake: number
  dt: number
  spawnFlame: boolean
  spawnRcs: number
}

interface Camera {
  scale: number
  x: (x: number) => number
  y: (y: number) => number
}

const POOL = 220

export function createScene() {
  const rng = mulberry32(42)
  const stars = Array.from({ length: 90 }, () => ({
    x: rng(),
    y: rng() * 0.72,
    r: rng() < 0.85 ? 0.7 : 1.5,
    phase: rng() * Math.PI * 2,
  }))
  const pebbles = Array.from({ length: 70 }, () => ({
    x: (rng() * 2 - 1) * 70,
    y: -rng() * 2.2,
    r: 0.08 + rng() * 0.16,
  }))
  const ridge = makeRidge(rng)
  const effects = new Effects()

  return {
    burst(state: RocketState, kind: 'land' | 'crash') {
      effects.burst(state.x, 0.2, kind)
    },
    draw(ctx: CanvasRenderingContext2D, w: number, h: number, frame: SceneFrame) {
      const cam = makeCamera(w, h)
      effects.update(frame.dt, frame)
      ctx.save()
      if (frame.shake > 0.02) {
        ctx.translate(Math.sin(frame.time * 46) * frame.shake * 8, Math.cos(frame.time * 37) * frame.shake * 5)
      }
      drawSky(ctx, w, h, cam)
      drawStars(ctx, w, h, stars, frame.time)
      drawMoon(ctx, w, h)
      drawRidge(ctx, cam, ridge)
      drawGround(ctx, w, h, cam, pebbles)
      drawPad(ctx, cam, frame.padHalf, frame.mood)
      drawTrail(ctx, cam, frame)
      drawShadow(ctx, cam, frame.state)
      if (frame.fired && frame.action === 1 && frame.state.y < 10) {
        drawExhaustLight(ctx, cam, frame.state)
      }
      effects.draw(ctx, cam, 'back')
      drawRocket(ctx, cam, frame)
      effects.draw(ctx, cam, 'front')
      ctx.restore()
      drawVignette(ctx, w, h)
      drawHud(ctx, frame)
    },
  }
}

class Effects {
  private readonly x = new Float32Array(POOL)
  private readonly y = new Float32Array(POOL)
  private readonly vx = new Float32Array(POOL)
  private readonly vy = new Float32Array(POOL)
  private readonly life = new Float32Array(POOL)
  private readonly max = new Float32Array(POOL)
  private readonly kind = new Uint8Array(POOL)
  private cursor = 0

  update(dt: number, frame: SceneFrame): void {
    for (let i = 0; i < POOL; i++) {
      if (this.life[i] <= 0) continue
      this.life[i] -= dt
      this.vy[i] -= 1.4 * dt
      this.x[i] += this.vx[i] * dt
      this.y[i] += this.vy[i] * dt
    }
    if (frame.spawnFlame) this.flame(frame.state, frame.time)
    if (frame.spawnRcs === 2 || frame.spawnRcs === 3) this.rcs(frame.state, frame.spawnRcs)
    if (frame.spawnFlame && frame.state.y < 8) this.dust(frame.state, frame.time)
  }

  burst(x: number, y: number, kind: 'land' | 'crash'): void {
    const n = kind === 'land' ? 26 : 34
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2
      const speed = kind === 'land' ? 2 + (i % 5) : 3 + (i % 7)
      this.spawn(
        x,
        y,
        Math.cos(a) * speed,
        Math.sin(a) * speed * 0.6 + 1.5,
        kind === 'land' ? 0.7 : 0.55,
        kind === 'land' ? 3 : 4,
      )
    }
  }

  draw(ctx: CanvasRenderingContext2D, cam: Camera, pass: 'back' | 'front'): void {
    for (let i = 0; i < POOL; i++) {
      if (this.life[i] <= 0) continue
      const back = this.kind[i] === 1 || this.kind[i] === 2
      if (pass === 'back' && !back) continue
      if (pass === 'front' && back) continue
      const t = Math.max(0, this.life[i] / this.max[i])
      const px = cam.x(this.x[i])
      const py = cam.y(this.y[i])
      const radius = (this.kind[i] === 2 ? 2.2 : 1.4) * cam.scale * (0.15 + 0.2 * t)
      ctx.globalAlpha = t
      const kind = this.kind[i]
      ctx.fillStyle =
        kind === 3 ? '#b9fff0' : kind === 4 ? '#ff8a4a' : kind === 5 ? '#d5ecff' : kind === 2 ? '#c9b8a2' : '#ffb15a'
      ctx.beginPath()
      ctx.arc(px, py, Math.max(0.6, radius), 0, Math.PI * 2)
      ctx.fill()
    }
    ctx.globalAlpha = 1
  }

  private flame(state: RocketState, time: number): void {
    const s = Math.sin(state.angle)
    const c = Math.cos(state.angle)
    for (let i = 0; i < 4; i++) {
      const jitter = Math.sin(time * 40 + i * 2) * 0.15
      this.spawn(
        state.x - s * 0.2 + c * jitter,
        state.y - c * 0.2 - s * jitter,
        -s * (7 + i) + state.vx * 0.15,
        -c * (7 + i) + state.vy * 0.15,
        0.28 + i * 0.04,
        1,
      )
    }
  }

  private rcs(state: RocketState, action: number): void {
    const s = Math.sin(state.angle)
    const c = Math.cos(state.angle)
    const side = action === 2 ? 1 : -1
    const ox = state.x + s * 5.1 + side * c * 0.85
    const oy = state.y + c * 5.1 - side * s * 0.85
    this.spawn(ox, oy, side * c * 5, -side * s * 5, 0.22, 5)
  }

  private dust(state: RocketState, time: number): void {
    const spread = Math.sin(time * 23) * 1.4
    this.spawn(state.x + spread, 0.15, spread * 0.8, 1.6 + Math.abs(spread), 0.45, 2)
  }

  private spawn(x: number, y: number, vx: number, vy: number, life: number, kind: number): void {
    let slot = -1
    for (let n = 0; n < 12; n++) {
      const i = (this.cursor + n) % POOL
      if (this.life[i] <= 0) {
        slot = i
        break
      }
    }
    if (slot < 0) slot = this.cursor % POOL
    this.cursor = slot + 1
    this.x[slot] = x
    this.y[slot] = y
    this.vx[slot] = vx
    this.vy[slot] = vy
    this.life[slot] = life
    this.max[slot] = life
    this.kind[slot] = kind
  }
}

function makeCamera(w: number, h: number): Camera {
  const spanX = 108
  const spanY = 58
  const scale = Math.min(w / spanX, h / spanY)
  const viewW = w / scale
  const viewH = h / scale
  const x0 = -viewW / 2
  const y0 = -viewH * 0.2
  return {
    scale,
    x: (x) => (x - x0) * scale,
    y: (y) => (y0 + viewH - y) * scale,
  }
}

function drawSky(ctx: CanvasRenderingContext2D, w: number, h: number, cam: Camera): void {
  const horizon = cam.y(0)
  const sky = ctx.createLinearGradient(0, 0, 0, horizon)
  sky.addColorStop(0, '#070b16')
  sky.addColorStop(0.55, '#1a2744')
  sky.addColorStop(0.82, '#c46a45')
  sky.addColorStop(1, '#e7a06a')
  ctx.fillStyle = sky
  ctx.fillRect(0, 0, w, Math.max(horizon, 0))
  const glow = ctx.createRadialGradient(w * 0.5, horizon, 10, w * 0.5, horizon, w * 0.45)
  glow.addColorStop(0, 'rgba(255, 196, 140, 0.35)')
  glow.addColorStop(1, 'rgba(255, 196, 140, 0)')
  ctx.fillStyle = glow
  ctx.fillRect(0, horizon - h * 0.3, w, h * 0.3)
}

function drawStars(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  stars: { x: number; y: number; r: number; phase: number }[],
  time: number,
): void {
  for (const star of stars) {
    const twinkle = 0.35 + 0.65 * (0.5 + 0.5 * Math.sin(time * 1.7 + star.phase))
    ctx.globalAlpha = twinkle
    ctx.fillStyle = '#f7f4ee'
    ctx.beginPath()
    ctx.arc(star.x * w, star.y * h, star.r, 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.globalAlpha = 1
}

function drawMoon(ctx: CanvasRenderingContext2D, w: number, h: number): void {
  const cx = w * 0.8
  const cy = h * 0.16
  const r = Math.min(w, h) * 0.075
  const glow = ctx.createRadialGradient(cx, cy, r * 0.6, cx, cy, r * 2.4)
  glow.addColorStop(0, 'rgba(243, 230, 207, 0.45)')
  glow.addColorStop(1, 'rgba(243, 230, 207, 0)')
  ctx.fillStyle = glow
  ctx.beginPath()
  ctx.arc(cx, cy, r * 2.4, 0, Math.PI * 2)
  ctx.fill()
  ctx.fillStyle = '#f4ead8'
  ctx.beginPath()
  ctx.arc(cx, cy, r, 0, Math.PI * 2)
  ctx.fill()
  ctx.fillStyle = 'rgba(180, 160, 140, 0.45)'
  ctx.beginPath()
  ctx.arc(cx - r * 0.25, cy - r * 0.1, r * 0.22, 0, Math.PI * 2)
  ctx.arc(cx + r * 0.28, cy + r * 0.22, r * 0.14, 0, Math.PI * 2)
  ctx.fill()
}

function makeRidge(rng: () => number): { x: number; y: number }[] {
  const points: { x: number; y: number }[] = []
  for (let x = -90; x <= 90; x += 4) {
    const far = Math.abs(x) < 22 ? 0 : Math.min(1, (Math.abs(x) - 22) / 16)
    const peak = far * (6 + rng() * 8 + 3 * Math.sin(x * 0.17))
    points.push({ x, y: peak })
  }
  return points
}

function drawRidge(ctx: CanvasRenderingContext2D, cam: Camera, ridge: { x: number; y: number }[]): void {
  ctx.beginPath()
  ctx.moveTo(cam.x(ridge[0].x), cam.y(0))
  for (const point of ridge) ctx.lineTo(cam.x(point.x), cam.y(point.y))
  ctx.lineTo(cam.x(ridge[ridge.length - 1].x), cam.y(0))
  ctx.closePath()
  ctx.fillStyle = '#141924'
  ctx.fill()
}

function drawGround(ctx: CanvasRenderingContext2D, w: number, h: number, cam: Camera, pebbles: { x: number; y: number; r: number }[]): void {
  const horizon = cam.y(0)
  const ground = ctx.createLinearGradient(0, horizon, 0, h)
  ground.addColorStop(0, '#4a4036')
  ground.addColorStop(0.08, '#2a241e')
  ground.addColorStop(1, '#12100e')
  ctx.fillStyle = ground
  ctx.fillRect(0, horizon, w, h - horizon)
  ctx.fillStyle = 'rgba(255,255,255,0.04)'
  for (const pebble of pebbles) {
    ctx.beginPath()
    ctx.arc(cam.x(pebble.x), cam.y(pebble.y), pebble.r * cam.scale, 0, Math.PI * 2)
    ctx.fill()
  }
}

function drawPad(ctx: CanvasRenderingContext2D, cam: Camera, padHalf: number, mood: Outcome): void {
  const x = cam.x(-padHalf)
  const y = cam.y(0.22)
  const width = padHalf * 2 * cam.scale
  const height = Math.max(4, 0.55 * cam.scale)
  ctx.fillStyle = '#1b242e'
  ctx.fillRect(x, y, width, height)
  ctx.fillStyle = mood === 'land' ? '#1c3d36' : '#243246'
  ctx.fillRect(x, cam.y(0.08), width, Math.max(3, 0.16 * cam.scale))
  ctx.strokeStyle = mood === 'land' ? 'rgba(125, 222, 196, 0.8)' : 'rgba(154, 215, 255, 0.7)'
  ctx.lineWidth = 2
  const chevron = 1.7
  ctx.beginPath()
  for (let cx = -padHalf + 1.2; cx < padHalf - 0.4; cx += chevron) {
    ctx.moveTo(cam.x(cx), cam.y(0.02))
    ctx.lineTo(cam.x(cx + 0.45), cam.y(0.1))
    ctx.lineTo(cam.x(cx + 0.9), cam.y(0.02))
  }
  ctx.stroke()
  const beacon = mood === 'land' ? '#7ddec4' : '#ffb15a'
  for (const side of [-1, 1]) {
    const bx = cam.x(side * padHalf)
    const by = cam.y(0.15)
    ctx.fillStyle = beacon
    ctx.shadowColor = beacon
    ctx.shadowBlur = 12
    ctx.beginPath()
    ctx.arc(bx, by, Math.max(3, 0.28 * cam.scale), 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.shadowBlur = 0
}

function drawTrail(ctx: CanvasRenderingContext2D, cam: Camera, frame: SceneFrame): void {
  if (frame.trailCount < 2) return
  ctx.beginPath()
  const start = (frame.trailHead - frame.trailCount + TRAIL) % TRAIL
  for (let i = 0; i < frame.trailCount; i++) {
    const idx = (start + i) % TRAIL
    const px = cam.x(frame.trailX[idx])
    const py = cam.y(frame.trailY[idx])
    if (i === 0) ctx.moveTo(px, py)
    else ctx.lineTo(px, py)
  }
  ctx.strokeStyle = 'rgba(255, 209, 168, 0.45)'
  ctx.lineWidth = 1.5
  ctx.stroke()
}

function drawShadow(ctx: CanvasRenderingContext2D, cam: Camera, state: RocketState): void {
  const altitude = Math.max(0, state.y)
  const fade = Math.max(0, 1 - altitude / 30)
  ctx.fillStyle = `rgba(0, 0, 0, ${0.35 * fade})`
  ctx.beginPath()
  ctx.ellipse(cam.x(state.x), cam.y(0.02), (2.2 + altitude * 0.03) * cam.scale, 0.35 * cam.scale, 0, 0, Math.PI * 2)
  ctx.fill()
}

function drawExhaustLight(ctx: CanvasRenderingContext2D, cam: Camera, state: RocketState): void {
  const x = cam.x(state.x)
  const y = cam.y(0)
  const radius = (8 + Math.max(0, 6 - state.y)) * cam.scale * 0.35
  const glow = ctx.createRadialGradient(x, y, 2, x, y, radius)
  glow.addColorStop(0, 'rgba(255, 170, 70, 0.35)')
  glow.addColorStop(1, 'rgba(255, 170, 70, 0)')
  ctx.fillStyle = glow
  ctx.beginPath()
  ctx.arc(x, y, radius, 0, Math.PI * 2)
  ctx.fill()
}

function drawRocket(ctx: CanvasRenderingContext2D, cam: Camera, frame: SceneFrame): void {
  const state = frame.state
  ctx.save()
  ctx.translate(cam.x(state.x), cam.y(state.y))
  ctx.scale(cam.scale, -cam.scale)
  ctx.rotate(-state.angle)
  if (frame.fired && frame.action === 1) drawFlame(ctx, frame.time)
  drawHull(ctx)
  if (frame.spawnRcs === 2 || frame.spawnRcs === 3) {
    ctx.fillStyle = 'rgba(180, 220, 255, 0.8)'
    const side = frame.spawnRcs === 2 ? 1 : -1
    ctx.beginPath()
    ctx.arc(side * 0.9, 5.1, 0.18, 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.restore()
}

function drawFlame(ctx: CanvasRenderingContext2D, time: number): void {
  const len = 2.4 + Math.sin(time * 34) * 0.45
  const outer = ctx.createLinearGradient(0, 0.1, 0, -len)
  outer.addColorStop(0, 'rgba(255, 250, 220, 0.95)')
  outer.addColorStop(0.35, 'rgba(255, 170, 40, 0.9)')
  outer.addColorStop(1, 'rgba(255, 70, 0, 0)')
  ctx.fillStyle = outer
  ctx.beginPath()
  ctx.moveTo(-0.32, 0.12)
  ctx.quadraticCurveTo(Math.sin(time * 50) * 0.2, -len * 0.45, 0, -len)
  ctx.quadraticCurveTo(Math.sin(time * 42) * 0.15, -len * 0.4, 0.32, 0.12)
  ctx.fill()
}

function drawHull(ctx: CanvasRenderingContext2D): void {
  const metal = ctx.createLinearGradient(-1.1, 0, 1.1, 0)
  metal.addColorStop(0, '#6a7382')
  metal.addColorStop(0.35, '#d5dbe6')
  metal.addColorStop(0.52, '#f7f9fc')
  metal.addColorStop(0.74, '#b4becc')
  metal.addColorStop(1, '#596273')
  ctx.fillStyle = metal
  ctx.beginPath()
  ctx.moveTo(-0.86, 0.72)
  ctx.lineTo(-0.86, 4.85)
  ctx.quadraticCurveTo(-0.86, 5.25, -0.42, 5.5)
  ctx.lineTo(0, 6.65)
  ctx.lineTo(0.42, 5.5)
  ctx.quadraticCurveTo(0.86, 5.25, 0.86, 4.85)
  ctx.lineTo(0.86, 0.72)
  ctx.quadraticCurveTo(0.86, 0.46, 0.5, 0.46)
  ctx.lineTo(-0.5, 0.46)
  ctx.quadraticCurveTo(-0.86, 0.46, -0.86, 0.72)
  ctx.fill()

  ctx.fillStyle = '#c4622d'
  ctx.beginPath()
  ctx.moveTo(-0.86, 1.55)
  ctx.lineTo(-1.75, 0.32)
  ctx.lineTo(-0.86, 0.55)
  ctx.fill()
  ctx.beginPath()
  ctx.moveTo(0.86, 1.55)
  ctx.lineTo(1.75, 0.32)
  ctx.lineTo(0.86, 0.55)
  ctx.fill()

  ctx.fillStyle = '#e2a15a'
  ctx.fillRect(-0.86, 2.15, 1.72, 0.16)

  ctx.fillStyle = '#3a342e'
  ctx.beginPath()
  ctx.moveTo(-0.36, 0.48)
  ctx.lineTo(-0.5, 0.02)
  ctx.lineTo(0.5, 0.02)
  ctx.lineTo(0.36, 0.48)
  ctx.fill()

  ctx.fillStyle = '#163248'
  ctx.beginPath()
  ctx.arc(0, 4.15, 0.34, 0, Math.PI * 2)
  ctx.fill()
  ctx.fillStyle = 'rgba(255,255,255,0.7)'
  ctx.beginPath()
  ctx.arc(-0.1, 4.28, 0.09, 0, Math.PI * 2)
  ctx.fill()

  ctx.strokeStyle = '#e6ebf3'
  ctx.lineWidth = 0.11
  ctx.lineCap = 'round'
  ctx.beginPath()
  ctx.moveTo(-0.55, 0.75)
  ctx.lineTo(-LEG_HALF, 0)
  ctx.moveTo(0.55, 0.75)
  ctx.lineTo(LEG_HALF, 0)
  ctx.stroke()
  ctx.fillStyle = '#f2f5f8'
  ctx.fillRect(-LEG_HALF - 0.28, -0.08, 0.56, 0.16)
  ctx.fillRect(LEG_HALF - 0.28, -0.08, 0.56, 0.16)
}

function drawVignette(ctx: CanvasRenderingContext2D, w: number, h: number): void {
  const vig = ctx.createRadialGradient(w / 2, h / 2, h * 0.25, w / 2, h / 2, h * 0.72)
  vig.addColorStop(0, 'rgba(0,0,0,0)')
  vig.addColorStop(1, 'rgba(0,0,0,0.38)')
  ctx.fillStyle = vig
  ctx.fillRect(0, 0, w, h)
}

function drawHud(ctx: CanvasRenderingContext2D, frame: SceneFrame): void {
  ctx.fillStyle = 'rgba(237, 241, 247, 0.92)'
  ctx.font = '600 13px "Avenir Next", "Segoe UI", sans-serif'
  ctx.textAlign = 'left'
  ctx.textBaseline = 'top'
  ctx.fillText(`ATTEMPT ${frame.episode}`, 16, 16)
  ctx.fillStyle = 'rgba(226, 214, 190, 0.75)'
  ctx.font = '12px "Avenir Next", "Segoe UI", sans-serif'
  ctx.fillText(frame.paused ? 'Paused' : frame.fixed ? 'Landing weights' : 'Training', 16, 36)
}
