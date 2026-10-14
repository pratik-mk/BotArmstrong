import { ACTION_LABELS, OBS_LABELS } from '../config'
import { clamp } from '../rng'

export interface NetSnap {
  input: ArrayLike<number>
  h1: ArrayLike<number>
  h2: ArrayLike<number>
  q: ArrayLike<number>
  action: number
  exploring: boolean
}

export function drawNetwork(ctx: CanvasRenderingContext2D, w: number, h: number, snap: NetSnap): void {
  ctx.clearRect(0, 0, w, h)
  ctx.fillStyle = '#12151c'
  ctx.fillRect(0, 0, w, h)

  const split = Math.min(168, w * 0.46)
  const inputBottom = h * 0.64
  const rowH = inputBottom / OBS_LABELS.length

  ctx.font = '11px "Avenir Next", "Segoe UI", sans-serif'
  ctx.textBaseline = 'middle'
  for (let i = 0; i < OBS_LABELS.length; i++) {
    const y = rowH * i + rowH / 2
    const value = snap.input[i] ?? 0
    ctx.fillStyle = '#9aa3b5'
    ctx.textAlign = 'left'
    ctx.fillText(OBS_LABELS[i], 8, y)
    const trackX = 42
    const trackW = split - trackX - 10
    const mid = trackX + trackW / 2
    ctx.fillStyle = '#262b36'
    roundRect(ctx, trackX, y - 3, trackW, 6, 3)
    ctx.fill()
    const mag = clamp(value / 1.6, -1, 1)
    ctx.fillStyle = value >= 0 ? '#e2a15a' : '#7eb6ff'
    const barW = Math.abs(mag) * (trackW / 2)
    roundRect(ctx, mag >= 0 ? mid : mid - barW, y - 3, Math.max(1, barW), 6, 3)
    ctx.fill()
  }

  drawHeat(ctx, split + 8, 8, w - split - 16, inputBottom * 0.46, snap.h1, 'Hidden 1')
  drawHeat(ctx, split + 8, 16 + inputBottom * 0.46, w - split - 16, inputBottom * 0.46 - 8, snap.h2, 'Hidden 2')

  const qTop = inputBottom + 8
  const qH = (h - qTop - 8) / ACTION_LABELS.length
  let maxAbs = 0.5
  for (let i = 0; i < snap.q.length; i++) maxAbs = Math.max(maxAbs, Math.abs(snap.q[i]))
  ctx.font = '11px "Avenir Next", "Segoe UI", sans-serif'
  for (let i = 0; i < ACTION_LABELS.length; i++) {
    const y = qTop + qH * i
    const chosen = i === snap.action
    ctx.fillStyle = chosen ? 'rgba(226,161,90,0.14)' : 'transparent'
    ctx.fillRect(8, y + 1, w - 16, qH - 2)
    ctx.fillStyle = chosen ? '#ffd7a8' : '#9aa3b5'
    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'
    ctx.fillText(ACTION_LABELS[i], 12, y + qH / 2)
    const origin = 78
    const room = w - origin - 54
    const q = snap.q[i] ?? 0
    const len = (q / maxAbs) * (room / 2)
    ctx.strokeStyle = '#343b4a'
    ctx.beginPath()
    ctx.moveTo(origin + room / 2, y + 4)
    ctx.lineTo(origin + room / 2, y + qH - 4)
    ctx.stroke()
    ctx.fillStyle = q >= 0 ? '#7ddec4' : '#ff6d7d'
    const x = len >= 0 ? origin + room / 2 : origin + room / 2 + len
    ctx.fillRect(x, y + qH / 2 - 4, Math.max(1, Math.abs(len)), 8)
    ctx.fillStyle = '#edf1f7'
    ctx.textAlign = 'right'
    ctx.fillText(q.toFixed(1), w - 12, y + qH / 2)
    if (chosen) {
      ctx.strokeStyle = snap.exploring ? '#9aa3b5' : '#e2a15a'
      ctx.setLineDash(snap.exploring ? [3, 3] : [])
      ctx.strokeRect(8.5, y + 1.5, w - 17, qH - 3)
      ctx.setLineDash([])
    }
  }
}

function drawHeat(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  values: ArrayLike<number>,
  label: string,
): void {
  ctx.fillStyle = '#9aa3b5'
  ctx.font = '10px "Avenir Next", "Segoe UI", sans-serif'
  ctx.textAlign = 'left'
  ctx.textBaseline = 'top'
  ctx.fillText(label, x, y)
  const top = y + 14
  const cols = 16
  const rows = 2
  const gap = 3
  const cellW = (w - gap * (cols - 1)) / cols
  const cellH = (h - 14 - gap) / rows
  let max = 0.35
  for (let i = 0; i < values.length; i++) max = Math.max(max, values[i])
  for (let i = 0; i < values.length; i++) {
    const col = i % cols
    const row = (i / cols) | 0
    const t = clamp(values[i] / max, 0, 1)
    ctx.fillStyle = heatColor(t)
    roundRect(ctx, x + col * (cellW + gap), top + row * (cellH + gap), cellW, cellH, 2)
    ctx.fill()
  }
}

function heatColor(t: number): string {
  const r = Math.round(28 + (255 - 28) * t)
  const g = Math.round(34 + (196 - 34) * t)
  const b = Math.round(48 + (120 - 48) * Math.sin(t * Math.PI))
  return `rgb(${r}, ${g}, ${b})`
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  const radius = Math.min(r, w / 2, h / 2)
  ctx.beginPath()
  ctx.moveTo(x + radius, y)
  ctx.arcTo(x + w, y, x + w, y + h, radius)
  ctx.arcTo(x + w, y + h, x, y + h, radius)
  ctx.arcTo(x, y + h, x, y, radius)
  ctx.arcTo(x, y, x + w, y, radius)
  ctx.closePath()
}
