export function drawCharts(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  returns: number[],
  lands: number[],
  epsilons: number[],
  losses: number[],
): void {
  ctx.clearRect(0, 0, w, h)
  ctx.fillStyle = '#12151c'
  ctx.fillRect(0, 0, w, h)
  const band = h / 4
  plot(ctx, 0, 0, w, band, returns, '#e2a15a', 'Return', false)
  plot(ctx, 0, band, w, band, rolling(lands, 20), '#7ddec4', 'Success', true)
  plot(ctx, 0, band * 2, w, band, epsilons.map((value) => value * 100), '#8eb4ff', 'Explore', true)
  plot(ctx, 0, band * 3, w, h - band * 3, losses, '#e07a7a', 'Loss', false)
}

function rolling(flags: number[], window: number): number[] {
  const out: number[] = []
  let sum = 0
  for (let i = 0; i < flags.length; i++) {
    sum += flags[i]
    if (i >= window) sum -= flags[i - window]
    const n = Math.min(i + 1, window)
    out.push((100 * sum) / n)
  }
  return out
}

function plot(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  values: number[],
  color: string,
  label: string,
  percent: boolean,
): void {
  ctx.fillStyle = '#9aa3b5'
  ctx.font = '11px "Avenir Next", "Segoe UI", sans-serif'
  ctx.textAlign = 'left'
  ctx.textBaseline = 'top'
  ctx.fillText(label, x + 8, y + 6)
  const left = 8
  const right = w - 8
  const top = y + 22
  const bottom = y + h - 8
  ctx.strokeStyle = 'rgba(255,255,255,0.06)'
  ctx.beginPath()
  ctx.moveTo(left, top)
  ctx.lineTo(right, top)
  ctx.moveTo(left, bottom)
  ctx.lineTo(right, bottom)
  ctx.stroke()
  if (values.length < 2) {
    ctx.fillStyle = '#6d7688'
    ctx.fillText('Waiting for flights', left, top + 8)
    return
  }
  let min = percent ? 0 : values[0]
  let max = percent ? 100 : values[0]
  if (!percent) {
    for (const value of values) {
      min = Math.min(min, value)
      max = Math.max(max, value)
    }
    if (max - min < 1) {
      min -= 1
      max += 1
    }
  }
  ctx.beginPath()
  for (let i = 0; i < values.length; i++) {
    const px = left + (i / (values.length - 1)) * (right - left)
    const py = bottom - ((values[i] - min) / (max - min)) * (bottom - top)
    if (i === 0) ctx.moveTo(px, py)
    else ctx.lineTo(px, py)
  }
  ctx.strokeStyle = color
  ctx.lineWidth = 1.6
  ctx.stroke()
  ctx.lineWidth = 1
  const last = values[values.length - 1]
  ctx.fillStyle = color
  ctx.textAlign = 'right'
  ctx.fillText(percent ? `${last.toFixed(0)}%` : last.toFixed(1), right, y + 6)
}
