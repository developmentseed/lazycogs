<script setup lang="ts">
import { useNav } from '@slidev/client'
import { sharedState } from '@slidev/client/state/shared.ts'
import { onMounted, onUnmounted, useTemplateRef, watch } from 'vue'
import { config, laser, laserDrag } from './laser'

const { isPresenter } = useNav()

const { color: COLOR, width: WIDTH, fade: FADE } = config
const GAP = 500 // ms between points that means "new stroke"
const LAYERS = 50
// Fraction of the remaining distance the head covers per frame while chasing a
// synced cursor. 0-1; only affects presenter/viewer mirroring, not local moves.
const SYNC_EASE = 1

const el = useTemplateRef<HTMLCanvasElement>('canvas')
let pts: { x: number, y: number, t: number }[] = []
let target: { x: number, y: number } | null = null // latest synced cursor
let head: { x: number, y: number } | null = null // what we've drawn up to
let raf = 0

function draw() {
  const canvas = el.value
  if (!canvas)
    return
  const ctx = canvas.getContext('2d')!
  const dpr = devicePixelRatio
  const w = canvas.clientWidth
  const h = canvas.clientHeight
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr
    canvas.height = h * dpr
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, w, h)

  chase()

  const now = performance.now()
  pts = pts.filter(p => now - p.t < FADE)
  const life = pts.map(p => 1 - (now - p.t) / FADE)

  // One stroke per layer, each covering pts[start..head]. A single stroke never
  // blends with itself, so the trail has no double-drawn joints; the gradient
  // comes from layers stacking up towards the head.
  ctx.lineCap = ctx.lineJoin = 'round'
  let below = 0 // alpha already laid down by the layers underneath
  for (let k = 0; k < LAYERS && below < 0.99; k++) {
    const start = Math.floor((pts.length - 1) * k / LAYERS)
    const want = life[start] // what the cumulative alpha should be at this band
    const alpha = (want - below) / (1 - below)
    below = want
    if (!(alpha > 0.002))
      continue
    ctx.globalAlpha = alpha
    ctx.strokeStyle = COLOR
    ctx.lineWidth = WIDTH * (0.15 + 0.85 * want)
    ctx.beginPath()
    ctx.moveTo(pts[start].x, pts[start].y)
    for (let i = start + 1; i < pts.length; i++) {
      if (pts[i].t - pts[i - 1].t > GAP)
        ctx.moveTo(pts[i].x, pts[i].y)
      else
        ctx.lineTo(pts[i].x, pts[i].y)
    }
    ctx.stroke()
  }

  ctx.globalAlpha = 1
  raf = (pts.length || target) ? requestAnimationFrame(draw) : 0
}

// The synced cursor arrives in bursts: vite-plugin-vue-server-ref debounces
// state by 10ms and a continuous mousemove keeps resetting that timer. So don't
// draw the samples, chase them — one eased step per frame turns the bursts into
// smooth motion, a few frames behind the presenter.
function chase() {
  if (!target)
    return
  const last = pts.at(-1)
  if (!head || !last || performance.now() - last.t > GAP)
    head = target // new stroke, start where the pointer is
  head = {
    x: head.x + (target.x - head.x) * SYNC_EASE,
    y: head.y + (target.y - head.y) * SYNC_EASE,
  }
  if (Math.hypot(target.x - head.x, target.y - head.y) < 0.3)
    target = null
  push(head.x, head.y)
}

function push(x: number, y: number) {
  pts.push({ x, y, t: performance.now() })
  if (!raf)
    raf = requestAnimationFrame(draw)
}

function onMove(e: PointerEvent) {
  const canvas = el.value
  if (!canvas)
    return
  const dragging = e.shiftKey && !!(e.buttons & 1)
  if (isPresenter.value) {
    // the synced cursor carries no modifier keys, so publish the gesture itself
    if (laserDrag.value !== dragging)
      laserDrag.value = dragging
    return
  }
  // toolbar toggle: just move the pointer. Otherwise hold shift and drag.
  if (!(laser.value || dragging))
    return
  // the slide container is CSS-scaled; map screen px back to canvas px
  const rect = canvas.getBoundingClientRect()
  const scale = canvas.clientWidth / (rect.width || 1)
  push((e.clientX - rect.left) * scale, (e.clientY - rect.top) * scale)
}

// Slidev already syncs the presenter's cursor (as % of the slide) to every
// window, so the presenter view and the audience view both draw from it.
watch(() => sharedState.cursor, (c) => {
  const canvas = el.value
  if (!c || !canvas || !(laser.value || laserDrag.value))
    return
  target = { x: c.x / 100 * canvas.clientWidth, y: c.y / 100 * canvas.clientHeight }
  if (!raf)
    raf = requestAnimationFrame(draw)
})

// shift+drag would otherwise advance the slide
function swallow(e: MouseEvent) {
  if (e.shiftKey) {
    e.stopPropagation()
    e.preventDefault()
  }
}

onMounted(() => {
  addEventListener('pointermove', onMove)
  addEventListener('click', swallow, true)
})
onUnmounted(() => {
  removeEventListener('pointermove', onMove)
  removeEventListener('click', swallow, true)
  cancelAnimationFrame(raf)
})
</script>

<template>
  <canvas ref="canvas" class="pointer-events-none fixed inset-0 z-100 h-full w-full" />
</template>
