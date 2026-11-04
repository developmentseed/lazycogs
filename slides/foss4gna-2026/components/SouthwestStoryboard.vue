<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ step: number }>()
const assets = import.meta.glob('../assets/southwest/*.{png,webp}', {
  eager: true,
  import: 'default',
  query: '?url',
}) as Record<string, string>
const months = [
  { key: 'mosaic-2025-03.png', label: 'March 2025' },
  { key: 'mosaic-2025-07.png', label: 'July 2025' },
  { key: 'mosaic-2025-11.png', label: 'November 2025' },
]
const mosaics = computed(() => months.map(month => ({
  ...month,
  src: assets[`../assets/southwest/${month.key}`],
})))
const passes = computed(() => Object.entries(assets)
  .filter(([path]) => /southwest\/datatake-2025-(03|07|11)-1\.png$/.test(path))
  .map(([path, src]) => {
    const filename = path.split('/').pop() ?? ''
    const dates: Record<string, string> = {
      'datatake-2025-03-1.png': 'Acquisition date in manifest',
      'datatake-2025-07-1.png': 'Acquisition date in manifest',
      'datatake-2025-11-1.png': 'Acquisition date in manifest',
    }
    return { src, label: `${dates[filename] ?? 'Acquisition date in manifest'} · representative datatake` }
  }))
</script>

<template>
  <section class="storyboard" :class="`step-${step}`" aria-label="Southwest imagery mosaic and representative source passes">
    <figure v-if="step === 0" class="hero-mosaic">
      <img v-if="mosaics[1]?.src" :src="mosaics[1].src" alt="July 2025 Southwest Sentinel-2 mosaic">
      <figcaption>Southwest · monthly Sentinel-2 mosaic</figcaption>
    </figure>
    <div v-else class="mosaic-row" :class="{ stacked: step === 1 || step === 2 }">
      <figure v-for="mosaic in mosaics" :key="mosaic.key" class="mosaic-plane">
        <img v-if="mosaic.src" :src="mosaic.src" :alt="`${mosaic.label} Southwest Sentinel-2 mosaic`">
        <figcaption>{{ mosaic.label }}</figcaption>
      </figure>
    </div>
    <div v-if="step === 2" class="pass-row">
      <figure v-for="pass in passes" :key="pass.src">
        <img :src="pass.src" :alt="`Representative orbit pass ${pass.label}`">
        <figcaption>{{ pass.label }} · representative pass selection</figcaption>
      </figure>
    </div>
    <div class="print-mosaics">
      <figure v-for="mosaic in mosaics" :key="mosaic.key">
        <img v-if="mosaic.src" :src="mosaic.src" :alt="`${mosaic.label} Southwest Sentinel-2 mosaic`">
        <figcaption>{{ mosaic.label }}</figcaption>
      </figure>
    </div>
    <p class="story-label">{{ step < 1 ? 'One mosaic, many dates' : step < 2 ? 'Monthly views reveal time' : step < 3 ? 'Representative source passes' : 'Mosaics reassembled by month' }}</p>
  </section>
</template>

<style scoped>
.storyboard {
  position: absolute;
  inset: 0;
  display: grid;
  grid-template-rows: 1fr auto;
  width: 100%;
  height: 100%;
  padding: 2.3rem 3rem 1rem;
  overflow: hidden;
  background: #111;
  color: #fff;
}
.mosaic-row {
  display: flex;
  justify-content: center;
  align-items: center;
  perspective: 1000px;
}
.mosaic-plane {
  width: min(76vw, 1000px);
  height: 68vh;
  margin: 0;
  transition: transform 700ms ease, width 700ms ease, opacity 500ms ease;
}
.mosaic-plane img {
  display: block;
  width: 100%;
  height: calc(100% - 2rem);
  object-fit: contain;
}
.mosaic-plane figcaption {
  text-align: center;
  font-size: 1rem;
  opacity: 0;
}
.hero-mosaic { display: grid; grid-template-rows: 1fr auto; min-height: 0; width: 100%; margin: 0; }
.hero-mosaic img { width: 100%; height: 100%; min-height: 0; object-fit: contain; }
.hero-mosaic figcaption { text-align: center; font-size: 1rem; }
.stacked .mosaic-plane {
  width: 30%;
  height: 55vh;
  transform: rotateY(-24deg) translateZ(calc(var(--i, 0) * 10px));
}
.stacked .mosaic-plane:nth-child(2) { transform: rotateY(0deg) translateZ(40px); }
.stacked .mosaic-plane:nth-child(3) { transform: rotateY(24deg) translateZ(0); }
.stacked .mosaic-plane figcaption { opacity: 1; }
.pass-row {
  position: absolute;
  inset: 8rem 4rem 5rem;
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 1rem;
  pointer-events: none;
}
.pass-row figure {
  width: 22%;
  height: 45vh;
  margin: 0;
  padding: 0.35rem;
  background: #111c;
  outline: 1px solid #fff8;
}
.pass-row img { width: 100%; height: calc(100% - 2rem); object-fit: contain; }
.pass-row figcaption { font-size: 0.72rem; text-align: center; }
.print-mosaics { display: none; }
.story-label { margin: 0; color: #fff; text-align: center; font-size: 1.15rem; }
.step-2 .mosaic-row, .step-2 .pass-row { opacity: 0.78; }
.step-3 .mosaic-plane { transform: none !important; }
.step-3 .pass-row { display: none; }
@media print {
  .storyboard { color: #111; background: #fff; }
  .hero-mosaic, .mosaic-row, .story-label { display: none; }
  .print-mosaics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; height: 100%; align-items: center; }
  .print-mosaics figure { min-width: 0; margin: 0; }
  .print-mosaics img { width: 100%; height: 45vh; object-fit: contain; }
  .print-mosaics figcaption { color: #111; text-align: center; }
  .pass-row { display: none; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition-duration: 0.01ms !important; }
}
</style>
