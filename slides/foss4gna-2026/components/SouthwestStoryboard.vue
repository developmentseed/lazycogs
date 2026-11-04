<script setup lang="ts">
import { computed } from 'vue'

type PassGroup = {
  month: string
  datatake_id: string
  acquisition_datetimes: string[]
  output: string
}

type SouthwestManifest = {
  months: string[]
  pass_groups: PassGroup[]
}

const props = defineProps<{ step: number }>()
const assets = import.meta.glob('../assets/southwest/*.{png,webp}', {
  eager: true,
  import: 'default',
  query: '?url',
}) as Record<string, string>
const manifestModules = import.meta.glob('../assets/southwest/manifest.json', {
  eager: true,
  import: 'default',
}) as Record<string, SouthwestManifest>
const manifest = Object.values(manifestModules)[0]
const months = computed(() => (manifest?.months ?? []).map(month => ({
  month,
  key: `mosaic-${month}.png`,
  label: new Date(`${month}-01T00:00:00Z`).toLocaleString('en-US', {
    month: 'long',
    year: 'numeric',
    timeZone: 'UTC',
  }),
  src: assets[`../assets/southwest/mosaic-${month}.png`],
})))
const passPairs = computed(() => (manifest?.months ?? []).flatMap(month => {
  const monthGroups = manifest.pass_groups.filter(group => group.month === month).slice(0, 2)
  if (monthGroups.length < 2) return []
  const mosaic = months.value.find(item => item.month === month)
  return [{
    month: mosaic?.label ?? month,
    dates: monthGroups.map(group => group.acquisition_datetimes[0]?.split('T')[0] ?? 'Date unavailable').join(' + '),
    passes: monthGroups.map(group => ({
      src: assets[`../assets/southwest/${group.output}`],
      output: group.output,
      date: group.acquisition_datetimes[0]?.split('T')[0] ?? 'Date unavailable',
    })),
  }]
}))
</script>

<template>
  <section class="storyboard" :class="`step-${props.step}`" aria-label="Southwest monthly mosaics and representative source datatakes">
    <figure v-if="props.step === 0" class="hero-mosaic">
      <img v-if="months[1]?.src" :src="months[1].src" :alt="`${months[1].label} Southwest Sentinel-2 mosaic`">
      <figcaption>{{ months[1]?.label }} · 1,200 m display preview</figcaption>
    </figure>
    <div v-else class="mosaic-row" :class="{ stacked: props.step === 1 }">
      <figure v-for="mosaic in months" :key="mosaic.key" class="mosaic-plane">
        <img v-if="mosaic.src" :src="mosaic.src" :alt="`${mosaic.label} Southwest Sentinel-2 mosaic preview`">
        <figcaption>{{ mosaic.label }} · 1,200 m preview</figcaption>
      </figure>
    </div>
    <div v-if="props.step === 2" class="pass-row">
      <figure v-for="pair in passPairs" :key="pair.month">
        <div class="pass-stack">
          <img v-for="pass in pair.passes" :key="pass.output" :src="pass.src" :alt="`Representative ${pair.month} source datatake from ${pass.date}`">
        </div>
        <figcaption><span>{{ pair.month }} · {{ pair.dates }}</span><span>two overlapping datatakes · 1,200 m previews</span></figcaption>
      </figure>
    </div>
    <div class="print-mosaics">
      <figure v-for="mosaic in months" :key="mosaic.key">
        <img v-if="mosaic.src" :src="mosaic.src" :alt="`${mosaic.label} Southwest Sentinel-2 mosaic preview`">
        <figcaption>{{ mosaic.label }} · 1,200 m preview</figcaption>
      </figure>
    </div>
    <p class="story-label">{{ props.step < 1 ? 'One mosaic, many dates' : props.step < 2 ? 'Parallel monthly views' : props.step < 3 ? 'Representative datatakes' : 'Mosaics reassembled by month' }}</p>
  </section>
</template>

<style scoped>
.storyboard {
  position: absolute;
  inset: 0;
  display: grid;
  grid-template-rows: minmax(0, 1fr) auto;
  width: 100%;
  height: 100%;
  padding: 3% 4% 2%;
  overflow: hidden;
  background: #111;
  color: #fff;
  perspective: 70rem;
}
.hero-mosaic {
  display: grid;
  grid-template-rows: minmax(0, 1fr) auto;
  min-width: 0;
  min-height: 0;
  margin: 0;
}
.hero-mosaic img {
  width: 100%;
  height: 100%;
  min-height: 0;
  object-fit: contain;
}
.hero-mosaic figcaption, .mosaic-plane figcaption, .pass-row figcaption {
  padding: .35rem;
  text-align: center;
  font-size: .78rem;
}
.pass-row figcaption { display: grid; gap: .1rem; font-size: .68rem; line-height: 1.1; }
.mosaic-row {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  align-items: center;
  min-width: 0;
  min-height: 0;
  transform-style: preserve-3d;
}
.mosaic-plane {
  display: grid;
  grid-template-rows: minmax(0, 1fr) auto;
  width: 100%;
  height: 88%;
  min-width: 0;
  min-height: 0;
  margin: 0;
}
.mosaic-plane img {
  width: 100%;
  height: 100%;
  min-height: 0;
  object-fit: contain;
}
.stacked .mosaic-plane img {
  transform: rotateY(-16deg) translateZ(var(--depth));
  transition: transform 650ms ease;
}
.stacked .mosaic-plane:nth-child(1) { --depth: -2.5rem; }
.stacked .mosaic-plane:nth-child(2) { --depth: 0; }
.stacked .mosaic-plane:nth-child(3) { --depth: 2.5rem; }
.pass-row {
  position: absolute;
  inset: 12% 5% 12%;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  align-items: center;
  gap: 1%;
  padding: 1%;
  background: #111;
}
.step-2 .mosaic-row { opacity: .18; }
.pass-row figure {
  display: grid;
  grid-template-rows: minmax(0, 1fr) auto;
  width: 100%;
  height: 100%;
  min-width: 0;
  min-height: 0;
  margin: 0;
  outline: 1px solid #fff8;
  background: #111;
}
.pass-stack { position: relative; min-width: 0; min-height: 0; }
.pass-stack img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  min-height: 0;
  object-fit: contain;
  opacity: .72;
}
.print-mosaics { display: none; }
.story-label {
  margin: 0;
  padding: .3rem;
  color: #fff;
  text-align: center;
  font-size: .9rem;
}
@media print {
  .storyboard { color: #111; background: #fff; }
  .hero-mosaic, .mosaic-row, .pass-row, .story-label { display: none; }
  .print-mosaics {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1%;
    min-height: 0;
    align-items: center;
  }
  .print-mosaics figure {
    display: grid;
    grid-template-rows: minmax(0, 1fr) auto;
    min-width: 0;
    min-height: 0;
    height: 100%;
    margin: 0;
  }
  .print-mosaics img {
    width: 100%;
    height: 100%;
    min-height: 0;
    object-fit: contain;
  }
  .print-mosaics figcaption { color: #111; text-align: center; font-size: .7rem; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition-duration: .01ms !important; }
}
</style>
