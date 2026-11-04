<script setup lang="ts">
import { computed } from 'vue'

type OverviewEvidence = {
  month: string
  item_id: string
  source_epsg: string
  source_pixel_size_m: number
  overview_resolutions_m: number[]
  target_resolution_in_source_crs_m: number
  selected_overview_resolution_m: number | null
  decision: string
}

type SouthwestManifest = {
  months: string[]
  overview_evidence: OverviewEvidence[]
}

const manifestModules = import.meta.glob('../assets/southwest/manifest.json', {
  eager: true,
  import: 'default',
}) as Record<string, SouthwestManifest>
const manifest = Object.values(manifestModules)[0]
const evidence = computed(() => (manifest?.months ?? []).flatMap(month => {
  const item = manifest?.overview_evidence.find(candidate => candidate.month === month)
  return item ? [item] : []
}))
</script>

<template>
  <section class="overview-evidence" aria-label="Measured source COG overview levels">
    <div v-for="item in evidence" :key="item.month" class="overview-row">
      <div class="overview-heading">{{ item.month }} · EPSG:{{ item.source_epsg }}</div>
      <div class="overview-levels" aria-label="Source pixel and overview resolutions">
        <span class="overview-level native" :style="{ width: '100%' }">{{ item.source_pixel_size_m }} m source</span>
        <span
          v-for="resolution in item.overview_resolutions_m"
          :key="resolution"
          class="overview-level"
          :class="{ selected: resolution === item.selected_overview_resolution_m }"
          :style="{ width: `${Math.max(24, 100 * Math.sqrt(item.source_pixel_size_m / resolution))}%` }"
        >{{ resolution }} m{{ resolution === item.selected_overview_resolution_m ? ' · selected' : '' }}</span>
      </div>
      <div class="overview-target">300 m output → {{ item.target_resolution_in_source_crs_m.toFixed(1) }} m in source CRS<br>{{ item.decision }}</div>
    </div>
    <p v-if="!evidence.length" class="overview-pending">Source overview evidence awaits the reviewed manifest.</p>
  </section>
</template>

<style scoped>
.overview-evidence {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  align-items: center;
  gap: 1%;
  min-width: 0;
  min-height: 0;
  padding: .3rem 0;
  overflow: hidden;
}
.overview-row {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  min-width: 0;
  min-height: 0;
  height: 100%;
  padding: .3rem;
  border: 1px solid #aaa;
  font-size: .58rem;
}
.overview-heading { font-weight: 700; }
.overview-levels {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: .08rem;
  min-height: 0;
  overflow: hidden;
}
.overview-level {
  display: block;
  max-width: 100%;
  padding: .08rem .2rem;
  background: #e8e8e8;
  text-align: center;
  white-space: nowrap;
}
.overview-level.native { background: #d8d8d8; }
.overview-level.selected { background: #cf3f02; color: #fff; font-weight: 700; }
.overview-target { color: #4d4d4d; font-size: .5rem; }
.overview-pending { grid-column: 1 / -1; text-align: center; }
</style>
