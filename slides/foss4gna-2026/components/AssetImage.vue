<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ asset: string; alt: string }>()
const assets = import.meta.glob('../assets/{hls,southwest-notebook,stac-geoparquet}/*.{svg,png,webp}', {
  eager: true,
  import: 'default',
  query: '?url',
}) as Record<string, string>

const src = computed(() => assets[`../assets/${props.asset}`])
</script>

<template>
  <img v-if="src" :src="src" :alt="alt" />
  <div v-else class="asset-unavailable">Verified asset unavailable: {{ asset }}</div>
</template>

<style scoped>
img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.asset-unavailable {
  display: grid;
  height: 100%;
  place-items: center;
  color: #8b2b19;
  border: 1px dashed currentColor;
  font: 0.7rem ui-monospace, monospace;
}
</style>
