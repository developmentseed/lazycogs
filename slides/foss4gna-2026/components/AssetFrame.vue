<script setup lang="ts">
import { computed } from 'vue'
import xarrayStyles from '../assets/deck/xarray.css?raw'

const props = defineProps<{ asset: string; title: string }>()
const assets = import.meta.glob('../assets/{hls,southwest-notebook}/*.html', {
  eager: true,
  import: 'default',
  query: '?raw',
}) as Record<string, string>

// Keep the original repr intact; apply the documentation's fixes inside its frame.
const document = computed(() => {
  const html = assets[`../assets/${props.asset}`]
  return html?.replace('</body>', `<style>${xarrayStyles}</style></body>`)
})
</script>

<template>
  <iframe
    v-if="document"
    class="asset-frame"
    :srcdoc="document"
    :title="title"
    loading="eager"
    sandbox="allow-same-origin"
    scrolling="no"
  />
  <div v-else class="asset-unavailable">Verified asset unavailable: {{ asset }}</div>
</template>

<style scoped>
.asset-frame,
.asset-unavailable {
  width: 100%;
  height: 100%;
  min-height: 10rem;
  border: 0;
  background: #fff;
}
.asset-unavailable {
  display: grid;
  place-items: center;
  color: #8b2b19;
  border: 1px dashed currentColor;
  font: 0.7rem ui-monospace, monospace;
}
</style>
