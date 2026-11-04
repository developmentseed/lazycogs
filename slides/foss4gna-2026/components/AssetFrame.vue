<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ asset: string; title: string }>()
const assets = import.meta.glob('../assets/{hls,southwest}/*.{html,svg,png,webp}', {
  eager: true,
  import: 'default',
  query: '?url',
}) as Record<string, string>

const src = computed(() => assets[`../assets/${props.asset}`])
</script>

<template>
  <iframe
    v-if="src"
    class="asset-frame"
    :src="src"
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
