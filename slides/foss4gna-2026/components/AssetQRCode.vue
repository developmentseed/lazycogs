<script setup lang="ts">
import { computed } from 'vue'
import docsQrCode from '../assets/deck/lazycogs-docs-qr.svg?url'
import hlsQrCode from '../assets/deck/hls-index-qr.svg?url'

const props = defineProps<{ value: string }>()
const codes: Record<string, string> = {
  'https://developmentseed.org/lazycogs/': docsQrCode,
  'https://maap-project.github.io/hls-stac-geoparquet-archive/': hlsQrCode,
}
const qrCode = computed(() => codes[props.value])
</script>

<template>
  <img v-if="qrCode" class="qr-code" :src="qrCode" :alt="`QR code for ${props.value}`">
</template>

<style scoped>
.qr-code { width: 8rem; height: 8rem; }
</style>
