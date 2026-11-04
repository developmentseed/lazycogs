<script setup lang="ts">
import qrCode from '../assets/deck/lazycogs-docs-qr.svg?url'

const props = defineProps<{ value: string }>()
</script>

<template>
  <img class="qr-code" :src="qrCode" :alt="`QR code for ${props.value}`">
</template>

<style scoped>
.qr-code { width: 8rem; height: 8rem; }
</style>
