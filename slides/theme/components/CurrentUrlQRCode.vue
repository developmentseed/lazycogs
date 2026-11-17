<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { resolveAssetUrl } from '../utils/assets'

interface Props {
  url?: string
  type?: 'png' | 'svg'
  image?: string
  width?: number
  height?: number
  dotsOptions?: Record<string, any>
  fullWidth?: boolean
  includeSlideNumber?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  type: 'png',
  image: '/images/logos/symbol--pos-mono@2x.png',
  width: 200,
  height: 200,
  dotsOptions: () => ({ type: 'classy-rounded', color: 'black' }),
  fullWidth: false,
  includeSlideNumber: false
})

const currentUrl = ref('')

onMounted(() => {
  let url = window.location.href

  // Strip slide number from URL by default (e.g., /14 -> /)
  if (!props.includeSlideNumber) {
    // Remove trailing slide number from the path
    url = url.replace(/\/\d+$/, '')
  }

  currentUrl.value = url
})

const qrCodeUrl = computed(() => props.url || currentUrl.value)

// When fullWidth is true, use a large pixel value that will be constrained by CSS
const computedWidth = computed(() => props.fullWidth ? 400 : props.width)
const computedHeight = computed(() => props.fullWidth ? 400 : props.height)

// Resolve the image URL to work with Vite's BASE_URL
const resolvedImage = computed(() => resolveAssetUrl(props.image))
</script>

<template>
  <a class="qrcode-container" :class="{ 'full-width': fullWidth }">
    <QRCode
      v-if="qrCodeUrl"
      :type="type"
      :image="resolvedImage"
      :width="computedWidth"
      :height="computedHeight"
      :data="qrCodeUrl"
      :dotsOptions="dotsOptions"
    />
  </a>
</template>

<style scoped>
.qrcode-container {
  display: flex;
  justify-content: center;
  align-items: center;
}

.qrcode-container.full-width {
  width: 100%;
}

.qrcode-container.full-width :deep(canvas),
.qrcode-container.full-width :deep(img),
.qrcode-container.full-width :deep(svg) {
  max-width: 100%;
  width: 100% !important;
  height: auto !important;
}
</style>
