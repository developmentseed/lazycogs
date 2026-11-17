<script setup lang="ts">
import { computed } from 'vue'
import { resolveAssetUrl } from '../utils/assets'

interface Props {
  src?: string
  alt?: string
  width?: string | number
  height?: string | number
  position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right' | 'center'
}

const props = withDefaults(defineProps<Props>(), {
  src: '/logo.png',
  alt: 'Company Logo',
  width: 'auto',
  height: '60px',
  position: 'top-right'
})

const positionClasses = computed(() => {
  const positions = {
    'top-left': 'top-6 left-6',
    'top-right': 'top-6 right-6',
    'bottom-left': 'bottom-6 left-6',
    'bottom-right': 'bottom-6 right-6',
    'center': 'top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2'
  }
  return positions[props.position] || positions['top-right']
})

const sizeStyle = computed(() => ({
  width: typeof props.width === 'number' ? `${props.width}px` : props.width,
  height: typeof props.height === 'number' ? `${props.height}px` : props.height
}))
</script>

<template>
  <div class="logo-container absolute z-10" :class="positionClasses">
    <img
      :src="resolveAssetUrl(src)"
      :alt="alt"
      :style="sizeStyle"
      class="logo-image"
    />
  </div>
</template>

<style scoped>
.logo-image {
  object-fit: contain;
}
</style>
