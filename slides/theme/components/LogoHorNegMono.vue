<script setup lang="ts">
import Logo from './Logo.vue'

interface Props {
  position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right' | 'center'
  height?: string | number
  width?: string | number
}

const props = withDefaults(defineProps<Props>(), {
  position: 'top-right',
  height: '24px',
  width: 'auto'
})
</script>

<template>
  <Logo
    src="/images/logos/hor--neg-mono@2x.png"
    alt="DevelopmentSeed Logo"
    :position="position"
    :height="height"
    :width="width"
  />
</template>
