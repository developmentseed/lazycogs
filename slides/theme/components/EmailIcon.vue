<script setup lang="ts">
interface Props {
  size?: string | number
  color?: string
}

const props = withDefaults(defineProps<Props>(), {
  size: '20',
  color: 'currentColor'
})
</script>

<template>
  <svg xmlns="http://www.w3.org/2000/svg" :width="size" :height="size" viewBox="0 0 24 24" :fill="color">
    <path d="M0 3v18h24v-18h-24zm21.518 2l-9.518 7.713-9.518-7.713h19.036zm-19.518 14v-11.817l10 8.104 10-8.104v11.817h-20z"/>
  </svg>
</template>
