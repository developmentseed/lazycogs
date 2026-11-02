<script setup lang="ts">
const props = withDefaults(
  defineProps<{
    color?: string;
    width?: string;
    height?: string;
    opacity?: number;
    zIndex?: number;
    position: {
      top?: string;
      bottom?: string;
      left?: string;
      right?: string;
    };
    multiply?: boolean;
  }>(),
  {
    color: "var(--slidev-theme-primary)",
    width: "16rem",
    height: "24rem",
    opacity: 1,
    zIndex: 0,
    multiply: true,
    position: {},
  }
);
</script>

<template>
  <div
    class="decorative-rectangle absolute rounded-md flex items-center justify-center p-4"
    :style="{
      backgroundColor: color,
      width: width,
      height: height,
      opacity: opacity,
      zIndex: zIndex,
      ...position,
      ...(multiply ? { mixBlendMode: 'multiply' } : {}),
    }"
  ></div>
  <div
    class="decorative-rectangle absolute rounded-md flex items-center justify-center p-4"
    :style="{
      width: width,
      height: height,
      opacity: opacity,
      zIndex: zIndex,
      ...position,
    }"
  >
    <slot />
  </div>
</template>

<style scoped>
/* Styles handled via props */
</style>
