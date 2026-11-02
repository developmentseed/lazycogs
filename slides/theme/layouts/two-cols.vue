<script setup lang="ts">
const props = defineProps({
  gap: {
    type: String,
    default: "4",
    description: "Gap between columns (tailwind spacing scale)",
  },
  leftRatio: {
    type: Number,
    default: 50,
    description: "Width percentage for left column (e.g., 50 for 50/50, 60 for 60/40)",
  },
});

const leftWidth = `${props.leftRatio}%`;
const rightWidth = `${100 - props.leftRatio}%`;
</script>

<template>
  <div
    class="slidev-layout two-cols h-full flex"
    :class="[`gap-${gap}`, $attrs.class]"
  >
    <!-- Left Column -->
    <div class="left-col flex flex-col" :style="{ width: leftWidth }">
      <slot />
    </div>

    <!-- Right Column -->
    <div class="right-col flex flex-col" :style="{ width: rightWidth }">
      <slot name="right" />
    </div>
  </div>
</template>

<style scoped>
.two-cols {
  padding: 1rem 2rem;
}

.left-col,
.right-col {
  overflow: auto;
}

/* Ensure proper spacing for content */
.left-col :deep(> *:first-child),
.right-col :deep(> *:first-child) {
  margin-top: 0;
}

.left-col :deep(> *:last-child),
.right-col :deep(> *:last-child) {
  margin-bottom: 0;
}
</style>
