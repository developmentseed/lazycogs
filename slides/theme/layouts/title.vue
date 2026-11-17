<script setup lang="ts">
import { resolveAssetUrl } from '../utils/assets'

const props = defineProps<{
  image?: string;
  subtitle?: string;
}>();
</script>

<template>
  <div class="slidev-layout title-layout h-full flex flex-col relative">
    <!-- Top two-thirds: Content -->
    <div class="title-content flex-grow flex flex-col p-16 text-left">
      <!-- Spacer for logo -->
      <div class="flex-grow-0 h-16"></div>

      <!-- Centered content -->
      <div class="flex-grow flex flex-col justify-center">
        <div class="grid grid-cols-2 gap-8">
          <div>
            <slot />
          </div>
          <div
            v-if="subtitle || $slots.subtitle"
            class="flex items-end text-muted subtitle-text"
          >
            <div class="text-lg my-1">
              <slot name="subtitle">{{ subtitle }}</slot>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Bottom third: Image -->
    <div class="title-image h-1/3 relative">
      <img
        v-if="image"
        :src="resolveAssetUrl(image)"
        class="absolute inset-0 w-full h-full object-cover z-10"
        alt=""
      />
      <slot name="image" />
    </div>
  </div>
</template>

<style scoped>
.title-layout {
  background-color: var(--slidev-theme-surface);
}

.title-content {
  min-height: 66.666%;
}

.title-content :deep(h1) {
  margin-bottom: 0 !important;
}

.subtitle-text {
  font-family: var(--slidev-font-family-heading);
}

.title-image {
  background-color: rgba(68, 63, 63, 0.05);
}
</style>
