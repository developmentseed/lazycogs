<script setup lang="ts">
import IconButton from '@slidev/client/internals/IconButton.vue'
import { laser } from './laser'
</script>

<template>
  <IconButton
    class="relative"
    :title="laser ? 'Disable laser pointer' : 'Enable laser pointer (or hold shift and drag)'"
    @click="laser = !laser"
  >
    <div class="i-ph:scribble-loop" />
    <div v-if="laser" class="absolute left-1 right-1 bottom-0 h-0.7 rounded-full bg-red-500" />
  </IconButton>
</template>
