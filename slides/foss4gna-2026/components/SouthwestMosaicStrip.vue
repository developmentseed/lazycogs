<script setup lang="ts">
type SouthwestManifest = { months: string[] }

const assets = import.meta.glob('../assets/southwest/*.png', {
  eager: true,
  import: 'default',
  query: '?url',
}) as Record<string, string>
const manifestModules = import.meta.glob('../assets/southwest/manifest.json', {
  eager: true,
  import: 'default',
}) as Record<string, SouthwestManifest>
const manifest = Object.values(manifestModules)[0]
const months = (manifest?.months ?? []).map(month => ({
  month,
  src: assets[`../assets/southwest/mosaic-${month}.png`],
}))
</script>

<template>
  <div class="southwest-mosaics" aria-label="Seasonal display previews">
    <figure v-for="item in months" :key="item.month">
      <img v-if="item.src" :src="item.src" :alt="`${item.month} Southwest RGB display preview`">
      <figcaption>{{ item.month }} · 1,200 m display preview</figcaption>
    </figure>
  </div>
</template>

<style scoped>
.southwest-mosaics {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  flex: 1;
  gap: .5rem;
  min-height: 0;
}
.southwest-mosaics figure {
  display: grid;
  grid-template-rows: minmax(0, 1fr) auto;
  min-width: 0;
  min-height: 0;
  margin: 0;
}
.southwest-mosaics img {
  width: 100%;
  height: 100%;
  min-height: 0;
  object-fit: contain;
}
.southwest-mosaics figcaption {
  padding: .2rem;
  text-align: center;
  font-size: .57rem;
}
</style>
