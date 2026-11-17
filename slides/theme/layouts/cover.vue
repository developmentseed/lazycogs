<script setup lang="ts">
import type { CSSProperties } from "vue";
import { resolveAssetUrl } from '../utils/assets'

const props = defineProps({
  background: {
    default: "",
  },
  dim: {
    type: Boolean,
    default: false,
  },
});

function isLightColor(color: string): boolean {
  // Simple hex color check
  if (color.startsWith("#")) {
    const hex = color.replace("#", "");
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    // Calculate relative luminance
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    return luminance > 0.5;
  }
  // For rgb/hsl, assume dark (could be enhanced)
  return false;
}

function handleBackground(background?: string, dim = false): CSSProperties {
  const isColor =
    background && ["#", "rgb", "hsl"].some((v) => background.indexOf(v) === 0);

  let textColor: string | undefined;
  if (background && !isColor) {
    // Image background - use white text
    textColor = "white";
  } else if (background && isColor) {
    // Color background - detect if light or dark
    textColor = isLightColor(background) ? "black" : "white";
  }

  const style = {
    background: isColor ? background : undefined,
    color: textColor,
    backgroundImage: isColor
      ? undefined
      : background
        ? dim
          ? `linear-gradient(#0005, #0008), url(${resolveAssetUrl(background)})`
          : `url("${resolveAssetUrl(background)}")`
        : undefined,
    backgroundRepeat: "no-repeat",
    backgroundPosition: "center",
    backgroundSize: "cover",
  };

  if (!style.background) delete style.background;

  return style;
}

const style = handleBackground(props.background, props.dim);
</script>

<template>
  <div
    class="slidev-layout cover h-full w-full flex items-center justify-center"
    :class="$attrs.class"
    :style="{ ...style, color: undefined }"
  >
    <div class="my-auto w-full cover-content" :style="{ color: style.color }">
      <slot />
    </div>
  </div>
</template>

<style scoped>
/* Ensure headings and text inherit the automatic color */
.cover-content :deep(h1),
.cover-content :deep(h2),
.cover-content :deep(h3),
.cover-content :deep(h4),
.cover-content :deep(h5),
.cover-content :deep(h6),
.cover-content :deep(p),
.cover-content :deep(a) {
  color: inherit;
}

/* Monaco container should not inherit text color and should be constrained in size */
.cover-content :deep(.slidev-monaco-container) {
  color: initial;
  max-height: 60vh;
  max-width: 100%;
  overflow: auto;
}

/* Ensure monaco editor itself respects the container size */
.cover-content :deep(.slidev-monaco-container .monaco-editor) {
  max-height: 20vh;
}
</style>
