<script setup lang="ts">
import { useIsSlideActive } from '@slidev/client'
const active = useIsSlideActive()
</script>

<template>
  <svg
    class="closing-logo"
    :class="{ 'is-active': active }"
    viewBox="0 0 200 200"
    role="img"
    aria-label="lazycogs"
  >
    <g transform="matrix(0.93327646,0,0,0.92351204,0.01535639,15.306394)">
      <rect width="200" height="179.91104" fill="#fff" x="9.9951649" y="13.507717" />
      <path class="stroke" d="m 106.20786,188.13759 89.7784,-46.31713 -89.7784,-46.317149 -89.778396,46.317149 z" />
      <path class="stroke" d="m 106.20786,157.2595 89.7784,-46.31713 -89.7784,-46.317153 -89.778396,46.317153 z" />
      <path class="stroke" d="m 106.20786,126.38141 89.7784,-46.317147 -89.7784,-46.31714 -89.778396,46.31714 z" />
      <path class="feature" d="m 76.281727,75.432549 q 10.474143,9.263429 20.948295,0" />
      <path class="feature eye-right-closed" d="m 115.1857,75.432549 q 10.47415,9.263429 20.94829,0" />
      <g class="googly-eyes">
        <ellipse cx="86.756" cy="75.433" rx="11" ry="10" />
        <ellipse cx="125.66" cy="75.433" rx="11" ry="10" />
        <circle class="pupil" cx="93.35" cy="75.433" r="3.4" />
        <circle class="pupil" cx="132.25" cy="75.433" r="3.4" />
      </g>
      <path class="feature" d="m 97.230022,90.871596 q 8.977838,9.263434 17.955678,0" />
      <path d="m 169.76502,56.351249 a 8,7.1964415 0 1 0 0,14.392883 8,7.1964415 0 1 1 0,-14.392883" fill="#000000" />
      <text xml:space="preserve" style="font-weight:800;font-size:40.2134px;font-family:'JetBrainsMono Nerd Font';font-variation-settings:normal;-inkscape-font-specification:'JetBrainsMono Nerd Font, Ultra-Bold';writing-mode:lr-tb;direction:ltr;fill:#d45500;stroke-width:0.538572;fill-opacity:1" x="11.903498" y="18.922346" transform="scale(0.99475497,1.0052727)"><tspan style="stroke-width:0.538572;fill:#d45500;fill-opacity:1" x="11.903498" y="18.922346">lazycogs</tspan></text>
    </g>
  </svg>
</template>

<style scoped>
.closing-logo {
  display: block;
}
.closing-logo .stroke {
  fill: #fff;
  stroke: #000;
  stroke-width: 2.5;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.closing-logo .feature {
  fill: none;
  stroke: #000;
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.closing-logo .googly-eyes {
  fill: #fff;
  stroke: #000;
  stroke-width: 2;
  opacity: 0;
}
.closing-logo .googly-eyes .pupil {
  fill: #000;
  stroke: none;
}
.closing-logo.is-active .googly-eyes {
  animation: googly-appear 8s steps(1, end) infinite;
}

@keyframes googly-appear {
  0%, 82%, 100% { opacity: 0; }
  64% { opacity: 1; }
}

@media (prefers-reduced-motion: reduce), print {
  .closing-logo.is-active .googly-eyes { animation: none; }
  .closing-logo .googly-eyes { opacity: 0; }
  .closing-logo .eye-right-closed { opacity: 1; }
}
</style>
