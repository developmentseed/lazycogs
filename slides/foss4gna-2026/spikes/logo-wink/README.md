# Closing-logo wink spike

Open `index.html` directly in a browser; it is self-contained and offline. The preview reuses the icon geometry and transform from `../../../../logo.svg`, including the original left-eye path. The right eye is the original closed path plus a small outlined open-eye variant. The open variant briefly replaces only that eye; the final/resting expression is unchanged. Use **Replay one-shot** to preview again. `?stage=wink` and `?stage=settled` select deterministic capture states.

## Recommendation

If the visual is approved, use one restrained one-shot on closing-slide entry, not a repeating animation. A single subtle eye-open/close beat can punctuate the ending; repeated motion would keep drawing attention during the five-minute discussion. The prototype's beat is 1.8 seconds: a brief open state around 0.5 seconds, then back to the original relaxed expression. Keep reduced-motion users at the static resting mark and ensure the static slide/export has that same fallback. Avoid introducing animation configuration unless the deck needs it elsewhere.

The production component prototype is `../../components/ClosingLogo.vue`. It inlines the combined `logo-text.svg` geometry, preserves its wordmark and original group transform, and starts a single native CSS animation only while Slidev reports the closing slide active. `useIsSlideActive()` is the public `@slidev/client` API; the installed implementation compares this page with `currentSlideNo`, so neighboring-slide preload does not trigger the animation. The companion open-eye path is a small authored contour; only the right eye's two opacity values animate. Reduced-motion and print styles keep the original resting expression. The embedded wordmark uses JetBrainsMono Nerd Font; font portability remains an existing caveat and typography is unchanged.

## Caveats

- `ClosingLogo.vue` is ready for the coordinator to add to the closing slide; this branch does not edit the deck itself, and timing/visual acceptance in the production deck remains for review.
- There is no rasterization or covering patch; only the real right-eye mark changes. The face already has relaxed closed eyes, so an opening beat is intentionally restrained.
- The combined wordmark source's right eye is a stroked quadratic path. The open-eye contour is a hand-authored companion shape, not an existing source path; review it against the brand mark in the production deck.
- CSS animation is ignored under `prefers-reduced-motion: reduce`; the resting closed-eye path remains visible.
