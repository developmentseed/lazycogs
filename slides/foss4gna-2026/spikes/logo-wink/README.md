# Closing-logo wink spike

Open `index.html` directly in a browser; it is self-contained and offline. The preview reuses the icon geometry and transform from `../../../../logo.svg`, including the original left-eye path. The right eye is the original closed path plus a small outlined open-eye variant. The open variant briefly replaces only that eye; the final/resting expression is unchanged. Use **Replay one-shot** to preview again. `?stage=wink` and `?stage=settled` select deterministic capture states.

## Recommendation

If the visual is approved, use one restrained one-shot on closing-slide entry, not a repeating animation. A single subtle eye-open/close beat can punctuate the ending; repeated motion would keep drawing attention during the five-minute discussion. The prototype's beat is 1.8 seconds: a brief open state around 0.5 seconds, then back to the original relaxed expression. Keep reduced-motion users at the static resting mark and ensure the static slide/export has that same fallback. Avoid introducing animation configuration unless the deck needs it elsewhere.

For integration, inline the original icon SVG in the closing slide so CSS can target the eye path. Preserve the existing viewBox, group transform, and every unanimated path. Add the open-eye variant alongside the original right-eye path and animate only their opacity with native CSS; scope the one-shot to the closing slide's entry. Do not put this spike into the main deck yet. The deck currently imports `logo-text.svg`, whose embedded wordmark uses JetBrainsMono Nerd Font and whose icon differs slightly in coordinates from `logo.svg`; preserve the combined wordmark and do not silently substitute typography. Check whether that font is available in the actual presentation/export environment before attempting to animate the combined asset. This spike isolates the symbol to assess eye geometry, not the wordmark composition.

## Caveats

- This is a visual prototype, not a Slidev integration or a claim that the timing is final.
- There is no rasterization or covering patch; only the real right-eye mark changes. The face already has relaxed closed eyes, so an opening beat is intentionally restrained.
- The source logo's right eye is a stroked quadratic path. The open-eye contour is a hand-authored companion shape, not an existing source path; review it against the brand mark before integration.
- CSS animation is ignored under `prefers-reduced-motion: reduce`; the resting closed-eye path remains visible.
