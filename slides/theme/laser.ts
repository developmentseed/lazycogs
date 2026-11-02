import { configs } from '@slidev/client/env.ts'
import { sharedState } from '@slidev/client/state/shared.ts'
import { showPresenterCursor } from '@slidev/client/state/storage.ts'
import { computed } from 'vue'

// ponytail: an extra key riding on Slidev's synced state, so toggling it in the
// presenter window reaches the audience window. Not in SharedState's type.
const shared = sharedState as typeof sharedState & { laser?: boolean, laserDrag?: boolean }

// tunable from slides.md headmatter: themeConfig: { laserColor, laserWidth, laserFade, laserEnabled }
const theme = configs.themeConfig as Record<string, any>
export const config = {
  color: theme.laserColor ?? '#ef4444',
  width: Number(theme.laserWidth ?? 6),
  fade: Number(theme.laserFade ?? 900),
}

export const laser = computed({
  get: () => shared.laser ?? theme.laserEnabled === true,
  set: (v) => {
    shared.laser = v
    // in presenter mode the trail is fed by Slidev's synced cursor, which only
    // gets published while the presenter cursor is enabled
    if (v)
      showPresenterCursor.value = true
  },
})

// shift+drag in the presenter window, published so viewers draw the same stroke
export const laserDrag = computed({
  get: () => !!shared.laserDrag,
  set: v => shared.laserDrag = v,
})
