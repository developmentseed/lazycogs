import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { UserConfig } from 'vite'

const deckRoot = fileURLToPath(new URL('.', import.meta.url))

const config: UserConfig = {
  publicDir: resolve(deckRoot, '../public'),
}

export default config
