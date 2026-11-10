import { registerPlugin } from '@capacitor/core'
import type { KneejerkPlugin } from './definitions'

const Kneejerk = registerPlugin<KneejerkPlugin>('Kneejerk', {
  web: () => import('./web').then((m) => new m.KneejerkWeb()),
})

export * from './definitions'
export { Kneejerk }
