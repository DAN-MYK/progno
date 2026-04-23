import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig(async () => ({
  plugins: [svelte()],
  server: {
    port: 5173,
    strictPort: false,
  },
  build: {
    target: ['chrome120', 'firefox121', 'safari17'],
  },
}))
