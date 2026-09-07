import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/annotated-video': 'http://127.0.0.1:8000',
      '/static': 'http://127.0.0.1:8000',
      '/upload': 'http://127.0.0.1:8000',
      '/process': 'http://127.0.0.1:8000',
      '/analytics': 'http://127.0.0.1:8000',
      '/trajectory': 'http://127.0.0.1:8000',
      '/reports': 'http://127.0.0.1:8000',
      '/highlights': 'http://127.0.0.1:8000',
      '/ball-speed': 'http://127.0.0.1:8000',
      '/shot-analysis': 'http://127.0.0.1:8000',
      '/scoreboard': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/job': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
