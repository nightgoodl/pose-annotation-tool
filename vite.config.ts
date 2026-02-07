import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  // Base path includes /cuhk-02 for proper API resolution through tunnel
  base: '/cuhk-02/',
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 7860,
    host: true,
    allowedHosts: [
      'wstunnel-http-train.meshy.art',
      '.meshy.art',
    ],
    proxy: {
      // Proxy /api to backend (for frontend absolute path calls)
      '/api': {
        target: 'http://localhost:8084',
        changeOrigin: true,
      },
      // Proxy /data to backend (for frontend absolute path calls)
      '/data': {
        target: 'http://localhost:8084',
        changeOrigin: true,
      },
    },
  },
  build: {
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'index.html'),
        mv: path.resolve(__dirname, 'index-mv.html'),
      },
    },
  },
})
