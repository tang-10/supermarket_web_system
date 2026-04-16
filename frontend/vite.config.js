import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000', // 转发接口请求
      '/ws': {
        target: 'ws://localhost:8000', // 转发 WebSocket
        ws: true
      },
      '/static': 'http://localhost:8000' // 转发静态视频
    }
  }
})