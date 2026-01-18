import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'

// 读取VERSION文件
const versionFile = path.resolve(__dirname, '../VERSION')
const version = fs.existsSync(versionFile)
  ? fs.readFileSync(versionFile, 'utf-8').trim()
  : '0.0.0'

export default defineConfig(({ mode }) => {
  // 从父目录 (Tier6+model) 加载 .env 文件
  const env = loadEnv(mode, path.resolve(__dirname, '..'), '')
  const apiPort = env.VITE_API_PORT || '8001'
  const apiTarget = `http://127.0.0.1:${apiPort}`

  return {
    plugins: [react()],
    define: {
      __APP_VERSION__: JSON.stringify(version),
    },
    server: {
      port: 3100,
      host: '127.0.0.1',
      proxy: {
        '/api': {
          target: apiTarget,
          changeOrigin: true,
        },
      },
      watch: {
        ignored: ['**/node_modules/**', '**/.git/**'],
      },
    },
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            'vendor-react': ['react', 'react-dom'],
            'vendor-three': ['three', '@react-three/fiber', '@react-three/drei', '@react-spring/three'],
            'vendor-ui': ['antd', '@ant-design/icons'],
            'vendor-chart': ['echarts', 'echarts-for-react'],
          },
        },
      },
    },
    optimizeDeps: {
      include: [
        'three',
        '@react-three/fiber',
        '@react-three/drei',
        '@react-spring/three',
        'react',
        'react-dom',
        'antd',
      ],
    },
  }
})
