import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': { target: 'http://localhost:8000', rewrite: path => path.startsWith('/api/v1/') ? path : path.replace(/^\/api/, '') } } },
  test: { environment: 'jsdom', setupFiles: './src/test-setup.js' },
});
