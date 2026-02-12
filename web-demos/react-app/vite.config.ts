import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import wasm from 'vite-plugin-wasm';
import path from 'path';

// !! DO NOT CHANGE THE WASM IMPORT STRATEGY !!
// Workers import from 'plonky2-wasm-webgpu' (bare specifier).
// The resolve.alias below points that to src/wasm/plonky2-wasm-webgpu/,
// which is populated by web-demos/build.sh copying wasm-pack output there.
// This survives npm install (unlike a node_modules copy) and lets Vite
// handle the .wasm files natively via vite-plugin-wasm.
// If you break this, the worker imports will fail at dev/build time.

export default defineConfig({
  plugins: [react(), wasm()],
  resolve: {
    alias: {
      'plonky2-wasm-webgpu': path.resolve(__dirname, 'src/wasm/plonky2-wasm-webgpu'),
    },
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  optimizeDeps: {
    exclude: ['plonky2-wasm-webgpu'],
  },
  worker: {
    plugins: () => [wasm()],
  },
});
