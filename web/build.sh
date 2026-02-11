#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building CPU-only WASM prover ==="
cd plonky2-wasm-prover
wasm-pack build --target web --release --out-dir ../benchmark-app/src/wasm/cpu-prover
cd ..

echo ""
echo "=== Building WebGPU WASM prover ==="
cd plonky2-webgpu-prover
wasm-pack build --target web --release --out-dir ../benchmark-app/src/wasm/webgpu-prover
cd ..

echo ""
echo "=== Installing benchmark app dependencies ==="
cd benchmark-app
npm install

echo ""
echo "=== Build complete! ==="
echo "To start the dev server: cd web/benchmark-app && npm run dev"
echo "Then open http://localhost:5173 in Chrome 113+ (for WebGPU support)"
