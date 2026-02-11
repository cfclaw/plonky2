#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Building CPU WASM package ==="
cd "$SCRIPT_DIR/plonky2-wasm-cpu"
wasm-pack build --target web --release

echo ""
echo "=== Building WebGPU WASM package ==="
cd "$SCRIPT_DIR/plonky2-wasm-webgpu"
wasm-pack build --target web --release

echo ""
echo "=== Installing React app dependencies ==="
cd "$SCRIPT_DIR/react-app"
npm install

echo ""
echo "=== Build complete! ==="
echo ""
echo "To start the development server:"
echo "  cd web-demos/react-app"
echo "  npm run dev"
echo ""
echo "Then open http://localhost:5173 in your browser."
echo "WebGPU requires Chrome 113+ or Edge 113+."
