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
echo "=== Copying WASM packages into react-app ==="
cd "$SCRIPT_DIR/react-app"
rm -rf node_modules/plonky2-wasm-cpu node_modules/plonky2-wasm-webgpu
mkdir -p node_modules/plonky2-wasm-cpu node_modules/plonky2-wasm-webgpu
cp -r "$SCRIPT_DIR/plonky2-wasm-cpu/pkg/"* node_modules/plonky2-wasm-cpu/
cp -r "$SCRIPT_DIR/plonky2-wasm-webgpu/pkg/"* node_modules/plonky2-wasm-webgpu/

echo ""
echo "=== Installing React app dependencies ==="
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
