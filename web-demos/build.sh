#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Building WASM package ==="
cd "$SCRIPT_DIR/plonky2-wasm-webgpu"
wasm-pack build --target web --release

echo ""
echo "=== Copying WASM package into react-app/src/wasm/ ==="
# !! DO NOT CHANGE THIS IMPORT STRATEGY !!
# The WASM pkg is copied into src/wasm/ and Vite resolves it via a
# resolve.alias in vite.config.ts.  This is intentional:
#   - Survives npm install (unlike node_modules copy)
#   - No symlinks, no @fs/ escapes, no fragile hacks
#   - Workers can import 'plonky2-wasm-webgpu' and Vite resolves it
# If you break this, the worker imports will fail at dev/build time.
rm -rf "$SCRIPT_DIR/react-app/src/wasm"
mkdir -p "$SCRIPT_DIR/react-app/src/wasm"
cp -r "$SCRIPT_DIR/plonky2-wasm-webgpu/pkg" "$SCRIPT_DIR/react-app/src/wasm/plonky2-wasm-webgpu"

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
