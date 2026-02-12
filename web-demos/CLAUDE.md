# Web Demos - Notes for Claude

## WASM Import Strategy - DO NOT CHANGE

Workers import from `'plonky2-wasm-webgpu'` (bare specifier). This resolves via:

1. `build.sh` runs `wasm-pack build` and copies `pkg/` into `react-app/src/wasm/plonky2-wasm-webgpu/`
2. `vite.config.ts` has a `resolve.alias` mapping `'plonky2-wasm-webgpu'` to that source folder
3. `vite-plugin-wasm` handles `.wasm` loading in both main thread and workers

**Never change worker imports to anything other than `'plonky2-wasm-webgpu'`.**
**Never copy WASM into `node_modules/` — it gets blown away by `npm install`.**
**Never add symlinks or `@fs/` path hacks.**

The `src/wasm/` folder is gitignored (build artifact). Run `web-demos/build.sh` to populate it.
