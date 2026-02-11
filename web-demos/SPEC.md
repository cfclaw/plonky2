# Plonky2 WASM + WebGPU Browser Proving — Specification

## Overview

This spec details the changes needed to:
1. Make the `plonky2_hw_acc_webgpu` crate compile to `wasm32-unknown-unknown`
2. Build two browser-ready WASM packages for end-to-end proof generation:
   - **CPU-only (WASM)**: Uses `PoseidonGoldilocksConfig` with standard CPU compute
   - **WebGPU-accelerated (WASM)**: Uses `PoseidonGoldilocksWebGpuConfig` with GPU compute
3. Provide a React app that loads both and compares performance side-by-side

The circuits used are from `examples/psy_bench_recursion`: `DummyPsyTypeCCircuit` (inner)
and `DummyPsyTypeCRecursiveVerifierCircuit` (recursive verifier).

---

## Part 1: WASM-Compatibility Changes to `plonky2_hw_acc_webgpu`

### Problem Analysis

The WebGPU crate currently targets native only. Key incompatibilities for `wasm32`:

| Issue | File | Fix |
|-------|------|-----|
| `pollster::block_on()` — blocks the thread, not allowed on WASM main thread | `context.rs` | Use `wgpu::Instance::poll()` loop or `wasm-bindgen-futures` async |
| `lazy_static` + `Mutex` global singleton | `context.rs` | Feature-gate: use `thread_local!` + `RefCell` on WASM |
| `std::sync::mpsc::channel` for buffer readback | `utils.rs` | Use `wasm-bindgen-futures` + `js_sys::Promise` on WASM |
| `env_logger` — not available on WASM | `Cargo.toml` | Make optional, use `web_sys::console` on WASM |
| `parking_lot::Mutex` | `context.rs` | Feature-gate or use `std::sync::Mutex` on WASM |
| `std::time::Instant` — not on WASM | examples | Use `web-time` crate (already a dep of plonky2) |
| `device.poll(wgpu::Maintain::Wait)` — blocks on native, needs async on WASM | `utils.rs` | Provide async variants for WASM |

### Changes to `plonky2_hw_acc_webgpu/Cargo.toml`

- Add `wasm-bindgen`, `wasm-bindgen-futures`, `js-sys`, `web-sys` as WASM-target deps
- Feature-gate `pollster`, `env_logger`, `parking_lot` as non-WASM only
- Add `console_log` + `console_error_panic_hook` for WASM logging/debugging
- Add `getrandom` with `"js"` feature for WASM random number generation

### Changes to `context.rs`

- Make `WebGpuContext::new()` async (returns `impl Future`)
- On WASM: use `wgpu::Instance::new()` with `wgpu::Backends::BROWSER_WEBGPU`
- Replace `pollster::block_on()` with `wasm_bindgen_futures::spawn_local()` on WASM
- Replace `lazy_static! { WEBGPU_CONTEXT }` with an async initialization function
- Provide `init_webgpu_context()` as the WASM entry point

### Changes to `utils.rs`

- Make `download_field_data()` async on WASM (buffer mapping is inherently async)
- Replace `std::sync::mpsc::channel` + `device.poll(Wait)` with `wasm_bindgen_futures`
- Keep the native synchronous version behind `#[cfg(not(target_arch = "wasm32"))]`

### Changes to `prover.rs`

- The `ProverCompute` trait methods are synchronous — they must stay sync for trait compat
- On WASM, use a blocking poll loop (`device.poll(Wait)` works on wgpu WASM via microtasks)
- Actually: wgpu's WASM backend handles `device.poll(Maintain::Wait)` correctly in recent versions
  by spinning an internal event loop — so **most prover code can stay as-is**

---

## Part 2: WASM-Only CPU Proving Package (`web-demos/plonky2-wasm-cpu`)

### Purpose
Compile the psy_bench_recursion circuit using standard `PoseidonGoldilocksConfig` (CPU prover)
to WASM, expose it via `wasm-bindgen`, and call from JavaScript.

### Structure
```
web-demos/plonky2-wasm-cpu/
├── Cargo.toml
├── src/
│   └── lib.rs          # wasm-bindgen entry points
```

### `Cargo.toml`
- Depends on `plonky2` with features: `["timing"]` and WITHOUT `parallel` (no rayon on WASM)
- Depends on `plonky2_field`, `plonky2_maybe_rayon` (no parallel), `plonky2_util`
- `wasm-bindgen`, `console_error_panic_hook`, `web-sys` (console, performance)
- `getrandom` with `"js"` feature
- `serde`, `serde_json` for proof serialization
- `crate-type = ["cdylib"]`

### `src/lib.rs`
- `#[wasm_bindgen] pub fn init()` — panic hook + console logger
- `#[wasm_bindgen] pub fn run_cpu_benchmark() -> JsValue` — full pipeline:
  1. Build `DummyPsyTypeCCircuit` with `PoseidonGoldilocksConfig`
  2. Generate 2 inner proofs
  3. Build `DummyPsyTypeCRecursiveVerifierCircuit`
  4. Generate recursive proof
  5. Verify all proofs
  6. Return JSON with timing data for each phase

### Build
```bash
wasm-pack build --target web --release
```

---

## Part 3: WebGPU + WASM Proving Package (`web-demos/plonky2-wasm-webgpu`)

### Purpose
Compile the psy_bench_recursion circuit using `PoseidonGoldilocksWebGpuConfig` (GPU-accelerated)
to WASM, expose it via `wasm-bindgen`.

### Structure
```
web-demos/plonky2-wasm-webgpu/
├── Cargo.toml
├── src/
│   └── lib.rs          # wasm-bindgen entry points (async)
```

### `Cargo.toml`
- Same base deps as CPU version
- Additionally depends on `plonky2_hw_acc_webgpu` (local path)
- `wgpu` with web features
- `wasm-bindgen-futures` for async GPU operations

### `src/lib.rs`
- `#[wasm_bindgen] pub async fn init_gpu()` — initialize WebGPU context
- `#[wasm_bindgen] pub fn run_webgpu_benchmark() -> JsValue` — full pipeline:
  1. Build `DummyPsyTypeCCircuit` with `PoseidonGoldilocksWebGpuConfig`
  2. Generate 2 inner proofs (GPU-accelerated FFT + Merkle)
  3. Build recursive verifier circuit
  4. Generate recursive proof (GPU-accelerated)
  5. Verify all proofs
  6. Return JSON with timing data

### Key Consideration
WebGPU initialization is async but proof generation itself can be sync because
wgpu's WASM backend handles `device.poll(Maintain::Wait)` internally.

---

## Part 4: React Comparison App (`web-demos/react-app`)

### Purpose
Single-page React app that loads both WASM modules and provides a UI to:
- Run CPU-only benchmark
- Run WebGPU benchmark
- Display side-by-side timing comparison

### Structure
```
web-demos/react-app/
├── package.json
├── vite.config.ts
├── index.html
├── tsconfig.json
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── components/
│   │   ├── BenchmarkRunner.tsx
│   │   └── ResultsTable.tsx
│   └── workers/
│       ├── cpu-worker.ts       # Web Worker for CPU proving
│       └── webgpu-worker.ts    # Web Worker for WebGPU proving
```

### Features
- **Web Workers**: Proofs run in workers to avoid blocking the UI thread
- **Progress display**: Shows current phase (circuit building, proving, verifying)
- **Results table**: Side-by-side timing comparison:
  - Circuit build time
  - Inner proof #1 time
  - Inner proof #2 time
  - Recursive proof time
  - Verification time
  - Total time
- **WebGPU detection**: Gracefully handles browsers without WebGPU support

### Tech Stack
- Vite (fast dev server with WASM support)
- React 18
- TypeScript
- No CSS framework (minimal inline styles for simplicity)

---

## Goals

1. **Correctness**: Both WASM packages produce valid plonky2 proofs that verify
2. **Apples-to-apples comparison**: Same circuit, same inputs, different compute backends
3. **Easy to build and run**: Single `npm run dev` after building WASM packages
4. **Browser-ready**: Works in Chrome 113+ (WebGPU), Firefox/Safari (CPU-only fallback)
5. **Minimal changes to existing code**: WebGPU crate changes are additive/conditional

---

## Build & Run Instructions (Target)

```bash
# 1. Build CPU WASM package
cd web-demos/plonky2-wasm-cpu
wasm-pack build --target web --release

# 2. Build WebGPU WASM package
cd ../plonky2-wasm-webgpu
wasm-pack build --target web --release

# 3. Run React app
cd ../react-app
npm install
npm run dev
# Open http://localhost:5173
```
