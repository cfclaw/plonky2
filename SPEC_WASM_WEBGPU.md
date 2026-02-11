# Plonky2 WASM + WebGPU Browser Proving: Technical Specification

## 1. Overview & Goals

Enable plonky2 zero-knowledge proofs to run entirely in the browser via WebAssembly,
with an optional WebGPU hardware-accelerated path. Deliver a React benchmark app that
lets users compare CPU-only WASM proving vs WebGPU-accelerated WASM proving side-by-side.

### Goals
1. **WASM-ready WebGPU crate**: Modify `plonky2_hw_acc_webgpu` so it compiles for
   `wasm32-unknown-unknown` alongside native targets.
2. **Pure WASM prover** (`web/plonky2-wasm-prover/`): A `wasm-bindgen` crate that
   exposes Fibonacci circuit proving/verifying using `PoseidonGoldilocksConfig`
   (CPU-only `CpuProverCompute`) to JavaScript.
3. **WebGPU WASM prover** (`web/plonky2-webgpu-prover/`): A `wasm-bindgen` crate
   that exposes the same Fibonacci circuit using `PoseidonGoldilocksWebGpuConfig`
   (GPU-accelerated `WebGpuProverCompute`) to JavaScript.
4. **React benchmark app** (`web/benchmark-app/`): A Vite + React app that loads
   both WASM modules and runs identical proofs, displaying timing comparisons.

---

## 2. Analysis of WASM Compatibility Issues in `plonky2_hw_acc_webgpu`

### 2.1 `pollster::block_on` (context.rs:36, 43)
- **Problem**: `pollster` blocks the thread waiting for an async future to resolve.
  On `wasm32`, there is no OS thread to block — the browser's main thread is
  single-threaded and blocking it deadlocks the page.
- **Fix**: Make `WebGpuContext::new()` async. On WASM, callers use
  `wasm_bindgen_futures::spawn_local()` or `await`. On native, callers use
  `pollster::block_on()`. Guard with `#[cfg(target_arch = "wasm32")]`.

### 2.2 `lazy_static` + `parking_lot::Mutex` (context.rs:158-168)
- **Problem**: `parking_lot` relies on OS futex/parking primitives unavailable in
  WASM. `lazy_static` initialization eagerly calls `WebGpuContext::new()` which
  is sync-blocking — impossible on WASM.
- **Fix**: On WASM, use `thread_local!` + `RefCell<Option<WebGpuContext>>` with
  an explicit async `init()` function. On native, keep existing `lazy_static`.

### 2.3 `std::sync::mpsc::channel` in `download_field_data` (utils.rs:78-83)
- **Problem**: `mpsc::channel` + `device.poll(Maintain::Wait)` are synchronous
  blocking patterns. On WASM, `device.poll()` is a no-op (browser runs the GPU
  event loop) and `recv()` would deadlock.
- **Fix**: Provide an async `download_field_data_async()` that uses
  `buffer_slice.map_async()` with a wasm-compatible callback. On WASM, await the
  future directly. On native, keep the existing sync path.

### 2.4 `env_logger` (Cargo.toml:12)
- **Problem**: `env_logger` reads environment variables and writes to stderr —
  neither exists in WASM.
- **Fix**: Make `env_logger` a non-WASM dependency. For WASM, optionally use
  `console_log` crate or just omit logger initialization.

### 2.5 Buffer size limits (context.rs:48-49)
- **Problem**: Requesting 1GB `max_storage_buffer_binding_size` may exceed browser
  WebGPU adapter limits (typically 128-256MB).
- **Fix**: On WASM, request `adapter.limits()` defaults or a more conservative
  256MB. On native, keep 1GB.

### 2.6 `plonky2_maybe_rayon` parallel feature
- **Problem**: Rayon thread pool cannot be used in WASM (no OS threads by default).
- **Fix**: Build all WASM targets with `default-features = false` on
  `plonky2_maybe_rayon` (disabling `parallel`). The crate already has sequential
  fallbacks for all parallel iterators.

---

## 3. Crate Architecture

```
plonky2/
├── plonky2_hw_acc_webgpu/     # Modified: WASM-compatible WebGPU prover
│   ├── Cargo.toml             # New features: "wasm" with wasm-bindgen-futures
│   ├── src/
│   │   ├── context.rs         # Async init, platform-specific context storage
│   │   ├── utils.rs           # Async download_field_data for WASM
│   │   ├── prover.rs          # Async-aware ProverCompute (mostly unchanged)
│   │   └── lib.rs
│   └── build.rs               # Unchanged (runs at compile time, not in WASM)
│
├── web/
│   ├── plonky2-wasm-prover/       # NEW: Pure CPU WASM prover
│   │   ├── Cargo.toml
│   │   └── src/lib.rs             # wasm-bindgen exports for prove/verify
│   │
│   ├── plonky2-webgpu-prover/     # NEW: WebGPU-accelerated WASM prover
│   │   ├── Cargo.toml
│   │   └── src/lib.rs             # wasm-bindgen exports with async GPU init
│   │
│   └── benchmark-app/             # NEW: React app
│       ├── package.json
│       ├── vite.config.ts
│       ├── index.html
│       └── src/
│           ├── App.tsx
│           ├── main.tsx
│           └── benchmark.ts       # Benchmark orchestration
```

---

## 4. Detailed Component Specifications

### 4.1 `plonky2_hw_acc_webgpu` Modifications

#### Cargo.toml changes:
```toml
[features]
default = []
wasm = ["wasm-bindgen-futures", "web-sys"]

[dependencies]
wgpu = "24"
log = "0.4"
anyhow = "1.0"
bytemuck = { version = "1", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
plonky2 = { path = "../plonky2", default-features = false, features = ["std"] }
plonky2_maybe_rayon = { path = "../maybe_rayon", default-features = false }
plonky2_util = { path = "../util", default-features = false }

# Native-only
[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
pollster = "0.4"
env_logger = "0.11"
lazy_static = "1.4"
parking_lot = "0.12"

# WASM-only
[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen-futures = "0.4"
web-sys = { version = "0.3", features = ["console"] }
```

#### context.rs changes:
- `WebGpuContext::new()` → `WebGpuContext::new_async() -> impl Future<Output = Result<Self>>`
- Native: `new()` calls `pollster::block_on(new_async())`
- WASM: Only exposes `new_async()`
- Context storage: native keeps `lazy_static`, WASM uses `thread_local!`

#### utils.rs changes:
- Add `download_field_data_async()` using async buffer mapping
- Native `download_field_data()` unchanged
- WASM calls async variant

### 4.2 Pure WASM Prover (`web/plonky2-wasm-prover/`)

**Purpose**: Compile plonky2 with `PoseidonGoldilocksConfig` to WASM, expose
`prove_fibonacci(n)` and `verify_proof(proof_bytes)` via `wasm-bindgen`.

**API**:
```rust
#[wasm_bindgen]
pub fn prove_fibonacci(num_steps: u32) -> Result<JsValue, JsError>;

#[wasm_bindgen]
pub fn verify_fibonacci(proof_json: &str) -> Result<bool, JsError>;
```

**Config**: Uses `PoseidonGoldilocksConfig` with `CpuProverCompute` — pure CPU,
no GPU, no threads. All computation single-threaded in WASM.

**Build**: `wasm-pack build --target web`

### 4.3 WebGPU WASM Prover (`web/plonky2-webgpu-prover/`)

**Purpose**: Same Fibonacci circuit but using `PoseidonGoldilocksWebGpuConfig`
with GPU-accelerated FFT and Merkle tree building.

**API**:
```rust
#[wasm_bindgen]
pub async fn init_webgpu() -> Result<(), JsError>;

#[wasm_bindgen]
pub async fn prove_fibonacci_webgpu(num_steps: u32) -> Result<JsValue, JsError>;

#[wasm_bindgen]
pub fn verify_fibonacci(proof_json: &str) -> Result<bool, JsError>;
```

**Note**: Proving must be async because GPU operations are async in the browser.
The `init_webgpu()` call must happen first to set up the GPU context.

**Build**: `wasm-pack build --target web`

### 4.4 React Benchmark App (`web/benchmark-app/`)

**Stack**: Vite + React + TypeScript

**Features**:
- Load both WASM modules on page load
- Configure Fibonacci circuit size (number of steps)
- "Run CPU Benchmark" button — calls pure WASM prover
- "Run WebGPU Benchmark" button — calls WebGPU WASM prover
- Display: proof generation time, verification time, proof size
- Side-by-side comparison table
- WebGPU availability detection with graceful fallback

---

## 5. Fibonacci Circuit (Shared Logic)

Both provers implement identical circuits for fair comparison:

```
Circuit: Fibonacci sequence computation
  Input: initial values a=0, b=1
  Steps: configurable N iterations (default 1000)
  Output: fib(N)
  Public inputs: initial_a, initial_b, result
  Config: standard_recursion_config (with reduced degree for small proofs)
```

The circuit builds N addition gates computing the Fibonacci sequence.
This is the canonical plonky2 benchmark — it exercises FFT, Merkle, and
Poseidon operations that benefit from GPU acceleration.

---

## 6. Build & Test Instructions

```bash
# 1. Build pure WASM prover
cd web/plonky2-wasm-prover
wasm-pack build --target web --release

# 2. Build WebGPU WASM prover
cd ../plonky2-webgpu-prover
wasm-pack build --target web --release

# 3. Start React dev server
cd ../benchmark-app
npm install
npm run dev
# Open http://localhost:5173 in Chrome 113+ (WebGPU support required)
```
