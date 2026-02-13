use anyhow::{anyhow, Result};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::ops::Square;
use plonky2::field::types::Field;
use std::collections::HashMap;

use crate::utils;

// Shader sources: goldilocks preamble + kernel-specific code, assembled by build.rs
const FFT_COSET_SCALE_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/fft_coset_scale.wgsl"));
const FFT_DIT_PASS_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/fft_dit_pass.wgsl"));
const BIT_REVERSE_COPY_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/bit_reverse_copy.wgsl"));
const IFFT_REORDER_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/ifft_reorder.wgsl"));
const MONT_CONVERT_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/mont_convert.wgsl"));
const TRANSPOSE_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/transpose.wgsl"));
const SCATTER_SALT_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/scatter_salt.wgsl"));
const MERKLE_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/merkle.wgsl"));

const MAX_DEGREE_LOG: usize = 18;
const MAX_RATE_BITS: usize = 4;

/// Default download chunk size for mobile/memory-constrained GPUs: 4 MiB.
/// Reduced from 16 MiB to lower peak staging buffer allocation. On iOS Safari,
/// each staging buffer contributes to cumulative GPU memory pressure even after
/// destroy(). Smaller chunks reduce the high-water mark per download cycle.
pub const MOBILE_DOWNLOAD_CHUNK_SIZE: u64 = 4 * 1024 * 1024;

pub struct WebGpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipelines: HashMap<String, wgpu::ComputePipeline>,
    /// Pre-computed twiddle factors in Montgomery form, keyed by log_n.
    pub twiddle_buffers: HashMap<usize, wgpu::Buffer>,
    /// Maximum bytes per GPU→CPU download chunk. When non-zero, large buffer
    /// downloads are split into multiple map/read cycles using a single staging
    /// buffer of this size, reducing peak GPU memory allocation. This prevents
    /// `BufferAsyncError` / device loss from OOM on mobile GPUs.
    /// Set to 0 for unlimited (single-shot download, the default).
    /// Recommended: [`MOBILE_DOWNLOAD_CHUNK_SIZE`] (16 MiB) for mobile.
    pub download_chunk_size: u64,
    /// The adapter's reported max_buffer_size, captured during initialization.
    /// Useful for diagnostics and auto-configuration. On constrained devices
    /// (e.g. iOS Safari reporting 256 MiB), chunked downloads are enabled
    /// automatically.
    pub max_buffer_size: u64,
}

impl WebGpuContext {
    /// Create a new WebGpuContext. This is async because GPU initialization is
    /// inherently async (adapter request, device request).
    pub async fn new_async() -> Result<Self> {
        let backends = if cfg!(target_arch = "wasm32") {
            wgpu::Backends::BROWSER_WEBGPU
        } else {
            wgpu::Backends::all()
        };

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or_else(|| anyhow!("No suitable GPU adapter found"))?;

        // Query adapter limits and clamp our requests to what the hardware
        // actually supports. This prevents device creation failures on
        // constrained GPUs (e.g. iOS Safari with max_buffer_size = 256 MiB).
        let adapter_limits = adapter.limits();
        let requested_max_buffer = (1u64 << 30).min(adapter_limits.max_buffer_size);
        let requested_max_storage = (1u32 << 30).min(adapter_limits.max_storage_buffer_binding_size);
        let requested_max_workgroups = 65535u32.min(adapter_limits.max_compute_workgroups_per_dimension);

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("plonky2_webgpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: requested_max_storage,
                    max_buffer_size: requested_max_buffer,
                    max_compute_workgroups_per_dimension: requested_max_workgroups,
                    ..wgpu::Limits::downlevel_defaults()
                },
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .map_err(|e| anyhow!("Failed to create WebGPU device: {}", e))?;

        // Auto-configure chunked downloads for constrained devices.
        // The WebGPU spec baseline maxBufferSize is 256 MiB, chosen to
        // accommodate the lowest iOS tier (gpuweb Issue #1371). If the adapter
        // reports ≤256 MiB, we're likely on a memory-constrained mobile device.
        let is_constrained = adapter_limits.max_buffer_size <= 256 * 1024 * 1024;
        let download_chunk_size = if is_constrained { MOBILE_DOWNLOAD_CHUNK_SIZE } else { 0 };

        let mut ctx = Self {
            device,
            queue,
            pipelines: HashMap::new(),
            twiddle_buffers: HashMap::new(),
            download_chunk_size,
            max_buffer_size: adapter_limits.max_buffer_size,
        };

        // Compile FFT pipelines
        ctx.compile_pipeline("coset_scale_and_bit_reverse_batched", FFT_COSET_SCALE_WGSL)?;
        ctx.compile_pipeline("fft_dit_pass_batched", FFT_DIT_PASS_WGSL)?;
        ctx.compile_pipeline("bit_reverse_copy_batched", BIT_REVERSE_COPY_WGSL)?;
        ctx.compile_pipeline("ifft_reorder_and_scale_batched", IFFT_REORDER_WGSL)?;

        // Compile utility pipelines (Montgomery conversion, transpose, salt scatter)
        ctx.compile_pipeline("mont_convert_batched", MONT_CONVERT_WGSL)?;
        ctx.compile_pipeline("transpose_and_bit_reverse", TRANSPOSE_WGSL)?;
        ctx.compile_pipeline("scatter_salt", SCATTER_SALT_WGSL)?;

        // Compile Merkle pipelines
        ctx.compile_pipeline("copy_row_leaves", MERKLE_WGSL)?;
        ctx.compile_pipeline("hash_row_leaves", MERKLE_WGSL)?;
        ctx.compile_pipeline("compress_nodes", MERKLE_WGSL)?;

        // Pre-compute twiddle factors in Montgomery form
        for log_n in 12..=(MAX_DEGREE_LOG + MAX_RATE_BITS) {
            let fwd_generator = GoldilocksField::primitive_root_of_unity(log_n);
            let root_table = fft_root_table_from_generator(fwd_generator, log_n);
            let twiddles_flat: Vec<u64> = root_table
                .into_iter()
                .flatten()
                .map(|f| utils::to_mont(f.0))
                .collect();

            let twiddle_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    twiddles_flat.as_ptr() as *const u8,
                    twiddles_flat.len() * std::mem::size_of::<u64>(),
                )
            };

            let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("twiddles_{}", log_n)),
                size: twiddle_bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            });
            buffer.slice(..).get_mapped_range_mut().copy_from_slice(twiddle_bytes);
            buffer.unmap();

            ctx.twiddle_buffers.insert(log_n, buffer);
        }

        Ok(ctx)
    }

    /// Synchronous constructor for native targets (convenience wrapper).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    fn compile_pipeline(&mut self, entry_point: &str, shader_source: &str) -> Result<()> {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry_point),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: None, // auto-layout from shader
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });

        self.pipelines.insert(entry_point.to_string(), pipeline);
        Ok(())
    }

    pub fn get_pipeline(&self, name: &str) -> Option<&wgpu::ComputePipeline> {
        self.pipelines.get(name)
    }
}

/// Generate twiddle table from a root of unity generator.
fn fft_root_table_from_generator(
    mut base: GoldilocksField,
    max_log_n: usize,
) -> Vec<Vec<GoldilocksField>> {
    let mut bases = Vec::with_capacity(max_log_n);
    bases.push(base);
    for _ in 1..max_log_n {
        base = base.square();
        bases.push(base);
    }
    bases.reverse();

    (1..=max_log_n)
        .map(|log_m| {
            let m = 1 << log_m;
            let base = bases[log_m - 1];
            base.powers().take(m / 2).collect()
        })
        .collect()
}

// ---- Native context management (lazy_static + parking_lot) ----
#[cfg(not(target_arch = "wasm32"))]
mod native_global {
    use super::*;
    use lazy_static::lazy_static;
    use parking_lot::Mutex;

    lazy_static! {
        pub static ref WEBGPU_CONTEXT: Mutex<Option<WebGpuContext>> = {
            match WebGpuContext::new() {
                Ok(ctx) => Mutex::new(Some(ctx)),
                Err(e) => {
                    log::error!("Failed to initialize WebGPU backend: {:?}", e);
                    Mutex::new(None)
                }
            }
        };
    }

    pub fn get_context<'a>() -> Option<parking_lot::MutexGuard<'a, Option<WebGpuContext>>> {
        WEBGPU_CONTEXT.try_lock()
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native_global::get_context;

// ---- WASM context management (thread_local + RefCell) ----
#[cfg(target_arch = "wasm32")]
mod wasm_global {
    use super::*;
    use std::cell::RefCell;

    thread_local! {
        pub(crate) static WEBGPU_CONTEXT: RefCell<Option<WebGpuContext>> = RefCell::new(None);
    }

    /// Initialize the WebGPU context. Must be called (and awaited) before any proving.
    pub async fn init_context() -> Result<()> {
        let ctx = WebGpuContext::new_async().await?;
        WEBGPU_CONTEXT.with(|cell| {
            *cell.borrow_mut() = Some(ctx);
        });
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm_global::init_context;

/// Set the download chunk size for GPU→CPU transfers on the global context.
/// When non-zero, large buffer downloads are split into chunks of at most
/// `bytes` (rounded down to a multiple of 8). This prevents OOM on mobile GPUs.
/// Pass 0 to disable chunking (single-shot downloads).
pub fn set_download_chunk_size(bytes: u64) -> anyhow::Result<()> {
    let aligned = bytes & !7; // round down to multiple of 8 (sizeof u64)
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut guard = native_global::WEBGPU_CONTEXT.lock();
        let ctx = guard.as_mut().ok_or_else(|| anyhow!("WebGPU context not initialized"))?;
        ctx.download_chunk_size = aligned;
    }
    #[cfg(target_arch = "wasm32")]
    {
        wasm_global::WEBGPU_CONTEXT.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let ctx = borrow.as_mut().ok_or_else(|| anyhow!("WebGPU context not initialized (call init_context first)"))?;
            ctx.download_chunk_size = aligned;
            Ok::<(), anyhow::Error>(())
        })?;
    }
    Ok(())
}

/// Configure the global context for mobile / memory-constrained GPUs.
/// Sets the download chunk size to [`MOBILE_DOWNLOAD_CHUNK_SIZE`] (4 MiB).
/// Must be called after context initialization.
pub fn configure_for_mobile() -> anyhow::Result<()> {
    set_download_chunk_size(MOBILE_DOWNLOAD_CHUNK_SIZE)
}

/// Register a JS function to be called at GPU phase boundaries.
///
/// The prover calls this function after destroying large GPU buffers and before
/// allocating new ones. The function **must return a `Promise`**; the prover
/// `await`s it, giving the browser event loop a turn to reclaim destroyed GPU
/// memory. This replaces the built-in `setTimeout(0)` fallback with JS-driven
/// yield control.
///
/// ```js
/// // Minimal: yield one event-loop turn
/// wasm.set_gpu_yield_callback(() => new Promise(r => setTimeout(r, 0)));
///
/// // With progress reporting to UI:
/// wasm.set_gpu_yield_callback(() => {
///   postMessage({ type: 'gpu_phase_done' });
///   return new Promise(r => setTimeout(r, 0));
/// });
/// ```
///
/// Pass `None` to clear the callback and revert to the built-in fallback.
#[cfg(target_arch = "wasm32")]
pub fn set_gpu_yield_callback(f: Option<js_sys::Function>) {
    crate::utils::set_gpu_yield_callback(f);
}

/// Unified context accessor. Calls `f` with a reference to the WebGpuContext.
/// Works on both native (via parking_lot MutexGuard) and WASM (via thread_local RefCell).
pub fn with_gpu_context<R>(f: impl FnOnce(&WebGpuContext) -> R) -> anyhow::Result<R> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let guard = get_context().ok_or_else(|| anyhow!("WebGPU context locked"))?;
        let ctx = guard.as_ref().ok_or_else(|| anyhow!("WebGPU context not initialized"))?;
        Ok(f(ctx))
    }
    #[cfg(target_arch = "wasm32")]
    {
        wasm_global::WEBGPU_CONTEXT.with(|cell| {
            let borrow = cell.borrow();
            match borrow.as_ref() {
                Some(ctx) => Ok(f(ctx)),
                None => Err(anyhow!("WebGPU context not initialized (call init_context first)")),
            }
        })
    }
}

/// Acquire a context guard that can be dereferenced to &WebGpuContext.
/// On native: holds a parking_lot MutexGuard.
/// On WASM: holds a RefCell borrow.
#[cfg(not(target_arch = "wasm32"))]
pub fn acquire_context() -> anyhow::Result<impl std::ops::Deref<Target = WebGpuContext> + 'static> {
    let guard = get_context().ok_or_else(|| anyhow!("WebGPU context locked"))?;
    // We need to return a type that derefs to WebGpuContext.
    // MutexGuard<Option<WebGpuContext>> -> we map it.
    // parking_lot::MutexGuard has MappedMutexGuard for this.
    if guard.is_none() {
        return Err(anyhow!("WebGPU context not initialized"));
    }
    Ok(parking_lot::MutexGuard::map(guard, |opt| opt.as_mut().unwrap()))
}

#[cfg(target_arch = "wasm32")]
pub fn acquire_context() -> anyhow::Result<std::cell::Ref<'static, WebGpuContext>> {
    // On WASM, we use thread_local RefCell.
    // We need to return a Ref that lives long enough.
    // thread_local! with_borrow doesn't allow returning the borrow.
    // Instead, we use a helper that maps the Ref.
    wasm_global::WEBGPU_CONTEXT.with(|cell| {
        let borrow = cell.borrow();
        if borrow.is_none() {
            return Err(anyhow!("WebGPU context not initialized (call init_context first)"));
        }
        // SAFETY: WASM is single-threaded and this thread_local lives for 'static.
        // We transmute the lifetime since thread_local guarantees the data lives
        // for the duration of the thread (which is the entire WASM lifetime).
        let borrow: std::cell::Ref<'_, Option<WebGpuContext>> = borrow;
        let mapped = std::cell::Ref::map(borrow, |opt| opt.as_ref().unwrap());
        let mapped: std::cell::Ref<'static, WebGpuContext> = unsafe { std::mem::transmute(mapped) };
        Ok(mapped)
    })
}
