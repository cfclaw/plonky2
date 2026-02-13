use crate::context::WebGpuContext;
use plonky2::field::goldilocks_field::GoldilocksField;
use wgpu;

/// Log to browser console on WASM, stderr on native.
pub fn log_gpu(msg: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::console::log_1(&msg.into());
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        eprintln!("{}", msg);
    }
}

/// Create a GPU storage buffer (for compute shader read/write).
pub fn create_storage_buffer(device: &wgpu::Device, size_bytes: u64, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Create a staging buffer for reading data back from GPU.
pub fn create_staging_buffer_read(device: &wgpu::Device, size_bytes: u64, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Upload a slice of GoldilocksField elements to a GPU buffer.
/// Elements are written as raw u64 bytes (little-endian).
pub fn upload_field_data(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    offset: u64,
    data: &[GoldilocksField],
) {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<GoldilocksField>(),
        )
    };
    queue.write_buffer(buffer, offset, bytes);
}

/// Upload a slice of u64 values to a GPU buffer at an offset.
pub fn upload_u64_data_at(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    offset: u64,
    data: &[u64],
) {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<u64>(),
        )
    };
    queue.write_buffer(buffer, offset, bytes);
}

/// Create a new GPU storage buffer and upload u64 data into it.
pub fn upload_u64_data(device: &wgpu::Device, data: &[u64], label: &str) -> wgpu::Buffer {
    let size_bytes = (data.len() * std::mem::size_of::<u64>()) as u64;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8)
    };
    buffer.slice(..).get_mapped_range_mut().copy_from_slice(bytes);
    buffer.unmap();
    buffer
}

/// Download data from a GPU buffer into a Vec<GoldilocksField>.
/// This is a blocking operation that maps the buffer, reads, and unmaps.
///
/// When `ctx.download_chunk_size` is non-zero and the total download exceeds
/// that size, the transfer is split into multiple map/read cycles using a
/// single reusable staging buffer. This reduces peak GPU memory and prevents
/// `BufferAsyncError` / device loss on memory-constrained GPUs (e.g. mobile).
///
/// On native: uses std::sync::mpsc channel + device.poll(Wait).
/// On WASM without async_prover: wgpu's WebGPU backend processes the callback
/// synchronously within poll(Wait) via the microtask queue.
#[cfg(not(feature = "async_prover"))]
pub fn download_field_data(
    ctx: &WebGpuContext,
    src_buffer: &wgpu::Buffer,
    num_elements: usize,
) -> Vec<GoldilocksField> {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let size_bytes = (num_elements * std::mem::size_of::<GoldilocksField>()) as u64;
    let chunk_size = ctx.download_chunk_size;

    // Use chunked path when configured and download is larger than one chunk.
    if chunk_size > 0 && size_bytes > chunk_size {
        return download_field_data_chunked_sync(device, queue, src_buffer, num_elements, chunk_size);
    }

    let staging = create_staging_buffer_read(device, size_bytes, "download_staging");

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("download_encoder"),
    });
    encoder.copy_buffer_to_buffer(src_buffer, 0, &staging, 0, size_bytes);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging.slice(..);

    use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
    let done = Arc::new(AtomicBool::new(false));
    let done_clone = done.clone();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        result.unwrap();
        done_clone.store(true, Ordering::SeqCst);
    });
    device.poll(wgpu::Maintain::Wait);
    assert!(done.load(Ordering::SeqCst), "Buffer mapping did not complete");

    let data = buffer_slice.get_mapped_range();
    let mut result = Vec::<GoldilocksField>::with_capacity(num_elements);
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const GoldilocksField,
            result.as_mut_ptr(),
            num_elements,
        );
        result.set_len(num_elements);
    }

    drop(data);
    staging.unmap();
    staging.destroy();

    result
}

/// Chunked synchronous download: copies `src_buffer` to CPU in fixed-size
/// chunks, reusing a single staging buffer to minimise peak GPU memory.
#[cfg(not(feature = "async_prover"))]
fn download_field_data_chunked_sync(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src_buffer: &wgpu::Buffer,
    num_elements: usize,
    chunk_bytes: u64,
) -> Vec<GoldilocksField> {
    use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

    let chunk_elems = (chunk_bytes as usize) / std::mem::size_of::<GoldilocksField>();
    let staging = create_staging_buffer_read(device, chunk_bytes, "download_staging_chunked");
    let mut result = Vec::<GoldilocksField>::with_capacity(num_elements);

    let mut offset = 0usize;
    while offset < num_elements {
        let count = (num_elements - offset).min(chunk_elems);
        let byte_count = (count * std::mem::size_of::<GoldilocksField>()) as u64;
        let byte_offset = (offset * std::mem::size_of::<GoldilocksField>()) as u64;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("download_chunk_encoder"),
        });
        encoder.copy_buffer_to_buffer(src_buffer, byte_offset, &staging, 0, byte_count);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..byte_count);
        let done = Arc::new(AtomicBool::new(false));
        let done_clone = done.clone();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            r.unwrap();
            done_clone.store(true, Ordering::SeqCst);
        });
        device.poll(wgpu::Maintain::Wait);
        assert!(done.load(Ordering::SeqCst), "Chunk buffer mapping did not complete");

        let data = slice.get_mapped_range();
        unsafe {
            let src_ptr = data.as_ptr() as *const GoldilocksField;
            let old_len = result.len();
            result.set_len(old_len + count);
            std::ptr::copy_nonoverlapping(src_ptr, result.as_mut_ptr().add(old_len), count);
        }
        drop(data);
        staging.unmap();

        offset += count;
    }

    staging.destroy();
    result
}

/// Async version of download_field_data for WASM.
/// Uses manual staging buffer with explicit destroy() to free GPU memory
/// immediately after reading. This is critical on memory-constrained devices
/// (e.g. iPhone Safari) where DownloadBuffer's deferred cleanup can cause
/// BufferAsyncError from accumulated memory pressure.
///
/// When `ctx.download_chunk_size` is non-zero and the total download exceeds
/// that size, the transfer is split into multiple map/read cycles using a
/// single reusable staging buffer of `download_chunk_size` bytes. Each chunk
/// cycle yields to the browser event loop, giving the GPU time to reclaim
/// memory between chunks.
///
/// On WASM, .await yields to the browser event loop so the GPU mapAsync
/// completion can fire.
#[cfg(feature = "async_prover")]
pub async fn download_field_data(
    ctx: &WebGpuContext,
    src_buffer: &wgpu::Buffer,
    num_elements: usize,
) -> Vec<GoldilocksField> {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let size_bytes = (num_elements * std::mem::size_of::<GoldilocksField>()) as u64;
    let chunk_size = ctx.download_chunk_size;

    // Use chunked path when configured and download is larger than one chunk.
    if chunk_size > 0 && size_bytes > chunk_size {
        let num_chunks = (size_bytes + chunk_size - 1) / chunk_size;
        log_gpu(&format!(
            "  [gpu] download: {} elems ({} bytes) [{} chunks × {} bytes]",
            num_elements, size_bytes, num_chunks, chunk_size,
        ));
        let result = download_field_data_chunked_async(device, queue, src_buffer, num_elements, chunk_size).await;
        log_gpu("  [gpu] download: complete");
        return result;
    }

    log_gpu(&format!("  [gpu] download: {} elems ({} bytes)", num_elements, size_bytes));

    // Copy source → staging and map, with retry (destroy + recreate) on OOM.
    let staging = copy_and_map_with_retry(
        device, queue, src_buffer, 0, size_bytes, "download_staging",
    ).await;

    // Read the mapped data.
    let data = staging.slice(..size_bytes).get_mapped_range();
    let mut result = Vec::<GoldilocksField>::with_capacity(num_elements);
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const GoldilocksField,
            result.as_mut_ptr(),
            num_elements,
        );
        result.set_len(num_elements);
    }

    drop(data);
    staging.unmap();
    // Explicitly destroy the staging buffer to free GPU memory immediately,
    // rather than waiting for JS GC. Prevents memory pressure on iOS Safari.
    staging.destroy();

    log_gpu("  [gpu] download: complete");

    result
}

/// Chunked async download: copies `src_buffer` to CPU in fixed-size chunks,
/// reusing a single staging buffer to minimise peak GPU memory. Each chunk
/// awaits the map operation, yielding to the browser event loop and giving
/// the GPU driver time to reclaim memory between iterations.
#[cfg(feature = "async_prover")]
async fn download_field_data_chunked_async(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src_buffer: &wgpu::Buffer,
    num_elements: usize,
    chunk_bytes: u64,
) -> Vec<GoldilocksField> {
    let chunk_elems = (chunk_bytes as usize) / std::mem::size_of::<GoldilocksField>();
    let mut result = Vec::<GoldilocksField>::with_capacity(num_elements);

    let mut offset = 0usize;
    while offset < num_elements {
        let count = (num_elements - offset).min(chunk_elems);
        let byte_count = (count * std::mem::size_of::<GoldilocksField>()) as u64;
        let byte_offset = (offset * std::mem::size_of::<GoldilocksField>()) as u64;

        // Copy chunk → staging and map, with retry (destroy + recreate) on OOM.
        let staging = copy_and_map_with_retry(
            device, queue, src_buffer, byte_offset, byte_count, "download_chunk_staging",
        ).await;

        let data = staging.slice(..byte_count).get_mapped_range();
        unsafe {
            let src_ptr = data.as_ptr() as *const GoldilocksField;
            let old_len = result.len();
            result.set_len(old_len + count);
            std::ptr::copy_nonoverlapping(src_ptr, result.as_mut_ptr().add(old_len), count);
        }
        drop(data);
        staging.unmap();
        staging.destroy();

        offset += count;
    }

    result
}

/// Maximum number of mapAsync retries on BufferAsyncError before giving up.
/// Increased from 3 to 6 to better handle cumulative memory pressure across
/// multiple proof commitments on mobile devices (iOS Safari).
#[cfg(feature = "async_prover")]
const MAP_ASYNC_MAX_RETRIES: u32 = 6;

/// Base delay in ms between mapAsync retries (doubles each attempt).
/// Increased from 100ms to 200ms to give the GPU driver more time to reclaim
/// memory. With 6 retries this gives: 200, 400, 800, 1600, 3200, 6400ms.
#[cfg(feature = "async_prover")]
const MAP_ASYNC_RETRY_BASE_MS: u32 = 200;

/// Copy `byte_count` bytes from `src_buffer` at `src_offset` into a freshly
/// created staging buffer, map it, and return the mapped staging buffer.
///
/// On iOS Safari, `mapAsync` can fail with `BufferAsyncError` when the GPU
/// process is under memory pressure.  Simply re-mapping the *same* staging
/// buffer does not free any GPU memory, so retries on the same buffer are
/// ineffective.  Instead, on failure we **destroy** the staging buffer
/// (releasing its GPU + IPC shared-memory allocation), wait for the GPU
/// process to reclaim memory, then create a **fresh** staging buffer and
/// re-submit the copy.  This gives the driver the best chance of finding
/// free memory on the next attempt.
///
/// Returns the staging buffer in the mapped state.  The caller must read
/// from it, call `unmap()`, then `destroy()`.
#[cfg(feature = "async_prover")]
async fn copy_and_map_with_retry(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src_buffer: &wgpu::Buffer,
    src_offset: u64,
    byte_count: u64,
    label: &str,
) -> wgpu::Buffer {
    for attempt in 0..=MAP_ASYNC_MAX_RETRIES {
        if attempt > 0 {
            let delay = MAP_ASYNC_RETRY_BASE_MS << (attempt - 1);
            log_gpu(&format!(
                "  [gpu] mapAsync retry {}/{} after {}ms (flush + recreate staging buffer)",
                attempt, MAP_ASYNC_MAX_RETRIES, delay,
            ));
            // Flush GPU memory first to let the driver reclaim destroyed buffers,
            // then sleep the requested backoff period.
            flush_gpu_memory(device).await;
            gpu_sleep_ms(delay).await;
        }

        // Create a fresh staging buffer for this attempt.
        let staging = create_staging_buffer_read(device, byte_count, label);

        // Copy from the source buffer into the staging buffer.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("download_retry_encoder"),
        });
        encoder.copy_buffer_to_buffer(src_buffer, src_offset, &staging, 0, byte_count);
        queue.submit(Some(encoder.finish()));

        // Try to map.
        let slice = staging.slice(..byte_count);
        let (tx, rx) = futures_channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });

        // On WASM this is a no-op; on native it drives completion.
        device.poll(wgpu::Maintain::Wait);

        match rx.await {
            Ok(Ok(())) => return staging,
            Ok(Err(e)) if attempt < MAP_ASYNC_MAX_RETRIES => {
                log_gpu(&format!("  [gpu] mapAsync failed: {:?}", e));
                // Unmap (reset internal state) then destroy to release the
                // staging buffer's GPU memory + IPC shared-memory allocation.
                staging.unmap();
                staging.destroy();
            }
            Ok(Err(e)) => {
                staging.unmap();
                staging.destroy();
                panic!(
                    "GPU buffer map failed after {} retries (BufferAsyncError: {:?}). \
                     Device may be lost due to memory pressure.",
                    MAP_ASYNC_MAX_RETRIES, e,
                );
            }
            Err(_) => panic!("GPU download channel cancelled"),
        }
    }
    unreachable!()
}

/// Async sleep that works in both browser main thread and Web Worker contexts.
/// On WASM, uses `globalThis.setTimeout` via `js_sys`. On native, blocks the
/// current thread (acceptable since the async executor is `pollster::block_on`).
#[cfg(feature = "async_prover")]
async fn gpu_sleep_ms(ms: u32) {
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        let promise = js_sys::Promise::new(&mut |resolve, _| {
            let global = js_sys::global();
            if let Ok(set_timeout) = js_sys::Reflect::get(&global, &wasm_bindgen::JsValue::from_str("setTimeout")) {
                if let Ok(func) = set_timeout.dyn_into::<js_sys::Function>() {
                    let _ = func.call2(
                        &wasm_bindgen::JsValue::undefined(),
                        &resolve,
                        &wasm_bindgen::JsValue::from(ms),
                    );
                }
            }
        });
        let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::sleep(std::time::Duration::from_millis(ms as u64));
    }
}

/// Flush GPU memory by polling the device to completion and yielding to the
/// browser event loop. On iOS Safari, `buffer.destroy()` marks the buffer for
/// deallocation, but the GPU process doesn't actually reclaim the memory until
/// the event loop runs. This function ensures destroyed buffers are truly freed
/// before we allocate new ones, preventing cumulative memory pressure across
/// multiple proof phases.
///
/// Call this after destroying large GPU buffers and before allocating new ones.
#[cfg(feature = "async_prover")]
pub async fn flush_gpu_memory(device: &wgpu::Device) {
    device.poll(wgpu::Maintain::Wait);
    // Yield to the event loop so the GPU process can reclaim destroyed buffers.
    // A 0ms setTimeout is sufficient — it just needs one event loop turn.
    gpu_sleep_ms(0).await;
}

/// Synchronous version of flush_gpu_memory for non-async targets.
/// Just polls the device to completion; no event loop yield is needed on native.
#[cfg(not(feature = "async_prover"))]
pub fn flush_gpu_memory(device: &wgpu::Device) {
    device.poll(wgpu::Maintain::Wait);
}

/// Convert a GoldilocksField element to Montgomery form.
/// a_mont = a * R mod p, where R = 2^64, p = 2^64 - 2^32 + 1.
/// R mod p = 2^32 - 1, so a_mont = a * (2^32 - 1) mod p.
pub fn to_mont(a: u64) -> u64 {
    const P: u128 = 18446744069414584321; // 2^64 - 2^32 + 1
    const R_MOD_P: u128 = 4294967295; // 2^32 - 1
    ((a as u128 * R_MOD_P) % P) as u64
}

/// Convert a Montgomery-form element back to standard form.
/// a = a_mont * R^{-1} mod p
/// R^{-1} mod p = inverse of (2^32 - 1) mod p
pub fn from_mont(a_mont: u64) -> u64 {
    // Montgomery reduction: mont_mul(a_mont, 1) = a_mont * 1 * R^{-1} = a * R * R^{-1} = a
    mont_mul(a_mont, 1)
}

/// Montgomery multiplication on CPU: (a * b * R^{-1}) mod p
/// Matches the Metal shader's branchless implementation exactly.
fn mont_mul(a: u64, b: u64) -> u64 {
    let xl = a.wrapping_mul(b);
    let xh = ((a as u128).wrapping_mul(b as u128) >> 64) as u64;

    // tmp = xl << 32 (64-bit truncating shift, matching Metal's ulong shift)
    let tmp = xl << 32;
    let (a_val, a_overflow_flag) = xl.overflowing_add(tmp);
    let a_overflow = a_overflow_flag as u64;

    let a_shr32 = a_val >> 32;
    let b_val = a_val.wrapping_sub(a_shr32).wrapping_sub(a_overflow);

    let (r, underflow) = xh.overflowing_sub(b_val);
    if underflow {
        r.wrapping_sub(0xFFFFFFFF) // subtract epsilon
    } else {
        r
    }
}

/// Convert a slice of GoldilocksField elements to Montgomery form.
pub fn fields_to_mont(fields: &[GoldilocksField]) -> Vec<u64> {
    fields.iter().map(|f| to_mont(f.0)).collect()
}

/// Convert a slice of Montgomery-form u64 values back to GoldilocksField.
pub fn fields_from_mont(mont_vals: &[u64]) -> Vec<GoldilocksField> {
    mont_vals.iter().map(|&v| GoldilocksField(from_mont(v))).collect()
}

/// Compute ceil_div(a, b)
pub fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}
