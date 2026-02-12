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
/// On native: uses std::sync::mpsc channel + device.poll(Wait).
/// On WASM without async_prover: wgpu's WebGPU backend processes the callback
/// synchronously within poll(Wait) via the microtask queue.
#[cfg(not(feature = "async_prover"))]
pub fn download_field_data(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src_buffer: &wgpu::Buffer,
    num_elements: usize,
) -> Vec<GoldilocksField> {
    let size_bytes = (num_elements * std::mem::size_of::<GoldilocksField>()) as u64;

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
    // Directly construct the Vec from the mapped bytes without an intermediate
    // slice-to-vec copy. The mapped range is only valid until we drop `data`,
    // so we copy once into a pre-allocated Vec.
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

/// Async version of download_field_data for WASM.
/// Uses manual staging buffer with explicit destroy() to free GPU memory
/// immediately after reading. This is critical on memory-constrained devices
/// (e.g. iPhone Safari) where DownloadBuffer's deferred cleanup can cause
/// BufferAsyncError from accumulated memory pressure.
/// On WASM, .await yields to the browser event loop so the GPU mapAsync
/// completion can fire.
#[cfg(feature = "async_prover")]
pub async fn download_field_data(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src_buffer: &wgpu::Buffer,
    num_elements: usize,
) -> Vec<GoldilocksField> {
    let size_bytes = (num_elements * std::mem::size_of::<GoldilocksField>()) as u64;

    log_gpu(&format!("  [gpu] download: {} elems ({} bytes)", num_elements, size_bytes));

    // Create staging buffer explicitly so we control its lifecycle.
    let staging = create_staging_buffer_read(device, size_bytes, "download_staging");

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("download_encoder"),
    });
    encoder.copy_buffer_to_buffer(src_buffer, 0, &staging, 0, size_bytes);
    queue.submit(Some(encoder.finish()));

    // Map the staging buffer for reading via oneshot channel.
    let buffer_slice = staging.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    // On WASM this is a no-op; on native it drives completion.
    device.poll(wgpu::Maintain::Wait);

    // Await the mapping — yields to browser event loop on WASM.
    rx.await
        .expect("GPU download channel cancelled")
        .expect("GPU buffer map failed (BufferAsyncError). Device may be lost due to memory pressure.");

    // Read the mapped data.
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
    // Explicitly destroy the staging buffer to free GPU memory immediately,
    // rather than waiting for JS GC. Prevents memory pressure on iOS Safari.
    staging.destroy();

    log_gpu("  [gpu] download: complete");

    result
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
