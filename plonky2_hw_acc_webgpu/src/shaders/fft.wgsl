// FFT compute shaders for Plonky2 WebGPU backend.
// This file is prepended with goldilocks.wgsl at shader compilation time.
// All data on GPU is in Montgomery form.
// Field elements stored as pairs of u32: [lo_0, hi_0, lo_1, hi_1, ...]

// ============================================================
// Uniform params (shared by all FFT kernels)
// ============================================================

struct FftParams {
    n: u32,
    log_n: u32,
    num_coeffs: u32,
    layer: u32,
    stride: u32,
    n_inv_lo: u32,
    n_inv_hi: u32,
    twiddle_offset: u32,
}

// ============================================================
// Kernel: coset_scale_and_bit_reverse_batched
//
// Bindings:
//   0: dst (storage, read_write) - output LDE buffer
//   1: src (storage, read)       - input coefficients
//   2: shifts (storage, read)    - powers of shift (Montgomery form)
//   3: params (uniform)
// ============================================================

@group(0) @binding(0) var<storage, read_write> dst: array<u32>;
@group(0) @binding(1) var<storage, read> src_buf: array<u32>;
@group(0) @binding(2) var<storage, read> shifts_buf: array<u32>;
@group(0) @binding(3) var<uniform> params: FftParams;

fn load_dst(idx: u32) -> vec2<u32> {
    let b = idx * 2u;
    return vec2<u32>(dst[b], dst[b + 1u]);
}
fn store_dst(idx: u32, val: vec2<u32>) {
    let b = idx * 2u;
    dst[b] = val.x;
    dst[b + 1u] = val.y;
}
fn load_src(idx: u32) -> vec2<u32> {
    let b = idx * 2u;
    return vec2<u32>(src_buf[b], src_buf[b + 1u]);
}
fn load_shift(idx: u32) -> vec2<u32> {
    let b = idx * 2u;
    return vec2<u32>(shifts_buf[b], shifts_buf[b + 1u]);
}

@compute @workgroup_size(256)
fn coset_scale_and_bit_reverse_batched(@builtin(global_invocation_id) gid: vec3<u32>) {
    let elem_idx = gid.x;
    let poly_idx = gid.y;
    if (elem_idx >= params.n) {
        return;
    }

    let dst_offset = poly_idx * params.n;
    let src_offset = poly_idx * params.num_coeffs;
    let src_idx = reverse_bits_32(elem_idx, params.log_n);

    if (src_idx < params.num_coeffs) {
        let val = load_src(src_offset + src_idx);
        let shift = load_shift(src_idx);
        store_dst(dst_offset + elem_idx, gl_mont_mul(val, shift));
    } else {
        store_dst(dst_offset + elem_idx, vec2<u32>(0u, 0u));
    }
}
