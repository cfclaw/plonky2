// Bit-reverse copy kernel - prepended with goldilocks.wgsl at compile time.

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

@group(0) @binding(0) var<storage, read_write> dst: array<u32>;
@group(0) @binding(1) var<storage, read> src_buf: array<u32>;
@group(0) @binding(2) var<uniform> params: FftParams;

fn store_dst(idx: u32, val: vec2<u32>) {
    let b = idx * 2u;
    dst[b] = val.x;
    dst[b + 1u] = val.y;
}
fn load_src(idx: u32) -> vec2<u32> {
    let b = idx * 2u;
    return vec2<u32>(src_buf[b], src_buf[b + 1u]);
}

@compute @workgroup_size(256)
fn bit_reverse_copy_batched(@builtin(global_invocation_id) gid: vec3<u32>) {
    let elem_idx = gid.x;
    let poly_idx = gid.y;
    if (elem_idx >= params.n) {
        return;
    }

    let base = poly_idx * params.n;
    let src_idx = reverse_bits_32(elem_idx, params.log_n);
    let val = load_src(base + src_idx);
    store_dst(base + elem_idx, val);
}
