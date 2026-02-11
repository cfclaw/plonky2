// FFT DIT butterfly pass - prepended with goldilocks.wgsl at compile time.
// All data in Montgomery form.

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

@group(0) @binding(0) var<storage, read_write> values: array<u32>;
@group(0) @binding(1) var<storage, read> twiddles: array<u32>;
@group(0) @binding(2) var<uniform> params: FftParams;

fn load_val(idx: u32) -> vec2<u32> {
    let b = idx * 2u;
    return vec2<u32>(values[b], values[b + 1u]);
}
fn store_val(idx: u32, val: vec2<u32>) {
    let b = idx * 2u;
    values[b] = val.x;
    values[b + 1u] = val.y;
}
fn load_tw(idx: u32) -> vec2<u32> {
    let b = idx * 2u;
    return vec2<u32>(twiddles[b], twiddles[b + 1u]);
}

@compute @workgroup_size(256)
fn fft_dit_pass_batched(@builtin(global_invocation_id) gid: vec3<u32>) {
    let butterfly_idx = gid.x;
    let poly_idx = gid.y;

    let n = params.n;
    if (butterfly_idx >= n / 2u) {
        return;
    }

    let base = poly_idx * params.stride;
    let layer = params.layer;

    let half_len = 1u << layer;
    let full_len = half_len << 1u;

    let block_idx = butterfly_idx / half_len;
    let idx_in_block = butterfly_idx % half_len;

    let u_idx = base + block_idx * full_len + idx_in_block;
    let v_idx = u_idx + half_len;

    let u = load_val(u_idx);
    let v = load_val(v_idx);
    let w = load_tw(params.twiddle_offset + idx_in_block);

    let t = gl_mont_mul(v, w);
    store_val(u_idx, gl_add(u, t));
    store_val(v_idx, gl_sub(u, t));
}
