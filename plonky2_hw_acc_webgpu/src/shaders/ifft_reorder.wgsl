// IFFT reorder and scale kernel - prepended with goldilocks.wgsl at compile time.
// All data in Montgomery form. n_inv is also in Montgomery form.

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
@group(0) @binding(1) var<uniform> params: FftParams;

fn load_val(idx: u32) -> vec2<u32> {
    let b = idx * 2u;
    return vec2<u32>(values[b], values[b + 1u]);
}
fn store_val(idx: u32, val: vec2<u32>) {
    let b = idx * 2u;
    values[b] = val.x;
    values[b + 1u] = val.y;
}

@compute @workgroup_size(256)
fn ifft_reorder_and_scale_batched(@builtin(global_invocation_id) gid: vec3<u32>) {
    let elem_idx = gid.x;
    let poly_idx = gid.y;
    let n = params.n;
    if (elem_idx >= n / 2u) {
        return;
    }

    let base = poly_idx * n;
    let n_inv_f = vec2<u32>(params.n_inv_lo, params.n_inv_hi);

    if (elem_idx == 0u) {
        let val0 = load_val(base);
        store_val(base, gl_mont_mul(val0, n_inv_f));

        if (n > 1u) {
            let val_half = load_val(base + n / 2u);
            store_val(base + n / 2u, gl_mont_mul(val_half, n_inv_f));
        }
        return;
    }

    let j = n - elem_idx;

    let val_i = load_val(base + elem_idx);
    let val_j = load_val(base + j);

    store_val(base + elem_idx, gl_mont_mul(val_j, n_inv_f));
    store_val(base + j, gl_mont_mul(val_i, n_inv_f));
}
