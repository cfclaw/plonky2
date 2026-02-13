// Coset shift inverse shader: pointwise multiply each polynomial coefficient
// by shift_inv^i (in Montgomery form).
//
// Used to convert IFFT result to coset IFFT result on GPU, avoiding a
// GPU→CPU→GPU round-trip for this operation.
//
// Input/Output: data[poly_idx * n + elem_idx] (in Montgomery form)
// Shift inv powers: shift_inv_powers[elem_idx] (in Montgomery form)

struct CosetShiftInvParams {
    n: u32,           // number of elements per polynomial
    num_polys: u32,   // number of polynomials
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> shift_inv_powers: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: CosetShiftInvParams;

@compute @workgroup_size(256)
fn coset_shift_inv_batched(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let elem_idx = gid.x;
    let poly_idx = gid.y;

    if (elem_idx >= params.n || poly_idx >= params.num_polys) {
        return;
    }

    let idx = poly_idx * params.n + elem_idx;
    data[idx] = gl_mont_mul(data[idx], shift_inv_powers[elem_idx]);
}
