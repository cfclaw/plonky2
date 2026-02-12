// Transpose shader: converts from polynomial-major layout to evaluation-major layout.
// Input:  src[poly_idx * lde_size + eval_idx]  (num_polys polynomials, each lde_size elements)
// Output: dst[rev_eval_idx * num_cols + poly_idx] (lde_size rows, num_cols columns)
//
// Also performs bit-reverse on the row (evaluation) index for Merkle tree ordering.
//
// Uses Y-batch pattern:
//   global_invocation_id.x = evaluation index
//   global_invocation_id.y = polynomial index

struct TransposeParams {
    lde_size: u32,    // number of evaluations per polynomial
    num_polys: u32,   // number of source polynomials
    num_cols: u32,    // total output columns (num_polys + salt_size)
    log_lde_size: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: TransposeParams;

@compute @workgroup_size(256)
fn transpose_and_bit_reverse(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let eval_idx = gid.x;
    let poly_idx = gid.y;

    if (eval_idx >= params.lde_size || poly_idx >= params.num_polys) {
        return;
    }

    // Bit-reverse the evaluation index for Merkle tree ordering
    let rev_eval_idx = reverse_bits_32(eval_idx, params.log_lde_size);

    let src_idx = poly_idx * params.lde_size + eval_idx;
    let dst_idx = rev_eval_idx * params.num_cols + poly_idx;

    dst[dst_idx] = src[src_idx];
}
