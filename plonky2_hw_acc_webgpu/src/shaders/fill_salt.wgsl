// Fill salt columns in the transposed leaf buffer.
//
// After GPU transpose, the leaf buffer has layout:
//   row r: [poly0_eval_r, poly1_eval_r, ..., 0, 0, ..., 0]
//   where the trailing zeros are salt column slots.
//
// This shader fills those salt columns with pre-generated random values.
// Salt data is stored contiguously: salt[r * salt_size + s] for row r, salt column s.
// Leaf layout: leaves[r * num_cols + num_polys + s] for row r, salt column s.

struct FillSaltParams {
    lde_size: u32,     // number of rows
    num_cols: u32,     // total columns per row (num_polys + salt_size)
    num_polys: u32,    // number of polynomial columns
    salt_size: u32,    // number of salt columns
}

@group(0) @binding(0) var<storage, read_write> leaves: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> salt_data: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: FillSaltParams;

@compute @workgroup_size(256)
fn fill_salt_batched(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let row = gid.x;
    let salt_col = gid.y;

    if (row >= params.lde_size || salt_col >= params.salt_size) {
        return;
    }

    let dst_idx = row * params.num_cols + params.num_polys + salt_col;
    let src_idx = row * params.salt_size + salt_col;
    leaves[dst_idx] = salt_data[src_idx];
}
