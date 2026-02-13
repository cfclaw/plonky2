// Scatter random salt values from a contiguous buffer into leaf buffer columns.
//
// The leaf buffer is row-major with `num_cols` columns per row.
// Polynomial data occupies columns [0..num_polys). This shader fills
// the salt columns [num_polys..num_polys+salt_size) from a contiguous
// source buffer of shape [lde_size x salt_size].
//
// Each thread handles one row.

struct ScatterSaltParams {
    lde_size: u32,
    num_polys: u32,
    num_cols: u32,
    salt_size: u32,
}

@group(0) @binding(0) var<storage, read_write> dst: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> salt_buf: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: ScatterSaltParams;

@compute @workgroup_size(256)
fn scatter_salt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.lde_size) {
        return;
    }
    let dst_base = row * params.num_cols + params.num_polys;
    let src_base = row * params.salt_size;
    for (var s = 0u; s < params.salt_size; s = s + 1u) {
        dst[dst_base + s] = salt_buf[src_base + s];
    }
}
