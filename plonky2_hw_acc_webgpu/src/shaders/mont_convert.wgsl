// Montgomery form conversion shader.
// Converts between standard and Montgomery representation on the GPU
// to avoid CPU-side conversion loops.
//
// Uses Y-batch pattern consistent with FFT shaders:
//   global_invocation_id.x = element index within polynomial
//   global_invocation_id.y = polynomial index

struct ConvertParams {
    n: u32,           // number of elements per polynomial
    num_polys: u32,   // number of polynomials
    direction: u32,   // 0 = to_mont, 1 = from_mont
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<vec2<u32>>;
@group(0) @binding(1) var<uniform> params: ConvertParams;

@compute @workgroup_size(256)
fn mont_convert_batched(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let elem_idx = gid.x;
    let poly_idx = gid.y;

    if (elem_idx >= params.n || poly_idx >= params.num_polys) {
        return;
    }

    let idx = poly_idx * params.n + elem_idx;
    let val = data[idx];
    if (params.direction == 0u) {
        data[idx] = gl_to_mont(val);
    } else {
        data[idx] = gl_from_mont(val);
    }
}
