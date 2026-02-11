// Merkle tree compute shaders for Plonky2 WebGPU backend.
// Prepended with goldilocks.wgsl + poseidon constants + poseidon functions at build time.
// Input leaf data is in standard form; conversion to/from Montgomery form happens in-shader.

// ============================================================
// Kernel: copy_row_leaves (NO-OP hash for leaf_size <= 4)
// ============================================================

struct MerkleParams {
    num_leaves: u32,
    leaf_size: u32,
    layer_idx: u32,
    pairs_per_subtree: u32,
    subtree_digest_len: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input_buf: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_buf: array<u32>;
@group(0) @binding(2) var<uniform> mparams: MerkleParams;

fn load_input_u64(idx: u32) -> u32 {
    // Input buffer stores raw u64 as pairs of u32 (lo, hi)
    return input_buf[idx];
}

fn store_output_u64_pair(idx: u32, lo: u32, hi: u32) {
    output_buf[idx * 2u] = lo;
    output_buf[idx * 2u + 1u] = hi;
}

@compute @workgroup_size(256)
fn copy_row_leaves(@builtin(global_invocation_id) gid: vec3<u32>) {
    let leaf_id = gid.x;
    if (leaf_id >= mparams.num_leaves) {
        return;
    }
    let in_off = leaf_id * mparams.leaf_size * 2u; // each element is 2 u32s
    let out_off = leaf_id * 4u * 2u; // 4 output elements, each 2 u32s

    for (var i = 0u; i < 4u; i++) {
        if (i < mparams.leaf_size) {
            output_buf[out_off + i * 2u] = input_buf[in_off + i * 2u];
            output_buf[out_off + i * 2u + 1u] = input_buf[in_off + i * 2u + 1u];
        } else {
            output_buf[out_off + i * 2u] = 0u;
            output_buf[out_off + i * 2u + 1u] = 0u;
        }
    }
}

// ============================================================
// Kernel: hash_row_leaves (Poseidon for leaf_size > 4)
// ============================================================

@compute @workgroup_size(256)
fn hash_row_leaves(@builtin(global_invocation_id) gid: vec3<u32>) {
    let leaf_id = gid.x;
    if (leaf_id >= mparams.num_leaves) {
        return;
    }
    let in_off = leaf_id * mparams.leaf_size * 2u;

    // Initialize state to zero (zero in Montgomery form is still 0)
    var s0 = vec2<u32>(0u, 0u);
    var s1 = vec2<u32>(0u, 0u);
    var s2 = vec2<u32>(0u, 0u);
    var s3 = vec2<u32>(0u, 0u);
    var s4 = vec2<u32>(0u, 0u);
    var s5 = vec2<u32>(0u, 0u);
    var s6 = vec2<u32>(0u, 0u);
    var s7 = vec2<u32>(0u, 0u);
    var s8 = vec2<u32>(0u, 0u);
    var s9 = vec2<u32>(0u, 0u);
    var s10 = vec2<u32>(0u, 0u);
    var s11 = vec2<u32>(0u, 0u);

    var curr = 0u;
    let leaf_size = mparams.leaf_size;

    // Absorb phase: load up to 8 elements per permutation (POSEIDON_RATE = 8)
    while (curr < leaf_size) {
        // Load elements into rate portion of state, converting to Montgomery form
        if (curr < leaf_size) {
            let v = vec2<u32>(input_buf[in_off + curr * 2u], input_buf[in_off + curr * 2u + 1u]);
            s0 = gl_to_mont(v); curr++;
        }
        if (curr < leaf_size) {
            let v = vec2<u32>(input_buf[in_off + curr * 2u], input_buf[in_off + curr * 2u + 1u]);
            s1 = gl_to_mont(v); curr++;
        }
        if (curr < leaf_size) {
            let v = vec2<u32>(input_buf[in_off + curr * 2u], input_buf[in_off + curr * 2u + 1u]);
            s2 = gl_to_mont(v); curr++;
        }
        if (curr < leaf_size) {
            let v = vec2<u32>(input_buf[in_off + curr * 2u], input_buf[in_off + curr * 2u + 1u]);
            s3 = gl_to_mont(v); curr++;
        }
        if (curr < leaf_size) {
            let v = vec2<u32>(input_buf[in_off + curr * 2u], input_buf[in_off + curr * 2u + 1u]);
            s4 = gl_to_mont(v); curr++;
        }
        if (curr < leaf_size) {
            let v = vec2<u32>(input_buf[in_off + curr * 2u], input_buf[in_off + curr * 2u + 1u]);
            s5 = gl_to_mont(v); curr++;
        }
        if (curr < leaf_size) {
            let v = vec2<u32>(input_buf[in_off + curr * 2u], input_buf[in_off + curr * 2u + 1u]);
            s6 = gl_to_mont(v); curr++;
        }
        if (curr < leaf_size) {
            let v = vec2<u32>(input_buf[in_off + curr * 2u], input_buf[in_off + curr * 2u + 1u]);
            s7 = gl_to_mont(v); curr++;
        }

        permute(&s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7, &s8, &s9, &s10, &s11);
    }

    // Squeeze: extract first 4 elements, convert back from Montgomery form
    let out_off = leaf_id * 4u * 2u;
    let o0 = gl_from_mont(s0);
    let o1 = gl_from_mont(s1);
    let o2 = gl_from_mont(s2);
    let o3 = gl_from_mont(s3);
    output_buf[out_off + 0u] = o0.x; output_buf[out_off + 1u] = o0.y;
    output_buf[out_off + 2u] = o1.x; output_buf[out_off + 3u] = o1.y;
    output_buf[out_off + 4u] = o2.x; output_buf[out_off + 5u] = o2.y;
    output_buf[out_off + 6u] = o3.x; output_buf[out_off + 7u] = o3.y;
}

// ============================================================
// Kernel: compress_nodes (2-to-1 hashing for internal tree nodes)
// Uses separate bindings since it needs linear_in, linear_out, and tree_out
// ============================================================

// NOTE: This kernel uses the same binding group but different buffer semantics.
// linear_in is binding 0 (input_buf), linear_out is binding 1 (output_buf).
// tree_out needs a separate binding.
@group(0) @binding(3) var<storage, read_write> tree_out: array<u32>;

@compute @workgroup_size(256)
fn compress_nodes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node_id = gid.x;
    // num_leaves is repurposed as num_pairs for this kernel
    let num_pairs = mparams.num_leaves;
    if (node_id >= num_pairs) {
        return;
    }

    // Read left and right children (4 elements each = 8 u32 pairs = 16 u32s)
    let in_base = node_id * 8u * 2u; // 8 field elements, 2 u32 each
    let left0 = vec2<u32>(input_buf[in_base + 0u], input_buf[in_base + 1u]);
    let left1 = vec2<u32>(input_buf[in_base + 2u], input_buf[in_base + 3u]);
    let left2 = vec2<u32>(input_buf[in_base + 4u], input_buf[in_base + 5u]);
    let left3 = vec2<u32>(input_buf[in_base + 6u], input_buf[in_base + 7u]);
    let right0 = vec2<u32>(input_buf[in_base + 8u], input_buf[in_base + 9u]);
    let right1 = vec2<u32>(input_buf[in_base + 10u], input_buf[in_base + 11u]);
    let right2 = vec2<u32>(input_buf[in_base + 12u], input_buf[in_base + 13u]);
    let right3 = vec2<u32>(input_buf[in_base + 14u], input_buf[in_base + 15u]);

    // Setup state, convert to Montgomery form
    var s0 = gl_to_mont(left0);
    var s1 = gl_to_mont(left1);
    var s2 = gl_to_mont(left2);
    var s3 = gl_to_mont(left3);
    var s4 = gl_to_mont(right0);
    var s5 = gl_to_mont(right1);
    var s6 = gl_to_mont(right2);
    var s7 = gl_to_mont(right3);
    var s8 = vec2<u32>(0u, 0u);
    var s9 = vec2<u32>(0u, 0u);
    var s10 = vec2<u32>(0u, 0u);
    var s11 = vec2<u32>(0u, 0u);

    // Permute
    permute(&s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7, &s8, &s9, &s10, &s11);

    // Extract parent hash (first 4 elements, back to standard form)
    let p0 = gl_from_mont(s0);
    let p1 = gl_from_mont(s1);
    let p2 = gl_from_mont(s2);
    let p3 = gl_from_mont(s3);

    // Write parent to linear output
    let out_base = node_id * 4u * 2u;
    output_buf[out_base + 0u] = p0.x; output_buf[out_base + 1u] = p0.y;
    output_buf[out_base + 2u] = p1.x; output_buf[out_base + 3u] = p1.y;
    output_buf[out_base + 4u] = p2.x; output_buf[out_base + 5u] = p2.y;
    output_buf[out_base + 6u] = p3.x; output_buf[out_base + 7u] = p3.y;

    // Write children to plonky2 tree layout
    let layer_idx = mparams.layer_idx;
    let pairs_per_subtree = mparams.pairs_per_subtree;
    let subtree_digest_len = mparams.subtree_digest_len;

    let sub_id = node_id / pairs_per_subtree;
    let local_pair_id = node_id % pairs_per_subtree;
    let p2_l = 1u << layer_idx;
    let local_idx = (local_pair_id * (p2_l << 1u)) + p2_l - 1u;
    let global_idx = (sub_id * subtree_digest_len) + (local_idx * 2u);

    // Write left child
    let tree_base_left = global_idx * 4u * 2u;
    tree_out[tree_base_left + 0u] = left0.x; tree_out[tree_base_left + 1u] = left0.y;
    tree_out[tree_base_left + 2u] = left1.x; tree_out[tree_base_left + 3u] = left1.y;
    tree_out[tree_base_left + 4u] = left2.x; tree_out[tree_base_left + 5u] = left2.y;
    tree_out[tree_base_left + 6u] = left3.x; tree_out[tree_base_left + 7u] = left3.y;

    // Write right child
    let tree_base_right = (global_idx + 1u) * 4u * 2u;
    tree_out[tree_base_right + 0u] = right0.x; tree_out[tree_base_right + 1u] = right0.y;
    tree_out[tree_base_right + 2u] = right1.x; tree_out[tree_base_right + 3u] = right1.y;
    tree_out[tree_base_right + 4u] = right2.x; tree_out[tree_base_right + 5u] = right2.y;
    tree_out[tree_base_right + 6u] = right3.x; tree_out[tree_base_right + 7u] = right3.y;
}
