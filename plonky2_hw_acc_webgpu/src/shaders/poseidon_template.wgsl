// Poseidon permutation for Goldilocks field (WGSL)
// Prepended with goldilocks.wgsl + generated RC_MONT constants at build time.
// All state values are in Montgomery form.

const POSEIDON_WIDTH: u32 = 12u;
const POSEIDON_HALF_FULL_ROUNDS: u32 = 4u;
const POSEIDON_PARTIAL_ROUNDS: u32 = 22u;

// S-box: x^7 using Montgomery multiplication
fn sbox_mont(x: vec2<u32>) -> vec2<u32> {
    let x2 = gl_mont_mul(x, x);
    let x4 = gl_mont_mul(x2, x2);
    let x3 = gl_mont_mul(x2, x);
    return gl_mont_mul(x4, x3);
}

// MDS layer using addition-based small-constant multiplication.
// MDS_MATRIX_CIRC = {17, 15, 41, 16, 2, 28, 13, 13, 39, 18, 34, 20}
// MDS_MATRIX_DIAG = {8, 0, 0, ...}
fn mds_layer(
    s0: ptr<function, vec2<u32>>, s1: ptr<function, vec2<u32>>,
    s2: ptr<function, vec2<u32>>, s3: ptr<function, vec2<u32>>,
    s4: ptr<function, vec2<u32>>, s5: ptr<function, vec2<u32>>,
    s6: ptr<function, vec2<u32>>, s7: ptr<function, vec2<u32>>,
    s8: ptr<function, vec2<u32>>, s9: ptr<function, vec2<u32>>,
    s10: ptr<function, vec2<u32>>, s11: ptr<function, vec2<u32>>,
) {
    // Load state into local vars
    var st = array<vec2<u32>, 12>(
        *s0, *s1, *s2, *s3, *s4, *s5, *s6, *s7, *s8, *s9, *s10, *s11
    );

    // Precompute powers of 2
    var d2: array<vec2<u32>, 12>;
    var d4: array<vec2<u32>, 12>;
    var d8: array<vec2<u32>, 12>;
    var d16: array<vec2<u32>, 12>;
    var d32: array<vec2<u32>, 12>;

    for (var i = 0u; i < 12u; i++) {
        d2[i] = gl_add(st[i], st[i]);
        d4[i] = gl_add(d2[i], d2[i]);
        d8[i] = gl_add(d4[i], d4[i]);
        d16[i] = gl_add(d8[i], d8[i]);
        d32[i] = gl_add(d16[i], d16[i]);
    }

    var result: array<vec2<u32>, 12>;

    for (var r = 0u; r < 12u; r++) {
        let i0  = (0u + r) % 12u;
        let i1  = (1u + r) % 12u;
        let i2  = (2u + r) % 12u;
        let i3  = (3u + r) % 12u;
        let i4  = (4u + r) % 12u;
        let i5  = (5u + r) % 12u;
        let i6  = (6u + r) % 12u;
        let i7  = (7u + r) % 12u;
        let i8  = (8u + r) % 12u;
        let i9  = (9u + r) % 12u;
        let i10 = (10u + r) % 12u;
        let i11 = (11u + r) % 12u;

        let c0  = gl_add(d16[i0], st[i0]);                                  // x17
        let c1  = gl_sub(d16[i1], st[i1]);                                  // x15
        let c2  = gl_add(gl_add(d32[i2], d8[i2]), st[i2]);                  // x41
        let c3  = d16[i3];                                                   // x16
        let c4  = d2[i4];                                                    // x2
        let c5  = gl_add(gl_add(d16[i5], d8[i5]), d4[i5]);                  // x28
        let c6  = gl_add(gl_add(d8[i6], d4[i6]), st[i6]);                   // x13
        let c7  = gl_add(gl_add(d8[i7], d4[i7]), st[i7]);                   // x13
        let c8  = gl_add(gl_add(gl_add(d32[i8], d4[i8]), d2[i8]), st[i8]);  // x39
        let c9  = gl_add(d16[i9], d2[i9]);                                  // x18
        let c10 = gl_add(d32[i10], d2[i10]);                                // x34
        let c11 = gl_add(d16[i11], d4[i11]);                                // x20

        let t01   = gl_add(c0, c1);
        let t23   = gl_add(c2, c3);
        let t45   = gl_add(c4, c5);
        let t67   = gl_add(c6, c7);
        let t89   = gl_add(c8, c9);
        let tab   = gl_add(c10, c11);

        let t0123 = gl_add(t01, t23);
        let t4567 = gl_add(t45, t67);
        let t89ab = gl_add(t89, tab);

        var res = gl_add(gl_add(t0123, t4567), t89ab);

        if (r == 0u) {
            res = gl_add(res, d8[0u]); // diag[0] = 8
        }

        result[r] = res;
    }

    *s0 = result[0]; *s1 = result[1]; *s2 = result[2]; *s3 = result[3];
    *s4 = result[4]; *s5 = result[5]; *s6 = result[6]; *s7 = result[7];
    *s8 = result[8]; *s9 = result[9]; *s10 = result[10]; *s11 = result[11];
}

// Poseidon permutation on 12 field elements (all in Montgomery form)
fn permute(
    s0: ptr<function, vec2<u32>>, s1: ptr<function, vec2<u32>>,
    s2: ptr<function, vec2<u32>>, s3: ptr<function, vec2<u32>>,
    s4: ptr<function, vec2<u32>>, s5: ptr<function, vec2<u32>>,
    s6: ptr<function, vec2<u32>>, s7: ptr<function, vec2<u32>>,
    s8: ptr<function, vec2<u32>>, s9: ptr<function, vec2<u32>>,
    s10: ptr<function, vec2<u32>>, s11: ptr<function, vec2<u32>>,
) {
    var round_ctr = 0u;

    // Phase 1: half full rounds (4 iterations)
    for (var i = 0u; i < POSEIDON_HALF_FULL_ROUNDS; i++) {
        let rc_base = round_ctr * 12u;
        *s0 = gl_add(*s0, RC_MONT[rc_base + 0u]);
        *s1 = gl_add(*s1, RC_MONT[rc_base + 1u]);
        *s2 = gl_add(*s2, RC_MONT[rc_base + 2u]);
        *s3 = gl_add(*s3, RC_MONT[rc_base + 3u]);
        *s4 = gl_add(*s4, RC_MONT[rc_base + 4u]);
        *s5 = gl_add(*s5, RC_MONT[rc_base + 5u]);
        *s6 = gl_add(*s6, RC_MONT[rc_base + 6u]);
        *s7 = gl_add(*s7, RC_MONT[rc_base + 7u]);
        *s8 = gl_add(*s8, RC_MONT[rc_base + 8u]);
        *s9 = gl_add(*s9, RC_MONT[rc_base + 9u]);
        *s10 = gl_add(*s10, RC_MONT[rc_base + 10u]);
        *s11 = gl_add(*s11, RC_MONT[rc_base + 11u]);

        *s0 = sbox_mont(*s0); *s1 = sbox_mont(*s1);
        *s2 = sbox_mont(*s2); *s3 = sbox_mont(*s3);
        *s4 = sbox_mont(*s4); *s5 = sbox_mont(*s5);
        *s6 = sbox_mont(*s6); *s7 = sbox_mont(*s7);
        *s8 = sbox_mont(*s8); *s9 = sbox_mont(*s9);
        *s10 = sbox_mont(*s10); *s11 = sbox_mont(*s11);

        mds_layer(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11);
        round_ctr++;
    }

    // Phase 2: partial rounds (22 iterations, S-box on element 0 only)
    for (var i = 0u; i < POSEIDON_PARTIAL_ROUNDS; i++) {
        let rc_base = round_ctr * 12u;
        *s0 = gl_add(*s0, RC_MONT[rc_base + 0u]);
        *s1 = gl_add(*s1, RC_MONT[rc_base + 1u]);
        *s2 = gl_add(*s2, RC_MONT[rc_base + 2u]);
        *s3 = gl_add(*s3, RC_MONT[rc_base + 3u]);
        *s4 = gl_add(*s4, RC_MONT[rc_base + 4u]);
        *s5 = gl_add(*s5, RC_MONT[rc_base + 5u]);
        *s6 = gl_add(*s6, RC_MONT[rc_base + 6u]);
        *s7 = gl_add(*s7, RC_MONT[rc_base + 7u]);
        *s8 = gl_add(*s8, RC_MONT[rc_base + 8u]);
        *s9 = gl_add(*s9, RC_MONT[rc_base + 9u]);
        *s10 = gl_add(*s10, RC_MONT[rc_base + 10u]);
        *s11 = gl_add(*s11, RC_MONT[rc_base + 11u]);

        *s0 = sbox_mont(*s0); // S-box on element 0 only

        mds_layer(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11);
        round_ctr++;
    }

    // Phase 3: half full rounds (4 iterations)
    for (var i = 0u; i < POSEIDON_HALF_FULL_ROUNDS; i++) {
        let rc_base = round_ctr * 12u;
        *s0 = gl_add(*s0, RC_MONT[rc_base + 0u]);
        *s1 = gl_add(*s1, RC_MONT[rc_base + 1u]);
        *s2 = gl_add(*s2, RC_MONT[rc_base + 2u]);
        *s3 = gl_add(*s3, RC_MONT[rc_base + 3u]);
        *s4 = gl_add(*s4, RC_MONT[rc_base + 4u]);
        *s5 = gl_add(*s5, RC_MONT[rc_base + 5u]);
        *s6 = gl_add(*s6, RC_MONT[rc_base + 6u]);
        *s7 = gl_add(*s7, RC_MONT[rc_base + 7u]);
        *s8 = gl_add(*s8, RC_MONT[rc_base + 8u]);
        *s9 = gl_add(*s9, RC_MONT[rc_base + 9u]);
        *s10 = gl_add(*s10, RC_MONT[rc_base + 10u]);
        *s11 = gl_add(*s11, RC_MONT[rc_base + 11u]);

        *s0 = sbox_mont(*s0); *s1 = sbox_mont(*s1);
        *s2 = sbox_mont(*s2); *s3 = sbox_mont(*s3);
        *s4 = sbox_mont(*s4); *s5 = sbox_mont(*s5);
        *s6 = sbox_mont(*s6); *s7 = sbox_mont(*s7);
        *s8 = sbox_mont(*s8); *s9 = sbox_mont(*s9);
        *s10 = sbox_mont(*s10); *s11 = sbox_mont(*s11);

        mds_layer(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11);
        round_ctr++;
    }
}
