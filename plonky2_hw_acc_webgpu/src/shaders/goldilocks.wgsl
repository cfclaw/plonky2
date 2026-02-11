// Goldilocks field arithmetic for WebGPU (WGSL)
// Field: p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
//
// Since WGSL lacks u64, we represent field elements as vec2<u32>:
//   .x = low 32 bits
//   .y = high 32 bits
//
// All GPU-side data is in Montgomery form. Conversion happens on the CPU side.

const GL_P_LO: u32 = 0x00000001u;
const GL_P_HI: u32 = 0xFFFFFFFFu;

// R^2 mod p where R = 2^64. For to_mont/from_mont on GPU (used by Poseidon).
// R^2 mod p = 0xFFFFFFFE00000001
const GL_R2_LO: u32 = 0x00000001u;
const GL_R2_HI: u32 = 0xFFFFFFFEu;

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    let carry_lo = select(0u, 1u, lo < a.x);
    let hi = a.y + b.y + carry_lo;
    return vec2<u32>(lo, hi);
}

fn add64_carry(a: vec2<u32>, b: vec2<u32>) -> u32 {
    let lo = a.x + b.x;
    let carry_lo = select(0u, 1u, lo < a.x);
    let s1 = a.y + b.y;
    let c1 = select(0u, 1u, s1 < a.y);
    let s2 = s1 + carry_lo;
    let c2 = select(0u, 1u, s2 < s1);
    return c1 + c2;
}

fn sub64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x - b.x;
    let borrow_lo = select(0u, 1u, a.x < b.x);
    let hi = a.y - b.y - borrow_lo;
    return vec2<u32>(lo, hi);
}

fn sub64_borrow(a: vec2<u32>, b: vec2<u32>) -> u32 {
    let borrow_lo = select(0u, 1u, a.x < b.x);
    let rhs = b.y + borrow_lo;
    let wrapped = select(0u, 1u, rhs < b.y);
    return select(select(0u, 1u, a.y < rhs), 1u, wrapped > 0u);
}

// Modular addition: a + b mod p
fn gl_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let p = vec2<u32>(GL_P_LO, GL_P_HI);
    let tmp = sub64(p, b);
    let underflow = sub64_borrow(a, tmp);
    let x1 = sub64(a, tmp);
    let adj = vec2<u32>(0u - underflow, 0u);
    return sub64(x1, adj);
}

// Modular subtraction: a - b mod p
fn gl_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let underflow = sub64_borrow(a, b);
    let x1 = sub64(a, b);
    let adj = vec2<u32>(0u - underflow, 0u);
    return sub64(x1, adj);
}

// 32x32 -> 64-bit multiply
fn mul32_wide(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;
    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;
    let mid = p1 + (p0 >> 16u);
    let mid2 = (mid & 0xFFFFu) + p2;
    let lo = (mid2 << 16u) | (p0 & 0xFFFFu);
    let hi = p3 + (mid >> 16u) + (mid2 >> 16u);
    return vec2<u32>(lo, hi);
}

// 64x64 -> 128-bit multiply
fn mul64_wide(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let p00 = mul32_wide(a.x, b.x);
    let p01 = mul32_wide(a.x, b.y);
    let p10 = mul32_wide(a.y, b.x);
    let p11 = mul32_wide(a.y, b.y);
    var w0 = p00.x;
    var w1 = p00.y;
    var w2 = p11.x;
    var w3 = p11.y;

    let s1 = w1 + p01.x;
    let c1 = select(0u, 1u, s1 < w1);
    w1 = s1;
    let s2 = w2 + p01.y + c1;
    let c2 = select(0u, 1u, s2 < w2 || (c1 > 0u && s2 <= w2));
    w2 = s2;
    w3 = w3 + c2;

    let s3 = w1 + p10.x;
    let c3 = select(0u, 1u, s3 < w1);
    w1 = s3;
    let s4 = w2 + p10.y + c3;
    let c4 = select(0u, 1u, s4 < w2 || (c3 > 0u && s4 <= w2));
    w2 = s4;
    w3 = w3 + c4;

    return vec4<u32>(w0, w1, w2, w3);
}

// Montgomery multiplication: (a * b * R^-1) mod p
fn gl_mont_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let prod = mul64_wide(a, b);
    let xl = vec2<u32>(prod.x, prod.y);
    let xh = vec2<u32>(prod.z, prod.w);
    let tmp = vec2<u32>(0u, xl.x);
    let a_val = add64(xl, tmp);
    let a_overflow = add64_carry(xl, tmp);
    let a_shr32 = vec2<u32>(a_val.y, 0u);
    let b_step1 = sub64(a_val, a_shr32);
    let b_val = sub64(b_step1, vec2<u32>(a_overflow, 0u));
    let r_underflow = sub64_borrow(xh, b_val);
    let r = sub64(xh, b_val);
    let adj = vec2<u32>(0u - r_underflow, 0u);
    return sub64(r, adj);
}

fn gl_to_mont(a: vec2<u32>) -> vec2<u32> {
    return gl_mont_mul(a, vec2<u32>(GL_R2_LO, GL_R2_HI));
}

fn gl_from_mont(a: vec2<u32>) -> vec2<u32> {
    return gl_mont_mul(a, vec2<u32>(1u, 0u));
}

fn reverse_bits_32(n: u32, bits: u32) -> u32 {
    var x = n;
    x = ((x & 0x55555555u) << 1u) | ((x >> 1u) & 0x55555555u);
    x = ((x & 0x33333333u) << 2u) | ((x >> 2u) & 0x33333333u);
    x = ((x & 0x0F0F0F0Fu) << 4u) | ((x >> 4u) & 0x0F0F0F0Fu);
    x = ((x & 0x00FF00FFu) << 8u) | ((x >> 8u) & 0x00FF00FFu);
    x = (x << 16u) | (x >> 16u);
    return x >> (32u - bits);
}
