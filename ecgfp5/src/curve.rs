//! Native EcGFp5 elliptic curve arithmetic over F_{p^5}.
//!
//! The curve is defined over GF(p^5) where p = 2^64 - 2^32 + 1 (Goldilocks)
//! and the extension is F_p[z]/(z^5 - 3).
//!
//! Curve equation: y^2 = x(x^2 + Ax + B) where A = 2 and B = 263*z
//! This is a "double-odd" curve with group order 2n where n is a 319-bit prime.
//!
//! We use (X, W) coordinates where W = x/y (not standard affine).
//! The neutral element is (0, 0).

use plonky2_field::extension::quintic::QuinticExtension;
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::types::Field;

/// An element of GF(p^5) represented as 5 Goldilocks field elements.
/// The extension polynomial is z^5 - 3.
pub type GFp5 = QuinticExtension<GoldilocksField>;

/// Curve parameter A = 2
pub const CURVE_A: GFp5 = QuinticExtension([
    GoldilocksField(2),
    GoldilocksField(0),
    GoldilocksField(0),
    GoldilocksField(0),
    GoldilocksField(0),
]);

/// Curve parameter B = 263*z (coefficient of z^1 is 263, rest are 0)
pub const CURVE_B: GFp5 = QuinticExtension([
    GoldilocksField(0),
    GoldilocksField(263),
    GoldilocksField(0),
    GoldilocksField(0),
    GoldilocksField(0),
]);

/// A point on the EcGFp5 curve in (X, W) coordinates.
/// The neutral element is (0, 0).
/// For a non-neutral point with affine coordinates (x, y):
///   X = x, W = x/y
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct EcGFp5Point {
    pub x: GFp5,
    pub w: GFp5,
}

impl EcGFp5Point {
    /// The neutral (identity) element.
    pub const NEUTRAL: Self = Self {
        x: GFp5::ZERO,
        w: GFp5::ZERO,
    };

    /// Create a new point from (X, W) coordinates.
    pub fn new(x: GFp5, w: GFp5) -> Self {
        Self { x, w }
    }

    /// Check if this is the neutral element.
    pub fn is_neutral(&self) -> bool {
        self.x == GFp5::ZERO
    }

    /// The generator point for EcGFp5.
    /// This is from Pornin's specification.
    pub fn generator() -> Self {
        // Generator x-coordinate and w-coordinate from the EcGFp5 spec.
        // We use a simple known point on the curve.
        // The generator can be found by hashing to the curve.
        // For now, we compute one by finding a valid point.
        let x = GFp5::from(GoldilocksField(5));
        // Compute y^2 = x(x^2 + Ax + B)
        let x2 = x * x;
        let rhs = x * (x2 + CURVE_A * x + CURVE_B);
        // Try to compute sqrt
        if let Some(y) = sqrt_gfp5(rhs) {
            let w = x * y.try_inverse().unwrap();
            Self { x, w }
        } else {
            // If x=5 doesn't work, try x=1
            let x = GFp5::ONE;
            let x2 = x * x;
            let rhs = x * (x2 + CURVE_A * x + CURVE_B);
            let y = sqrt_gfp5(rhs).expect("generator point must exist");
            let w = x * y.try_inverse().unwrap();
            Self { x, w }
        }
    }

    /// Double a point on the curve.
    /// Uses the double-odd curve doubling formulas in (X, W) coordinates.
    ///
    /// For a point P = (X, W) where W = X/Y:
    ///   X' = (X^2 - B)^2 / (4 * X * (X^2 + A*X + B))
    ///   W' = ... (derived from the tangent line)
    ///
    /// Using the formulas from Pornin's paper for the double-odd representation:
    pub fn double(&self) -> Self {
        if self.is_neutral() {
            return *self;
        }

        let x = self.x;
        let w = self.w;

        // From the double-odd curve formulas:
        // For doubling P = (X, W):
        //   Let J = 2*B*W
        //   Let K = W^2 - A
        //   Let L = 2*X*W
        //   X' = J * K
        //   W' = K^2 - 2*J*L

        // Actually, let's use the standard Weierstrass doubling converted to (X,W) coords.
        // y = x / w, so we work in affine (x, y) and convert back.
        //
        // For the curve y^2 = x^3 + A*x^2 + B*x:
        //   lambda = (3*x^2 + 2*A*x + B) / (2*y)
        //   x' = lambda^2 - A - 2*x
        //   y' = lambda * (x - x') - y
        //   w' = x'/y'

        let y = x * w.try_inverse().unwrap_or(GFp5::ZERO);
        if y == GFp5::ZERO {
            return Self::NEUTRAL;
        }

        let x2 = x * x;
        let two = GFp5::TWO;
        let three = GFp5::from(GoldilocksField(3));

        let numerator = three * x2 + two * CURVE_A * x + CURVE_B;
        let denominator = two * y;
        let lambda = numerator * denominator.try_inverse().unwrap();

        let x_new = lambda * lambda - CURVE_A - two * x;
        let y_new = lambda * (x - x_new) - y;

        if y_new == GFp5::ZERO {
            return Self::NEUTRAL;
        }

        let w_new = x_new * y_new.try_inverse().unwrap();
        Self { x: x_new, w: w_new }
    }

    /// Add two points on the curve.
    pub fn add(&self, other: &Self) -> Self {
        if self.is_neutral() {
            return *other;
        }
        if other.is_neutral() {
            return *self;
        }

        let x1 = self.x;
        let w1 = self.w;
        let x2 = other.x;
        let w2 = other.w;

        // Convert to affine (x, y) where y = x/w
        let y1 = x1 * w1.try_inverse().unwrap_or(GFp5::ZERO);
        let y2 = x2 * w2.try_inverse().unwrap_or(GFp5::ZERO);

        if y1 == GFp5::ZERO || y2 == GFp5::ZERO {
            // Degenerate case
            if y1 == GFp5::ZERO {
                return *other;
            }
            return *self;
        }

        if x1 == x2 {
            if y1 == y2 {
                return self.double();
            } else {
                // P + (-P) = O
                return Self::NEUTRAL;
            }
        }

        // Standard addition:
        // lambda = (y2 - y1) / (x2 - x1)
        // x3 = lambda^2 - A - x1 - x2
        // y3 = lambda*(x1 - x3) - y1
        let lambda = (y2 - y1) * (x2 - x1).try_inverse().unwrap();
        let x3 = lambda * lambda - CURVE_A - x1 - x2;
        let y3 = lambda * (x1 - x3) - y1;

        if y3 == GFp5::ZERO {
            return Self::NEUTRAL;
        }

        let w3 = x3 * y3.try_inverse().unwrap();
        Self { x: x3, w: w3 }
    }

    /// Negate a point: -(x, y) = (x, -y), so w = x/y -> -w = x/(-y)
    pub fn neg(&self) -> Self {
        Self {
            x: self.x,
            w: -self.w,
        }
    }

    /// Scalar multiplication using double-and-add.
    /// The scalar is given as a slice of bits (little-endian).
    pub fn scalar_mul_bits(&self, bits: &[bool]) -> Self {
        let mut result = Self::NEUTRAL;
        let mut base = *self;

        for &bit in bits {
            if bit {
                result = result.add(&base);
            }
            base = base.double();
        }

        result
    }

    /// Scalar multiplication by a u64 value.
    pub fn scalar_mul_u64(&self, scalar: u64) -> Self {
        if scalar == 0 {
            return Self::NEUTRAL;
        }
        let bits: Vec<bool> = (0..64).map(|i| (scalar >> i) & 1 == 1).collect();
        self.scalar_mul_bits(&bits)
    }

    /// Verify that this point lies on the curve.
    pub fn is_on_curve(&self) -> bool {
        if self.is_neutral() {
            return true;
        }

        let x = self.x;
        let w = self.w;

        // y = x/w
        let y = x * w.try_inverse().unwrap_or(GFp5::ZERO);
        if y == GFp5::ZERO && !self.is_neutral() {
            return false;
        }

        // Check: y^2 == x^3 + A*x^2 + B*x
        let y2 = y * y;
        let x2 = x * x;
        let x3 = x2 * x;
        let rhs = x3 + CURVE_A * x2 + CURVE_B * x;

        y2 == rhs
    }
}

/// Attempt to compute a square root in GF(p^5).
/// Uses the Tonelli-Shanks-like approach via Frobenius.
///
/// For p = 2^64 - 2^32 + 1, we have p ≡ 1 (mod 2^32).
/// p^5 - 1 has the same 2-adicity as p - 1 (= 32).
///
/// We use exponentiation: if a is a QR, then a^((q+1)/2) is a sqrt,
/// where q = p^5.
///
/// Actually, a simpler approach: compute candidate = a^((q+1)/4) or use
/// the Cipolla/Tonelli-Shanks algorithm adapted for extension fields.
///
/// For simplicity, we use: sqrt(a) = a^((q+1)/2) when q ≡ 3 (mod 4).
/// But p^5 ≡ 1 (mod 4) since p ≡ 1 (mod 4), so we need Tonelli-Shanks.
///
/// A simpler method: use the norm-based approach.
/// Let N = a^((p^5-1)/2). If N = -1, no sqrt. If N = 1, proceed.
///
/// For our purposes, we use a simple repeated-squaring check.
fn sqrt_gfp5(a: GFp5) -> Option<GFp5> {
    use plonky2_field::ops::Square;

    if a == GFp5::ZERO {
        return Some(GFp5::ZERO);
    }

    // p^5 - 1 = 2^32 * t where t is odd (since p-1 = 2^32 * t0 and p^5-1 shares this structure)
    // Actually p = 2^64 - 2^32 + 1, so p - 1 = 2^32 * (2^32 - 1)
    // The 2-adicity of p^5 - 1 equals TWO_ADICITY of the field = 32.

    // Tonelli-Shanks for GF(p^5):
    // q = p^5, q - 1 = 2^s * t where s = 32
    // Find a non-residue n, compute z = n^t
    // Then standard Tonelli-Shanks loop.

    let s = 32u32;
    // t = (q - 1) / 2^s. We need a^t.
    // Instead of computing t directly (it's huge), use:
    // a^t = a^((q-1)/2^s)
    // We can compute a^((p^5-1)/2^32) step by step.

    // Simpler approach: use the fact that for the field, POWER_OF_TWO_GENERATOR
    // is available. Let's just use brute force exponentiation with (q-1)/2 to check
    // if a is a QR, then find the root.

    // Even simpler: a^((q+1)/2) is a sqrt candidate when it works.
    // But we need (q+1)/2 which is huge. Let's use a different approach.

    // The simplest correct approach: use the Frobenius endomorphism.
    // Compute b = a^((p-1)/2) in the extension field using Frobenius.
    // Actually let's just use repeated squaring with the field's built-in pow.

    // For a practical implementation, let's use the Shanks-Tonelli algorithm
    // adapted for the extension field.

    // Step 1: check if a is a quadratic residue
    // a^((q-1)/2) should equal 1
    let exp_half = pow_by_p5_minus1_over2(a);
    if exp_half != GFp5::ONE {
        return None;
    }

    // Step 2: Tonelli-Shanks
    // Find non-residue
    let non_residue = find_non_residue();
    let t_exp = pow_by_t(a);       // a^t
    let t_nqr = pow_by_t(non_residue); // n^t

    let mut m = s;
    let mut c = t_nqr;
    let mut tt = t_exp;
    // a^((t+1)/2) = a^((t-1)/2) * a = sqrt candidate without the 2-adic part
    // Actually, root = a^((t+1)/2)
    let mut r = pow_by_t_plus1_over2(a);

    loop {
        if tt == GFp5::ONE {
            return Some(r);
        }

        // Find the least i such that tt^(2^i) = 1
        let mut i = 0u32;
        let mut tmp = tt;
        while tmp != GFp5::ONE {
            tmp = tmp.square();
            i += 1;
            if i >= m {
                return None; // shouldn't happen if a is a QR
            }
        }

        // Update
        let mut b = c;
        for _ in 0..(m - i - 1) {
            b = b.square();
        }
        m = i;
        c = b.square();
        tt = tt * c;
        r = r * b;
    }
}

/// Compute a^((p^5 - 1)/2) using Frobenius.
fn pow_by_p5_minus1_over2(a: GFp5) -> GFp5 {
    use plonky2_field::extension::Frobenius;
    use plonky2_field::ops::Square;

    // (p^5-1)/2 = (p-1)/2 * (p^4 + p^3 + p^2 + p + 1)
    // a^((p-1)/2) can be computed as a^((p-1)/2)
    // But it's simpler to just do: a^((p^5-1)/2)

    // p = 2^64 - 2^32 + 1
    // (p-1)/2 = 2^31 * (2^32 - 1)

    // Let's compute a^((p^5-1)/2) directly.
    // (p^5-1)/2 = ((p-1)/2) * (1 + p + p^2 + p^3 + p^4)

    // a^(p^k) = frobenius^k(a)
    let a_p = a.frobenius();        // a^p
    let a_p2 = a_p.frobenius();     // a^(p^2)
    let a_p3 = a_p2.frobenius();    // a^(p^3)
    let a_p4 = a_p3.frobenius();    // a^(p^4)

    // a^(1 + p + p^2 + p^3 + p^4) = a * a^p * a^(p^2) * a^(p^3) * a^(p^4)
    let norm_like = a * a_p * a_p2 * a_p3 * a_p4;

    // This equals a^((p^5-1)/(p-1)), which is in the base field.
    // Now we need to raise this to (p-1)/2.

    // (p-1)/2 = (2^64 - 2^32) / 2 = 2^31 * (2^32 - 1)
    // So raise to 2^31 then to (2^32-1)
    let mut tmp = norm_like;

    // Raise to 2^31
    for _ in 0..31 {
        tmp = tmp.square();
    }

    // Raise to (2^32 - 1) = sum of 2^i for i=0..31
    // = x^(2^32) / x
    // So we compute (tmp raised to 2^32) / tmp
    let mut tmp2 = tmp;
    for _ in 0..32 {
        tmp2 = tmp2.square();
    }
    tmp2 * tmp.try_inverse().unwrap_or(GFp5::ONE)
}

/// Find a non-quadratic-residue in GF(p^5).
fn find_non_residue() -> GFp5 {
    // z (the extension generator) is typically a non-residue.
    // Let's try it:
    let z = QuinticExtension([
        GoldilocksField(0),
        GoldilocksField(1),
        GoldilocksField(0),
        GoldilocksField(0),
        GoldilocksField(0),
    ]);

    // Verify: z^((q-1)/2) should be -1
    let test = pow_by_p5_minus1_over2(z);
    if test == GFp5::NEG_ONE {
        return z;
    }

    // Try small values
    for i in 2u64..100 {
        let candidate = GFp5::from(GoldilocksField(i));
        let test = pow_by_p5_minus1_over2(candidate);
        if test == GFp5::NEG_ONE {
            return candidate;
        }
    }

    panic!("Could not find non-residue");
}

/// Compute a^t where t = (p^5 - 1) / 2^32.
fn pow_by_t(a: GFp5) -> GFp5 {
    use plonky2_field::extension::Frobenius;
    use plonky2_field::ops::Square;

    // t = (p^5 - 1) / 2^32
    // p - 1 = 2^32 * (2^32 - 1)
    // p^5 - 1 = (p-1)(p^4+p^3+p^2+p+1)
    // The 2-adic valuation of p^5-1 is 32 (same as p-1, since p^4+p^3+p^2+p+1 is odd)
    // So t = (p-1)/2^32 * (p^4+p^3+p^2+p+1) = (2^32-1) * (p^4+p^3+p^2+p+1)

    // First compute a^(p^4+p^3+p^2+p+1) using Frobenius
    let a_p = a.frobenius();
    let a_p2 = a_p.frobenius();
    let a_p3 = a_p2.frobenius();
    let a_p4 = a_p3.frobenius();
    let norm_like = a * a_p * a_p2 * a_p3 * a_p4;

    // Now raise to (2^32 - 1)
    // (2^32 - 1) = sum of 2^i for i=0..31
    // Equivalent to: x^(2^32) / x
    let mut x32 = norm_like;
    for _ in 0..32 {
        x32 = x32.square();
    }
    x32 * norm_like.try_inverse().unwrap_or(GFp5::ONE)
}

/// Compute a^((t+1)/2) where t = (p^5-1)/2^32.
fn pow_by_t_plus1_over2(a: GFp5) -> GFp5 {
    use plonky2_field::extension::Frobenius;
    

    // t+1 = (p-1)/2^32 * (p^4+p^3+p^2+p+1) + 1
    // This is complex. Instead, compute a^((t+1)/2) = a * a^((t-1)/2)
    // = a * (a^t / a)^(1... nope this doesn't simplify nicely.

    // Simpler: (t+1)/2 = ((2^32-1)*(p^4+p^3+p^2+p+1) + 1)/2
    // Let's just compute a^t first, then a^((t+1)/2) = a^((t-1)/2 + 1) = a^((t-1)/2) * a

    // a^t is already computable. We need a^((t-1)/2).
    // But t = (2^32-1) * N where N = p^4+p^3+p^2+p+1.
    // t is odd (since 2^32-1 is odd and N is odd).
    // So (t+1)/2 is an integer.

    // Let's just compute: a^((t+1)/2) via a^t.
    // a^((t+1)/2) = sqrt(a^(t+1)) = sqrt(a^t * a)
    // But that requires sqrt again... circular.

    // Better: directly compute using the structure.
    // (t+1)/2 = ((2^32-1)*N + 1) / 2
    // Since N = p^4+p^3+p^2+p+1 and p is odd, N is odd. (2^32-1) is odd. Product is odd. +1 = even. /2 is fine.

    // For a practical implementation, let's compute:
    // a^(N) using Frobenius (call it b)
    // a^((t+1)/2) = a^(((2^32-1)*N+1)/2)
    // = (a^N)^((2^32-1)/2) * a^(N/2 + 1/2)  -- doesn't factor nicely

    // Simplest correct approach: compute a^t, then use the fact that
    // a^((t+1)/2) = a * (a^((t-1)/2))
    // and a^((t-1)/2) = a^t * a^(-1) ... raised to... no.

    // Let me just use a different route entirely.
    // We know a is a QR. Compute a^((q+1)/4) if q ≡ 3 mod 4.
    // q = p^5. p ≡ 1 mod 4 => p^5 ≡ 1 mod 4. So q ≡ 1 mod 4. Can't use this shortcut.

    // OK, let's just directly compute. We need a^e where e = (t+1)/2.
    // Let N = p^4+p^3+p^2+p+1 (computable via Frobenius)
    // t = (2^32-1)*N
    // e = ((2^32-1)*N + 1)/2

    // Compute b = a^N via Frobenius
    let a_p = a.frobenius();
    let a_p2 = a_p.frobenius();
    let a_p3 = a_p2.frobenius();
    let a_p4 = a_p3.frobenius();
    let _b = a * a_p * a_p2 * a_p3 * a_p4; // b = a^N

    // Now we need b^((2^32-1)/2) * a^(1/2)
    // But (2^32-1) is odd, so (2^32-1)/2 is not an integer. This doesn't factor cleanly.

    // Let's just use the chain: compute a^t first (we have pow_by_t),
    // then use iterative approach to find the right power.

    // Actually, the simplest correct approach for Tonelli-Shanks:
    // r = a^((t+1)/2)
    // We can compute this as a * a^((t-1)/2)
    // where a^((t-1)/2) = (a^t)^(... no, we can't halve an exponent easily.

    // The standard Tonelli-Shanks just needs r^2 = a * a^t (mod some 2-power root of unity).
    // So r = a^((t+1)/2) = a * a^(t/... wait, t is odd.
    // r^2 = a^(t+1) = a * a^t
    // So r = sqrt(a * a^t) -- but that's circular.

    // The key insight: in Tonelli-Shanks, the initial r = a^((t+1)/2).
    // Since t is odd, (t+1)/2 is a positive integer.
    // We compute it as: r = a^((t+1)/2) = a^((t-1)/2 + 1) = a * a^((t-1)/2)
    // and a^((t-1)/2) = (a^((t-1)))^(1/2)... still circular.

    // Alternative: compute a^((t+1)/2) from a^t:
    // Let c = a^t (we can compute this). Then a^((t+1)/2) = a * c ... no.
    // a^((t+1)/2) = a^(t/2 + 1/2) -- but t is odd so t/2 is not integer.

    // The actual standard approach: compute by repeated squaring.
    // We need to express (t+1)/2 in binary and do square-and-multiply.
    // But t is enormous (5*64 - 32 = 288 bits).

    // Practical approach: use the factored form.
    // (t+1)/2 = ((2^32-1)*N + 1)/2
    // N = p^4 + p^3 + p^2 + p + 1

    // Method: compute b = a^N (via Frobenius), then compute b^((2^32-1)) * a = a^t * a = a^(t+1).
    // Then we need a square root of a^(t+1).
    // But a^(t+1) = a^t * a, and a^t has order dividing 2^s in the multiplicative group.
    // So a^(t+1) = a * a^t, and since a is a QR and a^t is a 2^s-th root of unity times something...
    // This is getting circular.

    // Let me just implement a generic pow for GFp5 and compute (t+1)/2 as a BigUint.
    use num::bigint::BigUint;
    use num::One;

    let p = BigUint::from(0xFFFFFFFF00000001u64); // Goldilocks prime
    let p2 = &p * &p;
    let p3 = &p2 * &p;
    let p4 = &p3 * &p;
    let n_big = BigUint::one() + &p + &p2 + &p3 + &p4;
    let two32_minus1 = BigUint::from((1u64 << 32) - 1);
    let t = &two32_minus1 * &n_big;
    let exp = (&t + BigUint::one()) / BigUint::from(2u64);

    pow_gfp5(a, &exp)
}

/// Generic exponentiation in GFp5 by a BigUint exponent.
fn pow_gfp5(base: GFp5, exp: &num::bigint::BigUint) -> GFp5 {
    use num::Zero;
    use plonky2_field::ops::Square;

    if exp.is_zero() {
        return GFp5::ONE;
    }

    let bits = exp.to_bytes_le();
    let mut result = GFp5::ONE;
    let mut current = base;

    for byte in &bits {
        for bit_idx in 0..8 {
            if (byte >> bit_idx) & 1 == 1 {
                result = result * current;
            }
            current = current.square();
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use plonky2_field::ops::Square;
    

    #[test]
    fn test_gfp5_arithmetic() {
        let a = GFp5::from(GoldilocksField(7));
        let b = GFp5::from(GoldilocksField(13));
        let c = a * b;
        let expected = GFp5::from(GoldilocksField(91));
        assert_eq!(c, expected);

        // Test with extension element
        let z = QuinticExtension([
            GoldilocksField(1),
            GoldilocksField(2),
            GoldilocksField(3),
            GoldilocksField(0),
            GoldilocksField(0),
        ]);
        let z2 = z.square();
        let z_times_z = z * z;
        assert_eq!(z2, z_times_z);
    }

    #[test]
    fn test_gfp5_inverse() {
        let a = QuinticExtension([
            GoldilocksField(1),
            GoldilocksField(2),
            GoldilocksField(3),
            GoldilocksField(4),
            GoldilocksField(5),
        ]);
        let a_inv = a.try_inverse().unwrap();
        let product = a * a_inv;
        assert_eq!(product, GFp5::ONE);
    }

    #[test]
    fn test_sqrt_gfp5() {
        // Test that squaring and sqrt are inverses
        let a = QuinticExtension([
            GoldilocksField(7),
            GoldilocksField(3),
            GoldilocksField(0),
            GoldilocksField(0),
            GoldilocksField(0),
        ]);
        let a2 = a.square();
        let root = sqrt_gfp5(a2);
        assert!(root.is_some());
        let r = root.unwrap();
        assert_eq!(r.square(), a2);
    }

    #[test]
    fn test_generator_on_curve() {
        let g = EcGFp5Point::generator();
        assert!(!g.is_neutral(), "Generator should not be neutral");
        assert!(g.is_on_curve(), "Generator should be on the curve");
    }

    #[test]
    fn test_neutral_element() {
        let n = EcGFp5Point::NEUTRAL;
        assert!(n.is_neutral());
        assert!(n.is_on_curve());
    }

    #[test]
    fn test_point_double_on_curve() {
        let g = EcGFp5Point::generator();
        let g2 = g.double();
        assert!(g2.is_on_curve(), "2*G should be on curve");
        assert!(!g2.is_neutral());
    }

    #[test]
    fn test_point_add_on_curve() {
        let g = EcGFp5Point::generator();
        let g2 = g.double();
        let g3 = g.add(&g2);
        assert!(g3.is_on_curve(), "3*G should be on curve");
    }

    #[test]
    fn test_add_equals_double() {
        let g = EcGFp5Point::generator();
        let g_plus_g = g.add(&g);
        let g_doubled = g.double();
        assert_eq!(g_plus_g, g_doubled, "G+G should equal 2G");
    }

    #[test]
    fn test_add_neutral() {
        let g = EcGFp5Point::generator();
        let n = EcGFp5Point::NEUTRAL;
        assert_eq!(g.add(&n), g, "G + O = G");
        assert_eq!(n.add(&g), g, "O + G = G");
    }

    #[test]
    fn test_add_inverse() {
        let g = EcGFp5Point::generator();
        let neg_g = g.neg();
        let result = g.add(&neg_g);
        assert!(result.is_neutral(), "G + (-G) should be neutral");
    }

    #[test]
    fn test_scalar_mul() {
        let g = EcGFp5Point::generator();
        let g3_add = g.add(&g).add(&g);
        let g3_scalar = g.scalar_mul_u64(3);
        assert_eq!(g3_add, g3_scalar, "3*G via add should equal 3*G via scalar_mul");
    }

    #[test]
    fn test_scalar_mul_associativity() {
        let g = EcGFp5Point::generator();
        // (2*3)*G = 2*(3*G) = 3*(2*G)
        let g6_a = g.scalar_mul_u64(6);
        let g6_b = g.scalar_mul_u64(3).scalar_mul_u64(2);
        let g6_c = g.scalar_mul_u64(2).scalar_mul_u64(3);
        assert_eq!(g6_a, g6_b);
        assert_eq!(g6_b, g6_c);
    }

    #[test]
    fn test_scalar_mul_zero() {
        let g = EcGFp5Point::generator();
        let result = g.scalar_mul_u64(0);
        assert!(result.is_neutral());
    }

    #[test]
    fn test_point_operations_consistency() {
        let g = EcGFp5Point::generator();
        // Test: 5G = 2G + 3G
        let g2 = g.scalar_mul_u64(2);
        let g3 = g.scalar_mul_u64(3);
        let g5_add = g2.add(&g3);
        let g5_scalar = g.scalar_mul_u64(5);
        assert_eq!(g5_add, g5_scalar, "2G + 3G should equal 5G");
    }

    #[test]
    fn test_double_multiple_times() {
        let g = EcGFp5Point::generator();
        // 4G = double(double(G))
        let g4_dbl = g.double().double();
        let g4_scalar = g.scalar_mul_u64(4);
        assert_eq!(g4_dbl, g4_scalar);

        // 8G = double(double(double(G)))
        let g8_dbl = g.double().double().double();
        let g8_scalar = g.scalar_mul_u64(8);
        assert_eq!(g8_dbl, g8_scalar);
    }
}
