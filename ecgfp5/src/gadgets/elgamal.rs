//! ElGamal encryption circuit gadgets for EcGFp5.
//!
//! ElGamal encryption:
//!   - Private key: scalar x
//!   - Public key: Y = x * G
//!   - Encrypt message point M with randomness r:
//!     C1 = r * G
//!     C2 = M + r * Y
//!   - Decrypt: M = C2 - x * C1
//!
//! The circuit proves correct encryption: given public inputs (Y, C1, C2),
//! it verifies that the prover knows r and M such that the above holds.


use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::target::BoolTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::gadgets::curve::{CircuitBuilderEcGFp5, EcGFp5PointTarget};

/// ElGamal ciphertext target (two curve points).
#[derive(Copy, Clone, Debug)]
pub struct ElGamalCiphertextTarget {
    pub c1: EcGFp5PointTarget,
    pub c2: EcGFp5PointTarget,
}

/// Extension trait for ElGamal operations in the circuit.
pub trait CircuitBuilderElGamal<F: RichField + Extendable<D>, const D: usize> {
    /// Build a circuit that verifies ElGamal encryption.
    ///
    /// Given:
    ///   - generator G (constant)
    ///   - public key Y (public input)
    ///   - message point M (private witness)
    ///   - randomness r as bits (private witness)
    ///
    /// Computes and outputs:
    ///   - C1 = r * G
    ///   - C2 = M + r * Y
    ///
    /// The caller should register C1, C2, and Y as public inputs.
    fn elgamal_encrypt(
        &mut self,
        generator: EcGFp5PointTarget,
        public_key: EcGFp5PointTarget,
        message: EcGFp5PointTarget,
        randomness_bits: &[BoolTarget],
    ) -> ElGamalCiphertextTarget;
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderElGamal<F, D>
    for CircuitBuilder<F, D>
{
    fn elgamal_encrypt(
        &mut self,
        generator: EcGFp5PointTarget,
        public_key: EcGFp5PointTarget,
        message: EcGFp5PointTarget,
        randomness_bits: &[BoolTarget],
    ) -> ElGamalCiphertextTarget {
        // C1 = r * G
        let c1 = self.ecgfp5_scalar_mul(generator, randomness_bits);

        // r * Y
        let r_y = self.ecgfp5_scalar_mul(public_key, randomness_bits);

        // C2 = M + r*Y
        let c2 = self.ecgfp5_add(message, r_y);

        ElGamalCiphertextTarget { c1, c2 }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use super::*;
    use crate::curve::EcGFp5Point;
    use crate::gadgets::gfp5::WitnessWriteGFp5;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    #[test]
    fn test_elgamal_encrypt_decrypt() -> Result<()> {
        // Setup
        let g = EcGFp5Point::generator();
        let private_key: u64 = 42;
        let public_key = g.scalar_mul_u64(private_key);
        assert!(public_key.is_on_curve());

        let randomness: u64 = 17;
        let message = g.scalar_mul_u64(123); // Use a curve point as message
        assert!(message.is_on_curve());

        // Encrypt natively
        let c1 = g.scalar_mul_u64(randomness);
        let r_y = public_key.scalar_mul_u64(randomness);
        let c2 = message.add(&r_y);

        // Verify decryption: M = C2 - x*C1
        let x_c1 = c1.scalar_mul_u64(private_key);
        let decrypted = c2.add(&x_c1.neg());
        assert_eq!(decrypted, message, "Decryption should recover the message");

        // Now build the circuit
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let mut pw = PartialWitness::new();

        let g_target = builder.constant_ecgfp5(g);
        let pk_target = builder.add_virtual_ecgfp5_target();
        let msg_target = builder.add_virtual_ecgfp5_target();

        // Randomness bits (little-endian)
        let num_bits = 8;
        let mut r_bits = Vec::new();
        for i in 0..num_bits {
            let bit = ((randomness >> i) & 1) == 1;
            let b = builder.constant_bool(bit);
            r_bits.push(b);
        }

        let ct = builder.elgamal_encrypt(g_target, pk_target, msg_target, &r_bits);

        // Connect expected ciphertext
        let expected_c1 = builder.constant_ecgfp5(c1);
        let expected_c2 = builder.constant_ecgfp5(c2);
        builder.connect_ecgfp5(ct.c1, expected_c1);
        builder.connect_ecgfp5(ct.c2, expected_c2);

        // Set witness values
        pw.set_gfp5_target(pk_target.x, public_key.x)?;
        pw.set_gfp5_target(pk_target.w, public_key.w)?;
        pw.set_gfp5_target(msg_target.x, message.x)?;
        pw.set_gfp5_target(msg_target.w, message.w)?;

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;
        data.verify(proof)
    }
}
