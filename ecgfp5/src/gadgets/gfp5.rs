//! Circuit gadgets for GF(p^5) arithmetic.
//!
//! A `GFp5Target` represents an element of GF(p^5) as 5 base-field `Target`s.
//! Operations use the custom `GFp5MulGate` for multiplication and standard
//! arithmetic gates for addition/subtraction.

use std::string::String;
use std::vec::Vec;

use anyhow::Result;

use plonky2::field::extension::Extendable;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};

use crate::curve::GFp5;
use crate::gate::GFp5MulGate;

/// A GF(p^5) element in the circuit, represented as 5 base-field targets.
#[derive(Copy, Clone, Debug)]
pub struct GFp5Target {
    pub limbs: [Target; 5],
}

impl Default for GFp5Target {
    fn default() -> Self {
        Self {
            limbs: [Target::default(); 5],
        }
    }
}

impl GFp5Target {
    pub fn new(limbs: [Target; 5]) -> Self {
        Self { limbs }
    }
}

/// Extension trait for CircuitBuilder to add GF(p^5) operations.
pub trait CircuitBuilderGFp5<F: RichField + Extendable<D>, const D: usize> {
    fn add_virtual_gfp5_target(&mut self) -> GFp5Target;
    fn register_gfp5_public_input(&mut self, t: GFp5Target);
    fn connect_gfp5(&mut self, a: GFp5Target, b: GFp5Target);
    fn constant_gfp5(&mut self, val: GFp5) -> GFp5Target;
    fn zero_gfp5(&mut self) -> GFp5Target;
    fn one_gfp5(&mut self) -> GFp5Target;
    fn add_gfp5(&mut self, a: GFp5Target, b: GFp5Target) -> GFp5Target;
    fn sub_gfp5(&mut self, a: GFp5Target, b: GFp5Target) -> GFp5Target;
    fn neg_gfp5(&mut self, a: GFp5Target) -> GFp5Target;
    fn mul_gfp5(&mut self, a: GFp5Target, b: GFp5Target) -> GFp5Target;
    fn mul_gfp5_base(&mut self, a: GFp5Target, scalar: Target) -> GFp5Target;
    fn square_gfp5(&mut self, a: GFp5Target) -> GFp5Target;
    fn inv_gfp5(&mut self, a: GFp5Target) -> GFp5Target;
    fn select_gfp5(&mut self, b: BoolTarget, x: GFp5Target, y: GFp5Target) -> GFp5Target;
    fn is_equal_gfp5(&mut self, a: GFp5Target, b: GFp5Target) -> BoolTarget;
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderGFp5<F, D>
    for CircuitBuilder<F, D>
{
    fn add_virtual_gfp5_target(&mut self) -> GFp5Target {
        GFp5Target {
            limbs: [
                self.add_virtual_target(),
                self.add_virtual_target(),
                self.add_virtual_target(),
                self.add_virtual_target(),
                self.add_virtual_target(),
            ],
        }
    }

    fn register_gfp5_public_input(&mut self, t: GFp5Target) {
        for limb in t.limbs {
            self.register_public_input(limb);
        }
    }

    fn connect_gfp5(&mut self, a: GFp5Target, b: GFp5Target) {
        for i in 0..5 {
            self.connect(a.limbs[i], b.limbs[i]);
        }
    }

    fn constant_gfp5(&mut self, val: GFp5) -> GFp5Target {
        GFp5Target {
            limbs: [
                self.constant(F::from_canonical_u64(val.0[0].0)),
                self.constant(F::from_canonical_u64(val.0[1].0)),
                self.constant(F::from_canonical_u64(val.0[2].0)),
                self.constant(F::from_canonical_u64(val.0[3].0)),
                self.constant(F::from_canonical_u64(val.0[4].0)),
            ],
        }
    }

    fn zero_gfp5(&mut self) -> GFp5Target {
        self.constant_gfp5(GFp5::ZERO)
    }

    fn one_gfp5(&mut self) -> GFp5Target {
        self.constant_gfp5(GFp5::ONE)
    }

    fn add_gfp5(&mut self, a: GFp5Target, b: GFp5Target) -> GFp5Target {
        GFp5Target {
            limbs: [
                self.add(a.limbs[0], b.limbs[0]),
                self.add(a.limbs[1], b.limbs[1]),
                self.add(a.limbs[2], b.limbs[2]),
                self.add(a.limbs[3], b.limbs[3]),
                self.add(a.limbs[4], b.limbs[4]),
            ],
        }
    }

    fn sub_gfp5(&mut self, a: GFp5Target, b: GFp5Target) -> GFp5Target {
        GFp5Target {
            limbs: [
                self.sub(a.limbs[0], b.limbs[0]),
                self.sub(a.limbs[1], b.limbs[1]),
                self.sub(a.limbs[2], b.limbs[2]),
                self.sub(a.limbs[3], b.limbs[3]),
                self.sub(a.limbs[4], b.limbs[4]),
            ],
        }
    }

    fn neg_gfp5(&mut self, a: GFp5Target) -> GFp5Target {
        let zero = self.zero();
        GFp5Target {
            limbs: [
                self.sub(zero, a.limbs[0]),
                self.sub(zero, a.limbs[1]),
                self.sub(zero, a.limbs[2]),
                self.sub(zero, a.limbs[3]),
                self.sub(zero, a.limbs[4]),
            ],
        }
    }

    fn mul_gfp5(&mut self, a: GFp5Target, b: GFp5Target) -> GFp5Target {
        let gate = GFp5MulGate::new_from_config(&self.config);
        let (gate_row, op_idx) = self.find_slot(gate, &[], &[]);

        for j in 0..5 {
            let wire_a = Target::wire(gate_row, GFp5MulGate::wires_ith_input_a(op_idx).start + j);
            let wire_b = Target::wire(gate_row, GFp5MulGate::wires_ith_input_b(op_idx).start + j);
            self.connect(a.limbs[j], wire_a);
            self.connect(b.limbs[j], wire_b);
        }

        let out_start = GFp5MulGate::wires_ith_output(op_idx).start;
        GFp5Target {
            limbs: [
                Target::wire(gate_row, out_start),
                Target::wire(gate_row, out_start + 1),
                Target::wire(gate_row, out_start + 2),
                Target::wire(gate_row, out_start + 3),
                Target::wire(gate_row, out_start + 4),
            ],
        }
    }

    fn mul_gfp5_base(&mut self, a: GFp5Target, scalar: Target) -> GFp5Target {
        GFp5Target {
            limbs: [
                self.mul(a.limbs[0], scalar),
                self.mul(a.limbs[1], scalar),
                self.mul(a.limbs[2], scalar),
                self.mul(a.limbs[3], scalar),
                self.mul(a.limbs[4], scalar),
            ],
        }
    }

    fn square_gfp5(&mut self, a: GFp5Target) -> GFp5Target {
        self.mul_gfp5(a, a)
    }

    fn inv_gfp5(&mut self, a: GFp5Target) -> GFp5Target {
        let inv = self.add_virtual_gfp5_target();
        self.add_simple_generator(GFp5InvGenerator {
            input: a,
            output: inv,
        });
        let product = self.mul_gfp5(a, inv);
        let one = self.one_gfp5();
        self.connect_gfp5(product, one);
        inv
    }

    fn select_gfp5(&mut self, b: BoolTarget, x: GFp5Target, y: GFp5Target) -> GFp5Target {
        GFp5Target {
            limbs: [
                self.select(b, x.limbs[0], y.limbs[0]),
                self.select(b, x.limbs[1], y.limbs[1]),
                self.select(b, x.limbs[2], y.limbs[2]),
                self.select(b, x.limbs[3], y.limbs[3]),
                self.select(b, x.limbs[4], y.limbs[4]),
            ],
        }
    }

    fn is_equal_gfp5(&mut self, a: GFp5Target, b: GFp5Target) -> BoolTarget {
        let eq0 = self.is_equal(a.limbs[0], b.limbs[0]);
        let eq1 = self.is_equal(a.limbs[1], b.limbs[1]);
        let eq2 = self.is_equal(a.limbs[2], b.limbs[2]);
        let eq3 = self.is_equal(a.limbs[3], b.limbs[3]);
        let eq4 = self.is_equal(a.limbs[4], b.limbs[4]);
        let and01 = self.and(eq0, eq1);
        let and23 = self.and(eq2, eq3);
        let and0123 = self.and(and01, and23);
        self.and(and0123, eq4)
    }
}

/// Write GFp5 values into partial witness.
pub trait WitnessWriteGFp5 {
    fn set_gfp5_target(&mut self, target: GFp5Target, value: GFp5) -> Result<()>;
}

impl<T: WitnessWrite<GoldilocksField>> WitnessWriteGFp5 for T {
    fn set_gfp5_target(&mut self, target: GFp5Target, value: GFp5) -> Result<()> {
        for i in 0..5 {
            self.set_target(target.limbs[i], value.0[i])?;
        }
        Ok(())
    }
}

/// Generator that computes GF(p^5) inverse during witness generation.
#[derive(Debug, Clone)]
struct GFp5InvGenerator {
    input: GFp5Target,
    output: GFp5Target,
}

impl Default for GFp5InvGenerator {
    fn default() -> Self {
        Self {
            input: GFp5Target::default(),
            output: GFp5Target::default(),
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for GFp5InvGenerator {
    fn id(&self) -> String {
        "GFp5InvGenerator".into()
    }

    fn dependencies(&self) -> Vec<Target> {
        self.input.limbs.to_vec()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        use plonky2_field::extension::quintic::QuinticExtension;
        let a_vals: Vec<u64> = self
            .input
            .limbs
            .iter()
            .map(|&t| witness.get_target(t).to_canonical_u64())
            .collect();
        let a = QuinticExtension([
            GoldilocksField(a_vals[0]),
            GoldilocksField(a_vals[1]),
            GoldilocksField(a_vals[2]),
            GoldilocksField(a_vals[3]),
            GoldilocksField(a_vals[4]),
        ]);
        let a_inv = a.try_inverse().expect("GFp5 inverse of zero");
        for i in 0..5 {
            out_buffer.set_target(
                self.output.limbs[i],
                F::from_canonical_u64(a_inv.0[i].0),
            )?;
        }
        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        for &t in &self.input.limbs {
            dst.write_target(t)?;
        }
        for &t in &self.output.limbs {
            dst.write_target(t)?;
        }
        Ok(())
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let input = GFp5Target {
            limbs: [
                src.read_target()?,
                src.read_target()?,
                src.read_target()?,
                src.read_target()?,
                src.read_target()?,
            ],
        };
        let output = GFp5Target {
            limbs: [
                src.read_target()?,
                src.read_target()?,
                src.read_target()?,
                src.read_target()?,
                src.read_target()?,
            ],
        };
        Ok(Self { input, output })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2_field::extension::quintic::QuinticExtension;
    use plonky2_field::goldilocks_field::GoldilocksField;

    use super::*;
    use crate::curve::GFp5;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    fn make_gfp5(vals: [u64; 5]) -> GFp5 {
        QuinticExtension([
            GoldilocksField(vals[0]),
            GoldilocksField(vals[1]),
            GoldilocksField(vals[2]),
            GoldilocksField(vals[3]),
            GoldilocksField(vals[4]),
        ])
    }

    #[test]
    fn test_gfp5_add_circuit() -> Result<()> {
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let mut pw = PartialWitness::new();

        let a_val = make_gfp5([1, 2, 3, 4, 5]);
        let b_val = make_gfp5([10, 20, 30, 40, 50]);
        let expected = a_val + b_val;

        let a = builder.add_virtual_gfp5_target();
        let b = builder.add_virtual_gfp5_target();
        let c = builder.add_gfp5(a, b);
        let expected_t = builder.constant_gfp5(expected);
        builder.connect_gfp5(c, expected_t);

        pw.set_gfp5_target(a, a_val)?;
        pw.set_gfp5_target(b, b_val)?;

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;
        data.verify(proof)
    }

    #[test]
    fn test_gfp5_mul_circuit() -> Result<()> {
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let mut pw = PartialWitness::new();

        let a_val = make_gfp5([7, 3, 0, 0, 0]);
        let b_val = make_gfp5([11, 5, 0, 0, 0]);
        let expected = a_val * b_val;

        let a = builder.add_virtual_gfp5_target();
        let b = builder.add_virtual_gfp5_target();
        let c = builder.mul_gfp5(a, b);
        let expected_t = builder.constant_gfp5(expected);
        builder.connect_gfp5(c, expected_t);

        pw.set_gfp5_target(a, a_val)?;
        pw.set_gfp5_target(b, b_val)?;

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;
        data.verify(proof)
    }

    #[test]
    fn test_gfp5_mul_full_extension() -> Result<()> {
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let mut pw = PartialWitness::new();

        let a_val = make_gfp5([1, 2, 3, 4, 5]);
        let b_val = make_gfp5([6, 7, 8, 9, 10]);
        let expected = a_val * b_val;

        let a = builder.add_virtual_gfp5_target();
        let b = builder.add_virtual_gfp5_target();
        let c = builder.mul_gfp5(a, b);
        let expected_t = builder.constant_gfp5(expected);
        builder.connect_gfp5(c, expected_t);

        pw.set_gfp5_target(a, a_val)?;
        pw.set_gfp5_target(b, b_val)?;

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;
        data.verify(proof)
    }

    #[test]
    fn test_gfp5_inv_circuit() -> Result<()> {
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let mut pw = PartialWitness::new();

        let a_val = make_gfp5([1, 2, 3, 4, 5]);

        let a = builder.add_virtual_gfp5_target();
        let a_inv = builder.inv_gfp5(a);
        let product = builder.mul_gfp5(a, a_inv);
        let one = builder.one_gfp5();
        builder.connect_gfp5(product, one);

        pw.set_gfp5_target(a, a_val)?;

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;
        data.verify(proof)
    }
}
