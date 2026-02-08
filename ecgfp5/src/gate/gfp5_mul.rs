//! Custom gate for F_{p^5} multiplication.
//! Computes output = a * b in GF(p^5) = F_p[z]/(z^5 - 3) in a single gate row.
//!
//! Wire layout (per operation):
//!   wires[0..5]:   input a  (5 base field limbs)
//!   wires[5..10]:  input b  (5 base field limbs)
//!   wires[10..15]: output c (5 base field limbs)
//!
//! Total: 15 wires per operation.

use std::string::String;
use std::vec::Vec;
use std::format;
use core::ops::Range;

use anyhow::Result;

use plonky2::field::extension::Extendable;
use plonky2::gates::gate::Gate;
use plonky2::gates::util::StridedConstraintConsumer;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use plonky2::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBase};
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};

/// W = 3 is the irreducible polynomial parameter for z^5 - 3.
const W: u64 = 3;

/// Helper: compute GF(p^5) product constraints given 5 'a' limbs, 5 'b' limbs, 5 'c' limbs.
/// Returns 5 constraint values: c[i] - expected_c[i].
fn gfp5_mul_constraints<T: Copy + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>>(
    a: [T; 5],
    b: [T; 5],
    c: [T; 5],
    w: T,
) -> [T; 5] {
    let expected_c0 = a[0] * b[0] + w * (a[1] * b[4] + a[2] * b[3] + a[3] * b[2] + a[4] * b[1]);
    let expected_c1 = a[0] * b[1] + a[1] * b[0] + w * (a[2] * b[4] + a[3] * b[3] + a[4] * b[2]);
    let expected_c2 = a[0] * b[2] + a[1] * b[1] + a[2] * b[0] + w * (a[3] * b[4] + a[4] * b[3]);
    let expected_c3 = a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0] + w * a[4] * b[4];
    let expected_c4 = a[0] * b[4] + a[1] * b[3] + a[2] * b[2] + a[3] * b[1] + a[4] * b[0];

    [
        c[0] - expected_c0,
        c[1] - expected_c1,
        c[2] - expected_c2,
        c[3] - expected_c3,
        c[4] - expected_c4,
    ]
}

/// A custom gate that computes one or more GF(p^5) multiplications per gate row.
///
/// For each operation i, the gate enforces:
///   c[i] = a[i] * b[i] in GF(p^5)
///
/// Each constraint is degree 2 in the wires (products of two wire values),
/// but degree 3 overall since w * a_i * b_j is degree 2+1 = 3 when w is a constant.
/// However, w is baked in as a numeric constant (3), not a gate constant wire.
/// The constraint is actually degree 2 in the variables, but the Gate trait's `degree()`
/// means the degree of the constraint polynomial in the *wire polynomials*.
/// Since we have products of two wires, that's degree 2. But the gate framework
/// uses degree 3 as the standard for "multiply + add" style constraints.
#[derive(Debug, Clone)]
pub struct GFp5MulGate {
    pub num_ops: usize,
}

impl GFp5MulGate {
    const WIRES_PER_OP: usize = 15; // 5 + 5 + 5

    pub const fn new_from_config(config: &CircuitConfig) -> Self {
        Self {
            num_ops: Self::num_ops(config),
        }
    }

    pub const fn num_ops(config: &CircuitConfig) -> usize {
        config.num_routed_wires / Self::WIRES_PER_OP
    }

    pub const fn wires_ith_input_a(i: usize) -> Range<usize> {
        let start = Self::WIRES_PER_OP * i;
        start..start + 5
    }

    pub const fn wires_ith_input_b(i: usize) -> Range<usize> {
        let start = Self::WIRES_PER_OP * i + 5;
        start..start + 5
    }

    pub const fn wires_ith_output(i: usize) -> Range<usize> {
        let start = Self::WIRES_PER_OP * i + 10;
        start..start + 5
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for GFp5MulGate {
    fn id(&self) -> String {
        format!("{self:?}")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.num_ops)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let num_ops = src.read_usize()?;
        Ok(Self { num_ops })
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let w = F::Extension::from(F::from_canonical_u64(W));

        let mut constraints = Vec::with_capacity(self.num_ops * 5);
        for op in 0..self.num_ops {
            let a_start = Self::wires_ith_input_a(op).start;
            let b_start = Self::wires_ith_input_b(op).start;
            let c_start = Self::wires_ith_output(op).start;

            let a = [
                vars.local_wires[a_start],
                vars.local_wires[a_start + 1],
                vars.local_wires[a_start + 2],
                vars.local_wires[a_start + 3],
                vars.local_wires[a_start + 4],
            ];
            let b = [
                vars.local_wires[b_start],
                vars.local_wires[b_start + 1],
                vars.local_wires[b_start + 2],
                vars.local_wires[b_start + 3],
                vars.local_wires[b_start + 4],
            ];
            let c = [
                vars.local_wires[c_start],
                vars.local_wires[c_start + 1],
                vars.local_wires[c_start + 2],
                vars.local_wires[c_start + 3],
                vars.local_wires[c_start + 4],
            ];

            let constr = gfp5_mul_constraints(a, b, c, w);
            constraints.extend_from_slice(&constr);
        }

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        let w = F::from_canonical_u64(W);

        for op in 0..self.num_ops {
            let a_start = Self::wires_ith_input_a(op).start;
            let b_start = Self::wires_ith_input_b(op).start;
            let c_start = Self::wires_ith_output(op).start;

            let a = [
                vars.local_wires[a_start],
                vars.local_wires[a_start + 1],
                vars.local_wires[a_start + 2],
                vars.local_wires[a_start + 3],
                vars.local_wires[a_start + 4],
            ];
            let b = [
                vars.local_wires[b_start],
                vars.local_wires[b_start + 1],
                vars.local_wires[b_start + 2],
                vars.local_wires[b_start + 3],
                vars.local_wires[b_start + 4],
            ];
            let c = [
                vars.local_wires[c_start],
                vars.local_wires[c_start + 1],
                vars.local_wires[c_start + 2],
                vars.local_wires[c_start + 3],
                vars.local_wires[c_start + 4],
            ];

            let constr = gfp5_mul_constraints(a, b, c, w);
            for v in constr {
                yield_constr.one(v);
            }
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let w = builder.constant(F::from_canonical_u64(W));
        let w_ext = builder.convert_to_ext(w);

        let mut constraints = Vec::with_capacity(self.num_ops * 5);
        for op in 0..self.num_ops {
            let a_start = Self::wires_ith_input_a(op).start;
            let b_start = Self::wires_ith_input_b(op).start;
            let c_start = Self::wires_ith_output(op).start;

            let a: Vec<ExtensionTarget<D>> = (0..5).map(|j| vars.local_wires[a_start + j]).collect();
            let b: Vec<ExtensionTarget<D>> = (0..5).map(|j| vars.local_wires[b_start + j]).collect();
            let c: Vec<ExtensionTarget<D>> = (0..5).map(|j| vars.local_wires[c_start + j]).collect();

            // c0 = a0*b0 + W*(a1*b4 + a2*b3 + a3*b2 + a4*b1)
            let a0b0 = builder.mul_extension(a[0], b[0]);
            let a1b4 = builder.mul_extension(a[1], b[4]);
            let a2b3 = builder.mul_extension(a[2], b[3]);
            let a3b2 = builder.mul_extension(a[3], b[2]);
            let a4b1 = builder.mul_extension(a[4], b[1]);
            let sum04 = builder.add_many_extension([a1b4, a2b3, a3b2, a4b1]);
            let w_sum04 = builder.mul_extension(w_ext, sum04);
            let expected_c0 = builder.add_extension(a0b0, w_sum04);

            // c1 = a0*b1 + a1*b0 + W*(a2*b4 + a3*b3 + a4*b2)
            let a0b1 = builder.mul_extension(a[0], b[1]);
            let a1b0 = builder.mul_extension(a[1], b[0]);
            let a2b4 = builder.mul_extension(a[2], b[4]);
            let a3b3 = builder.mul_extension(a[3], b[3]);
            let a4b2 = builder.mul_extension(a[4], b[2]);
            let sum14 = builder.add_many_extension([a2b4, a3b3, a4b2]);
            let w_sum14 = builder.mul_extension(w_ext, sum14);
            let expected_c1 = builder.add_many_extension([a0b1, a1b0, w_sum14]);

            // c2 = a0*b2 + a1*b1 + a2*b0 + W*(a3*b4 + a4*b3)
            let a0b2 = builder.mul_extension(a[0], b[2]);
            let a1b1 = builder.mul_extension(a[1], b[1]);
            let a2b0 = builder.mul_extension(a[2], b[0]);
            let a3b4 = builder.mul_extension(a[3], b[4]);
            let a4b3 = builder.mul_extension(a[4], b[3]);
            let sum24 = builder.add_extension(a3b4, a4b3);
            let w_sum24 = builder.mul_extension(w_ext, sum24);
            let expected_c2 = builder.add_many_extension([a0b2, a1b1, a2b0, w_sum24]);

            // c3 = a0*b3 + a1*b2 + a2*b1 + a3*b0 + W*a4*b4
            let a0b3 = builder.mul_extension(a[0], b[3]);
            let a1b2 = builder.mul_extension(a[1], b[2]);
            let a2b1 = builder.mul_extension(a[2], b[1]);
            let a3b0 = builder.mul_extension(a[3], b[0]);
            let a4b4 = builder.mul_extension(a[4], b[4]);
            let w_a4b4 = builder.mul_extension(w_ext, a4b4);
            let expected_c3 = builder.add_many_extension([a0b3, a1b2, a2b1, a3b0, w_a4b4]);

            // c4 = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0
            let a0b4 = builder.mul_extension(a[0], b[4]);
            let a1b3 = builder.mul_extension(a[1], b[3]);
            let a2b2 = builder.mul_extension(a[2], b[2]);
            let a3b1 = builder.mul_extension(a[3], b[1]);
            let a4b0 = builder.mul_extension(a[4], b[0]);
            let expected_c4 = builder.add_many_extension([a0b4, a1b3, a2b2, a3b1, a4b0]);

            constraints.push(builder.sub_extension(c[0], expected_c0));
            constraints.push(builder.sub_extension(c[1], expected_c1));
            constraints.push(builder.sub_extension(c[2], expected_c2));
            constraints.push(builder.sub_extension(c[3], expected_c3));
            constraints.push(builder.sub_extension(c[4], expected_c4));
        }

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        (0..self.num_ops)
            .map(|i| {
                WitnessGeneratorRef::new(
                    GFp5MulGenerator::<F, D> {
                        row,
                        i,
                        _phantom: core::marker::PhantomData,
                    }
                    .adapter(),
                )
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.num_ops * Self::WIRES_PER_OP
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        3
    }

    fn num_constraints(&self) -> usize {
        self.num_ops * 5
    }
}

#[derive(Clone, Debug, Default)]
pub struct GFp5MulGenerator<F: RichField + Extendable<D>, const D: usize> {
    row: usize,
    i: usize,
    _phantom: core::marker::PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for GFp5MulGenerator<F, D>
{
    fn id(&self) -> String {
        "GFp5MulGenerator".into()
    }

    fn dependencies(&self) -> Vec<Target> {
        let a_range = GFp5MulGate::wires_ith_input_a(self.i);
        let b_range = GFp5MulGate::wires_ith_input_b(self.i);
        a_range
            .chain(b_range)
            .map(|j| Target::wire(self.row, j))
            .collect()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let a_range = GFp5MulGate::wires_ith_input_a(self.i);
        let b_range = GFp5MulGate::wires_ith_input_b(self.i);
        let c_range = GFp5MulGate::wires_ith_output(self.i);

        let a: Vec<F> = a_range
            .map(|j| witness.get_target(Target::wire(self.row, j)))
            .collect();
        let b: Vec<F> = b_range
            .map(|j| witness.get_target(Target::wire(self.row, j)))
            .collect();

        let w = F::from_canonical_u64(W);

        // GF(p^5) multiplication: output = a * b mod (z^5 - W)
        let c0 = a[0] * b[0] + w * (a[1] * b[4] + a[2] * b[3] + a[3] * b[2] + a[4] * b[1]);
        let c1 = a[0] * b[1] + a[1] * b[0] + w * (a[2] * b[4] + a[3] * b[3] + a[4] * b[2]);
        let c2 = a[0] * b[2] + a[1] * b[1] + a[2] * b[0] + w * (a[3] * b[4] + a[4] * b[3]);
        let c3 = a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0] + w * a[4] * b[4];
        let c4 = a[0] * b[4] + a[1] * b[3] + a[2] * b[2] + a[3] * b[1] + a[4] * b[0];

        let outputs = [c0, c1, c2, c3, c4];
        for (j, &val) in c_range.zip(outputs.iter()) {
            out_buffer.set_target(Target::wire(self.row, j), val)?;
        }

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_usize(self.i)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let i = src.read_usize()?;
        Ok(Self {
            row,
            i,
            _phantom: core::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::gates::gate_testing::{test_eval_fns, test_low_degree};
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    use super::*;

    #[test]
    fn low_degree() {
        let gate = GFp5MulGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_low_degree::<GoldilocksField, _, 4>(gate);
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = GFp5MulGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_eval_fns::<F, C, _, D>(gate)
    }
}
