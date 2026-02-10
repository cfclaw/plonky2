#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::marker::PhantomData;
use core::ops::Range;

use anyhow::Result;

use crate::hash::poseidon2::{Poseidon2, SPONGE_WIDTH};
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::gates::gate::Gate;
use plonky2::gates::util::StridedConstraintConsumer;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::{ExtensionAlgebraTarget, ExtensionTarget};
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBase};
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};
use plonky2_field::extension::algebra::ExtensionAlgebra;
use plonky2_field::extension::FieldExtension;

/// Combined gate that computes BOTH Poseidon2's external and internal linear layers
/// as degree-1 constraints over extension algebra elements.
///
/// Wire layout (D=2, SPONGE_WIDTH=12):
/// - Wires 0..24: input (12 extension elements)
/// - Wires 24..48: external layer output (12 extension elements)
/// - Wires 48..72: internal layer output (12 extension elements)
///
/// The gate always verifies both outputs. The caller uses whichever output
/// is needed (external or internal) and leaves the other unconnected.
///
/// By combining both operations in one gate type, the recursive circuit has
/// 13 gates (matching Poseidon) instead of 14, keeping 3 selector groups.
#[derive(Debug, Default)]
pub struct Poseidon2MdsGate<F: RichField + Extendable<D> + Poseidon2, const D: usize>(
    PhantomData<F>,
);

impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> Poseidon2MdsGate<F, D> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }

    pub(crate) const fn wires_input(i: usize) -> Range<usize> {
        assert!(i < SPONGE_WIDTH);
        i * D..(i + 1) * D
    }

    pub(crate) const fn wires_external_output(i: usize) -> Range<usize> {
        assert!(i < SPONGE_WIDTH);
        (SPONGE_WIDTH + i) * D..(SPONGE_WIDTH + i + 1) * D
    }

    pub(crate) const fn wires_internal_output(i: usize) -> Range<usize> {
        assert!(i < SPONGE_WIDTH);
        (2 * SPONGE_WIDTH + i) * D..(2 * SPONGE_WIDTH + i + 1) * D
    }

    // ========================================================================
    // External linear layer computation
    // ========================================================================

    fn external_layer_algebra(
        input: &[ExtensionAlgebra<F::Extension, D>; SPONGE_WIDTH],
    ) -> [ExtensionAlgebra<F::Extension, D>; SPONGE_WIDTH] {
        let mut state = *input;

        for chunk_start in (0..SPONGE_WIDTH).step_by(4) {
            let s0 = state[chunk_start];
            let s1 = state[chunk_start + 1];
            let s2 = state[chunk_start + 2];
            let s3 = state[chunk_start + 3];

            let t0 = s0 + s1;
            let t1 = s2 + s3;
            let t2 = s1 + s1 + t1;
            let t3 = s3 + s3 + t0;
            let t1_2 = t1 + t1;
            let t4 = t1_2 + t1_2 + t3;
            let t0_2 = t0 + t0;
            let t5 = t0_2 + t0_2 + t2;
            let t6 = t3 + t5;
            let t7 = t2 + t4;

            state[chunk_start] = t6;
            state[chunk_start + 1] = t5;
            state[chunk_start + 2] = t7;
            state[chunk_start + 3] = t4;
        }

        let sum0 = state[0] + state[4] + state[8];
        let sum1 = state[1] + state[5] + state[9];
        let sum2 = state[2] + state[6] + state[10];
        let sum3 = state[3] + state[7] + state[11];

        state[0] = state[0] + sum0;
        state[4] = state[4] + sum0;
        state[8] = state[8] + sum0;
        state[1] = state[1] + sum1;
        state[5] = state[5] + sum1;
        state[9] = state[9] + sum1;
        state[2] = state[2] + sum2;
        state[6] = state[6] + sum2;
        state[10] = state[10] + sum2;
        state[3] = state[3] + sum3;
        state[7] = state[7] + sum3;
        state[11] = state[11] + sum3;

        state
    }

    fn external_layer_field(
        input: &[F::Extension; SPONGE_WIDTH],
    ) -> [F::Extension; SPONGE_WIDTH] {
        let mut state = *input;

        for chunk_start in (0..SPONGE_WIDTH).step_by(4) {
            let s0 = state[chunk_start];
            let s1 = state[chunk_start + 1];
            let s2 = state[chunk_start + 2];
            let s3 = state[chunk_start + 3];

            let t0 = s0 + s1;
            let t1 = s2 + s3;
            let t2 = s1 + s1 + t1;
            let t3 = s3 + s3 + t0;
            let t1_2 = t1 + t1;
            let t4 = t1_2 + t1_2 + t3;
            let t0_2 = t0 + t0;
            let t5 = t0_2 + t0_2 + t2;
            let t6 = t3 + t5;
            let t7 = t2 + t4;

            state[chunk_start] = t6;
            state[chunk_start + 1] = t5;
            state[chunk_start + 2] = t7;
            state[chunk_start + 3] = t4;
        }

        let sum0 = state[0] + state[4] + state[8];
        let sum1 = state[1] + state[5] + state[9];
        let sum2 = state[2] + state[6] + state[10];
        let sum3 = state[3] + state[7] + state[11];

        state[0] += sum0; state[4] += sum0; state[8] += sum0;
        state[1] += sum1; state[5] += sum1; state[9] += sum1;
        state[2] += sum2; state[6] += sum2; state[10] += sum2;
        state[3] += sum3; state[7] += sum3; state[11] += sum3;

        state
    }

    fn external_layer_algebra_circuit(
        builder: &mut CircuitBuilder<F, D>,
        input: &[ExtensionAlgebraTarget<D>; SPONGE_WIDTH],
    ) -> [ExtensionAlgebraTarget<D>; SPONGE_WIDTH] {
        let mut state = *input;

        for chunk_start in (0..SPONGE_WIDTH).step_by(4) {
            let s0 = state[chunk_start];
            let s1 = state[chunk_start + 1];
            let s2 = state[chunk_start + 2];
            let s3 = state[chunk_start + 3];

            let t0 = builder.add_ext_algebra(s0, s1);
            let t1 = builder.add_ext_algebra(s2, s3);
            let s1_2 = builder.add_ext_algebra(s1, s1);
            let t2 = builder.add_ext_algebra(s1_2, t1);
            let s3_2 = builder.add_ext_algebra(s3, s3);
            let t3 = builder.add_ext_algebra(s3_2, t0);
            let t1_2 = builder.add_ext_algebra(t1, t1);
            let t1_4 = builder.add_ext_algebra(t1_2, t1_2);
            let t4 = builder.add_ext_algebra(t1_4, t3);
            let t0_2 = builder.add_ext_algebra(t0, t0);
            let t0_4 = builder.add_ext_algebra(t0_2, t0_2);
            let t5 = builder.add_ext_algebra(t0_4, t2);
            let t6 = builder.add_ext_algebra(t3, t5);
            let t7 = builder.add_ext_algebra(t2, t4);

            state[chunk_start] = t6;
            state[chunk_start + 1] = t5;
            state[chunk_start + 2] = t7;
            state[chunk_start + 3] = t4;
        }

        let sum0 = {
            let a = builder.add_ext_algebra(state[0], state[4]);
            builder.add_ext_algebra(a, state[8])
        };
        let sum1 = {
            let a = builder.add_ext_algebra(state[1], state[5]);
            builder.add_ext_algebra(a, state[9])
        };
        let sum2 = {
            let a = builder.add_ext_algebra(state[2], state[6]);
            builder.add_ext_algebra(a, state[10])
        };
        let sum3 = {
            let a = builder.add_ext_algebra(state[3], state[7]);
            builder.add_ext_algebra(a, state[11])
        };

        state[0] = builder.add_ext_algebra(state[0], sum0);
        state[4] = builder.add_ext_algebra(state[4], sum0);
        state[8] = builder.add_ext_algebra(state[8], sum0);
        state[1] = builder.add_ext_algebra(state[1], sum1);
        state[5] = builder.add_ext_algebra(state[5], sum1);
        state[9] = builder.add_ext_algebra(state[9], sum1);
        state[2] = builder.add_ext_algebra(state[2], sum2);
        state[6] = builder.add_ext_algebra(state[6], sum2);
        state[10] = builder.add_ext_algebra(state[10], sum2);
        state[3] = builder.add_ext_algebra(state[3], sum3);
        state[7] = builder.add_ext_algebra(state[7], sum3);
        state[11] = builder.add_ext_algebra(state[11], sum3);

        state
    }

    // ========================================================================
    // Internal linear layer computation
    // ========================================================================

    fn internal_layer_algebra(
        input: &[ExtensionAlgebra<F::Extension, D>; SPONGE_WIDTH],
    ) -> [ExtensionAlgebra<F::Extension, D>; SPONGE_WIDTH] {
        let mut sum = ExtensionAlgebra::ZERO;
        for x in input.iter() {
            sum = sum + *x;
        }

        let mut result = [ExtensionAlgebra::ZERO; SPONGE_WIDTH];
        for i in 0..SPONGE_WIDTH {
            let diag = F::Extension::from_canonical_u64(<F as Poseidon2>::INTERNAL_MATRIX_DIAG[i]);
            result[i] = input[i].scalar_mul(diag) + sum;
        }
        result
    }

    fn internal_layer_field(
        input: &[F::Extension; SPONGE_WIDTH],
    ) -> [F::Extension; SPONGE_WIDTH] {
        let mut sum = F::Extension::ZERO;
        for x in input.iter() {
            sum = sum + *x;
        }

        core::array::from_fn(|i| {
            let diag = F::Extension::from_canonical_u64(<F as Poseidon2>::INTERNAL_MATRIX_DIAG[i]);
            input[i] * diag + sum
        })
    }

    fn internal_layer_algebra_circuit(
        builder: &mut CircuitBuilder<F, D>,
        input: &[ExtensionAlgebraTarget<D>; SPONGE_WIDTH],
    ) -> [ExtensionAlgebraTarget<D>; SPONGE_WIDTH] {
        let mut sum = input[0];
        for i in 1..SPONGE_WIDTH {
            sum = builder.add_ext_algebra(sum, input[i]);
        }

        let mut result = [builder.zero_ext_algebra(); SPONGE_WIDTH];
        for i in 0..SPONGE_WIDTH {
            let diag = builder.constant_extension(F::Extension::from_canonical_u64(
                <F as Poseidon2>::INTERNAL_MATRIX_DIAG[i],
            ));
            result[i] = builder.scalar_mul_add_ext_algebra(diag, input[i], sum);
        }
        result
    }
}

impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> Gate<F, D>
    for Poseidon2MdsGate<F, D>
{
    fn id(&self) -> String {
        format!("{self:?}<WIDTH={SPONGE_WIDTH}>")
    }

    fn serialize(
        &self,
        _dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<()> {
        Ok(())
    }

    fn deserialize(_src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Self::new())
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let inputs: [_; SPONGE_WIDTH] = (0..SPONGE_WIDTH)
            .map(|i| vars.get_local_ext_algebra(Self::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let computed_external = Self::external_layer_algebra(&inputs);
        let computed_internal = Self::internal_layer_algebra(&inputs);

        let mut constraints = Vec::with_capacity(2 * SPONGE_WIDTH * D);

        // External layer constraints
        for i in 0..SPONGE_WIDTH {
            let out = vars.get_local_ext_algebra(Self::wires_external_output(i));
            constraints.extend((out - computed_external[i]).to_basefield_array());
        }

        // Internal layer constraints
        for i in 0..SPONGE_WIDTH {
            let out = vars.get_local_ext_algebra(Self::wires_internal_output(i));
            constraints.extend((out - computed_internal[i]).to_basefield_array());
        }

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        let inputs: [_; SPONGE_WIDTH] = (0..SPONGE_WIDTH)
            .map(|i| vars.get_local_ext(Self::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let computed_external = Self::external_layer_field(&inputs);
        let computed_internal = Self::internal_layer_field(&inputs);

        // External layer constraints
        yield_constr.many(
            (0..SPONGE_WIDTH)
                .map(|i| vars.get_local_ext(Self::wires_external_output(i)))
                .zip(computed_external)
                .flat_map(|(out, computed)| (out - computed).to_basefield_array()),
        );

        // Internal layer constraints
        yield_constr.many(
            (0..SPONGE_WIDTH)
                .map(|i| vars.get_local_ext(Self::wires_internal_output(i)))
                .zip(computed_internal)
                .flat_map(|(out, computed)| (out - computed).to_basefield_array()),
        );
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let inputs: [_; SPONGE_WIDTH] = (0..SPONGE_WIDTH)
            .map(|i| vars.get_local_ext_algebra(Self::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let computed_external = Self::external_layer_algebra_circuit(builder, &inputs);
        let computed_internal = Self::internal_layer_algebra_circuit(builder, &inputs);

        let mut constraints = Vec::with_capacity(2 * SPONGE_WIDTH * D);

        // External layer constraints
        for i in 0..SPONGE_WIDTH {
            let out = vars.get_local_ext_algebra(Self::wires_external_output(i));
            constraints.extend(
                builder
                    .sub_ext_algebra(out, computed_external[i])
                    .to_ext_target_array(),
            );
        }

        // Internal layer constraints
        for i in 0..SPONGE_WIDTH {
            let out = vars.get_local_ext_algebra(Self::wires_internal_output(i));
            constraints.extend(
                builder
                    .sub_ext_algebra(out, computed_internal[i])
                    .to_ext_target_array(),
            );
        }

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        let gen = Poseidon2MdsGenerator::<F, D> {
            row,
            _phantom: PhantomData,
        };
        vec![WitnessGeneratorRef::new(gen.adapter())]
    }

    fn num_wires(&self) -> usize {
        3 * D * SPONGE_WIDTH
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn num_constraints(&self) -> usize {
        2 * SPONGE_WIDTH * D
    }
}

#[derive(Clone, Debug, Default)]
pub struct Poseidon2MdsGenerator<F: RichField + Extendable<D> + Poseidon2, const D: usize> {
    row: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> SimpleGenerator<F, D>
    for Poseidon2MdsGenerator<F, D>
{
    fn id(&self) -> String {
        "Poseidon2MdsGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        (0..SPONGE_WIDTH)
            .flat_map(|i| {
                Target::wires_from_range(self.row, Poseidon2MdsGate::<F, D>::wires_input(i))
            })
            .collect()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let get_local_target = |wire_range| ExtensionTarget::from_range(self.row, wire_range);
        let get_local_ext =
            |wire_range| witness.get_extension_target(get_local_target(wire_range));

        let inputs: [F::Extension; SPONGE_WIDTH] = (0..SPONGE_WIDTH)
            .map(|i| get_local_ext(Poseidon2MdsGate::<F, D>::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // Compute both outputs
        let external_outputs = Poseidon2MdsGate::<F, D>::external_layer_field(&inputs);
        let internal_outputs = Poseidon2MdsGate::<F, D>::internal_layer_field(&inputs);

        for (i, &out) in external_outputs.iter().enumerate() {
            out_buffer.set_extension_target(
                get_local_target(Poseidon2MdsGate::<F, D>::wires_external_output(i)),
                out,
            )?;
        }

        for (i, &out) in internal_outputs.iter().enumerate() {
            out_buffer.set_extension_target(
                get_local_target(Poseidon2MdsGate::<F, D>::wires_internal_output(i)),
                out,
            )?;
        }

        Ok(())
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<()> {
        dst.write_usize(self.row)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        Ok(Self {
            row,
            _phantom: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use plonky2::gates::gate_testing::{test_eval_fns, test_low_degree};
    use plonky2::plonk::config::GenericConfig;
    use crate::Poseidon2GoldilocksConfig;

    use super::*;

    #[test]
    fn mds_gate_low_degree() {
        const D: usize = 2;
        type C = Poseidon2GoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = Poseidon2MdsGate::<F, D>::new();
        test_low_degree(gate)
    }

    #[test]
    fn mds_gate_eval_fns() -> anyhow::Result<()> {
        const D: usize = 2;
        type C = Poseidon2GoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = Poseidon2MdsGate::<F, D>::new();
        test_eval_fns::<F, C, _, D>(gate)
    }
}
