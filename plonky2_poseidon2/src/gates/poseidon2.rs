#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::marker::PhantomData;

use anyhow::Result;

use crate::gates::poseidon_mds::Poseidon2MdsGate;
use crate::hash::poseidon2::{
    Poseidon2, HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS, SPONGE_WIDTH,
};
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::gates::gate::Gate;
use plonky2::gates::util::StridedConstraintConsumer;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use plonky2::iop::target::Target;
use plonky2::iop::wire::Wire;
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBase};
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};
use plonky2_field::ops::Square;
/// Evaluates a full Poseidon2 permutation with 12 state elements.
///
/// Includes a swap flag to swap the first four inputs with the next four,
/// useful for Merkle proofs.
#[derive(Debug, Default)]
pub struct Poseidon2Gate<F: RichField + Extendable<D> + Poseidon2, const D: usize>(PhantomData<F>);

impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> Poseidon2Gate<F, D> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }

    /// The wire index for the `i`th input to the permutation.
    pub(crate) const fn wire_input(i: usize) -> usize {
        i
    }

    /// The wire index for the `i`th output to the permutation.
    pub(crate) const fn wire_output(i: usize) -> usize {
        SPONGE_WIDTH + i
    }

    /// If this is set to 1, the first four inputs will be swapped with the next four inputs.
    pub(crate) const WIRE_SWAP: usize = 2 * SPONGE_WIDTH;

    const START_DELTA: usize = 2 * SPONGE_WIDTH + 1;

    /// A wire which stores `swap * (input[i + 4] - input[i])`; used to compute the swapped inputs.
    const fn wire_delta(i: usize) -> usize {
        assert!(i < 4);
        Self::START_DELTA + i
    }

    const START_FULL_0: usize = Self::START_DELTA + 4;

    /// A wire which stores the input of the `i`-th S-box of the `round`-th round of the
    /// first set of full rounds.
    const fn wire_full_sbox_0(round: usize, i: usize) -> usize {
        debug_assert!(
            round != 0,
            "First round S-box inputs are not stored as wires"
        );
        debug_assert!(round < HALF_N_FULL_ROUNDS);
        debug_assert!(i < SPONGE_WIDTH);
        Self::START_FULL_0 + SPONGE_WIDTH * (round - 1) + i
    }

    const START_PARTIAL: usize =
        Self::START_FULL_0 + SPONGE_WIDTH * (HALF_N_FULL_ROUNDS - 1);

    /// A wire which stores the input of the S-box of the `round`-th round of the partial rounds.
    const fn wire_partial_sbox(round: usize) -> usize {
        debug_assert!(round < N_PARTIAL_ROUNDS);
        Self::START_PARTIAL + round
    }

    const START_FULL_1: usize = Self::START_PARTIAL + N_PARTIAL_ROUNDS;

    /// A wire which stores the input of the `i`-th S-box of the `round`-th round of the
    /// second set of full rounds.
    const fn wire_full_sbox_1(round: usize, i: usize) -> usize {
        debug_assert!(round < HALF_N_FULL_ROUNDS);
        debug_assert!(i < SPONGE_WIDTH);
        Self::START_FULL_1 + SPONGE_WIDTH * round + i
    }

    /// End of wire indices, exclusive.
    const fn end() -> usize {
        Self::START_FULL_1 + SPONGE_WIDTH * HALF_N_FULL_ROUNDS
    }
}

impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> Gate<F, D> for Poseidon2Gate<F, D> {
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
        Ok(Poseidon2Gate::new())
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let mut constraints = Vec::with_capacity(self.num_constraints());

        // Inline S-box for F::Extension (x^7)
        let sbox = |x: F::Extension| {
            let x2 = x.square();
            let x4 = x2.square();
            let x3 = x * x2;
            x3 * x4
        };

        // 1. Swap Logic
        let swap = vars.local_wires[Self::WIRE_SWAP];
        constraints.push(swap * (swap - F::Extension::ONE));

        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            constraints.push(swap * (input_rhs - input_lhs) - delta_i);
        }

        let mut state = [F::Extension::ZERO; SPONGE_WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            state[i] = vars.local_wires[Self::wire_input(i)] + delta_i;
            state[i + 4] = vars.local_wires[Self::wire_input(i + 4)] - delta_i;
        }
        for i in 8..SPONGE_WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        // 2. Initial Linear
        Self::external_linear_layer_extension(&mut state);

        // 3. Initial Full Rounds
        for r in 0..HALF_N_FULL_ROUNDS {
            for i in 0..SPONGE_WIDTH {
                let c = F::Extension::from_canonical_u64(
                    <F as Poseidon2>::FULL_ROUND_CONSTANTS[r * SPONGE_WIDTH + i],
                );
                state[i] += c;
            }

            for i in 0..SPONGE_WIDTH {
                if r != 0 {
                    let wire = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    constraints.push(state[i] - wire);
                    state[i] = wire;
                }
                state[i] = sbox(state[i]);
            }

            Self::external_linear_layer_extension(&mut state);
        }

        // 4. Partial Rounds
        for r in 0..N_PARTIAL_ROUNDS {
            let c = F::Extension::from_canonical_u64(
                <F as Poseidon2>::PARTIAL_ROUND_CONSTANTS[r],
            );
            state[0] += c;

            let wire = vars.local_wires[Self::wire_partial_sbox(r)];
            constraints.push(state[0] - wire);
            state[0] = wire;
            state[0] = sbox(state[0]);

            Self::internal_linear_layer_extension(&mut state);
        }

        // 5. Terminal Full Rounds
        for r in 0..HALF_N_FULL_ROUNDS {
            let rc_idx = HALF_N_FULL_ROUNDS + r;
            for i in 0..SPONGE_WIDTH {
                let c = F::Extension::from_canonical_u64(
                    <F as Poseidon2>::FULL_ROUND_CONSTANTS[rc_idx * SPONGE_WIDTH + i],
                );
                state[i] += c;
            }

            for i in 0..SPONGE_WIDTH {
                let wire = vars.local_wires[Self::wire_full_sbox_1(r, i)];
                constraints.push(state[i] - wire);
                state[i] = wire;
                state[i] = sbox(state[i]);
            }

            Self::external_linear_layer_extension(&mut state);
        }

        // 6. Output
        for i in 0..SPONGE_WIDTH {
            constraints.push(state[i] - vars.local_wires[Self::wire_output(i)]);
        }

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        let swap = vars.local_wires[Self::WIRE_SWAP];
        yield_constr.one(swap * (swap - F::ONE));

        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            yield_constr.one(swap * (input_rhs - input_lhs) - delta_i);
        }

        let mut state = [F::ZERO; SPONGE_WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            state[i] = vars.local_wires[Self::wire_input(i)] + delta_i;
            state[i + 4] = vars.local_wires[Self::wire_input(i + 4)] - delta_i;
        }
        for i in 8..SPONGE_WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        // Native calls for base field
        F::external_linear_layer(&mut state);

        for r in 0..HALF_N_FULL_ROUNDS {
            for i in 0..SPONGE_WIDTH {
                state[i] += F::from_canonical_u64(
                    <F as Poseidon2>::FULL_ROUND_CONSTANTS[r * SPONGE_WIDTH + i],
                );
            }

            for i in 0..SPONGE_WIDTH {
                if r != 0 {
                    let wire = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    yield_constr.one(state[i] - wire);
                    state[i] = wire;
                }
                state[i] = <F as Poseidon2>::sbox_monomial(state[i]);
            }
            F::external_linear_layer(&mut state);
        }

        for r in 0..N_PARTIAL_ROUNDS {
            state[0] += F::from_canonical_u64(<F as Poseidon2>::PARTIAL_ROUND_CONSTANTS[r]);

            let wire = vars.local_wires[Self::wire_partial_sbox(r)];
            yield_constr.one(state[0] - wire);
            state[0] = wire;
            state[0] = <F as Poseidon2>::sbox_monomial(state[0]);

            F::internal_linear_layer(&mut state);
        }

        for r in 0..HALF_N_FULL_ROUNDS {
            let rc_idx = HALF_N_FULL_ROUNDS + r;
            for i in 0..SPONGE_WIDTH {
                state[i] += F::from_canonical_u64(
                    <F as Poseidon2>::FULL_ROUND_CONSTANTS[rc_idx * SPONGE_WIDTH + i],
                );
            }

            for i in 0..SPONGE_WIDTH {
                let wire = vars.local_wires[Self::wire_full_sbox_1(r, i)];
                yield_constr.one(state[i] - wire);
                state[i] = wire;
                state[i] = <F as Poseidon2>::sbox_monomial(state[i]);
            }
            F::external_linear_layer(&mut state);
        }

        for i in 0..SPONGE_WIDTH {
            yield_constr.one(state[i] - vars.local_wires[Self::wire_output(i)]);
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        // Use MDS gates when we have enough routed wires, reducing builder operations
        // in recursive circuits. Each MDS gate replaces ~50 arithmetic operations with
        // a single degree-1 gate row.
        let use_mds_gate = builder.config.num_routed_wires
            >= Poseidon2MdsGate::<F, D>::new().num_wires();

        let mut constraints = Vec::with_capacity(self.num_constraints());

        let swap = vars.local_wires[Self::WIRE_SWAP];
        constraints.push(builder.mul_sub_extension(swap, swap, swap));

        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let diff = builder.sub_extension(input_rhs, input_lhs);
            constraints.push(builder.mul_sub_extension(swap, diff, delta_i));
        }

        let mut state = [builder.zero_extension(); SPONGE_WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            state[i] = builder.add_extension(input_lhs, delta_i);
            state[i + 4] = builder.sub_extension(input_rhs, delta_i);
        }
        for i in 8..SPONGE_WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        if use_mds_gate {
            Self::external_layer_via_gate(builder, &mut state);
        } else {
            <F as Poseidon2>::external_linear_layer_circuit(builder, &mut state);
        }

        for r in 0..HALF_N_FULL_ROUNDS {
            for i in 0..SPONGE_WIDTH {
                let c = F::Extension::from_canonical_u64(
                    <F as Poseidon2>::FULL_ROUND_CONSTANTS[r * SPONGE_WIDTH + i],
                );
                let c_tgt = builder.constant_extension(c);
                state[i] = builder.add_extension(state[i], c_tgt);
            }

            for i in 0..SPONGE_WIDTH {
                if r != 0 {
                    let wire = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    constraints.push(builder.sub_extension(state[i], wire));
                    state[i] = wire;
                }
                state[i] = <F as Poseidon2>::sbox_monomial_circuit(builder, state[i]);
            }

            if use_mds_gate {
                Self::external_layer_via_gate(builder, &mut state);
            } else {
                <F as Poseidon2>::external_linear_layer_circuit(builder, &mut state);
            }
        }

        for r in 0..N_PARTIAL_ROUNDS {
            let c = F::Extension::from_canonical_u64(
                <F as Poseidon2>::PARTIAL_ROUND_CONSTANTS[r],
            );
            let c_tgt = builder.constant_extension(c);
            state[0] = builder.add_extension(state[0], c_tgt);

            let wire = vars.local_wires[Self::wire_partial_sbox(r)];
            constraints.push(builder.sub_extension(state[0], wire));
            state[0] = wire;
            state[0] = <F as Poseidon2>::sbox_monomial_circuit(builder, state[0]);

            if use_mds_gate {
                Self::internal_layer_via_gate(builder, &mut state);
            } else {
                <F as Poseidon2>::internal_linear_layer_circuit(builder, &mut state);
            }
        }

        for r in 0..HALF_N_FULL_ROUNDS {
            let rc_idx = HALF_N_FULL_ROUNDS + r;
            for i in 0..SPONGE_WIDTH {
                let c = F::Extension::from_canonical_u64(
                    <F as Poseidon2>::FULL_ROUND_CONSTANTS[rc_idx * SPONGE_WIDTH + i],
                );
                let c_tgt = builder.constant_extension(c);
                state[i] = builder.add_extension(state[i], c_tgt);
            }

            for i in 0..SPONGE_WIDTH {
                let wire = vars.local_wires[Self::wire_full_sbox_1(r, i)];
                constraints.push(builder.sub_extension(state[i], wire));
                state[i] = wire;
                state[i] = <F as Poseidon2>::sbox_monomial_circuit(builder, state[i]);
            }

            if use_mds_gate {
                Self::external_layer_via_gate(builder, &mut state);
            } else {
                <F as Poseidon2>::external_linear_layer_circuit(builder, &mut state);
            }
        }

        for i in 0..SPONGE_WIDTH {
            constraints.push(
                builder.sub_extension(state[i], vars.local_wires[Self::wire_output(i)]),
            );
        }

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        let gen = Poseidon2Generator::<F, D> {
            row,
            _phantom: PhantomData,
        };
        vec![WitnessGeneratorRef::new(gen.adapter())]
    }

    fn num_wires(&self) -> usize {
        Self::end()
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        7
    }

    fn num_constraints(&self) -> usize {
        1 + 4 + (3 * 12) + 22 + (4 * 12) + 12
    }
}

// Helpers for MDS gate integration and Extension algebra linear layers
impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> Poseidon2Gate<F, D> {
    /// Apply external linear layer via the combined MDS gate.
    /// This replaces ~50 arithmetic operations with a single gate row + copy constraints.
    fn external_layer_via_gate(
        builder: &mut CircuitBuilder<F, D>,
        state: &mut [ExtensionTarget<D>; SPONGE_WIDTH],
    ) {
        let gate = Poseidon2MdsGate::<F, D>::new();
        let index = builder.add_gate(gate, vec![]);
        for i in 0..SPONGE_WIDTH {
            let input_wire = Poseidon2MdsGate::<F, D>::wires_input(i);
            builder.connect_extension(state[i], ExtensionTarget::from_range(index, input_wire));
        }
        for i in 0..SPONGE_WIDTH {
            let output_wire = Poseidon2MdsGate::<F, D>::wires_external_output(i);
            state[i] = ExtensionTarget::from_range(index, output_wire);
        }
    }

    /// Apply internal linear layer via the combined MDS gate.
    /// This replaces ~47 arithmetic operations with a single gate row + copy constraints.
    fn internal_layer_via_gate(
        builder: &mut CircuitBuilder<F, D>,
        state: &mut [ExtensionTarget<D>; SPONGE_WIDTH],
    ) {
        let gate = Poseidon2MdsGate::<F, D>::new();
        let index = builder.add_gate(gate, vec![]);
        for i in 0..SPONGE_WIDTH {
            let input_wire = Poseidon2MdsGate::<F, D>::wires_input(i);
            builder.connect_extension(state[i], ExtensionTarget::from_range(index, input_wire));
        }
        for i in 0..SPONGE_WIDTH {
            let output_wire = Poseidon2MdsGate::<F, D>::wires_internal_output(i);
            state[i] = ExtensionTarget::from_range(index, output_wire);
        }
    }

    fn external_linear_layer_extension(state: &mut [F::Extension; SPONGE_WIDTH]) {
        // We replicate the Goldilocks-12 logic for extensions.
        // This makes assumptions about the matrix structure matching the specific Goldilocks implementation.
        // Ideally, Poseidon2 trait should provide this.
        
        // Block Mix 4x4
        for chunk in state.chunks_exact_mut(4) {
             let s0 = chunk[0];
             let s1 = chunk[1];
             let s2 = chunk[2];
             let s3 = chunk[3];
             
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
             
             chunk[0] = t6;
             chunk[1] = t5;
             chunk[2] = t7;
             chunk[3] = t4;
        }

        // Column Sums
        let sum0 = state[0] + state[4] + state[8];
        let sum1 = state[1] + state[5] + state[9];
        let sum2 = state[2] + state[6] + state[10];
        let sum3 = state[3] + state[7] + state[11];

        state[0] += sum0; state[4] += sum0; state[8] += sum0;
        state[1] += sum1; state[5] += sum1; state[9] += sum1;
        state[2] += sum2; state[6] += sum2; state[10] += sum2;
        state[3] += sum3; state[7] += sum3; state[11] += sum3;
    }

    fn internal_linear_layer_extension(state: &mut [F::Extension; SPONGE_WIDTH]) {
        let mut sum = F::Extension::ZERO;
        for x in state.iter() {
            sum += *x;
        }
        for i in 0..SPONGE_WIDTH {
            let diag = F::Extension::from_canonical_u64(<F as Poseidon2>::INTERNAL_MATRIX_DIAG[i]);
            state[i] = state[i] * diag + sum;
        }
    }
}

#[derive(Debug, Default)]
pub struct Poseidon2Generator<F: RichField + Extendable<D> + Poseidon2, const D: usize> {
    row: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> SimpleGenerator<F, D>
    for Poseidon2Generator<F, D>
{
    fn id(&self) -> String {
        "Poseidon2Generator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        (0..SPONGE_WIDTH)
            .map(|i| Poseidon2Gate::<F, D>::wire_input(i))
            .chain(Some(Poseidon2Gate::<F, D>::WIRE_SWAP))
            .map(|column| Target::wire(self.row, column))
            .collect()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let local_wire = |column| Wire {
            row: self.row,
            column,
        };

        let mut state = (0..SPONGE_WIDTH)
            .map(|i| witness.get_wire(local_wire(Poseidon2Gate::<F, D>::wire_input(i))))
            .collect::<Vec<_>>();

        let swap_value = witness.get_wire(local_wire(Poseidon2Gate::<F, D>::WIRE_SWAP));
        debug_assert!(swap_value == F::ZERO || swap_value == F::ONE);

        for i in 0..4 {
            let delta_i = swap_value * (state[i + 4] - state[i]);
            out_buffer.set_wire(local_wire(Poseidon2Gate::<F, D>::wire_delta(i)), delta_i)?;
        }

        if swap_value == F::ONE {
            for i in 0..4 {
                state.swap(i, 4 + i);
            }
        }

        let mut state: [F; SPONGE_WIDTH] = state.try_into().unwrap();

        F::external_linear_layer(&mut state);

        for r in 0..HALF_N_FULL_ROUNDS {
            for i in 0..SPONGE_WIDTH {
                state[i] += F::from_canonical_u64(
                    <F as Poseidon2>::FULL_ROUND_CONSTANTS[r * SPONGE_WIDTH + i],
                );
            }

            if r != 0 {
                for i in 0..SPONGE_WIDTH {
                    out_buffer.set_wire(
                        local_wire(Poseidon2Gate::<F, D>::wire_full_sbox_0(r, i)),
                        state[i],
                    )?;
                }
            }

            for i in 0..SPONGE_WIDTH {
                state[i] = <F as Poseidon2>::sbox_monomial(state[i]);
            }

            F::external_linear_layer(&mut state);
        }

        for r in 0..N_PARTIAL_ROUNDS {
            state[0] += F::from_canonical_u64(<F as Poseidon2>::PARTIAL_ROUND_CONSTANTS[r]);

            out_buffer.set_wire(
                local_wire(Poseidon2Gate::<F, D>::wire_partial_sbox(r)),
                state[0]
            )?;

            state[0] = <F as Poseidon2>::sbox_monomial(state[0]);
            F::internal_linear_layer(&mut state);
        }

        for r in 0..HALF_N_FULL_ROUNDS {
            let rc_idx = HALF_N_FULL_ROUNDS + r;
            for i in 0..SPONGE_WIDTH {
                state[i] += F::from_canonical_u64(
                    <F as Poseidon2>::FULL_ROUND_CONSTANTS[rc_idx * SPONGE_WIDTH + i],
                );
            }

            for i in 0..SPONGE_WIDTH {
                out_buffer.set_wire(
                    local_wire(Poseidon2Gate::<F, D>::wire_full_sbox_1(r, i)),
                    state[i],
                )?;
                state[i] = <F as Poseidon2>::sbox_monomial(state[i]);
            }

            F::external_linear_layer(&mut state);
        }

        for i in 0..SPONGE_WIDTH {
            out_buffer.set_wire(local_wire(Poseidon2Gate::<F, D>::wire_output(i)), state[i])?
        }

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
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
    use super::*;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::gates::gate_testing::{test_eval_fns, test_low_degree};
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn test_poseidon2_gate_low_degree() {
        type F = GoldilocksField;
        let gate = Poseidon2Gate::<F, 4>::new();
        test_low_degree(gate)
    }

    #[test]
    fn test_poseidon2_gate_eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = Poseidon2Gate::<F, D>::new();
        test_eval_fns::<F, C, _, D>(gate)
    }
}