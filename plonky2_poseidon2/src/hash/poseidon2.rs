
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::fmt::Debug;
use plonky2_field::extension::Extendable;
use serde::{Deserialize, Serialize};

use plonky2::hash::hash_types::{HashOut, RichField};
use plonky2::hash::hashing::{compress, hash_n_to_hash_no_pad, PlonkyPermutation};
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::{AlgebraicHasher, Hasher};

pub const SPONGE_RATE: usize = 8;
pub const SPONGE_CAPACITY: usize = 4;
pub const SPONGE_WIDTH: usize = SPONGE_RATE + SPONGE_CAPACITY;

pub const HALF_N_FULL_ROUNDS: usize = 4;
pub const N_FULL_ROUNDS_TOTAL: usize = 2 * HALF_N_FULL_ROUNDS;
pub const N_PARTIAL_ROUNDS: usize = 22;
pub const N_ROUNDS: usize = N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS;


/// Trait defining the parameters and operations for the Poseidon2 permutation.
pub trait Poseidon2: RichField {
    /// Diagonal for the internal matrix (M_internal = D + J).
    const INTERNAL_MATRIX_DIAG: [u64; SPONGE_WIDTH];

    /// Constants for the full rounds (Initial and Terminal).
    /// Size: (HALF_N_FULL_ROUNDS * 2) * SPONGE_WIDTH
    const FULL_ROUND_CONSTANTS: [u64; N_FULL_ROUNDS_TOTAL * SPONGE_WIDTH];

    /// Constants for the partial rounds.
    /// In partial rounds, we only add a constant to the first element (state[0]).
    const PARTIAL_ROUND_CONSTANTS: [u64; N_PARTIAL_ROUNDS];

    /// S-box monomial x^7
    #[inline(always)]
    fn sbox_monomial(x: Self) -> Self {
        let x2 = x.square();
        let x4 = x2.square();
        let x3 = x * x2;
        x3 * x4
    }

    /// Circuit version of S-box x^7
    fn sbox_monomial_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        x: ExtensionTarget<D>,
    ) -> ExtensionTarget<D>
    where
        Self: RichField + Extendable<D>,
    {
        builder.exp_u64_extension(x, 7)
    }

    /// External Linear Layer (M_ext).
    /// Optimized implementation should use the 4x4 block + column sum decomposition.
    fn external_linear_layer(state: &mut [Self; SPONGE_WIDTH]);

    /// Circuit version of External Linear Layer.
    fn external_linear_layer_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &mut [ExtensionTarget<D>; SPONGE_WIDTH],
    ) where
        Self: RichField + Extendable<D>;

    /// Internal Linear Layer (M_int = D + J).
    fn internal_linear_layer(state: &mut [Self; SPONGE_WIDTH]);

    /// Circuit version of Internal Linear Layer.
    fn internal_linear_layer_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &mut [ExtensionTarget<D>; SPONGE_WIDTH],
    ) where
        Self: RichField + Extendable<D>;

    /// The full Poseidon2 permutation function (Native).
    #[inline(always)]
    fn poseidon2(mut state: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        // 1. Initial External Linear Layer
        Self::external_linear_layer(&mut state);

        // 2. Initial Full Rounds
        for r in 0..HALF_N_FULL_ROUNDS {
            for i in 0..SPONGE_WIDTH {
                state[i] += Self::from_canonical_u64(Self::FULL_ROUND_CONSTANTS[r * SPONGE_WIDTH + i]);
                state[i] = Poseidon2::sbox_monomial(state[i]);
            }
            Self::external_linear_layer(&mut state);
        }

        // 3. Partial Rounds
        for r in 0..N_PARTIAL_ROUNDS {
            state[0] += Self::from_canonical_u64(Self::PARTIAL_ROUND_CONSTANTS[r]);
            state[0] = Poseidon2::sbox_monomial(state[0]);
            Self::internal_linear_layer(&mut state);
        }

        // 4. Terminal Full Rounds
        for r in 0..HALF_N_FULL_ROUNDS {
            let rc_idx = HALF_N_FULL_ROUNDS + r;
            for i in 0..SPONGE_WIDTH {
                state[i] += Self::from_canonical_u64(Self::FULL_ROUND_CONSTANTS[rc_idx * SPONGE_WIDTH + i]);
                state[i] = Poseidon2::sbox_monomial(state[i]);
            }
            Self::external_linear_layer(&mut state);
        }

        state
    }
}

/// PlonkyPermutation wrapper.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Poseidon2Permutation<T> {
    pub state: [T; SPONGE_WIDTH],
}

impl<T> AsRef<[T]> for Poseidon2Permutation<T> {
    fn as_ref(&self) -> &[T] {
        &self.state
    }
}

pub trait Poseidon2Permuter: Sized {
    fn permute(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH];
}

impl<F: Poseidon2> Poseidon2Permuter for F {
    fn permute(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        <F as Poseidon2>::poseidon2(input)
    }
}

impl Poseidon2Permuter for Target {
    fn permute(_input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        panic!("Circuit permutation must use AlgebraicHasher::permute_swapped");
    }
}

impl<T: Copy + Debug + Default + Eq + Poseidon2Permuter + Send + Sync + Serialize>
    PlonkyPermutation<T> for Poseidon2Permutation<T>
{
    const RATE: usize = SPONGE_RATE;
    const WIDTH: usize = SPONGE_WIDTH;

    fn new<I: IntoIterator<Item = T>>(elts: I) -> Self {
        let mut perm = Self {
            state: [T::default(); SPONGE_WIDTH],
        };
        perm.set_from_iter(elts, 0);
        perm
    }

    fn set_elt(&mut self, elt: T, idx: usize) {
        self.state[idx] = elt;
    }

    fn set_from_slice(&mut self, elts: &[T], start_idx: usize) {
        let begin = start_idx;
        let end = start_idx + elts.len();
        self.state[begin..end].copy_from_slice(elts);
    }

    fn set_from_iter<I: IntoIterator<Item = T>>(&mut self, elts: I, start_idx: usize) {
        for (s, e) in self.state[start_idx..].iter_mut().zip(elts) {
            *s = e;
        }
    }

    fn permute(&mut self) {
        self.state = T::permute(self.state);
    }

    fn squeeze(&self) -> &[T] {
        &self.state[..Self::RATE]
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Poseidon2Hash;

impl<F: RichField + Poseidon2> Hasher<F> for Poseidon2Hash {
    const HASH_SIZE: usize = 32;
    type Hash = HashOut<F>;
    type Permutation = Poseidon2Permutation<F>;

    fn hash_no_pad(input: &[F]) -> Self::Hash {
        hash_n_to_hash_no_pad::<F, Self::Permutation>(input)
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
        compress::<F, Self::Permutation>(left, right)
    }
}

impl<F: RichField + Poseidon2> AlgebraicHasher<F> for Poseidon2Hash {
    type AlgebraicPermutation = Poseidon2Permutation<Target>;

    fn permute_swapped<const D: usize>(
        inputs: Self::AlgebraicPermutation,
        swap: BoolTarget,
        builder: &mut CircuitBuilder<F, D>,
    ) -> Self::AlgebraicPermutation
    where
        F: RichField + Extendable<D>,
    {
        use crate::gates::poseidon2::Poseidon2Gate;

        let gate = Poseidon2Gate::<F, D>::new();
        let gate_index = builder.add_gate(gate, vec![]);

        let swap_wire = Target::wire(gate_index, Poseidon2Gate::<F, D>::WIRE_SWAP);
        builder.connect(swap.target, swap_wire);

        let input_targets = inputs.as_ref();
        for i in 0..SPONGE_WIDTH {
            let in_wire = Target::wire(gate_index, Poseidon2Gate::<F, D>::wire_input(i));
            builder.connect(input_targets[i], in_wire);
        }

        Poseidon2Permutation {
            state: (0..SPONGE_WIDTH)
                .map(|i| Target::wire(gate_index, Poseidon2Gate::<F, D>::wire_output(i)))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }
}