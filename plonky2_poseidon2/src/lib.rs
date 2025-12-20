use plonky2::plonk::config::{CpuProverCompute, GenericConfig};
use plonky2_field::{extension::quadratic::QuadraticExtension, goldilocks_field::GoldilocksField};
use serde::Serialize;

use crate::hash::poseidon2::Poseidon2Hash;

pub mod hash;
pub mod gates;


/// Configuration using Poseidon2 over the Goldilocks field.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Serialize)]
pub struct Poseidon2GoldilocksConfig;
impl GenericConfig<2> for Poseidon2GoldilocksConfig {
    type F = GoldilocksField;
    type FE = QuadraticExtension<Self::F>;
    type Hasher = Poseidon2Hash;
    type InnerHasher = Poseidon2Hash;
    type Compute = CpuProverCompute;
}
