//! Example demonstrating circuit caching to avoid expensive rebuilds.
//!
//! On the first run, the circuit is built from scratch and saved to disk.
//! On subsequent runs, the circuit is loaded from the cache file, skipping
//! the expensive build step entirely.
//!
//! Run twice to observe the difference:
//!
//! ```sh
//! # First run: builds and caches the circuit
//! cargo run --example circuit_caching
//!
//! # Second run: loads from cache (much faster)
//! cargo run --example circuit_caching
//!
//! # To force a rebuild, delete the cache file
//! rm fibonacci_circuit.bin
//! ```

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use plonky2::field::types::Field;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_cache::build_or_load_circuit;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::util::serialization::{DefaultGateSerializer, DefaultGeneratorSerializer};

/// Build a Fibonacci circuit that proves knowledge of the 100th Fibonacci
/// number starting from two initial values.
fn build_fibonacci_circuit<
    F: Field + plonky2::hash::hash_types::RichField + plonky2::field::extension::Extendable<D>,
    const D: usize,
    C: GenericConfig<D, F = F>,
>(
    config: CircuitConfig,
) -> plonky2::plonk::circuit_data::CircuitData<F, C, D> {
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let initial_a = builder.add_virtual_target();
    let initial_b = builder.add_virtual_target();
    let mut prev_target = initial_a;
    let mut cur_target = initial_b;
    for _ in 0..99 {
        let temp = builder.add(prev_target, cur_target);
        prev_target = cur_target;
        cur_target = temp;
    }

    builder.register_public_input(initial_a);
    builder.register_public_input(initial_b);
    builder.register_public_input(cur_target);

    builder.build::<C>()
}

fn main() -> Result<()> {
    // Initialize logging so we can see cache hit/miss messages.
    env_logger::init();

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    let gate_serializer = DefaultGateSerializer;
    let generator_serializer = DefaultGeneratorSerializer::<C, D>::default();

    let cache_path = Path::new("fibonacci_circuit.bin");

    // ---- Build or load the circuit ----
    let start = Instant::now();
    let data = build_or_load_circuit(
        cache_path,
        &gate_serializer,
        &generator_serializer,
        || {
            println!("Building circuit from scratch...");
            let config = CircuitConfig::standard_recursion_config();
            build_fibonacci_circuit::<F, D, C>(config)
        },
    )
    .expect("Failed to build or load circuit");
    let elapsed = start.elapsed();

    if cache_path.exists() {
        let file_size = std::fs::metadata(cache_path).unwrap().len();
        println!(
            "Circuit ready in {:.3}s (cache file: {} bytes)",
            elapsed.as_secs_f64(),
            file_size,
        );
    }

    // ---- Generate a proof ----
    let start = Instant::now();
    let mut pw = PartialWitness::new();
    pw.set_target(data.prover_only.public_inputs[0], F::ZERO)?;
    pw.set_target(data.prover_only.public_inputs[1], F::ONE)?;
    let proof = data.prove(pw)?;
    println!("Proof generated in {:.3}s", start.elapsed().as_secs_f64());

    println!(
        "100th Fibonacci number mod |F| (starting with {}, {}) is: {}",
        proof.public_inputs[0], proof.public_inputs[1], proof.public_inputs[2]
    );

    // ---- Verify the proof ----
    let start = Instant::now();
    data.verify(proof)?;
    println!("Proof verified in {:.3}s", start.elapsed().as_secs_f64());

    Ok(())
}
