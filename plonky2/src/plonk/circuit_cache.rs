//! Circuit caching utilities for avoiding expensive circuit rebuilds.
//!
//! Building a plonky2 circuit involves several computationally expensive steps:
//! computing selector polynomials, generating sigma polynomials, performing
//! IFFTs, computing low-degree extensions, and constructing Merkle trees.
//! For production systems where the same circuit is used repeatedly with
//! different witnesses, this module provides utilities to serialize a built
//! circuit to disk and reload it on subsequent runs, skipping the entire
//! build process.
//!
//! # Usage
//!
//! The primary entry point is [`build_or_load_circuit`], which accepts a
//! cache file path and a closure that builds the circuit. On the first call
//! the circuit is built and saved; on subsequent calls it is loaded from disk.
//!
//! ```rust,no_run
//! use std::path::Path;
//! use plonky2::plonk::circuit_builder::CircuitBuilder;
//! use plonky2::plonk::circuit_cache::build_or_load_circuit;
//! use plonky2::plonk::circuit_data::CircuitConfig;
//! use plonky2::plonk::config::PoseidonGoldilocksConfig;
//! use plonky2::util::serialization::{DefaultGateSerializer, DefaultGeneratorSerializer};
//!
//! const D: usize = 2;
//! type C = PoseidonGoldilocksConfig;
//! type F = <C as plonky2::plonk::config::GenericConfig<D>>::F;
//!
//! let gate_serializer = DefaultGateSerializer;
//! let generator_serializer = DefaultGeneratorSerializer::<C, D>::default();
//!
//! let data = build_or_load_circuit(
//!     Path::new("my_circuit.bin"),
//!     &gate_serializer,
//!     &generator_serializer,
//!     || {
//!         let config = CircuitConfig::standard_recursion_config();
//!         let mut builder = CircuitBuilder::<F, D>::new(config);
//!         // ... define circuit ...
//!         builder.build::<C>()
//!     },
//! ).expect("Failed to build or load circuit");
//!
//! // Use `data` for proving / verification as usual.
//! ```
//!
//! For finer-grained control, use [`save_circuit_data`] and [`load_circuit_data`]
//! directly, or the convenience methods added to [`CircuitData`], [`ProverCircuitData`],
//! and [`VerifierCircuitData`].

#[cfg(feature = "std")]
use std::fs;
#[cfg(feature = "std")]
use std::path::Path;

use crate::field::extension::Extendable;
use crate::hash::hash_types::RichField;
use crate::plonk::circuit_data::{CircuitData, ProverCircuitData, VerifierCircuitData};
use crate::plonk::config::GenericConfig;
use crate::util::serialization::{GateSerializer, WitnessGeneratorSerializer};

/// Error type for circuit cache operations.
#[derive(Debug)]
pub enum CacheError {
    /// The cache file was not found or could not be read.
    IoError(String),
    /// The cached data could not be deserialized (e.g., stale cache).
    DeserializationError(String),
    /// The circuit data could not be serialized.
    SerializationError(String),
}

impl core::fmt::Display for CacheError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CacheError::IoError(msg) => write!(f, "I/O error: {}", msg),
            CacheError::DeserializationError(msg) => {
                write!(f, "deserialization error: {}", msg)
            }
            CacheError::SerializationError(msg) => {
                write!(f, "serialization error: {}", msg)
            }
        }
    }
}

/// Save [`CircuitData`] to a file at the given path.
///
/// The data is serialized using the plonky2 binary format via `to_bytes()`.
/// Both gate and generator serializers are required because the prover data
/// contains witness generators.
#[cfg(feature = "std")]
pub fn save_circuit_data<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    data: &CircuitData<F, C, D>,
    path: &Path,
    gate_serializer: &dyn GateSerializer<F, D>,
    generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
) -> Result<(), CacheError> {
    let bytes = data
        .to_bytes(gate_serializer, generator_serializer)
        .map_err(|e| CacheError::SerializationError(format!("{:?}", e)))?;
    fs::write(path, bytes).map_err(|e| CacheError::IoError(e.to_string()))?;
    Ok(())
}

/// Load [`CircuitData`] from a file at the given path.
///
/// Returns a deserialization error if the file contents are invalid or
/// incompatible (e.g., the circuit definition has changed since the cache
/// was written).
#[cfg(feature = "std")]
pub fn load_circuit_data<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    path: &Path,
    gate_serializer: &dyn GateSerializer<F, D>,
    generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
) -> Result<CircuitData<F, C, D>, CacheError> {
    let bytes = fs::read(path).map_err(|e| CacheError::IoError(e.to_string()))?;
    CircuitData::from_bytes(&bytes, gate_serializer, generator_serializer)
        .map_err(|e| CacheError::DeserializationError(format!("{:?}", e)))
}

/// Save [`ProverCircuitData`] to a file at the given path.
#[cfg(feature = "std")]
pub fn save_prover_circuit_data<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    data: &ProverCircuitData<F, C, D>,
    path: &Path,
    gate_serializer: &dyn GateSerializer<F, D>,
    generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
) -> Result<(), CacheError> {
    let bytes = data
        .to_bytes(gate_serializer, generator_serializer)
        .map_err(|e| CacheError::SerializationError(format!("{:?}", e)))?;
    fs::write(path, bytes).map_err(|e| CacheError::IoError(e.to_string()))?;
    Ok(())
}

/// Load [`ProverCircuitData`] from a file at the given path.
#[cfg(feature = "std")]
pub fn load_prover_circuit_data<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    path: &Path,
    gate_serializer: &dyn GateSerializer<F, D>,
    generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
) -> Result<ProverCircuitData<F, C, D>, CacheError> {
    let bytes = fs::read(path).map_err(|e| CacheError::IoError(e.to_string()))?;
    ProverCircuitData::from_bytes(&bytes, gate_serializer, generator_serializer)
        .map_err(|e| CacheError::DeserializationError(format!("{:?}", e)))
}

/// Save [`VerifierCircuitData`] to a file at the given path.
#[cfg(feature = "std")]
pub fn save_verifier_circuit_data<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    data: &VerifierCircuitData<F, C, D>,
    path: &Path,
    gate_serializer: &dyn GateSerializer<F, D>,
) -> Result<(), CacheError> {
    let bytes = data
        .to_bytes(gate_serializer)
        .map_err(|e| CacheError::SerializationError(format!("{:?}", e)))?;
    fs::write(path, bytes).map_err(|e| CacheError::IoError(e.to_string()))?;
    Ok(())
}

/// Load [`VerifierCircuitData`] from a file at the given path.
#[cfg(feature = "std")]
pub fn load_verifier_circuit_data<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    path: &Path,
    gate_serializer: &dyn GateSerializer<F, D>,
) -> Result<VerifierCircuitData<F, C, D>, CacheError> {
    let bytes = fs::read(path).map_err(|e| CacheError::IoError(e.to_string()))?;
    VerifierCircuitData::from_bytes(bytes, gate_serializer)
        .map_err(|e| CacheError::DeserializationError(format!("{:?}", e)))
}

/// Build a circuit or load it from a cache file.
///
/// This is the primary convenience function for circuit caching. It works
/// as follows:
///
/// 1. If a file exists at `cache_path`, attempt to deserialize it into
///    [`CircuitData`]. If deserialization succeeds, return the loaded data.
/// 2. If the file does not exist **or** deserialization fails (e.g., stale
///    cache from a changed circuit), call `build_fn` to build a fresh
///    circuit.
/// 3. Serialize the freshly built circuit to `cache_path` for future use.
/// 4. Return the circuit data.
///
/// # Cache Invalidation
///
/// There is no automatic fingerprinting of the circuit definition. If you
/// change the circuit, delete the cache file to force a rebuild. A
/// deserialization failure (e.g., from a structurally incompatible cached
/// file) will also trigger a rebuild automatically.
///
/// # Errors
///
/// Returns [`CacheError`] only if the build succeeds but the result cannot
/// be serialized to disk. Loading failures are handled transparently by
/// falling back to building.
#[cfg(feature = "std")]
pub fn build_or_load_circuit<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    cache_path: &Path,
    gate_serializer: &dyn GateSerializer<F, D>,
    generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
    build_fn: impl FnOnce() -> CircuitData<F, C, D>,
) -> Result<CircuitData<F, C, D>, CacheError> {
    // Try to load from cache first.
    if cache_path.exists() {
        match load_circuit_data::<F, C, D>(cache_path, gate_serializer, generator_serializer) {
            Ok(data) => {
                log::info!("Loaded circuit from cache: {}", cache_path.display());
                return Ok(data);
            }
            Err(e) => {
                log::warn!(
                    "Failed to load circuit cache at {}: {}. Rebuilding.",
                    cache_path.display(),
                    e,
                );
            }
        }
    }

    // Build the circuit.
    log::info!("Building circuit (no valid cache found)...");
    let data = build_fn();

    // Save to cache for next time.
    save_circuit_data(&data, cache_path, gate_serializer, generator_serializer)?;
    log::info!("Saved circuit cache to: {}", cache_path.display());

    Ok(data)
}

/// Build a prover circuit or load it from a cache file.
///
/// Same semantics as [`build_or_load_circuit`] but for [`ProverCircuitData`].
#[cfg(feature = "std")]
pub fn build_or_load_prover_circuit<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    cache_path: &Path,
    gate_serializer: &dyn GateSerializer<F, D>,
    generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
    build_fn: impl FnOnce() -> ProverCircuitData<F, C, D>,
) -> Result<ProverCircuitData<F, C, D>, CacheError> {
    if cache_path.exists() {
        match load_prover_circuit_data::<F, C, D>(
            cache_path,
            gate_serializer,
            generator_serializer,
        ) {
            Ok(data) => {
                log::info!(
                    "Loaded prover circuit from cache: {}",
                    cache_path.display()
                );
                return Ok(data);
            }
            Err(e) => {
                log::warn!(
                    "Failed to load prover circuit cache at {}: {}. Rebuilding.",
                    cache_path.display(),
                    e,
                );
            }
        }
    }

    log::info!("Building prover circuit (no valid cache found)...");
    let data = build_fn();

    save_prover_circuit_data(&data, cache_path, gate_serializer, generator_serializer)?;
    log::info!("Saved prover circuit cache to: {}", cache_path.display());

    Ok(data)
}

/// Build a verifier circuit or load it from a cache file.
///
/// Same semantics as [`build_or_load_circuit`] but for [`VerifierCircuitData`].
#[cfg(feature = "std")]
pub fn build_or_load_verifier_circuit<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    cache_path: &Path,
    gate_serializer: &dyn GateSerializer<F, D>,
    build_fn: impl FnOnce() -> VerifierCircuitData<F, C, D>,
) -> Result<VerifierCircuitData<F, C, D>, CacheError> {
    if cache_path.exists() {
        match load_verifier_circuit_data::<F, C, D>(cache_path, gate_serializer) {
            Ok(data) => {
                log::info!(
                    "Loaded verifier circuit from cache: {}",
                    cache_path.display()
                );
                return Ok(data);
            }
            Err(e) => {
                log::warn!(
                    "Failed to load verifier circuit cache at {}: {}. Rebuilding.",
                    cache_path.display(),
                    e,
                );
            }
        }
    }

    log::info!("Building verifier circuit (no valid cache found)...");
    let data = build_fn();

    save_verifier_circuit_data(&data, cache_path, gate_serializer)?;
    log::info!(
        "Saved verifier circuit cache to: {}",
        cache_path.display()
    );

    Ok(data)
}

#[cfg(test)]
#[cfg(feature = "std")]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;
    use crate::field::types::Field;
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::PoseidonGoldilocksConfig;
    use crate::util::serialization::{DefaultGateSerializer, DefaultGeneratorSerializer};

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Returns a unique temporary file path for each test invocation.
    fn temp_path(suffix: &str) -> std::path::PathBuf {
        let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        std::env::temp_dir().join(format!("plonky2_cache_test_{}_{}_{}", pid, id, suffix))
    }

    /// Remove a file if it exists, ignoring errors.
    fn cleanup(path: &Path) {
        let _ = fs::remove_file(path);
    }

    fn build_fibonacci_circuit() -> CircuitData<F, C, D> {
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let initial_a = builder.add_virtual_target();
        let initial_b = builder.add_virtual_target();
        let mut prev_target = initial_a;
        let mut cur_target = initial_b;
        for _ in 0..10 {
            let temp = builder.add(prev_target, cur_target);
            prev_target = cur_target;
            cur_target = temp;
        }

        builder.register_public_input(initial_a);
        builder.register_public_input(initial_b);
        builder.register_public_input(cur_target);

        builder.build::<C>()
    }

    #[test]
    fn test_save_and_load_circuit_data() {
        let data = build_fibonacci_circuit();
        let gate_serializer = DefaultGateSerializer;
        let generator_serializer = DefaultGeneratorSerializer::<C, D>::default();

        let path = temp_path("circuit.bin");
        cleanup(&path);

        // Save.
        save_circuit_data(&data, &path, &gate_serializer, &generator_serializer).unwrap();
        assert!(path.exists());

        // Load.
        let loaded =
            load_circuit_data::<F, C, D>(&path, &gate_serializer, &generator_serializer).unwrap();

        cleanup(&path);

        // The loaded circuit should produce the same proofs.
        let mut pw = PartialWitness::new();
        pw.set_target(loaded.prover_only.public_inputs[0], F::ZERO)
            .unwrap();
        pw.set_target(loaded.prover_only.public_inputs[1], F::ONE)
            .unwrap();
        let proof = loaded.prove(pw).unwrap();
        loaded.verify(proof).unwrap();
    }

    #[test]
    fn test_build_or_load_circuit() {
        let gate_serializer = DefaultGateSerializer;
        let generator_serializer = DefaultGeneratorSerializer::<C, D>::default();

        let path = temp_path("build_or_load.bin");
        cleanup(&path);

        // First call: builds.
        let data1 = build_or_load_circuit(
            &path,
            &gate_serializer,
            &generator_serializer,
            build_fibonacci_circuit,
        )
        .unwrap();
        assert!(path.exists());

        // Second call: loads from cache.
        let data2 = build_or_load_circuit::<F, C, D>(
            &path,
            &gate_serializer,
            &generator_serializer,
            || panic!("build_fn should not be called when cache exists"),
        )
        .unwrap();

        cleanup(&path);

        // Both should have the same circuit digest.
        assert_eq!(
            data1.verifier_only.circuit_digest,
            data2.verifier_only.circuit_digest
        );

        // Prove and verify with loaded data.
        let mut pw = PartialWitness::new();
        pw.set_target(data2.prover_only.public_inputs[0], F::ZERO)
            .unwrap();
        pw.set_target(data2.prover_only.public_inputs[1], F::ONE)
            .unwrap();
        let proof = data2.prove(pw).unwrap();
        data2.verify(proof).unwrap();
    }

    #[test]
    fn test_stale_cache_triggers_rebuild() {
        let gate_serializer = DefaultGateSerializer;
        let generator_serializer = DefaultGeneratorSerializer::<C, D>::default();

        let path = temp_path("stale.bin");

        // Write garbage to the cache file.
        fs::write(&path, b"not a valid circuit").unwrap();

        // build_or_load should detect the bad cache and rebuild.
        let data = build_or_load_circuit(
            &path,
            &gate_serializer,
            &generator_serializer,
            build_fibonacci_circuit,
        )
        .unwrap();

        cleanup(&path);

        // Verify the rebuilt circuit works.
        let mut pw = PartialWitness::new();
        pw.set_target(data.prover_only.public_inputs[0], F::ZERO)
            .unwrap();
        pw.set_target(data.prover_only.public_inputs[1], F::ONE)
            .unwrap();
        let proof = data.prove(pw).unwrap();
        data.verify(proof).unwrap();
    }

    #[test]
    fn test_save_and_load_prover_circuit_data() {
        let data = build_fibonacci_circuit().prover_data();
        let gate_serializer = DefaultGateSerializer;
        let generator_serializer = DefaultGeneratorSerializer::<C, D>::default();

        let path = temp_path("prover.bin");
        cleanup(&path);

        save_prover_circuit_data(&data, &path, &gate_serializer, &generator_serializer).unwrap();
        let loaded = load_prover_circuit_data::<F, C, D>(
            &path,
            &gate_serializer,
            &generator_serializer,
        )
        .unwrap();

        cleanup(&path);

        let mut pw = PartialWitness::new();
        pw.set_target(loaded.prover_only.public_inputs[0], F::ZERO)
            .unwrap();
        pw.set_target(loaded.prover_only.public_inputs[1], F::ONE)
            .unwrap();
        let proof = loaded.prove(pw).unwrap();
        assert_eq!(proof.public_inputs[0], F::ZERO);
        assert_eq!(proof.public_inputs[1], F::ONE);
    }

    #[test]
    fn test_save_and_load_verifier_circuit_data() {
        let circuit = build_fibonacci_circuit();
        let gate_serializer = DefaultGateSerializer;

        let path = temp_path("verifier.bin");
        cleanup(&path);

        let verifier = circuit.verifier_data();
        save_verifier_circuit_data(&verifier, &path, &gate_serializer).unwrap();

        let loaded =
            load_verifier_circuit_data::<F, C, D>(&path, &gate_serializer).unwrap();

        cleanup(&path);

        assert_eq!(
            verifier.verifier_only.circuit_digest,
            loaded.verifier_only.circuit_digest,
        );

        // Prove with original circuit, verify with loaded verifier.
        let mut pw = PartialWitness::new();
        pw.set_target(circuit.prover_only.public_inputs[0], F::ZERO)
            .unwrap();
        pw.set_target(circuit.prover_only.public_inputs[1], F::ONE)
            .unwrap();
        let proof = circuit.prove(pw).unwrap();
        loaded.verify(proof).unwrap();
    }
}
