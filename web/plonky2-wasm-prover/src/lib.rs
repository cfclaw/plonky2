use plonky2::field::types::Field;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use serde::Serialize;
use wasm_bindgen::prelude::*;

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;

#[derive(Serialize)]
struct ProveResult {
    proof_bytes: Vec<u8>,
    public_inputs: Vec<String>,
    prove_time_ms: f64,
    verify_time_ms: f64,
    proof_size: usize,
}

fn performance_now() -> f64 {
    // Use web-sys performance.now() for high-res timing in browser
    js_sys::Date::now()
}

/// Build and prove a Fibonacci circuit with the given number of iterations.
/// Returns JSON with proof bytes, timing, and public inputs.
#[wasm_bindgen]
pub fn prove_fibonacci(num_steps: u32) -> Result<String, JsError> {
    // Build circuit
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let initial_a = builder.add_virtual_target();
    let initial_b = builder.add_virtual_target();
    builder.register_public_input(initial_a);
    builder.register_public_input(initial_b);

    let mut prev_target = initial_a;
    let mut cur_target = initial_b;
    for _ in 0..num_steps {
        let temp = builder.add(prev_target, cur_target);
        prev_target = cur_target;
        cur_target = temp;
    }
    builder.register_public_input(cur_target);

    let start = performance_now();
    let data = builder.build::<C>();

    let mut pw = PartialWitness::new();
    pw.set_target(initial_a, F::ZERO)
        .map_err(|e| JsError::new(&format!("Failed to set witness: {}", e)))?;
    pw.set_target(initial_b, F::ONE)
        .map_err(|e| JsError::new(&format!("Failed to set witness: {}", e)))?;

    let proof = data.prove(pw)
        .map_err(|e| JsError::new(&format!("Proving failed: {}", e)))?;
    let prove_time = performance_now() - start;

    // Serialize proof
    let proof_bytes = serde_json::to_vec(&proof)
        .map_err(|e| JsError::new(&format!("Serialization failed: {}", e)))?;

    // Verify
    let verify_start = performance_now();
    data.verify(proof.clone())
        .map_err(|e| JsError::new(&format!("Verification failed: {}", e)))?;
    let verify_time = performance_now() - verify_start;

    let public_inputs: Vec<String> = proof.public_inputs.iter().map(|f| format!("{:?}", f)).collect();

    let result = ProveResult {
        proof_size: proof_bytes.len(),
        proof_bytes: vec![], // Don't send the full bytes to JS to save memory
        public_inputs,
        prove_time_ms: prove_time,
        verify_time_ms: verify_time,
    };

    serde_json::to_string(&result)
        .map_err(|e| JsError::new(&format!("JSON serialization failed: {}", e)))
}

/// Simple health check to verify WASM module loaded correctly.
#[wasm_bindgen]
pub fn health_check() -> String {
    "plonky2-wasm-prover loaded (CPU-only, PoseidonGoldilocksConfig)".to_string()
}
