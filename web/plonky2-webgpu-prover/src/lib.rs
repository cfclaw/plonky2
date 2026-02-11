use plonky2::field::types::Field;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::GenericConfig;
use plonky2_hw_acc_webgpu::prover::PoseidonGoldilocksWebGpuConfig;
use serde::Serialize;
use wasm_bindgen::prelude::*;

const D: usize = 2;
type C = PoseidonGoldilocksWebGpuConfig;
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
    js_sys::Date::now()
}

/// Initialize the WebGPU context. Must be called once before proving.
#[wasm_bindgen]
pub async fn init_webgpu() -> Result<(), JsError> {
    plonky2_hw_acc_webgpu::context::init_context()
        .await
        .map_err(|e| JsError::new(&format!("WebGPU init failed: {}", e)))
}

/// Build and prove a Fibonacci circuit using WebGPU-accelerated proving.
/// init_webgpu() must be called first.
#[wasm_bindgen]
pub fn prove_fibonacci_webgpu(num_steps: u32) -> Result<String, JsError> {
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
        proof_bytes: vec![],
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
    "plonky2-webgpu-prover loaded (WebGPU-accelerated, PoseidonGoldilocksWebGpuConfig)".to_string()
}
