use std::cell::RefCell;
use wasm_bindgen::prelude::*;
use serde::Serialize;

use plonky2::gates::arithmetic_base::ArithmeticGate;
use plonky2::gates::arithmetic_extension::ArithmeticExtensionGate;
use plonky2::gates::base_sum::BaseSumGate;
use plonky2::gates::constant::ConstantGate;
use plonky2::gates::coset_interpolation::CosetInterpolationGate;
use plonky2::gates::gate::GateRef;
use plonky2::gates::multiplication_extension::MulExtensionGate;
use plonky2::gates::noop::NoopGate;
use plonky2::gates::poseidon::PoseidonGate;
use plonky2::gates::poseidon_mds::PoseidonMdsGate;
use plonky2::gates::random_access::RandomAccessGate;
use plonky2::gates::reducing::ReducingGate;
use plonky2::gates::reducing_extension::ReducingExtensionGate;
use plonky2::plonk::verifier_helper::verify_proof_borrowed;
use plonky2_field::interpolation::barycentric_weights;
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::types::Sample;
use plonky2_field::extension::Extendable;

use plonky2::hash::hash_types::{HashOut, HashOutTarget, RichField};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{
    CircuitConfig, CircuitData, CommonCircuitData, VerifierCircuitTarget, VerifierOnlyCircuitData,
};
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig, PoseidonGoldilocksConfig};
use plonky2::plonk::proof::{ProofWithPublicInputs, ProofWithPublicInputsTarget};
use plonky2::plonk::prover::prove;
use plonky2::util::timing::TimingTree;
use plonky2_hw_acc_webgpu::prover::PoseidonGoldilocksWebGpuConfig;
use log::Level;

// Concrete type aliases for circuit state caching
type CpuInner = DummyPsyTypeCCircuit<GoldilocksField, PoseidonGoldilocksConfig, 2>;
type CpuRecursive = DummyPsyTypeCRecursiveVerifierCircuit<GoldilocksField, PoseidonGoldilocksConfig, 2>;
type GpuInner = DummyPsyTypeCCircuit<GoldilocksField, PoseidonGoldilocksWebGpuConfig, 2>;
type GpuRecursive = DummyPsyTypeCRecursiveVerifierCircuit<GoldilocksField, PoseidonGoldilocksWebGpuConfig, 2>;

thread_local! {
    static CPU_CIRCUITS: RefCell<Option<(CpuInner, CpuRecursive)>> = RefCell::new(None);
    static GPU_CIRCUITS: RefCell<Option<(GpuInner, GpuRecursive)>> = RefCell::new(None);
}

#[wasm_bindgen]
pub fn init() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub async fn init_gpu() -> Result<(), JsValue> {
    plonky2_hw_acc_webgpu::context::init_context()
        .await
        .map_err(|e| JsValue::from_str(&format!("GPU init failed: {}", e)))
}

/// Configure GPU for mobile / memory-constrained devices.
/// Enables chunked GPU→CPU downloads (4 MiB chunks) to prevent
/// BufferAsyncError / device loss from OOM on iOS Safari.
/// Must be called after `init_gpu()`. Note: if the adapter reports
/// max_buffer_size ≤ 256 MiB, mobile mode is enabled automatically.
#[wasm_bindgen]
pub fn configure_gpu_for_mobile() -> Result<(), JsValue> {
    plonky2_hw_acc_webgpu::context::configure_for_mobile()
        .map_err(|e| JsValue::from_str(&format!("configure_for_mobile failed: {}", e)))
}

/// Register a JS callback for GPU phase boundary yields.
///
/// The prover calls this function between GPU phases (after destroying buffers,
/// before allocating new ones). The callback **must return a Promise** — the
/// prover awaits it, giving the browser event loop a turn to reclaim destroyed
/// GPU memory.
///
/// This replaces the built-in `setTimeout(0)` fallback with JS-controlled
/// yield behavior, enabling progress reporting and explicit scheduling.
///
/// ```js
/// // Simple: yield one event-loop turn between GPU phases
/// wasm.set_gpu_yield_callback(() => new Promise(r => setTimeout(r, 0)));
///
/// // With progress reporting:
/// wasm.set_gpu_yield_callback(() => {
///   postMessage({ type: 'gpu_phase_done' });
///   return new Promise(r => setTimeout(r, 0));
/// });
/// ```
#[wasm_bindgen]
pub fn set_gpu_yield_callback(f: js_sys::Function) {
    plonky2_hw_acc_webgpu::context::set_gpu_yield_callback(Some(f));
}

/// Clear the GPU yield callback, reverting to the built-in setTimeout(0) fallback.
#[wasm_bindgen]
pub fn clear_gpu_yield_callback() {
    plonky2_hw_acc_webgpu::context::set_gpu_yield_callback(None);
}

/// Set the GPU download chunk size in bytes. When non-zero, large buffer
/// downloads are split into chunks of at most this size. Pass 0 to disable.
#[wasm_bindgen]
pub fn set_gpu_download_chunk_size(bytes: u32) -> Result<(), JsValue> {
    plonky2_hw_acc_webgpu::context::set_download_chunk_size(bytes as u64)
        .map_err(|e| JsValue::from_str(&format!("set_download_chunk_size failed: {}", e)))
}

/// Returns the GPU adapter's max_buffer_size in bytes. Useful for diagnostics.
/// Returns 0 if the context is not initialized.
#[wasm_bindgen]
pub fn get_gpu_max_buffer_size() -> u64 {
    plonky2_hw_acc_webgpu::context::with_gpu_context(|ctx| ctx.max_buffer_size)
        .unwrap_or(0)
}

/// Returns the current download chunk size in bytes (0 = unlimited).
#[wasm_bindgen]
pub fn get_gpu_download_chunk_size() -> u64 {
    plonky2_hw_acc_webgpu::context::with_gpu_context(|ctx| ctx.download_chunk_size)
        .unwrap_or(0)
}

#[derive(Serialize)]
pub struct BenchmarkResult {
    pub circuit_build_ms: f64,
    pub inner_proof_1_ms: f64,
    pub inner_proof_2_ms: f64,
    pub recursive_circuit_build_ms: f64,
    pub recursive_proof_ms: f64,
    pub verification_ms: f64,
    pub total_ms: f64,
    pub success: bool,
    pub error: Option<String>,
}

#[derive(Serialize)]
pub struct CircuitInitResult {
    pub circuit_build_ms: f64,
    pub recursive_circuit_build_ms: f64,
    pub total_ms: f64,
    pub success: bool,
    pub error: Option<String>,
}

#[derive(Serialize)]
pub struct ProofResult {
    pub inner_proof_1_ms: f64,
    pub inner_proof_2_ms: f64,
    pub recursive_proof_ms: f64,
    pub verification_ms: f64,
    pub total_ms: f64,
    pub success: bool,
    pub error: Option<String>,
}

fn now_ms() -> f64 {
    js_sys::Date::now()
}

fn new_coset_gate_with_max_degree<F: RichField + Extendable<D>, const D: usize>(
    subgroup_bits: usize,
    max_degree: usize,
) -> CosetInterpolationGate<F, D> {
    let mut base_gate = CosetInterpolationGate::<F, D>::default();
    assert!(max_degree > 1, "need at least quadratic constraints");
    let n_points = 1 << subgroup_bits;
    let n_intermediates = (n_points - 2) / (max_degree - 1);
    let degree = (n_points - 2) / (n_intermediates + 1) + 2;
    let barycentric_weights = barycentric_weights(
        &F::two_adic_subgroup(subgroup_bits)
            .into_iter()
            .map(|x| (x, F::ZERO))
            .collect::<Vec<_>>(),
    );
    base_gate.subgroup_bits = subgroup_bits;
    base_gate.degree = degree;
    base_gate.barycentric_weights = barycentric_weights;
    base_gate
}

fn pad_circuit_degree<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    target_degree: usize,
) {
    while builder.num_gates() < (1u64 << (target_degree as u64)) as usize {
        builder.add_gate(NoopGate, vec![]);
    }
}

fn add_qed_type_c_common_gates<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
) {
    builder.add_gate_to_gate_set(GateRef::new(ConstantGate::new(builder.config.num_constants)));
    builder.add_gate_to_gate_set(GateRef::new(RandomAccessGate::new_from_config(
        &builder.config, 4,
    )));
    let coset_gate = GateRef::new(new_coset_gate_with_max_degree::<F, D>(4, 8));
    builder.add_gate_to_gate_set(coset_gate);
    builder.add_gate_to_gate_set(GateRef::new(PoseidonGate::<F, D>::new()));
    builder.add_gate_to_gate_set(GateRef::new(PoseidonMdsGate::<F, D>::new()));
    builder.add_gate_to_gate_set(GateRef::new(ReducingGate::<D>::new(43)));
    builder.add_gate_to_gate_set(GateRef::new(ReducingExtensionGate::<D>::new(32)));
    builder.add_gate_to_gate_set(GateRef::new(ArithmeticGate::new_from_config(&builder.config)));
    builder.add_gate_to_gate_set(GateRef::new(ArithmeticExtensionGate::new_from_config(
        &builder.config,
    )));
    builder.add_gate_to_gate_set(GateRef::new(MulExtensionGate::new_from_config(
        &builder.config,
    )));
    builder.add_gate_to_gate_set(GateRef::new(BaseSumGate::<2>::new_from_config::<F>(
        &builder.config,
    )));
}

struct DummyPsyTypeCCircuit<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    dummy_public_inputs: HashOutTarget,
    circuit_data: CircuitData<F, C, D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    DummyPsyTypeCCircuit<F, C, D>
where
    C::Hasher: AlgebraicHasher<F>,
{
    fn new(config: &CircuitConfig) -> Self {
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        let dummy_public_inputs = builder.add_virtual_hash();
        let mut res_linear = builder.hash_n_to_hash_no_pad::<C::Hasher>(
            vec![dummy_public_inputs.elements, dummy_public_inputs.elements].concat(),
        );
        for _ in 0..8000 {
            res_linear = builder.hash_n_to_hash_no_pad::<C::Hasher>(
                vec![res_linear.elements, dummy_public_inputs.elements].concat(),
            );
        }
        let zero = builder.zero();
        let is_first_zero = builder.is_equal(res_linear.elements[0], zero);
        let is_second_zero = builder.is_equal(res_linear.elements[1], zero);
        let are_both_zero = builder.and(is_first_zero, is_second_zero);
        let is_one_non_zero = builder.not(are_both_zero);
        let one = builder.one();
        builder.connect(is_one_non_zero.target, one);
        builder.register_public_inputs(&dummy_public_inputs.elements);
        add_qed_type_c_common_gates(&mut builder);
        pad_circuit_degree(&mut builder, 12);
        let circuit_data = builder.build::<C>();
        DummyPsyTypeCCircuit {
            dummy_public_inputs,
            circuit_data,
        }
    }

    async fn prove(
        &self,
        dummy_public_inputs: HashOut<F>,
    ) -> anyhow::Result<ProofWithPublicInputs<F, C, D>> {
        let mut pw = PartialWitness::new();
        pw.set_hash_target(self.dummy_public_inputs, dummy_public_inputs)?;
        let mut timing = TimingTree::new("prove dummy psy type c", Level::Debug);
        let proof = prove::<F, C, D>(
            &self.circuit_data.prover_only,
            &self.circuit_data.common,
            pw,
            &mut timing,
        ).await?;
        Ok(proof)
    }
}

struct DummyPsyTypeCRecursiveVerifierCircuit<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    proof_a_target: ProofWithPublicInputsTarget<D>,
    proof_a_verifier_data_target: VerifierCircuitTarget,
    proof_b_target: ProofWithPublicInputsTarget<D>,
    proof_b_verifier_data_target: VerifierCircuitTarget,
    circuit_data: CircuitData<F, C, D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    DummyPsyTypeCRecursiveVerifierCircuit<F, C, D>
where
    C::Hasher: AlgebraicHasher<F>,
{
    fn new(
        config: &CircuitConfig,
        child_cap_height: usize,
        child_common_data: &CommonCircuitData<F, D>,
    ) -> Self {
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        let proof_a_target = builder.add_virtual_proof_with_pis(child_common_data);
        let proof_a_verifier_data_target = builder.add_virtual_verifier_data(child_cap_height);
        let proof_b_target = builder.add_virtual_proof_with_pis(child_common_data);
        let proof_b_verifier_data_target = builder.add_virtual_verifier_data(child_cap_height);
        builder.verify_proof::<C>(
            &proof_a_target,
            &proof_a_verifier_data_target,
            child_common_data,
        );
        builder.verify_proof::<C>(
            &proof_b_target,
            &proof_b_verifier_data_target,
            child_common_data,
        );
        let public_inputs_output = builder.hash_n_to_hash_no_pad::<C::Hasher>(
            [
                proof_a_target.public_inputs.clone(),
                proof_b_target.public_inputs.clone(),
            ]
            .concat(),
        );
        builder.register_public_inputs(&public_inputs_output.elements);
        add_qed_type_c_common_gates(&mut builder);
        let circuit_data = builder.build::<C>();
        DummyPsyTypeCRecursiveVerifierCircuit {
            proof_a_target,
            proof_a_verifier_data_target,
            proof_b_target,
            proof_b_verifier_data_target,
            circuit_data,
        }
    }

    async fn prove(
        &self,
        left_proof: &ProofWithPublicInputs<F, C, D>,
        right_proof: &ProofWithPublicInputs<F, C, D>,
        verifier_data: &VerifierOnlyCircuitData<C, D>,
    ) -> anyhow::Result<ProofWithPublicInputs<F, C, D>> {
        let mut pw = PartialWitness::new();
        pw.set_verifier_data_target(&self.proof_a_verifier_data_target, verifier_data)?;
        pw.set_verifier_data_target(&self.proof_b_verifier_data_target, verifier_data)?;
        pw.set_proof_with_pis_target(&self.proof_a_target, left_proof)?;
        pw.set_proof_with_pis_target(&self.proof_b_target, right_proof)?;
        let mut timing = TimingTree::new("prove recursive", Level::Trace);
        let proof = prove::<F, C, D>(
            &self.circuit_data.prover_only,
            &self.circuit_data.common,
            pw,
            &mut timing,
        ).await?;
        Ok(proof)
    }
}

// Phase 1: Build both circuits, return them with timing
fn build_circuits_inner<C: GenericConfig<2, F = GoldilocksField>>() -> (
    DummyPsyTypeCCircuit<GoldilocksField, C, 2>,
    DummyPsyTypeCRecursiveVerifierCircuit<GoldilocksField, C, 2>,
    CircuitInitResult,
)
where
    C::Hasher: AlgebraicHasher<GoldilocksField>,
{
    let total_start = now_ms();
    let config = CircuitConfig::standard_recursion_config();

    let t = now_ms();
    let inner = DummyPsyTypeCCircuit::<GoldilocksField, C, 2>::new(&config);
    let circuit_build_ms = now_ms() - t;

    let t = now_ms();
    let recursive = DummyPsyTypeCRecursiveVerifierCircuit::<GoldilocksField, C, 2>::new(
        &config,
        inner.circuit_data.verifier_only.constants_sigmas_cap.height(),
        &inner.circuit_data.common,
    );
    let recursive_circuit_build_ms = now_ms() - t;

    let result = CircuitInitResult {
        circuit_build_ms,
        recursive_circuit_build_ms,
        total_ms: now_ms() - total_start,
        success: true,
        error: None,
    };

    (inner, recursive, result)
}

// Phase 2: Generate proofs and verify using pre-built circuits
async fn run_proofs_inner<C: GenericConfig<2, F = GoldilocksField>>(
    inner: &DummyPsyTypeCCircuit<GoldilocksField, C, 2>,
    recursive: &DummyPsyTypeCRecursiveVerifierCircuit<GoldilocksField, C, 2>,
) -> Result<ProofResult, String>
where
    C::Hasher: AlgebraicHasher<GoldilocksField>,
{
    let total_start = now_ms();

    let t = now_ms();
    let dummy_proof_1 = inner
        .prove(HashOut::rand())
        .await
        .map_err(|e| format!("Inner proof 1 failed: {}", e))?;
    let inner_proof_1_ms = now_ms() - t;

    let t = now_ms();
    let dummy_proof_2 = inner
        .prove(HashOut::rand())
        .await
        .map_err(|e| format!("Inner proof 2 failed: {}", e))?;
    let inner_proof_2_ms = now_ms() - t;

    let t = now_ms();
    let recursive_proof = recursive
        .prove(
            &dummy_proof_1,
            &dummy_proof_2,
            &inner.circuit_data.verifier_only,
        )
        .await
        .map_err(|e| format!("Recursive proof failed: {}", e))?;
    let recursive_proof_ms = now_ms() - t;

    let t = now_ms();
    inner
        .circuit_data
        .verify(dummy_proof_1)
        .map_err(|e| format!("Inner proof 1 verification failed: {}", e))?;
    inner
        .circuit_data
        .verify(dummy_proof_2)
        .map_err(|e| format!("Inner proof 2 verification failed: {}", e))?;
    recursive
        .circuit_data
        .verify(recursive_proof.clone())
        .map_err(|e| format!("Recursive proof verification failed: {}", e))?;

    for _ in 0..9 {
        verify_proof_borrowed(
            &recursive_proof,
            &recursive.circuit_data.verifier_only,
            &recursive.circuit_data.common,
        )
        .map_err(|e| format!("Repeated verification failed: {}", e))?;
    }
    let verification_ms = now_ms() - t;

    Ok(ProofResult {
        inner_proof_1_ms,
        inner_proof_2_ms,
        recursive_proof_ms,
        verification_ms,
        total_ms: now_ms() - total_start,
        success: true,
        error: None,
    })
}

// Combined benchmark (calls both phases, no caching)
async fn run_benchmark_inner<C: GenericConfig<2, F = GoldilocksField>>() -> Result<BenchmarkResult, String>
where
    C::Hasher: AlgebraicHasher<GoldilocksField>,
{
    let total_start = now_ms();

    let (inner, recursive, init_result) = build_circuits_inner::<C>();
    let proof_result = run_proofs_inner::<C>(&inner, &recursive).await?;

    Ok(BenchmarkResult {
        circuit_build_ms: init_result.circuit_build_ms,
        inner_proof_1_ms: proof_result.inner_proof_1_ms,
        inner_proof_2_ms: proof_result.inner_proof_2_ms,
        recursive_circuit_build_ms: init_result.recursive_circuit_build_ms,
        recursive_proof_ms: proof_result.recursive_proof_ms,
        verification_ms: proof_result.verification_ms,
        total_ms: now_ms() - total_start,
        success: true,
        error: None,
    })
}

fn error_result(e: String) -> BenchmarkResult {
    BenchmarkResult {
        circuit_build_ms: 0.0,
        inner_proof_1_ms: 0.0,
        inner_proof_2_ms: 0.0,
        recursive_circuit_build_ms: 0.0,
        recursive_proof_ms: 0.0,
        verification_ms: 0.0,
        total_ms: 0.0,
        success: false,
        error: Some(e),
    }
}

fn init_error_result(e: String) -> CircuitInitResult {
    CircuitInitResult {
        circuit_build_ms: 0.0,
        recursive_circuit_build_ms: 0.0,
        total_ms: 0.0,
        success: false,
        error: Some(e),
    }
}

fn proof_error_result(e: String) -> ProofResult {
    ProofResult {
        inner_proof_1_ms: 0.0,
        inner_proof_2_ms: 0.0,
        recursive_proof_ms: 0.0,
        verification_ms: 0.0,
        total_ms: 0.0,
        success: false,
        error: Some(e),
    }
}

// --- Legacy combined benchmarks ---

#[wasm_bindgen]
pub async fn run_cpu_benchmark() -> JsValue {
    let result = match run_benchmark_inner::<PoseidonGoldilocksConfig>().await {
        Ok(r) => r,
        Err(e) => error_result(e),
    };
    serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
}

#[wasm_bindgen]
pub async fn run_webgpu_benchmark() -> JsValue {
    let result = match run_benchmark_inner::<PoseidonGoldilocksWebGpuConfig>().await {
        Ok(r) => r,
        Err(e) => error_result(e),
    };
    serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
}

// --- Two-phase API: build circuits (cached in thread_local) ---

#[wasm_bindgen]
pub fn build_circuits_cpu() -> JsValue {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (inner, recursive, result) = build_circuits_inner::<PoseidonGoldilocksConfig>();
        CPU_CIRCUITS.with(|c| *c.borrow_mut() = Some((inner, recursive)));
        result
    }));
    match result {
        Ok(r) => serde_wasm_bindgen::to_value(&r).unwrap_or(JsValue::NULL),
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Unknown panic during circuit build".to_string()
            };
            serde_wasm_bindgen::to_value(&init_error_result(msg)).unwrap_or(JsValue::NULL)
        }
    }
}

#[wasm_bindgen]
pub fn build_circuits_webgpu() -> JsValue {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (inner, recursive, result) = build_circuits_inner::<PoseidonGoldilocksWebGpuConfig>();
        GPU_CIRCUITS.with(|c| *c.borrow_mut() = Some((inner, recursive)));
        result
    }));
    match result {
        Ok(r) => serde_wasm_bindgen::to_value(&r).unwrap_or(JsValue::NULL),
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Unknown panic during circuit build".to_string()
            };
            serde_wasm_bindgen::to_value(&init_error_result(msg)).unwrap_or(JsValue::NULL)
        }
    }
}

// --- Two-phase API: run proofs (using cached circuits) ---

#[wasm_bindgen]
pub async fn run_proofs_cpu() -> JsValue {
    let circuits = CPU_CIRCUITS.with(|c| c.borrow_mut().take());
    let Some((inner, recursive)) = circuits else {
        return serde_wasm_bindgen::to_value(&proof_error_result(
            "Circuits not initialized. Call build_circuits_cpu first.".into(),
        ))
        .unwrap_or(JsValue::NULL);
    };
    let result = match run_proofs_inner::<PoseidonGoldilocksConfig>(&inner, &recursive).await {
        Ok(r) => r,
        Err(e) => proof_error_result(e),
    };
    // Put circuits back for reuse
    CPU_CIRCUITS.with(|c| *c.borrow_mut() = Some((inner, recursive)));
    serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
}

#[wasm_bindgen]
pub async fn run_proofs_webgpu() -> JsValue {
    let circuits = GPU_CIRCUITS.with(|c| c.borrow_mut().take());
    let Some((inner, recursive)) = circuits else {
        return serde_wasm_bindgen::to_value(&proof_error_result(
            "Circuits not initialized. Call build_circuits_webgpu first.".into(),
        ))
        .unwrap_or(JsValue::NULL);
    };
    let result = match run_proofs_inner::<PoseidonGoldilocksWebGpuConfig>(&inner, &recursive).await {
        Ok(r) => r,
        Err(e) => proof_error_result(e),
    };
    // Put circuits back for reuse
    GPU_CIRCUITS.with(|c| *c.borrow_mut() = Some((inner, recursive)));
    serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
}
