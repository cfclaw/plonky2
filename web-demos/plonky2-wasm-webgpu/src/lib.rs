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
use plonky2_field::types::Sample;
use plonky2_field::extension::Extendable;

use plonky2::hash::hash_types::{HashOut, HashOutTarget, RichField};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{
    CircuitConfig, CircuitData, CommonCircuitData, VerifierCircuitTarget, VerifierOnlyCircuitData,
};
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig};
use plonky2::plonk::proof::{ProofWithPublicInputs, ProofWithPublicInputsTarget};
use plonky2::plonk::prover::prove;
use plonky2::util::timing::TimingTree;
use plonky2_hw_acc_webgpu::prover::PoseidonGoldilocksWebGpuConfig;
use log::Level;

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

    fn prove(
        &self,
        dummy_public_inputs: HashOut<F>,
    ) -> anyhow::Result<ProofWithPublicInputs<F, C, D>> {
        let mut pw = PartialWitness::new();
        pw.set_hash_target(self.dummy_public_inputs, dummy_public_inputs)?;
        let mut timing = TimingTree::new("prove dummy psy type c (WebGPU/WASM)", Level::Debug);
        let proof = prove::<F, C, D>(
            &self.circuit_data.prover_only,
            &self.circuit_data.common,
            pw,
            &mut timing,
        )?;
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

    fn prove(
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
        let mut timing = TimingTree::new("prove recursive (WebGPU/WASM)", Level::Trace);
        let proof = prove::<F, C, D>(
            &self.circuit_data.prover_only,
            &self.circuit_data.common,
            pw,
            &mut timing,
        )?;
        Ok(proof)
    }
}

fn run_benchmark_inner() -> Result<BenchmarkResult, String> {
    const D: usize = 2;
    type C = PoseidonGoldilocksWebGpuConfig;
    type F = <C as GenericConfig<D>>::F;

    let total_start = now_ms();

    // 1. Build inner circuit
    let t = now_ms();
    let config = CircuitConfig::standard_recursion_config();
    let dummy_inner = DummyPsyTypeCCircuit::<F, C, D>::new(&config);
    let circuit_build_ms = now_ms() - t;

    // 2. Generate inner proof 1
    let t = now_ms();
    let dummy_proof_1 = dummy_inner
        .prove(HashOut::rand())
        .map_err(|e| format!("Inner proof 1 failed: {}", e))?;
    let inner_proof_1_ms = now_ms() - t;

    // 3. Generate inner proof 2
    let t = now_ms();
    let dummy_proof_2 = dummy_inner
        .prove(HashOut::rand())
        .map_err(|e| format!("Inner proof 2 failed: {}", e))?;
    let inner_proof_2_ms = now_ms() - t;

    // 4. Build recursive circuit
    let t = now_ms();
    let dummy_recursive = DummyPsyTypeCRecursiveVerifierCircuit::<F, C, D>::new(
        &config,
        dummy_inner
            .circuit_data
            .verifier_only
            .constants_sigmas_cap
            .height(),
        &dummy_inner.circuit_data.common,
    );
    let recursive_circuit_build_ms = now_ms() - t;

    // 5. Generate recursive proof
    let t = now_ms();
    let recursive_proof = dummy_recursive
        .prove(
            &dummy_proof_1,
            &dummy_proof_2,
            &dummy_inner.circuit_data.verifier_only,
        )
        .map_err(|e| format!("Recursive proof failed: {}", e))?;
    let recursive_proof_ms = now_ms() - t;

    // 6. Verify all proofs
    let t = now_ms();
    dummy_inner
        .circuit_data
        .verify(dummy_proof_1)
        .map_err(|e| format!("Inner proof 1 verification failed: {}", e))?;
    dummy_inner
        .circuit_data
        .verify(dummy_proof_2)
        .map_err(|e| format!("Inner proof 2 verification failed: {}", e))?;
    dummy_recursive
        .circuit_data
        .verify(recursive_proof.clone())
        .map_err(|e| format!("Recursive proof verification failed: {}", e))?;

    for _ in 0..9 {
        verify_proof_borrowed(
            &recursive_proof,
            &dummy_recursive.circuit_data.verifier_only,
            &dummy_recursive.circuit_data.common,
        )
        .map_err(|e| format!("Repeated verification failed: {}", e))?;
    }
    let verification_ms = now_ms() - t;

    let total_ms = now_ms() - total_start;

    Ok(BenchmarkResult {
        circuit_build_ms,
        inner_proof_1_ms,
        inner_proof_2_ms,
        recursive_circuit_build_ms,
        recursive_proof_ms,
        verification_ms,
        total_ms,
        success: true,
        error: None,
    })
}

#[wasm_bindgen]
pub fn run_webgpu_benchmark() -> JsValue {
    let result = match run_benchmark_inner() {
        Ok(r) => r,
        Err(e) => BenchmarkResult {
            circuit_build_ms: 0.0,
            inner_proof_1_ms: 0.0,
            inner_proof_2_ms: 0.0,
            recursive_circuit_build_ms: 0.0,
            recursive_proof_ms: 0.0,
            verification_ms: 0.0,
            total_ms: 0.0,
            success: false,
            error: Some(e),
        },
    };
    serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
}
