import init, { init as initPanic, init_gpu, run_webgpu_benchmark } from 'plonky2-wasm-webgpu';

self.onmessage = async (e: MessageEvent) => {
  if (e.data.type === 'run') {
    try {
      self.postMessage({ type: 'status', status: 'Initializing WASM module...' });
      await init();
      initPanic();

      self.postMessage({ type: 'status', status: 'Initializing WebGPU device...' });
      await init_gpu();

      self.postMessage({ type: 'status', status: 'Running WebGPU benchmark (this will take a while)...' });
      const result = run_webgpu_benchmark();

      self.postMessage({ type: 'result', result });
    } catch (err: any) {
      self.postMessage({
        type: 'result',
        result: {
          success: false,
          error: `Worker error: ${err.message || err}`,
          circuit_build_ms: 0,
          inner_proof_1_ms: 0,
          inner_proof_2_ms: 0,
          recursive_circuit_build_ms: 0,
          recursive_proof_ms: 0,
          verification_ms: 0,
          total_ms: 0,
        },
      });
    }
  }
};
