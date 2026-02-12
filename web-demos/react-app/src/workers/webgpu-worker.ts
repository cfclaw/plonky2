import init, { init as initPanic, init_gpu, build_circuits_webgpu, run_proofs_webgpu } from 'plonky2-wasm-webgpu';

let wasmInitialized = false;

self.onmessage = async (e: MessageEvent) => {
  try {
    if (e.data.type === 'init' || e.data.type === 'run') {
      if (!wasmInitialized) {
        self.postMessage({ type: 'status', status: 'Initializing WASM module...' });
        await init();
        initPanic();
        self.postMessage({ type: 'status', status: 'Initializing WebGPU device...' });
        await init_gpu();
        wasmInitialized = true;
      }

      self.postMessage({ type: 'status', status: 'Building circuits...' });
      const initResult = build_circuits_webgpu();
      self.postMessage({ type: 'init_result', result: initResult });

      if (!initResult.success) return;

      if (e.data.type === 'run') {
        self.postMessage({ type: 'status', status: 'Generating and verifying proofs...' });
        const proofResult = await run_proofs_webgpu();
        self.postMessage({ type: 'proof_result', result: proofResult });
      }
    } else if (e.data.type === 'prove') {
      self.postMessage({ type: 'status', status: 'Generating and verifying proofs...' });
      const proofResult = await run_proofs_webgpu();
      self.postMessage({ type: 'proof_result', result: proofResult });
    }
  } catch (err: any) {
    self.postMessage({
      type: 'error',
      error: `Worker error: ${err.message || err}`,
    });
  }
};
